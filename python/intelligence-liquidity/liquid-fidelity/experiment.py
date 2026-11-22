"""Freeze occupancy models on development buildings, then score Graz once."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import torch
from torch import nn
from torch.nn import functional as F


SEEDS = (41, 42, 43)
ARMS = ("co2", "logistic", "gru", "liquid", "wide_liquid", "moe", "delta")
WIDTH = 16
NAMES = {
    "uci": "occupancy-fresh.zip",
    "aalborg": "dataset__2023_09_02__2023_10_04.csv",
    "calibrate": "aalto-1.csv",
    "select": "aalto-2.csv",
    "review": "aalto-ventilation-test.csv",
    "test": "graz.zip",
}
SHA = {
    "uci": "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a",
    "aalborg": "c9c4cd86cac80898fa5ea91dcab4ba3142298d4af85ae6a6481338d00054e11e",
}
MD5 = {
    "calibrate": "5a370c371c92d9a186dacd5cf3a0e0d2",
    "select": "c6e22d6b03d224c34b3d3882f0e93b6e",
    "review": "6bd29e069ccce6703cb4df3974741906",
    "test": "58c8fc3cac8723c7fa1a7e1e61e86a40",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source(data, key):
    raw = (data / NAMES[key]).read_bytes()
    if key in SHA:
        valid = sha(raw) == SHA[key]
    else:
        valid = hashlib.md5(raw).hexdigest() == MD5[key]
    if not valid:
        raise ValueError(f"Source checksum mismatch for {key}")
    return raw


def rows(raw, delimiter=","):
    return csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=delimiter)


def timestamp(value):
    time = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return int(time.replace(tzinfo=time.tzinfo or timezone.utc).timestamp())


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def reading(time, co2, temp, occupancy):
    count = number(occupancy)
    label = float(count > 0) if math.isfinite(count) and count >= 0 else math.nan
    return timestamp(time), number(co2), number(temp), label


def training(raw):
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw["uci"])) as archive:
        names = [name for name in archive.namelist()
                 if Path(name).name == "datatraining.txt"]
        if len(names) != 1:
            raise ValueError("Cannot identify UCI training file")
        parsed = csv.reader(io.StringIO(archive.read(names[0]).decode("utf-8-sig")))
        columns = next(parsed)
        uci = [dict(zip(columns, row[1:], strict=True)) for row in parsed]
    result["UCI"] = [reading(row["date"], row["CO2"], row["Temperature"], row["Occupancy"])
                     for row in uci]
    office = list(rows(raw["aalborg"], ";"))
    for room in ("Room" + name for name in "ABCDEF"):
        result["Aalborg-" + room] = [
            reading(row["timestamp"], row[f"{room}:Sensor__CO2"],
                    row[f"{room}:Sensor__room_temperature"], row[f"{room}:People__amount"])
            for row in office
        ]
    return result


def aalto(raw):
    return [reading(row["time"], row["Room CO2 (ppm)"], row["Temperature (°C)"],
                    row["Occupant count"]) for row in rows(raw)]


def windows(records):
    """Eight causal samples spanning 35 minutes. Preserve exact endpoint labels."""
    ordered = sorted(records)
    times = [row[0] for row in ordered]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Duplicate timestamps within recording")
    features, clocks, labels, endpoints = [], [], [], []
    cursor = [0] * 8
    segment = 0
    for end, (now, co2, temp, label) in enumerate(ordered):
        if end and now - times[end - 1] > 450:
            segment = end
        if not math.isfinite(co2) or not math.isfinite(temp):
            segment = end + 1
            continue
        if segment > end or now - times[segment] < 2100:
            continue
        indices = []
        for step in range(8):
            target = now - (7 - step) * 300
            index = max(segment, cursor[step])
            while index < end and times[index + 1] <= target:
                index += 1
            cursor[step] = index
            if not 0 <= target - times[index] < 300:
                break
            indices.append(index)
        if len(indices) != 8 or len(set(indices)) != 8 or not math.isfinite(label):
            continue
        features.append([[ordered[i][1], ordered[i][2]] for i in indices])
        clocks.append([5.0] + [(times[b] - times[a]) / 60 for a, b in zip(indices, indices[1:])])
        labels.append(label)
        endpoints.append(now)
    if not features:
        raise ValueError("No labeled causal windows")
    return (*map(torch.tensor, (features, clocks, labels)), endpoints)


def inputs(x):
    # Fixed physical units. First-reading offsets let the same model see change
    # in a colder room without treating the absolute temperatures as equal.
    levels = (x - x.new_tensor((500, 22))) / x.new_tensor((200, 3))
    change = (x - x[:, :1]) / x.new_tensor((100, 1))
    previous = torch.cat((x[:, :1], x[:, :-1]), 1)
    slope = (x - previous) / x.new_tensor((100, 1))
    return torch.cat((levels, change, slope), -1).clamp(-10, 10)


class LiquidCell(nn.Module):
    def __init__(self, n_in, width):
        super().__init__()
        self.drive = nn.Linear(n_in + width, 2 * width)

    def forward(self, x, state, minutes):
        target, rate = self.drive(torch.cat((x, state), -1)).chunk(2, -1)
        change = -torch.expm1(-(F.softplus(rate) + 1e-4) * minutes[:, None] / 5)
        return state + change * (target.tanh() - state)


class DeltaMemory(nn.Module):
    def __init__(self, width, slots=8):
        super().__init__()
        self.slots = slots
        self.project = nn.Linear(width, slots * 3)
        self.write = nn.Linear(width, 1)
        self.decay = nn.Parameter(torch.full((slots,), -2.0))
        self.read = nn.Linear(slots, width, bias=False)

    def forward(self, x, prior, minutes):
        key, query, value = self.project(x).chunk(3, -1)
        key, query = F.normalize(key, dim=-1), F.normalize(query, dim=-1)
        elapsed = minutes[:, None] / 5
        old = prior * torch.exp(-F.softplus(self.decay) * elapsed)[:, :, None]
        residual = value.tanh() - torch.bmm(key[:, None], old).squeeze(1)
        gate = self.write(x).sigmoid() * -torch.expm1(-elapsed)
        board = old + key[:, :, None] * (residual * gate)[:, None, :]
        return board, self.read(torch.bmm(query[:, None], board).squeeze(1))


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.width = 32 if arm == "wide_liquid" else WIDTH
        self.mixture = arm in ("moe", "delta")
        if arm in ("co2", "logistic"):
            self.readout = nn.Linear(1 if arm == "co2" else 12, 1)
            return
        self.core = (nn.GRUCell(6, self.width) if arm == "gru" else LiquidCell(6, self.width))
        self.readout = nn.Linear(self.width + 6, 1)
        if self.mixture:
            self.experts = nn.ModuleList(LiquidCell(self.width + n, self.width)
                                         for n in (2, 4, 6))
            self.router = nn.Linear(self.width + 6, 3)
            self.expert_heads = nn.ModuleList(nn.Linear(self.width, 1) for _ in range(3))
        self.memory = DeltaMemory(self.width) if arm == "delta" else None

    def forward(self, x, clock):
        f = inputs(x)
        latest = f[:, -1]
        if self.arm == "co2":
            return self.readout(latest[:, :1]).squeeze(-1), None, None
        if self.arm == "logistic":
            return self.readout(torch.cat((latest, f.mean(1)), -1)).squeeze(-1), None, None
        state = x.new_zeros(len(x), self.width)
        if self.mixture:
            experts = [torch.zeros_like(state) for _ in self.experts]
            route = []
            if self.memory is not None:
                board = x.new_zeros(len(x), self.memory.slots, self.memory.slots)
        for sample, minutes in zip(f.unbind(1), clock.unbind(1), strict=True):
            if self.arm == "gru":
                state = self.core(sample, state)
            else:
                state = self.core(sample, state, minutes)
            if self.mixture:
                views = (sample[:, :2], sample[:, 2:], sample)
                weights = self.router(torch.cat((state, sample), -1)).softmax(-1)
                experts = [
                    cell(torch.cat((state, view), -1), prior, minutes)
                    for cell, view, prior in zip(self.experts, views, experts, strict=True)
                ]
                combined = state + (weights[:, :, None] * torch.stack(experts, 1)).sum(1)
                if self.memory is not None:
                    board, retrieved = self.memory(combined, board, minutes)
                    combined = combined + retrieved
                route.append(weights)
        if self.mixture:
            head = torch.cat([layer(prior) for layer, prior
                              in zip(self.expert_heads, experts, strict=True)], -1)
            logit = self.readout(torch.cat((combined, latest), -1)).squeeze(-1)
            logit = logit + (head * weights).sum(-1)
            return logit, torch.stack(route, 1), head
        return self.readout(torch.cat((state, latest), -1)).squeeze(-1), None, None


def train_loss(model, x, time, y):
    logit, route, head = model(x, time)
    loss = F.binary_cross_entropy_with_logits(logit, y)
    if route is not None:
        # Keep every expert learning. A shared load term discourages one expert
        # from monopolizing the mixture without prescribing individual routes.
        balance = ((route.mean((0, 1)) - 1 / 3) ** 2).sum()
        expert_loss = F.binary_cross_entropy_with_logits(head, y[:, None].expand_as(head))
        loss = loss + 2 * balance + 0.2 * expert_loss
    return loss


@torch.no_grad()
def predictions(model, dataset):
    x, time, _, _ = dataset
    batches = [model(a, b) for a, b in zip(x.split(1024), time.split(1024), strict=True)]
    return tuple(torch.cat(column) if column[0] is not None else None
                 for column in zip(*batches, strict=True))


def calibrate(raw, y):
    """Fit one positive scale and one offset to probabilities on Aalto-1."""
    log_scale = torch.zeros((), requires_grad=True)
    offset = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS((log_scale, offset), max_iter=25, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        z = raw * log_scale.clamp(-3, 3).exp() + offset
        objective = F.binary_cross_entropy_with_logits(z, y) + 0.005 * log_scale.square()
        objective.backward()
        return objective

    opt.step(loss)
    return float(log_scale.detach().clamp(-3, 3).exp()), float(offset.detach())


def probability(raw, calibration):
    scale, offset = calibration
    return (raw * scale + offset).sigmoid()


def rates(prob, y, cutoff):
    pred = prob >= cutoff
    present = y.bool()
    return float(pred[present].float().mean()), float((~pred[~present]).float().mean())


def threshold(prob, y):
    candidates = [i / 100 for i in range(5, 96)]
    return max(candidates, key=lambda p: (sum(rates(prob, y, p)), -abs(p - 0.5)))


def score(prob, y, cutoff, decision=None):
    present = y.bool()
    pred = prob >= cutoff if decision is None else decision
    tpr, tnr = rates(pred.float(), y, 0.5)
    # Average ranks for ties, including actual quantized sensor readings.
    ordered, idx = prob.sort()
    _, group, counts = ordered.unique_consecutive(return_inverse=True, return_counts=True)
    average_ranks = counts.cumsum(0).double() - (counts - 1) / 2
    p = int(present.sum())
    n = len(present) - p
    auc = float((average_ranks[group][present[idx]].sum() - p * (p + 1) / 2) / (p * n))
    calibration_error = 0.0
    for ids in torch.bucketize(prob, torch.linspace(0.1, 0.9, 9)).unique():
        mask = torch.bucketize(prob, torch.linspace(0.1, 0.9, 9)) == ids
        calibration_error += float(mask.float().mean() * (prob[mask].mean() - y[mask].mean()).abs())
    return {
        "rows": len(y), "occupied": p, "accuracy": float((pred == present).float().mean()),
        "balanced_accuracy": (tpr + tnr) / 2, "occupied_recall": tpr, "empty_recall": tnr,
        "auroc": auc, "brier": float((prob - y).square().mean()), "calibration_error": calibration_error,
        "false_empty": int((present & ~pred).sum()), "false_occupied": int((~present & pred).sum()),
    }


def revise(base, proposal, verifier, cutoff, margin, gain_limit):
    """Score changing a decision using an independent sensor-only verifier."""
    first = base >= cutoff
    second = proposal >= cutoff
    # Reward +1 for a correct decision, -1 for a false alarm, -2 for a miss.
    yes_value = 2 * verifier - 1
    no_value = 1 - 3 * verifier
    gain = torch.where(second, yes_value - no_value, no_value - yes_value)
    attempted = (base - cutoff).abs() <= margin
    changed = attempted & (first != second) & (gain >= gain_limit)
    return torch.where(changed, proposal, base), attempted, changed


def review_policy(base, proposal, verifier, y, cutoff):
    first = base >= cutoff
    candidates = []
    for margin in (0.03, 0.06, 0.10, 0.15, 0.2):
        for gain in (0.05, 0.15, 0.3, 0.6):
            out, attempted, changed = revise(base, proposal, verifier, cutoff, margin, gain)
            second = out >= cutoff
            if not 0 < int(changed.sum()) or float(attempted.float().mean()) > 0.25:
                continue
            old_cost = ((first != y.bool()) * (1 + y)).sum()
            new_cost = ((second != y.bool()) * (1 + y)).sum()
            delta = float(old_cost - new_cost)
            balanced = sum(rates(out, y, cutoff)) - sum(rates(base, y, cutoff))
            corrected = int((changed & (second == y.bool())).sum())
            broken = int(changed.sum()) - corrected
            if delta > 0 and balanced >= 0 and corrected > broken:
                candidates.append(((balanced, delta, -int(attempted.sum())), {
                    "margin": margin, "gain_limit": gain, "attempted": int(attempted.sum()),
                    "corrected": corrected, "broken": broken, "saved_cost": delta,
                }))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def alternate(expert_logits):
    # The median asks the experts for a majority recommendation.
    return expert_logits.median(-1).values


def fit(data, out, steps):
    raw = {key: source(data, key) for key in NAMES if key != "test"}
    train = {k: windows(v) for k, v in training(raw).items()}
    tune = {key: windows(aalto(raw[key])) for key in ("calibrate", "select", "review")}
    for name, bundle in list(train.items()) + list(tune.items()):
        y = bundle[2]
        if not 0 < y.sum() < len(y):
            raise ValueError(f"Both labels required for {name}")
    for left, right in (("calibrate", "select"), ("calibrate", "review"), ("select", "review")):
        if set(tune[left][3]) & set(tune[right][3]):
            raise ValueError(f"Development recordings overlap: {left} and {right}")
    # Give both training sites equal opportunity, then round-robin Aalborg rooms.
    train_keys = tuple(train)
    x, t, y = (torch.cat([train[key][col] for key in train_keys]) for col in range(3))
    offsets = [0]
    for key in train_keys:
        offsets.append(offsets[-1] + len(train[key][2]))
    out.mkdir()
    report = {
        "seed_count": len(SEEDS), "steps": steps, "pytorch": str(torch.__version__),
        "source_sha256": {key: sha(value) for key, value in raw.items()},
        "code_sha256": sha(Path(__file__).read_bytes()), "models": {},
        "datasets": {
            key: {"rows": len(ds[2]), "occupied": int(ds[2].sum())}
            for key, ds in {**train, **tune}.items()
        },
    }
    for seed in SEEDS:
        generator = torch.Generator().manual_seed(seed)
        plan = [
            torch.cat((
                torch.randint(0, offsets[1], (64,), generator=generator),
                *(torch.randint(a, b, (10,), generator=generator)
                  for a, b in zip(offsets[1:-1], offsets[2:], strict=True)),
            ))
            for _ in range(steps)
        ]
        for arm in ARMS:
            torch.manual_seed(seed)
            model = Model(arm)
            opt = torch.optim.AdamW(model.parameters(), lr=0.005)
            best = None
            for step, idx in enumerate(plan, 1):
                opt.zero_grad()
                loss = train_loss(model, x[idx], t[idx], y[idx])
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                opt.step()
                if step in (steps // 3, 2 * steps // 3, steps):
                    z = predictions(model, tune["calibrate"])[0]
                    calibration = calibrate(z, tune["calibrate"][2])
                    dev = probability(predictions(model, tune["select"])[0], calibration)
                    dev_y = tune["select"][2]
                    cutoff = threshold(dev, dev_y)
                    result = score(dev, dev_y, cutoff)
                    rank = (result["auroc"], -result["brier"])
                    if best is None or rank > best[0]:
                        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                        best = rank, state, calibration, cutoff, step, result
            _, state, calibration, cutoff, step, result = best
            model.load_state_dict(state)
            model_id = f"{arm}-{seed}"
            weights = out / f"{model_id}.pt"
            torch.save(state, weights)
            entry = {
                "arm": arm, "seed": seed, "weights_sha256": sha(weights.read_bytes()),
                "calibration": calibration, "cutoff": cutoff, "step": step,
                "development": result, "parameters": sum(p.numel() for p in model.parameters()),
            }
            report["models"][model_id] = entry
            if arm == "logistic":
                verifier = model
                verifier_cal = calibration
            if arm == "delta":
                z_cal, _, experts = predictions(model, tune["calibrate"])
                entry["alternate_calibration"] = calibrate(alternate(experts), tune["calibrate"][2])
                r, routes, experts = predictions(model, tune["review"])
                entry["review_routes"] = routes.mean((0, 1)).tolist()
                alt = probability(alternate(experts), entry["alternate_calibration"])
                checker = probability(predictions(verifier, tune["review"])[0], verifier_cal)
                entry["policy"] = review_policy(
                    probability(r, calibration), alt, checker, tune["review"][2], cutoff)
            print(model_id, "step", step, "dev balanced", round(result["balanced_accuracy"], 4),
                  "review", entry.get("policy"), flush=True)
    manifest = out / "frozen.json"
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Frozen SHA-256", sha(manifest.read_bytes()), flush=True)


def restore(out, expected):
    raw = (out / "frozen.json").read_bytes()
    if sha(raw) != expected:
        raise ValueError("Frozen manifest checksum mismatch")
    manifest = json.loads(raw)
    if manifest["code_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("Experiment code changed after freeze")
    models = {}
    for key, entry in manifest["models"].items():
        weights = (out / f"{key}.pt").read_bytes()
        if sha(weights) != entry["weights_sha256"]:
            raise ValueError(f"Frozen weights changed: {key}")
        model = Model(entry["arm"])
        model.load_state_dict(torch.load(io.BytesIO(weights), weights_only=True))
        models[key] = model.eval()
    return manifest, models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=450)
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    torch.set_num_threads(1)
    fit(args.data, args.out, args.steps)
