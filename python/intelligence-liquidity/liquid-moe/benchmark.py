"""CPU benchmark. The score command is the only Aalborg reader."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from time import perf_counter
import zipfile

import torch
from torch.nn import functional as F

import model

ARMS = ("current", "gru", "liquid", "moe", "moe_delta")
UCI = "occupancy-fresh.zip"
EXTERNAL = "dataset__2023_09_02__2023_10_04.csv"
DIGESTS = {
    UCI: "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a",
    EXTERNAL: "c9c4cd86cac80898fa5ea91dcab4ba3142298d4af85ae6a6481338d00054e11e",
}
UCI_FILES = ("datatraining.txt", "datatest.txt", "datatest2.txt")
ROOMS = tuple("Room" + letter for letter in "ABCDEF")
BATCH = 64


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"SHA-256 mismatch: {path}")
    return data


def sources():
    folder = Path(__file__).resolve().parent
    return {name: digest((folder / name).read_bytes()) for name in
            ("benchmark.py", "model.py", "liquid-moe-protocol.md")}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def timestamp(value):
    point = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return point.replace(tzinfo=point.tzinfo or timezone.utc).timestamp()


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return math.nan


def csv_rows(data, delimiter):
    return csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""),
                          delimiter=delimiter)


def uci_parts(data, include_tests=False):
    archive = verified(data / UCI, DIGESTS[UCI])
    result = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in UCI_FILES if include_tests else UCI_FILES[:1]:
            matches = [member for member in bundle.namelist()
                       if Path(member).name == name and not member.endswith("/")]
            require(len(matches) == 1, f"Expected exactly one archive member {name}")
            with bundle.open(matches[0]) as member:
                reader = csv.reader(io.TextIOWrapper(member, encoding="utf-8-sig"))
                columns = next(reader)
                rows = []
                for row in reader:
                    require(len(row) == len(columns) + 1, "Unexpected UCI record shape")
                    rows.append(dict(zip(columns, row[1:], strict=True)))
            if name == "datatraining.txt":
                require(len(rows) > 6500, "UCI training split is incomplete")
                splits = {"train": rows[:6500], "validation": rows[6500:]}
            else:
                splits = {name: rows}
            for split, subset in splits.items():
                result[split] = [
                    (timestamp(row["date"]), number(row["CO2"]),
                     number(row["Temperature"]), number(row["Occupancy"]))
                    for row in subset
                ]
    return result


def external_rooms(data):
    reader = csv_rows(verified(data / EXTERNAL, DIGESTS[EXTERNAL]), ";")
    headers = reader.fieldnames or []
    canonical = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in headers}
    time_key = canonical["timestamp"]
    fields = ("sensorco2", "sensorroomtemperature", "peopleamount")
    records = {room: [] for room in ROOMS}
    if all(field in canonical for field in fields):
        room_key = next((canonical[key] for key in ("room", "roomid", "roomname")
                         if key in canonical), None)
        require(room_key is not None, "Missing Aalborg room column")
        keys = [canonical[field] for field in fields]
        for row in reader:
            room = row[room_key].strip()
            if room in records:
                records[room].append(external_row(row, time_key, keys))
    else:
        by_room = {}
        for room in ROOMS:
            keys = []
            for field in fields:
                matches = [name for key, name in canonical.items()
                           if key in (room.lower() + field, field + room.lower())]
                require(len(matches) == 1, f"Missing or ambiguous {field} for {room}")
                keys.append(matches[0])
            by_room[room] = keys
        for row in reader:
            for room, keys in by_room.items():
                records[room].append(external_row(row, time_key, keys))
    return records


def external_row(row, time_key, keys):
    co2, temperature, people = (number(row[key]) for key in keys)
    label = float(people > 0) if math.isfinite(people) and people >= 0 else math.nan
    return timestamp(row[time_key]), co2, temperature, label


def windows(rows):
    """At each observed endpoint use trailing offsets 35, 30, ..., 0 minutes."""
    rows = sorted(rows, key=lambda row: row[0])
    require(all(a[0] < b[0] for a, b in zip(rows, rows[1:])),
            "Duplicate timestamps within a room or UCI partition")
    feature_windows, clocks, labels = [], [], []
    cursors = [0] * 8
    segment = 0
    for end, (now, co2, temperature, label) in enumerate(rows):
        if not (math.isfinite(co2) and math.isfinite(temperature)):
            segment = end + 1
            continue
        if end and now - rows[end - 1][0] > 450:
            segment = end
        if segment > end or now - rows[segment][0] < 2100:
            continue
        indices = []
        for step in range(8):
            target = now - (7 - step) * 300
            cursor = max(cursors[step], segment)
            while cursor < end and rows[cursor + 1][0] <= target:
                cursor += 1
            cursors[step] = cursor
            if not 0 <= target - rows[cursor][0] < 300:
                break
            indices.append(cursor)
        if len(indices) != 8 or len(set(indices)) != 8:
            continue
        selected = [rows[index] for index in indices]
        feature_windows.append([[r[1], r[2]] for r in selected])
        clocks.append([0.0] + [(b[0] - a[0]) / 60 for a, b in
                              zip(selected, selected[1:])])
        labels.append(label)
    require(bool(labels), "No usable causal windows")
    return tuple(torch.tensor(values, dtype=torch.float32)
                 for values in (feature_windows, clocks, labels))


def normalize(dataset, mean, scale):
    x, clock, y = dataset
    return (x - mean) / scale, clock, y


@torch.no_grad()
def predict(net, dataset, cutoff=None):
    logits, masks, attempts, counts = [], [], [], {}
    x, clock, _ = dataset
    start = perf_counter()
    for start_index in range(0, len(x), BATCH):
        xs = x[start_index:start_index + BATCH]
        ts = clock[start_index:start_index + BATCH]
        out = (net(xs, ts) if cutoff is None else
               model.review(net, xs, ts, torch.sigmoid(torch.tensor(cutoff)).item()))
        score = out["logits"]
        logits.append(score)
        accepted = out.get("accepted", torch.ones_like(score, dtype=torch.bool))
        attempted = out.get("attempted", torch.zeros_like(score, dtype=torch.bool))
        masks.append(accepted.bool())
        attempts.append(attempted.bool())
        trace = out.get("trace", {})
        for key, value in (out | (trace if isinstance(trace, dict) else {})).items():
            if not isinstance(value, torch.Tensor):
                continue
            if "expert" in key and not value.is_floating_point() and value.dtype != torch.bool:
                for expert, count in zip(*torch.unique(value, return_counts=True)):
                    bucket = counts.setdefault(key, {})
                    index = str(expert.item())
                    bucket[index] = bucket.get(index, 0) + count.item()
            if any(term in key for term in ("committed", "rejected", "revised", "calls")):
                bucket = counts.setdefault(key, {})
                bucket["total"] = bucket.get("total", 0) + value.sum().item()
    seconds = perf_counter() - start
    return torch.cat(logits), torch.cat(masks), torch.cat(attempts), seconds, counts


def cutoff_for(logits, y):
    candidates = [i / 4 for i in range(-32, 33)]
    positive = y == 1
    require(bool(positive.any() and (~positive).any()), "Validation needs both classes")
    best = None
    for threshold in candidates:
        decision = logits >= threshold
        balanced = (decision[positive].double().mean() +
                    (~decision[~positive]).double().mean()).item() / 2
        rank = (balanced, -abs(threshold), -threshold)
        if best is None or rank > best[0]:
            best = rank, threshold
    return best[1]


def batch_plan(seed, steps, population):
    generator = torch.Generator().manual_seed(1000003 + seed)
    return torch.randint(population, (steps, BATCH), generator=generator)


def fit(data, out, steps, seeds):
    require(steps > 0, "Steps must be positive")
    require(not (out / "fit.json").exists() and not (out / "frozen.json").exists(),
            "Use a fresh output directory")
    src = sources()
    parts = uci_parts(data)
    train, validation = (windows(parts[key]) for key in ("train", "validation"))
    for dataset in (train, validation):
        require(bool(((dataset[2] == 0) | (dataset[2] == 1)).all()), "Invalid UCI label")
    mean = train[0].mean(dim=(0, 1))
    scale = train[0].std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    train, validation = (normalize(dataset, mean, scale) for dataset in (train, validation))
    positives = train[2].sum()
    require(0 < positives < len(train[2]), "Training needs both classes")
    pos_weight = (len(train[2]) - positives) / positives
    receipt = {"schema": 1, "sources": src, "digests": DIGESTS, "torch": str(torch.__version__),
               "mean": mean.tolist(), "scale": scale.tolist(), "steps": steps,
               "seeds": seeds, "batch_size": BATCH, "train_windows": len(train[2]),
               "validation_windows": len(validation[2]), "pos_weight": pos_weight.item(),
               "optimizer": {"name": "Adam", "lr": 0.003},
               "batch_plans": {}, "runs": []}
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        plan = batch_plan(seed, steps, len(train[2]))
        receipt["batch_plans"][str(seed)] = digest(json.dumps(plan.tolist()).encode())
        for arm in ARMS:
            torch.manual_seed(seed)
            net = model.build(arm)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
            snapshot_steps = {step for step in range(24, steps + 1, 24)} | {steps}
            best = None
            start = perf_counter()
            for step, indices in enumerate(plan, 1):
                net.train()
                optimizer.zero_grad()
                x, clock, y = (values[indices] for values in train)
                loss = model.training_loss(net, x, clock, y, pos_weight)
                require(bool(torch.isfinite(loss)), f"Nonfinite loss: {arm}/{seed}/{step}")
                loss.backward()
                optimizer.step()
                if step in snapshot_steps:
                    net.eval()
                    logits = predict(net, validation)[0]
                    val_loss = F.binary_cross_entropy_with_logits(logits, validation[2]).item()
                    require(math.isfinite(val_loss), "Nonfinite validation loss")
                    if best is None or val_loss < best[0]:
                        cutoff = cutoff_for(logits, validation[2])
                        state = {key: value.detach().clone()
                                 for key, value in net.state_dict().items()}
                        best = val_loss, step, cutoff, state
            val_loss, snapshot, cutoff, state = best
            filename = f"{arm}-{seed}.pt"
            with (out / filename).open("xb") as stream:
                torch.save(state, stream)
            receipt["runs"].append({"arm": arm, "seed": seed, "file": filename,
                                    "sha256": digest((out / filename).read_bytes()),
                                    "snapshot": snapshot, "cutoff": cutoff,
                                    "validation_loss": val_loss,
                                    "parameters": sum(p.numel() for p in net.parameters()),
                                    "fit_seconds": perf_counter() - start})
    require(sources() == src, "Source changed during fitting")
    write_json(out / "fit.json", receipt)


def audit(out, data):
    fit_bytes = (out / "fit.json").read_bytes()
    receipt = json.loads(fit_bytes)
    require(receipt["schema"] == 1 and receipt["sources"] == sources(), "Source changed")
    require(receipt["digests"] == DIGESTS, "Data declaration changed")
    verified(data / UCI, DIGESTS[UCI])
    expected = [(seed, arm) for seed in receipt["seeds"] for arm in ARMS]
    require([(row["seed"], row["arm"]) for row in receipt["runs"]] == expected,
            "Incomplete or reordered fit")
    checkpoints = {}
    for row in receipt["runs"]:
        name = f'{row["arm"]}-{row["seed"]}.pt'
        require(row["file"] == name, "Unexpected checkpoint name")
        checkpoints[name] = verified(out / name, row["sha256"])
    manifest = {"fit_sha256": digest(fit_bytes), "sources": receipt["sources"],
                "digests": DIGESTS,
                "checkpoints": {name: digest(data) for name, data in checkpoints.items()}}
    return receipt, manifest, checkpoints


def auc(logits, y):
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = logits.argsort()
    _, counts = torch.unique_consecutive(logits[order], return_counts=True)
    ends = counts.cumsum(0)
    ranks = ((ends.double() * 2 - counts + 1) / 2).repeat_interleave(counts)
    return (ranks[y[order]].sum().item() - positives * (positives + 1) / 2) / (
        positives * negatives)


def metrics(scores, mask, attempted, labels, cutoff):
    labeled = torch.isfinite(labels)
    valid = mask & labeled & torch.isfinite(scores)
    y = labels == 1
    decision = scores >= cutoff
    tp = int((valid & decision & y).sum())
    tn = int((valid & ~decision & ~y).sum())
    fp = int((valid & decision & ~y).sum())
    fn = int((valid & ~decision & y).sum())
    total, covered = int(labeled.sum()), tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    attempt_count = int((attempted & labeled).sum())
    payoff = tp + tn - 2 * fn - fp - 0.20 * (total - covered) - 0.06 * attempt_count
    n_positive = int((labeled & y).sum())
    n_negative = total - n_positive
    return {"all_window_balanced_accuracy": (tp / n_positive + tn / n_negative) / 2
            if n_positive and n_negative else None,
            "windows": len(labels), "labeled": total, "covered": covered,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "attempted": int(attempted.sum()),
            "coverage": covered / total if total else None,
            "accuracy": (tp + tn) / covered if covered else None,
            "all_window_accuracy": (tp + tn) / total if total else None,
            "balanced_accuracy": (sensitivity + specificity) / 2
            if sensitivity is not None and specificity is not None else None,
            "sensitivity": sensitivity, "specificity": specificity,
            "auroc": auc(scores[valid], y[valid]), "payoff": payoff,
            "mean_payoff": payoff / total if total else None}


def score(data, out, frozen_sha256):
    require(not (out / "scores.json").exists(), "Use a fresh replay directory for scoring")
    frozen = verified(out / "frozen.json", frozen_sha256)
    receipt, manifest, checkpoints = audit(out, data)
    require(json.loads(frozen) == manifest, "Frozen manifest does not match fit")
    parts = uci_parts(data, include_tests=True)
    mean = torch.tensor(receipt["mean"])
    scale = torch.tensor(receipt["scale"])
    groups = {
        "aalborg": external_rooms(data),
        "uci_original_tests": {name: {name: parts[name]} for name in UCI_FILES[1:]},
    }
    datasets = {
        "aalborg": {room: normalize(windows(rows), mean, scale)
                    for room, rows in groups["aalborg"].items()},
        **{name: {name: normalize(windows(rooms[name]), mean, scale)}
           for name, rooms in groups["uci_original_tests"].items()},
    }
    records = []
    for row in receipt["runs"]:
        net = model.build(row["arm"])
        net.load_state_dict(torch.load(io.BytesIO(checkpoints[row["file"]]),
                                       map_location="cpu", weights_only=True))
        net.eval()
        for site, rooms in datasets.items():
            ordinary = {room: predict(net, ds) for room, ds in rooms.items()}
            variants = [(row["arm"], ordinary)]
            if row["arm"] == "moe_delta":
                reviewed = {room: predict(net, ds, row["cutoff"]) for room, ds in rooms.items()}
                full = {}
                for room, (logits, accepted, attempted, seconds, counts) in reviewed.items():
                    base = ordinary[room]
                    full[room] = (torch.where(accepted, logits, base[0]),
                                  torch.ones_like(accepted), attempted, seconds + base[3], counts)
                variants.extend([("moe_delta_review", reviewed),
                                 ("moe_delta_review_full", full)])
            for variant, predictions in variants:
                room_metrics = {}
                for room, (logits, accepted, attempted, seconds, counts) in predictions.items():
                    values = metrics(logits, accepted, attempted, rooms[room][2], row["cutoff"])
                    room_metrics[room] = values | {"seconds": seconds, "route_counts": counts}
                columns = [torch.cat([p[index] for p in predictions.values()])
                           for index in range(3)]
                labels = torch.cat([ds[2] for ds in rooms.values()])
                pooled = metrics(*columns, labels, row["cutoff"])
                pooled["seconds"] = sum(values["seconds"] for values in room_metrics.values())
                pooled_counts = {}
                for values in room_metrics.values():
                    for key, counts in values["route_counts"].items():
                        bucket = pooled_counts.setdefault(key, {})
                        for route, count in counts.items():
                            bucket[route] = bucket.get(route, 0) + count
                pooled["route_counts"] = pooled_counts
                records.append({"seed": row["seed"], "arm": variant, "dataset": site,
                                "parameters": row["parameters"], "fit_seconds": row["fit_seconds"],
                                "cutoff": row["cutoff"], "pooled": pooled, "rooms": room_metrics})
    write_json(out / "scores.json", {"frozen_sha256": frozen_sha256, "results": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fit", "freeze", "score"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--out", type=Path, required=True)
        if name == "fit":
            command.add_argument("--steps", type=int, default=144)
            command.add_argument("--seed", type=int)
        if name == "score":
            command.add_argument("--frozen-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        print(json.dumps({name: len(windows(rows)[2])
                          for name, rows in uci_parts(args.data).items()}))
    elif args.command == "fit":
        fit(args.data, args.out, args.steps,
            [args.seed] if args.seed is not None else [41, 42, 43])
    elif args.command == "freeze":
        _, manifest, _ = audit(args.out, args.data)
        write_json(args.out / "frozen.json", manifest)
        print(digest((args.out / "frozen.json").read_bytes()))
    else:
        score(args.data, args.out, args.frozen_sha256)


if __name__ == "__main__":
    main()
