"""CPU occupancy experiment. Fit first, freeze, then invoke score."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from adaptive import Adaptive
from policy import ABSTAIN, RETRY, callback, fit_policy, outcome

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attention-benchmark"))
from data import fitting_blocks, minute, normalizer, occupancy_windows, office
from models import Network

INPUT = Path("/tmp/sndv-liquid-attention-inputs")
UCI = Path("/tmp/sndv-ltc-data/occupancy-fresh.zip")
V1 = INPUT / "adaptive-v1-refuted" / "adaptive.py"
AALTO_HASH = "bf0617f9c273e64f100f2e9303e4104d5346ebab1c3060fa8a9cc88850e596b2"
NAMES = ("current", "gru", "library", "liquid", "v1", "v2", "no_delta", "no_correction")
SEEDS = (41, 42, 43)
BUDGETS = (25, 50, 100)
LIMIT = 240
VALIDATE_EVERY = 24
MIN_STEPS = 120
PATIENCE = 3
BATCH = 64


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(name):
    if name in ("v2", "no_delta", "no_correction"):
        return Adaptive(width=8, delta=name != "no_delta", correction=name != "no_correction")
    if name == "v1":
        spec = importlib.util.spec_from_file_location("retained_v1", V1)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        return original.Adaptive("occupancy", "adaptive")
    return Network("occupancy", name)


def pack(windows, stats):
    center, scale = map(torch.tensor, stats)
    return ((torch.tensor([w["x"] for w in windows]) - center) / scale,
            torch.tensor([w["gaps"] for w in windows], dtype=torch.float32),
            torch.tensor([w["y"] for w in windows], dtype=torch.bool))


def batches(data, size=256):
    return zip(*(x.split(size) for x in data))


def forward(model, x, clock, choose=None):
    if isinstance(model, Adaptive):
        return model(x, clock, choose)
    if hasattr(model, "critics"):
        predictions, scores, accepted, trace = model(x, clock)
        return [p[:, 0] for p in predictions], scores, accepted, trace
    return [model(x, clock)[0][:, 0]], [], torch.ones(len(x), dtype=torch.bool), []


@torch.no_grad()
def predict(model, data, policy=None):
    results, coverage, costs, traces = [], [], [], []
    for x, clock, y in batches(data):
        decisions = []
        pred, _, accepted, trace = forward(
            model, x, clock, None if policy is None else callback(policy, decisions))
        results.append(pred[-1])
        coverage.append(accepted)
        attempted = sum((t.get("attempted", torch.zeros_like(y)).long() for t in trace),
                        torch.zeros_like(y, dtype=torch.long))
        if hasattr(model, "critics") and not isinstance(model, Adaptive):
            attempted = sum((t["revise"].long() for t in trace), torch.zeros_like(attempted))
        costs.append(attempted)
        if decisions:
            traces.append([dict(state=s.tolist(), q=q.nan_to_num(neginf=-1e9).tolist(),
                                predicted_quality=p.tolist(), action=a.tolist(),
                                retained=trace[i + 1].get("retained", torch.zeros_like(y)).tolist())
                           for i, (s, q, p, a) in enumerate(decisions)])
    return torch.cat(results), torch.cat(coverage), torch.cat(costs), traces


def balanced(logits, y, cutoff):
    guess = logits >= cutoff
    return float(((guess[y]).float().mean() + (~guess[~y]).float().mean()) / 2)


def calibration(logits, y):
    options = torch.unique(torch.cat((torch.quantile(logits, torch.linspace(0, 1, 101)),
                                     torch.tensor([0.])))).tolist()
    return max(options, key=lambda c: (balanced(logits, y, c), -abs(c)))


def train(name, seed, budget, directory):
    torch.manual_seed(seed)
    raw = office(UCI, "datatraining.txt")
    blocks = fitting_blocks(raw, budget)
    stats = normalizer([r[2] for block in blocks for r in block])
    fit = pack(occupancy_windows(blocks, 5, 1), stats)
    validation = pack(occupancy_windows([raw[6500:]], 5, 1), stats)
    model = build(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    positives = int(fit[2].sum())
    weight = torch.tensor((len(fit[2]) - positives) / positives)
    generator = torch.Generator().manual_seed(seed)
    curves, best, snapshots = [], None, {}
    fit_start = perf_counter()

    def scheduled(stage, proposal, score, prior, remaining):
        return torch.randint(0, 2, (len(proposal),), generator=generator)

    for step in range(1, LIMIT + 1):
        ids = torch.randint(len(fit[2]), (BATCH,), generator=generator)
        x, clock, y = (part[ids] for part in fit)
        pred, scores, _, _ = forward(model, x, clock, scheduled)
        label = y.float()
        loss = torch.stack([F.binary_cross_entropy_with_logits(p, label, pos_weight=weight)
                            for p in pred]).mean()
        if scores:
            loss = loss + 0.2 * torch.stack([
                F.binary_cross_entropy_with_logits(s, ((p.detach() >= 0) == y).float())
                for s, p in zip(scores, pred)]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % VALIDATE_EVERY:
            continue
        dev, _, _, _ = predict(model, validation)
        value = float(F.binary_cross_entropy_with_logits(dev, validation[2].float(), pos_weight=weight))
        if not torch.isfinite(dev).all():
            raise RuntimeError("Nonfinite validation predictions")
        curves.append(dict(step=step, loss=value, balanced=balanced(dev, validation[2], 0),
                           fit_loss=float(loss.detach())))
        if best is None or value < best[0]:
            best = value, step, {k: v.clone() for k, v in model.state_dict().items()}
        if step in (MIN_STEPS, LIMIT):
            snapshots[step] = curves[-1]
        if name not in ("v1", "v2") and step >= MIN_STEPS and step - best[1] >= PATIENCE * VALIDATE_EVERY:
            break
    network_seconds = perf_counter() - fit_start
    model.load_state_dict(best[2])
    dev, _, _, _ = predict(model, validation)
    cutoff = calibration(dev, validation[2])
    policy_start = perf_counter()
    policies = fit_policy(model, batches(fit), cutoff) if name in ("v2", "no_delta") else {}
    policy_seconds = perf_counter() - policy_start
    points = {}
    for key, policy in policies.items():
        routed, accepted, attempted, _ = predict(model, validation, policy)
        points[key] = dict(coverage=float(accepted.float().mean()), retries=int(attempted.sum()),
                           payoff=float((torch.where(accepted, outcome(routed, validation[2], cutoff),
                                                     ABSTAIN) + RETRY * attempted).mean()))
    checkpoint = dict(weights=best[2], stats=stats, cutoff=cutoff, policies=policies)
    key = f"{name}-{seed}-{budget}"
    path = directory / f"{key}.pt"
    torch.save(checkpoint, path)
    reload = torch.load(path, weights_only=True)
    clone = build(name)
    clone.load_state_dict(reload["weights"])
    assert torch.equal(dev, predict(clone, validation)[0])
    receipt = dict(name=name, seed=seed, budget=budget, steps=step, best_step=best[1],
                   fit_windows=len(fit[2]), fit_positive=positives, pos_weight=float(weight),
                   validation_windows=len(validation[2]), validation_positive=int(validation[2].sum()),
                   cutoff=cutoff, reload_exact=True, curve=curves, staged=snapshots,
                   network_seconds=network_seconds, policy_seconds=policy_seconds,
                   validation_policy=points, checkpoint_sha256=digest(path))
    (directory / f"{key}.json").write_text(json.dumps(receipt, indent=2))
    print(key, step, best[1], round(network_seconds, 1), round(policy_seconds, 1), flush=True)


def aalto():
    path = INPUT / "aalto-ventilation-test.csv"
    assert digest(path) == AALTO_HASH
    fields = ("Temperature (°C)", "Relative Humidity (%)", "Room CO2 (ppm)")
    raw = []
    with path.open(newline="") as file:
        for i, row in enumerate(csv.DictReader(file)):
            count = float(row["Occupant count"])
            if not (count >= 0 and count.is_integer()):
                raise ValueError(f"Invalid occupant count at row {i}")
            raw.append((i, minute(row["time"]), tuple(float(row[k]) for k in fields), int(count > 0)))
    gaps = [b[1] - a[1] for a, b in zip(raw, raw[1:])]
    if not all(g > 0 for g in gaps):
        raise ValueError("Aalto timestamps must increase strictly in input order")
    windows = occupancy_windows([raw], 5, 1)
    return windows, dict(source_rows=len(raw), first=raw[0][1], last=raw[-1][1],
                         min_gap=min(gaps), max_gap=max(gaps), window_count=len(windows),
                         membership_sha256=hashlib.sha256(json.dumps(
                             [(w["id"], w["time"]) for w in windows]).encode()).hexdigest())


def metrics(logits, y, covered, cutoff):
    guess = logits >= cutoff
    tp = int((guess & y & covered).sum())
    tn = int((~guess & ~y & covered).sum())
    fp = int((guess & ~y & covered).sum())
    fn = int((~guess & y & covered).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, abstain_positive=int((y & ~covered).sum()),
                abstain_negative=int((~y & ~covered).sum()), coverage=float(covered.float().mean()),
                accuracy=(tp + tn) / max(1, tp + tn + fp + fn),
                balanced=(tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
                effective_accuracy=(tp + tn) / len(y),
                effective_balanced=(tp / int(y.sum()) + tn / int((~y).sum())) / 2)


def ranking(logits, y):
    order = logits.argsort(descending=True)
    logits, y = logits[order], y[order]
    ends = torch.cat((logits[1:] != logits[:-1], torch.tensor([True])))
    tp = y.cumsum(0)[ends].double()
    fp = (~y).cumsum(0)[ends].double()
    tpr = torch.cat((torch.tensor([0.]), tp / tp[-1]))
    fpr = torch.cat((torch.tensor([0.]), fp / fp[-1]))
    return dict(auroc=float(torch.trapezoid(tpr, fpr)),
                average_precision=float((tpr.diff() * tp / (tp + fp)).sum()))


def freeze_sources():
    paths = [HERE / n for n in ("adaptive.py", "policy.py", "experiment.py", "verify.py", "context_checks.py")]
    paths += [HERE.parent / "attention-benchmark" / n for n in ("data.py", "models.py")]
    paths += [V1, V1.with_name("graz.py"), UCI]
    return {str(p): digest(p) for p in paths}


def score(directory):
    freeze = json.loads((HERE / "freeze.json").read_text())
    assert freeze["source_sha256"] == freeze_sources()
    assert freeze["aalto_sha256"] == AALTO_HASH
    for path in directory.glob("*.pt"):
        assert digest(path) == freeze["checkpoint_sha256"][path.name]
    expected = {f"{name}-{seed}-{budget}.pt" for name in NAMES
                for seed in SEEDS for budget in BUDGETS}
    assert set(freeze["checkpoint_sha256"]) == expected
    windows, membership = aalto()
    (directory / "aalto-membership.json").write_text(json.dumps(dict(
        **membership, ids=[w["id"] for w in windows], labels=[w["y"] for w in windows])))
    print(json.dumps(membership), flush=True)
    for path in sorted(directory.glob("*.pt")):
        model = build(path.stem.split("-")[0])
        saved = torch.load(path, weights_only=True)
        model.load_state_dict(saved["weights"])
        data = pack(windows, saved["stats"])
        y, cutoff = data[2], saved["cutoff"]
        direct, _, _, _ = predict(model, data)
        modes = dict(base=None, **saved["policies"])
        result = {}
        raw = {}
        for mode, policy in modes.items():
            logits, covered, attempted, traces = predict(model, data, policy)
            retry_only = attempted > 0
            before = (direct >= cutoff) == y
            after = (logits >= cutoff) == y
            payoff = torch.where(covered, outcome(logits, y, cutoff), ABSTAIN) + RETRY * attempted
            result[mode] = dict(original=metrics(logits, y, covered, 0),
                                selected=metrics(logits, y, covered, cutoff),
                                ranking_all_proposals=ranking(logits, y),
                                retries=int(attempted.sum()),
                                retry_help=int((~before & after & retry_only & covered).sum()),
                                retry_hurt=int((before & ~after & retry_only & covered).sum()),
                                reward_mean=float(payoff.mean()),
                                reward_breakdown=dict(
                                    correct=int((after & covered).sum()),
                                    false_alarm=int(((logits >= cutoff) & ~y & covered).sum()),
                                    miss=int(((logits < cutoff) & y & covered).sum()),
                                    abstain=int((~covered).sum()),
                                    retry=int(attempted.sum())))
            raw[mode] = dict(logits=logits.tolist(), covered=covered.tolist(),
                             attempts=attempted.tolist(), actions=[[t["action"] for t in b] for b in traces],
                             trace_examples=[dict(batch=i, trace=[{k: v[:2] for k, v in t.items()}
                                                                  for t in b])
                                             for i, b in enumerate(traces[:3])])
        if "0.1" in raw:
            result["process_changed_routes"] = sum(
                any(a != b for a, b in zip(s, t)) for sa, sb in
                zip(raw["0.0"]["actions"], raw["0.1"]["actions"])
                for s, t in zip(zip(*sa), zip(*sb)))
        (directory / f"{path.stem}-aalto.json").write_text(json.dumps(
            dict(summary=result, ids=[w["id"] for w in windows], routes=raw)))
        print(path.stem, json.dumps(result), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "score", "freeze"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("name", nargs="?", choices=NAMES)
    parser.add_argument("seed", nargs="?", type=int, choices=SEEDS)
    parser.add_argument("budget", nargs="?", type=int, choices=BUDGETS)
    args = parser.parse_args()
    if args.mode == "fit":
        train(args.name, args.seed, args.budget, args.directory)
    elif args.mode == "freeze":
        print(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                              source_sha256=freeze_sources(), aalto_sha256=AALTO_HASH,
                              checkpoint_sha256={p.name: digest(p) for p in sorted(args.directory.glob("*.pt"))}),
                         indent=2))
    else:
        score(args.directory)
