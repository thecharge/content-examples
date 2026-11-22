"""Train, snapshot and evaluate fixed offline occupancy and heating comparisons."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from data import (OCC_FEATURES, SOURCES, WINDOW, apartments, fitting_blocks, heat_rows,
                  heat_windows, normalizer, occupancy_windows, office, support_bucket,
                  valid_history)
from models import HEAT_MODELS, OCC_MODELS, SUBSTEPS, UNITS, Network

SEEDS = (41, 42, 43)
BUDGETS = (25, 50, 100)
BATCH = 256
EPOCHS = 12
PATIENCE = 2
LR = 0.01
SUPPORT_MIN = 8
AUX_WEIGHT = 0.25
COMFORT_WEIGHT = 2
THRESHOLD = 0.5
OCC_SITES = {"datatest.txt": (5, 1), "datatest2.txt": (5, 1), "apartments": (20, 15)}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def freeze():
    files = ("protocol.md", "benchmark.py", "data.py", "models.py", "requirements.txt")
    base = Path(__file__).parent
    return dict(sources=SOURCES, files={f: hashlib.sha256((base / f).read_bytes()).hexdigest()
                                        for f in files},
                versions={d: importlib.metadata.version(d) for d in ("torch", "ncps", "numpy")},
                python=platform.python_version(), budgets=BUDGETS, seeds=SEEDS,
                occupancy_models=OCC_MODELS, heating_models=HEAT_MODELS, window=WINDOW,
                features=OCC_FEATURES, units=UNITS, substeps=SUBSTEPS, batch=BATCH,
                max_epochs=EPOCHS, patience=PATIENCE, learning_rate=LR,
                support_min=SUPPORT_MIN, aux_weight=AUX_WEIGHT, comfort_weight=COMFORT_WEIGHT,
                threshold=THRESHOLD, occupancy_windows=OCC_SITES)


def pack(rows, x_stats, y_stats=None):
    x_mean, x_scale = map(torch.tensor, x_stats)
    x = (torch.tensor([r["x"] for r in rows]) - x_mean) / x_scale
    gaps = torch.tensor([r["gaps"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["y"] for r in rows], dtype=torch.float32)
    if y_stats is not None:
        y_mean, y_scale = map(torch.tensor, y_stats)
        return x, gaps, (y - y_mean) / y_scale, torch.tensor([r["action"] for r in rows]), None, None
    mask = torch.tensor([r["next_x"] is not None for r in rows])
    next_x = torch.tensor([r["next_x"] if r["next_x"] is not None else x_stats[0] for r in rows])
    return x, gaps, y[:, None], None, (next_x - x_mean) / x_scale, mask


def loss(model, data, ids, aux):
    x, g, y, action, future, mask = data
    pred, _, next_reading = model(x[ids], g[ids], None if action is None else action[ids])
    if action is not None:
        return F.mse_loss(pred, y[ids])
    value = F.binary_cross_entropy_with_logits(pred, y[ids])
    if aux and next_reading is not None:
        included = mask[ids]
        if included.any():
            value = value + AUX_WEIGHT * F.mse_loss(next_reading[included], future[ids][included])
    return value


def weights(model):
    return {key: value.tolist() for key, value in model.state_dict().items()}


def reload(task, name, state):
    model = Network(task, name)
    model.load_state_dict({key: torch.tensor(value) for key, value in state.items()})
    return model.eval()


def arguments(data, ids):
    x, g, _, action, _, _ = data
    return x[ids], g[ids], None if action is None else action[ids]


def train(task, name, seed, fit, validation):
    torch.manual_seed(seed)
    model = Network(task, name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    order = torch.Generator().manual_seed(seed)
    fit_count, val_count = len(fit[2]), len(validation[2])
    updates, best, best_state, curve = 0, math.inf, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for ids in torch.randperm(fit_count, generator=order).split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss(model, fit, ids, aux=True).backward()
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            value = sum(len(ids) * loss(model, validation, ids, aux=False).item()
                        for ids in torch.arange(val_count).split(BATCH)) / val_count
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        curve.append(value)
        if value < best:
            best, selected, best_state = value, epoch, weights(model)
            with torch.no_grad():
                sample = arguments(validation, slice(0, 4))
                reference = model(*sample)
        if epoch - selected >= PATIENCE:
            break
    restored = reload(task, name, json.loads(json.dumps(best_state)))
    with torch.no_grad():
        for expected, actual in zip(reference, restored(*sample)):
            assert (actual is None) if expected is None else torch.equal(expected, actual)
    return restored, dict(weights=best_state, parameters=params, epochs=len(curve), updates=updates,
                          selected_epoch=selected, validation_loss=curve, reload_exact=True)


@torch.no_grad()
def score_occ(model, rows, stats):
    x, g, y, _, _, _ = pack(rows, stats)
    logits, traces = [], []
    for start in range(0, len(y), BATCH):
        pred, attention, _ = model(x[start:start+BATCH], g[start:start+BATCH])
        logits.append(pred[:, 0])
        if start == 0:
            for index in range(min(3, len(pred))):
                traces.append(dict(id=rows[index]["id"], gaps=rows[index]["gaps"],
                                   probability=pred[index, 0].sigmoid().item(),
                                   weights=None if attention is None else attention[index].tolist()))
    logits = torch.cat(logits)
    assert torch.isfinite(logits).all()
    actual = y[:, 0].bool()
    positive = logits.sigmoid() >= THRESHOLD
    tp, tn = int((positive & actual).sum()), int((~positive & ~actual).sum())
    fp, fn = int((positive & ~actual).sum()), int((~positive & actual).sum())
    balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) and (tn + fp) else None
    return dict(n=len(rows), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp+tn)/len(rows),
                balanced_accuracy=balanced, bce=F.binary_cross_entropy_with_logits(logits, y[:, 0]).item(),
                traces=traces)


def occupancy(path, danish, name, seed, budget, fit_only):
    source = office(path, "datatraining.txt")
    blocks = fitting_blocks(source, budget)
    fit = occupancy_windows(blocks, 5, 1)
    validation = occupancy_windows([source[6500:]], 5, 1)
    assert {r["y"] for r in fit} == {0, 1}
    stats = normalizer([r[2] for block in blocks for r in block])
    model, receipt = train("occupancy", name, seed, pack(fit, stats), pack(validation, stats))
    receipt.update(x_stats=stats, fit_ranges=[[block[0][0], block[-1][0]] for block in blocks],
                   fit_rows=sum(map(len, blocks)), fit_windows=len(fit),
                   fit_positive=sum(r["y"] for r in fit),
                   fit_aux_windows=sum(r["next_id"] is not None for r in fit),
                   validation_windows=len(validation))
    if fit_only:
        return receipt
    tests = {}
    for member in ("datatest.txt", "datatest2.txt"):
        rows = office(path, member)
        tests[member] = score_occ(model, occupancy_windows([rows], *OCC_SITES[member]), stats)
    groups = defaultdict(list)
    raw_groups = apartments(danish)
    for (apartment, _), rows in raw_groups.items():
        groups[apartment].extend(occupancy_windows([rows], *OCC_SITES["apartments"]))
    assert len(groups) == 5
    tests["apartments"] = score_occ(model, [r for rows in groups.values() for r in rows], stats)
    tests["per_apartment"] = {key: score_occ(model, rows, stats) if rows else {"n": 0}
                               for key, rows in groups.items()}
    receipt["tests"] = tests
    return receipt


def reward(outcomes):
    return -outcomes[..., 1] - COMFORT_WEIGHT * outcomes[..., 2]


def counts_and_means(fit):
    groups = defaultdict(list)
    actions = defaultdict(list)
    for row in fit:
        groups[(*row["bucket"], row["action"])].append(row["y"])
        actions[row["action"]].append(row["y"])
    counts = {key: len(values) for key, values in groups.items()}
    means = {key: torch.tensor(values).mean(0) for key, values in groups.items()}
    fallback = {key: torch.tensor(values).mean(0) for key, values in actions.items()}
    return counts, means, fallback


def heating_prediction(model, rows, action, stats, target_stats, means, fallback):
    if model is None:
        return torch.stack([means.get((*r["bucket"], action), fallback[action]) for r in rows]), None
    x, g, *_ = pack(rows, stats, target_stats)
    y_mean, y_scale = map(torch.tensor, target_stats)
    logits, attention, _ = model(x, g, torch.full((len(x),), action))
    raw = logits * y_scale + y_mean
    return torch.cat((raw[:, :1], raw[:, 1:].clamp_min(0)), dim=1), attention


def recommendation(row, predictions, counts, attention=None):
    coverage = [counts.get((*row["bucket"], action), 0) for action in (0, 1)]
    values = reward(predictions)
    invalid = (len(row["x"]) != WINDOW or len(row["gaps"]) != WINDOW
               or any(len(reading) != 2 for reading in row["x"])
               or not valid_history(row["x"], row["gaps"], 1)
               or not torch.isfinite(predictions).all())
    reason = "invalid_input" if invalid else "low_action_support" if min(coverage) < SUPPORT_MIN else None
    rank = None if reason else torch.argsort(values, descending=True, stable=True).tolist()
    return dict(action=None if rank is None else rank[0], rank=rank, reason=reason, support=coverage,
                candidates=[dict(action=a, temperature_c=float(predictions[a, 0]),
                                 kwh=float(predictions[a, 1]), degree_hours=float(predictions[a, 2]),
                                 reward=float(values[a])) for a in (0, 1)] if not invalid else None,
                attention=attention)


@torch.no_grad()
def score_heat(model, rows, stats, target_stats, counts, means, fallback):
    logs = []
    for start in range(0, len(rows), BATCH):
        part = rows[start:start+BATCH]
        off, attention = heating_prediction(model, part, 0, stats, target_stats, means, fallback)
        on, _ = heating_prediction(model, part, 1, stats, target_stats, means, fallback)
        for i, row in enumerate(part):
            predictions = torch.stack((off[i], on[i]))
            trace = recommendation(row, predictions, counts,
                                   None if attention is None else attention[i].tolist())
            logged = row["action"]
            actual = torch.tensor(row["y"])
            estimate = predictions[logged]
            errors = (estimate - actual).tolist() + [float(reward(estimate) - reward(actual))]
            logs.append(dict(id=row["id"], logged=logged, errors=errors, observed=row["y"], **trace))
    errors = torch.tensor([row["errors"] for row in logs])
    assert torch.isfinite(errors).all()
    picks = Counter("abstain" if r["action"] is None else ("off", "on")[r["action"]] for r in logs)
    traces = {}
    for row in logs:
        key = "abstain" if row["action"] is None else ("off", "on")[row["action"]]
        traces.setdefault(key, row)
    return dict(n=len(rows), predictions=dict(picks), logged=dict(Counter(str(r["logged"]) for r in logs)),
                supports=dict(Counter(str(r["support"]) for r in logs)),
                error_order=["temperature_c", "thermal_kwh", "degree_hours", "reward"],
                mae=errors.abs().mean(0).tolist(), rmse=errors.square().mean(0).sqrt().tolist(),
                traces=traces, records=logs)


def heating(path, name, seed, fit_only):
    pre = heat_rows(path, False)
    cutoff = min(pre) + (max(pre) + 1 - min(pre)) * 4 // 5
    fit, fit_skips = heat_windows({key: row for key, row in pre.items() if key < cutoff})
    valid, valid_skips = heat_windows({key: row for key, row in pre.items() if key >= cutoff})
    stats = normalizer([x for row in fit for x in row["x"]])
    target_stats = normalizer([row["y"] for row in fit])
    counts, means, fallback = counts_and_means(fit)
    model = None
    table = [[list(key), count, means[key].tolist()] for key, count in sorted(counts.items())]
    receipt = dict(parameters=0, epochs=0, updates=0, lookup=table,
                   action_means={key: value.tolist() for key, value in fallback.items()})
    if name != "mean":
        model, receipt = train("heat", name, seed, pack(fit, stats, target_stats),
                               pack(valid, stats, target_stats))
    receipt.update(x_stats=stats, y_stats=target_stats, fit_windows=len(fit),
                   validation_windows=len(valid), validation_start_utc_minute=cutoff,
                   fit_skips=fit_skips, validation_skips=valid_skips,
                   fit_actions=dict(Counter(str(r["action"]) for r in fit)),
                   support_table=[[list(key), value] for key, value in sorted(counts.items())])
    if not fit_only:
        test, skips = heat_windows(heat_rows(path, True))
        receipt.update(test=score_heat(model, test, stats, target_stats, counts, means, fallback),
                       holdout_skips=skips)
    return receipt


def selfcheck():
    torch.manual_seed(SEEDS[0])
    rows = [(i, i, (1., 2., float(i)), i % 2) for i in range(20)]
    first = occupancy_windows([rows[:10], rows[10:]], 5, 1)
    assert [row["id"] for row in first] == [7, 8, 9, 17, 18, 19]
    assert [row["next_id"] for row in first] == [8, 9, None, 18, 19, None]
    changed = rows[:8] + [(8, 8, (9., 9., 9.), 0)] + rows[9:]
    second = occupancy_windows([changed[:10], changed[10:]], 5, 1)
    assert first[0]["next_x"] != second[0]["next_x"]
    x, g, *_ = pack(first[:1], ([0.] * 3, [1.] * 3))
    x2, g2, *_ = pack(second[:1], ([0.] * 3, [1.] * 3))
    for name in OCC_MODELS:
        model = Network("occupancy", name).eval()
        with torch.no_grad():
            a, attention, _ = model(x, g)
            b, _, _ = model(x2, g2)
            assert torch.equal(a, b)
            if attention is not None:
                assert torch.isfinite(attention).all() and (attention >= 0).all()
                torch.testing.assert_close(attention.sum(dim=1), torch.ones(len(x)))
            assert model(x.expand(7, -1, -1), g.expand(7, -1))[0].shape == (7, 1)
    row = dict(x=[(21., 2.)] * WINDOW, gaps=[1.] * WINDOW,
               bucket=support_bucket(21, 2, 21))
    outcomes = torch.tensor([[21., 0.1, 1.], [21., 0.9, 0.]])
    support = {(*row["bucket"], a): SUPPORT_MIN for a in (0, 1)}
    assert recommendation(row, outcomes, {})["action"] is None
    assert recommendation(row, outcomes, support)["rank"] == [1, 0]
    assert recommendation(row, outcomes.flip(0), support)["rank"] == [0, 1]
    assert recommendation(dict(row, gaps=[0.] * WINDOW), outcomes, support)["action"] is None
    print("Fixed-seed leakage, attention, baseline shape, support and reward rank checks passed.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("check")
    for task, names in (("occupancy", OCC_MODELS), ("heat", HEAT_MODELS)):
        run = sub.add_parser(task)
        run.add_argument("source")
        run.add_argument("--model", choices=names)
        run.add_argument("--seed", type=int, choices=SEEDS, required=True)
        run.add_argument("--fit-only", action="store_true")
        run.add_argument("--summary", action="store_true", help="Omit bulky heating row-level records")
        if task == "occupancy":
            run.add_argument("--danish")
            run.add_argument("--budget", type=int, choices=BUDGETS)
    args = cli.parse_args()
    if args.command == "check":
        return selfcheck()
    config = freeze()
    if args.command == "freeze":
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    if args.command == "occupancy":
        if not args.fit_only and not args.danish:
            cli.error("--danish is required for occupancy evaluation")
        for budget in ((args.budget,) if args.budget else BUDGETS):
            for name in ((args.model,) if args.model else OCC_MODELS):
                receipt = occupancy(args.source, args.danish, name, args.seed, budget, args.fit_only)
                print(json.dumps(dict(task=args.command, model=name, seed=args.seed, budget=budget,
                                      config=config["files"], **receipt), allow_nan=False), flush=True)
    else:
        for name in ((args.model,) if args.model else HEAT_MODELS):
            receipt = heating(args.source, name, args.seed, args.fit_only)
            if args.summary and "test" in receipt:
                del receipt["test"]["records"]
            print(json.dumps(dict(task=args.command, model=name, seed=args.seed,
                                  config=config["files"], **receipt), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
