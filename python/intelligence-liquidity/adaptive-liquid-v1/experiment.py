"""Run the frozen occupancy and logged-action heating experiment."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from adaptive import Adaptive
import benchmark as b
from data import (apartments, fitting_blocks, heat_rows, heat_windows, normalizer,
                  occupancy_windows, office, support_bucket, valid_history, WINDOW)
import graz

VARIANTS = ("adaptive", "no_feedback", "delta_only", "hard_only", "no_delta")
QUALITY = 0.1
ROOT = Path(__file__).resolve().parent
INPUT = Path("/tmp/sndv-liquid-attention-inputs")
SOURCES = dict(occupancy="/tmp/sndv-ltc-data/occupancy-fresh.zip",
               heat="/tmp/sndv-external/six-buildings.zip")
GRAZ = INPUT / "graz.zip"


def freeze():
    config = b.freeze()
    config.update(adaptive_files={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("adaptive.py", "experiment.py", "graz.py", "protocol.md")},
                  variants=VARIANTS, heating_quality_weight=QUALITY, graz_sha256=graz.SHA256)
    return config


def train_adaptive(task, name, seed, fit, validation):
    torch.manual_seed(seed)
    model = Adaptive(task, name)
    optimizer = torch.optim.Adam(model.parameters(), lr=b.LR)
    generator = torch.Generator().manual_seed(seed)
    updates, best, selected, curve = 0, math.inf, 0, []
    critic_counts = [[0, 0, 0] for _ in range(3)]
    supervised = F.binary_cross_entropy_with_logits if task == "occupancy" else F.mse_loss
    for epoch in range(1, b.EPOCHS + 1):
        model.train()
        for ids in torch.randperm(len(fit[2]), generator=generator).split(b.BATCH):
            args = b.arguments(fit, ids)
            target = fit[2][ids]
            proposals, logits, _, _ = model(*args, force=updates % 4 == 0)
            # Perturb critic features computed strictly from the observed prefix.
            # The model detaches all critic inputs before either score is computed.
            _, corrupt_scores, _, _ = model(*args, corrupt=True)
            objective = sum(supervised(p, target) for p in proposals) / 3
            for i, (proposal, score, corrupt_score) in enumerate(zip(proposals, logits, corrupt_scores)):
                if task == "occupancy":
                    error = (proposal.detach().sigmoid() - target).abs().squeeze(-1)
                else:
                    error = (proposal.detach() - target).square().mean(-1).sqrt()
                good = (error <= (0.5 if task == "occupancy" else 0.75)).float()
                critic_counts[i][0] += int(good.sum())
                critic_counts[i][1] += len(good) - int(good.sum())
                critic_counts[i][2] += len(good)
                objective += (F.binary_cross_entropy_with_logits(score, good)
                              + F.softplus(corrupt_score).mean()) / 6
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            value = sum(len(ids) * supervised(model(*b.arguments(validation, ids))[0][-1],
                                              validation[2][ids]).item()
                        for ids in torch.arange(len(validation[2])).split(b.BATCH)) / len(validation[2])
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        curve.append(value)
        if value < best:
            best, selected = value, epoch
            state = b.weights(model)
            with torch.no_grad():
                sample = b.arguments(validation, slice(0, 4))
                reference = model(*sample)[0][-1]
        if epoch - selected >= b.PATIENCE:
            break
    restored = restore(task, name, json.loads(json.dumps(state)))
    with torch.no_grad():
        assert torch.equal(reference, restored(*sample)[0][-1])
    return restored, dict(weights=state, parameters=sum(p.numel() for p in model.parameters()),
                          epochs=len(curve), updates=updates, selected_epoch=selected,
                          critic_counts=critic_counts, validation_loss=curve, reload_exact=True)


def restore(task, name, weights):
    if name not in VARIANTS:
        return b.reload(task, name, weights)
    model = Adaptive(task, name)
    model.load_state_dict({key: torch.tensor(value) for key, value in weights.items()})
    return model.eval()


def train(task, name, seed, fit, validation, stats, target_stats=None):
    start = perf_counter()
    method = train_adaptive if name in VARIANTS else b.train
    model, receipt = method(task, name, seed, b.pack(fit, stats, target_stats),
                            b.pack(validation, stats, target_stats))
    receipt.update(fit_seconds=perf_counter() - start, x_stats=stats, y_stats=target_stats,
                   fit_windows=len(fit), validation_windows=len(validation))
    return model, receipt


def features(rows, stats):
    mean, scale = map(torch.tensor, stats)
    return ((torch.tensor([r["x"] for r in rows]) - mean) / scale,
            torch.tensor([r["gaps"] for r in rows], dtype=torch.float32))


def predict(model, rows, stats, action=None):
    x, gaps = features(rows, stats)
    actions = None if action is None else torch.full((len(rows),), action, dtype=torch.long)
    if isinstance(model, Adaptive):
        proposals, scores, accepted, trace = model(x, gaps, actions)
        return proposals[-1], accepted, trace, torch.stack(scores, 1).sigmoid().mean(1)
    output, _, _ = model(x, gaps, actions)
    return output, torch.isfinite(output).all(1), [], torch.zeros(len(rows))


def trace_at(trace, index):
    return [{key: value[index].item() for key, value in stage.items()} for stage in trace]


def occupancy_metrics(logits, actual):
    n = len(logits)
    if not n:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "bce": None}
    positive, truth = logits.sigmoid() >= b.THRESHOLD, actual.bool()
    tp, tn = int((positive & truth).sum()), int((~positive & ~truth).sum())
    fp, fn = int((positive & ~truth).sum()), int((~positive & truth).sum())
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / n,
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2
                if tp + fn and tn + fp else None,
                bce=F.binary_cross_entropy_with_logits(logits, actual).item())


@torch.no_grad()
def score_occupancy(model, rows, stats):
    start = perf_counter()
    logits, accepted, examples = [], [], {}
    route = [Counter() for _ in range(3)]
    for begin in range(0, len(rows), b.BATCH):
        part = rows[begin:begin + b.BATCH]
        pred, mask, trace, score = predict(model, part, stats)
        logits.append(pred[:, 0])
        accepted.append(mask)
        for i, stage in enumerate(trace):
            for key in ("initial_pass", "revise", "pass_after", "abstain", "hard"):
                route[i][key] += int(stage[key].sum())
            replayed = stage["revise"] & (stage["upstream_change"] > 0) & (stage["replay_change"] > 0)
            route[i]["changed_and_replayed"] += int(replayed.sum())
            for key, selection in (("pass", stage["pass_after"]), ("revise", replayed),
                                   ("abstain", stage["abstain"])):
                indexes = selection.nonzero()
                if len(indexes) and f"{i}:{key}" not in examples:
                    j = int(indexes[0].item())
                    examples[f"{i}:{key}"] = dict(id=part[j]["id"], actual=part[j]["y"],
                                                   probability=pred[j, 0].sigmoid().item(),
                                                   score=score[j].item(), stages=trace_at(trace, j))
    if not logits:
        return {"all": occupancy_metrics(torch.empty(0), torch.empty(0))}
    logits, accepted = torch.cat(logits), torch.cat(accepted)
    actual = torch.tensor([r["y"] for r in rows], dtype=torch.float32)
    assert torch.isfinite(logits).all()
    return dict(all=occupancy_metrics(logits, actual),
                accepted=occupancy_metrics(logits[accepted], actual[accepted]),
                coverage=accepted.float().mean().item(), routes=route, examples=examples,
                seconds=perf_counter() - start)


def occupancy(name, seed, budget, mode, checkpoint=None):
    if checkpoint is None:
        raw = office(SOURCES["occupancy"], "datatraining.txt")
        blocks = fitting_blocks(raw, budget)
        fit = occupancy_windows(blocks, 5, 1)
        validation = occupancy_windows([raw[6500:]], 5, 1)
        stats = normalizer([r[2] for block in blocks for r in block])
        model, receipt = train("occupancy", name, seed, fit, validation, stats)
        receipt.update(fit_ranges=[[block[0][0], block[-1][0]] for block in blocks],
                       fit_rows=sum(map(len, blocks)), fit_positive=sum(r["y"] for r in fit))
    else:
        receipt = checkpoint
        stats = receipt["x_stats"]
        model = restore("occupancy", name, receipt["weights"])
    if mode == "fit":
        return receipt
    tests = {}
    for member in ("datatest.txt", "datatest2.txt"):
        rows = occupancy_windows([office(SOURCES["occupancy"], member)], 5, 1)
        tests[member] = score_occupancy(model, rows, stats)
    groups = defaultdict(list)
    for (apartment, _), rows in apartments(INPUT / "apartments.csv").items():
        groups[apartment].extend(occupancy_windows([rows], 20, 15))
    tests["apartments"] = score_occupancy(model, sum(groups.values(), []), stats)
    if mode == "all":
        for room, (rows, audit) in graz.load(GRAZ).items():
            tests[room] = dict(join=audit, **score_occupancy(model, rows, stats))
    return dict(receipt, tests=tests)


def heat_candidates(model, rows, stats, target_stats, means, fallback, counts):
    out, traces, masks, qualities = [], [], [], []
    mean, scale = map(torch.tensor, target_stats)
    for action in (0, 1):
        if model is None:
            values = torch.stack([means.get((*r["bucket"], action), fallback[action]) for r in rows])
            mask, quality, trace = torch.ones(len(rows), dtype=torch.bool), torch.zeros(len(rows)), []
        else:
            pred, mask, trace, quality = predict(model, rows, stats, action)
            values = pred * scale + mean
            values = torch.cat((values[:, :1], values[:, 1:].clamp_min(0)), 1)
        out.append(values)
        traces.append(trace)
        masks.append(mask)
        qualities.append(quality)
    for i, row in enumerate(rows):
        values = torch.stack([result[i] for result in out])
        support = [counts.get((*row["bucket"], a), 0) for a in (0, 1)]
        valid = valid_history(row["x"], row["gaps"], 1) and torch.isfinite(values).all().item()
        hard = valid and min(support) >= b.SUPPORT_MIN
        physical = b.reward(values).tolist()
        learned = [float(q[i]) for q in qualities]
        bonus = QUALITY if isinstance(model, Adaptive) and model.variant != "hard_only" else 0.
        total = [r + bonus * q for r, q in zip(physical, learned)]
        candidates = [a for a in (0, 1) if hard and bool(masks[a][i])]
        rank = sorted(candidates, key=lambda a: -total[a])
        yield values, dict(support=support, hard=hard, rank=rank, action=rank[0] if rank else None,
                           physical_reward=physical, learned_process=learned, final_reward=total,
                           candidates=values.tolist(), stages=[trace_at(t, i) for t in traces])


@torch.no_grad()
def score_heat(model, rows, stats, target_stats, means, fallback, counts):
    start = perf_counter()
    records = []
    for begin in range(0, len(rows), b.BATCH):
        part = rows[begin:begin + b.BATCH]
        for row, (values, trace) in zip(part, heat_candidates(model, part, stats, target_stats,
                                                            means, fallback, counts)):
            actual = torch.tensor(row["y"])
            predicted = values[row["action"]]
            error = (predicted - actual).tolist() + [float(b.reward(predicted) - b.reward(actual))]
            records.append(dict(id=row["id"], observed=row["y"], logged=row["action"],
                                errors=error, **trace))
    errors = torch.tensor([r["errors"] for r in records])
    assert torch.isfinite(errors).all()
    return dict(n=len(rows), mae=errors.abs().mean(0).tolist(),
                rmse=errors.square().mean(0).sqrt().tolist(),
                picks=dict(Counter(str(r["action"]) for r in records)),
                records=records, seconds=perf_counter() - start)


def heating(name, seed, mode):
    pre = heat_rows(SOURCES["heat"], False)
    cutoff = min(pre) + (max(pre) + 1 - min(pre)) * 4 // 5
    fit, fit_skips = heat_windows({k: r for k, r in pre.items() if k < cutoff})
    valid, valid_skips = heat_windows({k: r for k, r in pre.items() if k >= cutoff})
    stats = normalizer([x for r in fit for x in r["x"]])
    target_stats = normalizer([r["y"] for r in fit])
    start = perf_counter()
    counts, means, fallback = b.counts_and_means(fit)
    model = None
    receipt = dict(parameters=0, epochs=0, updates=0, fit_seconds=perf_counter() - start,
                   fit_windows=len(fit), validation_windows=len(valid), x_stats=stats,
                   y_stats=target_stats)
    if name != "mean":
        model, receipt = train("heat", name, seed, fit, valid, stats, target_stats)
    receipt.update(lookup=[[list(k), v, means[k].tolist()] for k, v in sorted(counts.items())],
                   action_means={k: v.tolist() for k, v in fallback.items()}, cutoff=cutoff,
                   fit_skips=fit_skips, validation_skips=valid_skips)
    if mode != "fit":
        test, skips = heat_windows(heat_rows(SOURCES["heat"], True))
        receipt.update(test=score_heat(model, test, stats, target_stats, means, fallback, counts),
                       holdout_skips=skips)
    return receipt


@torch.no_grad()
def infer(receipt, request):
    task, model_name = receipt["task"], receipt["model"]
    model = None if model_name == "mean" else restore(task, model_name, receipt["weights"])
    x, gaps = request["x"], request["gaps"]
    width = 3 if task == "occupancy" else 2
    if (len(x) != WINDOW or len(gaps) != WINDOW or any(len(v) != width for v in x)
            or not valid_history(x, gaps, 20 if task == "occupancy" else 1)):
        return {"action": None, "reason": "invalid_history"}
    row = {"x": x, "gaps": gaps}
    if task == "occupancy":
        pred, accepted, trace, score = predict(model, [row], receipt["x_stats"])
        return dict(probability=pred.sigmoid().item(), accepted=accepted.item(),
                    learned_process=score.item(), stages=trace_at(trace, 0))
    prior = request["prior_c"]
    if not math.isfinite(prior):
        return {"action": None, "reason": "invalid_prior"}
    row["bucket"] = support_bucket(*x[-1], prior)
    counts = {tuple(k): n for k, n, _ in receipt["lookup"]}
    means = {tuple(k): torch.tensor(v) for k, _, v in receipt["lookup"]}
    fallback = {int(k): torch.tensor(v) for k, v in receipt["action_means"].items()}
    _, result = next(heat_candidates(model, [row], receipt["x_stats"], receipt["y_stats"],
                                     means, fallback, counts))
    return result


def check():
    b.selfcheck()
    torch.manual_seed(41)
    model = Adaptive("occupancy", "adaptive")
    x, gaps = torch.randn(4, 8, 3), torch.ones(4, 8)
    torch.testing.assert_close(model.delta(x)[:, :4], model.delta(x[:, :4]))
    with torch.no_grad():
        model.scores[1].bias.fill_(-10.)
        pred, _, _, trace = model(x, gaps, force=True)
        assert not trace[1]["initial_pass"].any()
        assert trace[1]["revise"].all() and (trace[1]["upstream_change"] > 0).all()
        assert (trace[1]["replay_change"] > 0).all()
        x[0, -1, 0] = float("nan")
        for score in model.scores:
            score.bias.fill_(20.)
        _, _, accept, invalid = model(x, gaps)
        assert not accept[0] and not invalid[-1]["hard"][0]
    x[0, -1, 0] = 0.
    _, scores, _, _ = model(x, gaps)
    sum(score.sum() for score in scores).backward()
    assert all(p.grad is None for layer in (model.delta, model.stages, model.heads, model.corrections)
               for p in layer.parameters())
    assert all(any(p.grad is not None for p in critic.parameters()) for critic in model.critics)
    return {"delta_prefix": True, "hard_veto": True, "forced_disagreement": trace_at(trace, 0),
            "critic_gradient_isolation": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("freeze", "check", "graz-audit", "occupancy", "heat", "infer"))
    cli.add_argument("--model")
    cli.add_argument("--seed", type=int, choices=b.SEEDS, default=41)
    cli.add_argument("--budget", type=int, choices=b.BUDGETS, default=100)
    cli.add_argument("--mode", choices=("fit", "explored", "all"), default="explored")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--request", type=Path)
    cli.add_argument("--summary", action="store_true")
    cli.add_argument("--freeze", type=Path)
    args = cli.parse_args()
    config = freeze()
    if args.command == "freeze":
        result = config
    elif args.command == "check":
        result = check()
    elif args.command == "graz-audit":
        result = {name: audit for name, (_, audit) in graz.load(GRAZ).items()}
    elif args.command == "infer":
        result = infer(json.loads(args.checkpoint.read_text()), json.loads(args.request.read_text()))
    else:
        task = args.command
        if args.mode == "all" and (not args.freeze or json.loads(args.freeze.read_text()) !=
                                   json.loads(json.dumps(config))):
            cli.error("Graz evaluation requires a matching pre-evaluation freeze")
        names = b.OCC_MODELS if task == "occupancy" else b.HEAT_MODELS
        if args.model not in names + VARIANTS:
            cli.error("Choose a model from " + str(names + VARIANTS))
        if task == "occupancy":
            checkpoint = None if not args.checkpoint else json.loads(args.checkpoint.read_text())
            if checkpoint and any(checkpoint[k] != getattr(args, k) for k in ("model", "seed", "budget")):
                cli.error("Checkpoint identity does not match requested comparison")
            result = occupancy(args.model, args.seed, args.budget, args.mode, checkpoint)
        else:
            result = heating(args.model, args.seed, args.mode)
        result.update(task=task, model=args.model, seed=args.seed,
                      budget=args.budget if task == "occupancy" else None,
                      config=config["adaptive_files"])
        if args.summary:
            result.pop("weights", None)
            if "test" in result:
                result["test"].pop("records")
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
