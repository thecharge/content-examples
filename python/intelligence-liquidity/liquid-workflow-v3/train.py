"""Train the two blackboards, then score data kept out of fitting."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from data import INTERVAL, heater, occupancy, sha
from blackboard import Blackboard, DISCOUNT, action_loss, classification_loss, sensor_loss
from online import Event, Online

ROOT = Path(__file__).resolve().parent
MAX_WINDOW = 16
WARMUP = 3
EPSILON = .0001


def trace(result, evidence, names):
    def round_(stage):
        return {
            "proposals": [{"expert": i, "vote": int(stage.votes[0, i]),
                           "values": stage.proposals.values[0, i].tolist(),
                           "available": bool(stage.available[0, i]),
                           "review_support": float(stage.validity[0, i].sigmoid()),
                           "counter_values": stage.counter_values[0, i].tolist(),
                           "objection": bool(stage.contradicts[0, i])} for i in range(3)],
            "consensus": int(stage.decision[0]),
            "consensus_values": stage.aggregate[0].tolist(),
            "quorum": bool(stage.quorum[0]),
            "consensus_objection": bool(stage.consensus_objection[0]),
        }
    return dict(evidence={name: float(value) if value.isfinite() else None
                          for name, value in zip(names, evidence[0], strict=True)},
                first=round_(result.first), revised_experts=result.revised[0].nonzero().flatten().tolist(),
                final=round_(result.final), accepted=bool(result.accepted[0]),
                last_committed_decision=int(result.board.last_decision[0]),
                last_committed_votes=result.board.votes[0].tolist(),
                committed_decisions=int(result.board.commits[0]))


def boundaries(t, gap):
    return [0, *((t.diff() != gap).nonzero().flatten() + 1).tolist(), len(t)]


def windows(streams, gap):
    buckets = {}
    for data in streams:
        edges = boundaries(data[0], gap)
        for start, end in zip(edges, edges[1:]):
            for left in range(start, end, MAX_WINDOW-WARMUP-1):
                right = min(left + MAX_WINDOW, end)
                if right - left > WARMUP + 1:
                    buckets.setdefault(right-left, []).append(tuple(v[left:right] for v in data[1:]))
    return [tuple(torch.stack(v) for v in zip(*parts, strict=True)) for parts in buckets.values()]


def returns(times, rewards, gap):
    expected = rewards.clone()
    for i in range(len(expected)-2, -1, -1):
        if times[i+1] - times[i] == gap:
            expected[i] += DISCOUNT * expected[i+1]
    return expected


def epoch_loss(model, target, batch, gap, warm_start, weights):
    x, y, *rewards = batch
    minutes = x.new_full((len(x),), gap / 60)
    state, target_state, costs = None, None, []
    for i in range(x.shape[1] - 1):
        result = model(x[:, i], minutes, state, force_commit=warm_start)
        state = result.board
        if rewards:
            with torch.no_grad():
                current = target(x[:, i], minutes, target_state, force_commit=warm_start)
                target_state = current.board
                future = target(x[:, i+1], minutes, target_state, force_commit=warm_start)
                expected = rewards[0][:, i] + DISCOUNT * future.final.aggregate.amax(-1).clamp_max(0.)
        if i >= WARMUP:
            supervised = action_loss(result, y[:, i], expected) if rewards else classification_loss(result, y[:, i], weights)
            costs.append(supervised + .25 * sensor_loss(result, x[:, i+1]))
    return torch.stack(costs).mean()


def balanced(y, correct):
    groups = [correct[y == c].float().mean().item() for c in (0, 1) if (y == c).any()]
    return sum(groups) / len(groups)


def auc(y, p):
    pair = p[y == 1, None] - p[y == 0][None, :]
    return (pair.gt(0).float() + pair.eq(0).float() / 2).mean().item() if pair.numel() else None


def measure_class(y, scores, accepted):
    correct = scores.argmax(-1).eq(y) & accepted
    return dict(rows=len(y), occupied=int(y.sum()), correct=int(correct.sum()),
                coverage=accepted.float().mean().item(), balanced_accuracy=balanced(y, correct),
                score_auc=auc(y, scores.softmax(-1)[:, 1]))


def replay(model, data, *, live=False, unchecked=False, evidence=False):
    gap = INTERVAL[model.task]
    t, x, y, *reward = data
    state, last, forecasts, previous_reading = None, None, None, None
    learner = Online(model, gap) if live else None
    starting = {key: value.detach().clone() for key, value in model.named_parameters()} if live else {}
    scores, covered, receipts, errors, persistence, updates = [], [], [], [], [], []
    challenged, repaired, committed = 0, 0, 0
    for i, stamp in enumerate(t.tolist()):
        current = x[i:i+1]
        joined = last is not None and stamp - last == gap
        if not joined:
            state, forecasts, previous_reading = None, None, None
        minutes = x.new_tensor([gap / 60])
        event = Event(stamp, current, minutes, state)
        with torch.no_grad():
            result = model(current, minutes, state, force_commit=unchecked)
            state = result.board
            scores.append(result.final.aggregate[0])
            covered.append(result.accepted[0] if not unchecked else torch.ones_like(result.accepted[0]))
            challenged += int(result.revised.sum())
            repaired += int((result.revised & (result.first.votes != result.final.votes)).sum())
            committed += int(result.accepted[0])
            if evidence and (i == 0 or result.revised.any() or not result.accepted[0]) and len(receipts) < 6:
                receipts.append(dict(row=i, **trace(result, current, model.sensors)))
            if forecasts is not None:
                width = 5 if model.task == "occupancy" else 2
                measured = current[0, :width]
                usable = measured.isfinite() & previous_reading.isfinite()
                if usable.any():
                    errors.extend((forecasts[usable] - measured[usable]).abs().tolist())
                    persistence.extend((previous_reading[usable] - measured[usable]).abs().tolist())
            forecasts = result.forecast[0, :5 if model.task == "occupancy" else 2]
            previous_reading = current[0, :len(forecasts)]
        if learner:
            observed = dict(action=y[i-1:i], reward=reward[0][i-1:i]) if joined and reward else {}
            update = learner.feedback(event, result, **observed)
            if update is not None:
                updates.append(update)
        last = stamp
    summary = dict(challenged_proposals=challenged, changed_votes=repaired, committed=committed,
                   online_sensor_updates=len(updates), online_reward_updates=sum(v["reward_loss"] is not None for v in updates),
                   forecast_mae=sum(errors)/len(errors) if errors else None,
                   persistence_mae=sum(persistence)/len(persistence) if persistence else None,
                   trace=receipts)
    if live:
        for key, prefix in (("online_parameter_change_l1", ""), ("online_liquid_parameter_change_l1", "propose."),
                            ("online_falsifier_parameter_change_l1", "evidence_filter.")):
            summary[key] = sum((value.detach() - starting[name]).abs().sum().item()
                               for name, value in model.named_parameters() if name.startswith(prefix))
    p, called = torch.stack(scores), torch.stack(covered)
    if model.task == "heat":
        recorded = returns(t, reward[0], gap)
        error = (p.gather(1, y[:, None]).squeeze(1) - recorded).abs()
        votes = p.argmax(-1)[called]
        summary.update(rows=len(y), on=int(votes.sum()), off=len(votes)-int(votes.sum()),
                       coverage=called.float().mean().item(), logged_return_mae=error.mean().item(),
                       logged_return_mae_committed=error[called].mean().item() if called.any() else None,
                       recorded_on=int(y.sum()), recorded_off=len(y)-int(y.sum()))
    else:
        summary.update(measure_class(y, p, called))
    return summary


def train(task, fitting, validation, epochs, out):
    torch.manual_seed(51)
    model = Blackboard(task)
    target = deepcopy(model) if task == "heat" else None
    opt = torch.optim.AdamW(model.parameters(), lr=.008)
    groups = windows(fitting, INTERVAL[task])
    weights = None
    if task == "occupancy":
        counts = torch.cat([stream[2] for stream in fitting]).bincount(minlength=2).float()
        weights = counts.mean() / counts.clamp_min(1)
    curve, best, best_ranking, best_epoch = [], float("inf"), 0., 0
    if not groups or epochs < 1:
        raise ValueError("Fitting requires contiguous observations and at least one epoch")
    for epoch in range(1, epochs+1):
        for data in groups:
            for ids in torch.randperm(len(data[0])).split(128):
                opt.zero_grad()
                cost = epoch_loss(model, target, tuple(v[ids] for v in data), INTERVAL[task], epoch <= 3, weights)
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
        if target is not None:
            target.load_state_dict(model.state_dict())
        model.eval()
        val = replay(model, validation)
        model.train()
        error = val["logged_return_mae"] if task == "heat" else 1-val["balanced_accuracy"]
        ranking = val.get("score_auc") or 0.
        curve.append(dict(epoch=epoch, error=error, coverage=val["coverage"], score_auc=val.get("score_auc")))
        if error < best-EPSILON or (abs(error-best) < EPSILON and ranking > best_ranking+EPSILON):
            best, best_ranking, best_epoch = error, ranking, epoch
            pending = out / f"{task}.pending"
            torch.save(model.state_dict(), pending)
            pending.replace(out / f"{task}.pt")
        print(task, curve[-1], flush=True)
        if epoch-best_epoch >= 5:
            break
    return dict(curve=curve, best_epoch=best_epoch, training_windows=sum(len(g[0]) for g in groups),
                weights_sha256=sha(out / f"{task}.pt"), validation_rows=len(validation[0]))


@torch.no_grad()
def calibrate(model, validation):
    times, evidence, labels = validation
    proposal, critics = [], []
    before, last = None, None
    for i, stamp in enumerate(times.tolist()):
        if last is None or stamp - last != INTERVAL["occupancy"]:
            before = None
        values = evidence[i:i+1]
        result = model(values, values.new_tensor([5.]), before, force_commit=True)
        first = result.first
        proposal.append(first.proposals.values[0, :, 1] - first.proposals.values[0, :, 0])
        predictions = torch.cat((first.counter_values[0], result.independent), dim=0)
        critics.append(predictions[:, 1] - predictions[:, 0])
        before, last = result.board, stamp

    def thresholds(samples):
        offsets = []
        for scores in torch.stack(samples).T:
            grid = torch.quantile(scores, torch.linspace(.05, .95, 37)).unique()
            scores_by_cutoff = [(balanced(labels, (scores >= threshold).eq(labels.bool())), -abs(threshold.item()))
                                for threshold in grid]
            best = max(range(len(grid)), key=scores_by_cutoff.__getitem__)
            offsets.append(grid[best])
        return torch.stack(offsets)

    model.proposal_offset.copy_(thresholds(proposal))
    model.review_offset.copy_(thresholds(critics))
    return dict(proposals=model.proposal_offset.tolist(), reviewers=model.review_offset.tolist())


def baseline(training, x):
    features = torch.cat([s[1] for s in training])
    labels = torch.cat([s[2] for s in training])
    model = nn.Linear(features.shape[1]*2, 2)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=60, line_search_fn="strong_wolfe")

    def pack(value):
        return torch.cat((value.nan_to_num(), value.isfinite().float()), -1)

    train_x = pack(features)
    weights = labels.bincount(minlength=2).float().reciprocal()
    def objective():
        opt.zero_grad()
        loss = F.cross_entropy(model(train_x), labels, weight=weights)
        loss.backward()
        return loss
    opt.step(objective)
    return model(pack(x)).detach()


def run(args):
    task = args.task
    args.out.mkdir(parents=True, exist_ok=True)
    if task == "occupancy":
        sources = {}
        streams = []
        for home, days in enumerate((14,14,14,7,7,7), 1):
            data, hashes = occupancy(args.data, home, days)
            sources.update(hashes)
            streams.append(data)
            print(f"H{home}", len(data[0]), "occupied", int(data[-1].sum()), flush=True)
        training, validation, held_out = streams[:3], streams[3], list(zip(("H5", "H6"), streams[4:], strict=True))
    else:
        training, validation, test = heater(args.data)
        training, held_out = [training], [("heat-holdout", test)]
        sources = {args.data.name: sha(args.data)}
    record = train(task, training, validation, args.epochs, args.out)
    path = args.out / f"{task}.pt"
    weights = torch.load(path, weights_only=True)
    if task == "occupancy":
        model = Blackboard(task)
        model.load_state_dict(weights)
        record["h4_calibration"] = calibrate(model, validation)
        calibrated = replay(model, validation)
        record["h4_after_calibration"] = {key: calibrated[key] for key in ("balanced_accuracy", "coverage", "score_auc")}
        weights = model.state_dict()
        pending = path.with_suffix(".pending")
        torch.save(weights, pending)
        pending.replace(path)
        record["weights_sha256"] = sha(path)
    comparisons = {}
    for name, data in held_out:
        comparisons[name] = {}
        for mode in ("frozen", "live", "forced_commit"):
            model = Blackboard(task)
            model.load_state_dict(weights)
            model.eval()
            comparisons[name][mode] = replay(model, data, live=mode == "live",
                                             unchecked=mode == "forced_commit", evidence=mode != "forced_commit")
        if task == "occupancy":
            _, x, y = data
            torch.manual_seed(51)
            logits = baseline(training, x)
            comparisons[name]["logistic"] = measure_class(y, logits, torch.ones_like(y, dtype=torch.bool))
        print(name, {k: round(v["coverage"], 3) for k, v in comparisons[name].items()}, flush=True)
    prior = ROOT.parent / "liquid-workflow-v2" / "workflow.py"
    record.update(data=sources, torch_version=torch.__version__, code={p.name: sha(p) for p in sorted(ROOT.glob("*.py"))},
                  prior_workflow_sha256=sha(prior), comparisons=comparisons)
    (args.out / f"{task}.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=INTERVAL)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    torch.set_num_threads(1)
    run(parser.parse_args())
