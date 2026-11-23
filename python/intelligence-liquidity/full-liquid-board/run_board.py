"""Fit and compare one liquid board across independent verification policies."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch.nn import functional as F
import xgboost as xgb

from board import Board, claim_loss, observation_loss, reward_loss
from data import heater, occupancy, sha
from train import auc, balanced, windows

GAP = {"occupancy": 300, "heat": 900}


def pool(streams, gap):
    groups = windows(streams, gap)
    length = max(x.shape[1] for x, *_ in groups)
    return tuple(torch.cat([F.pad(column, (0, 0, 0, length - column.shape[1]), value=float("nan"))
                            if column.ndim == 3 else
                            F.pad(column, (0, length - column.shape[1]),
                                  value=float("nan") if column.is_floating_point() else 0)
                            for column in columns])
                 for columns in zip(*groups, strict=True))


def train_batch(model, batch, weights, target):
    x, labels, *rewards = batch
    minutes = x.new_full((len(x),), GAP[model.task] / 60)
    state, target_state, losses = None, None, []
    for i in range(x.shape[1] - 1):
        reading, later = x[:, i], x[:, i+1]
        result = model(reading, minutes, state)
        state = result.state
        if target is not None:
            with torch.no_grad():
                now = target(reading, minutes, target_state)
                target_state = now.state
                follow = target(later, minutes, target_state)
        if i < 3:
            continue
        observed = observation_loss(model, result, reading, later)
        if target is None:
            cost = claim_loss(model, result, reading, labels[:, i], weights)
        else:
            cost = reward_loss(result, labels[:, i], rewards[0][:, i], follow.action_value,
                               later.isfinite().any(-1))
        losses.append(cost + .25 * observed)
    return torch.stack(losses).mean()


def tree_review(model, training):
    x = torch.cat([stream[1] for stream in training])
    y = torch.cat([stream[2] for stream in training])
    reviewers = []
    for view in model.views:
        other = ~view
        batch = xgb.DMatrix(x[:, other].numpy(), label=y.numpy(), nthread=1)
        reviewer = xgb.train(dict(max_depth=2, eta=.05, nthread=1, tree_method="hist",
                                  objective="binary:logistic", seed=51), batch, num_boost_round=100)
        reviewers.append((other, reviewer))
    return reviewers


def review_all(reviewers, x):
    return torch.stack([torch.from_numpy(model.predict(xgb.DMatrix(x[:, cols].numpy(), nthread=1)))
                        for cols, model in reviewers], dim=-1)


def trace(result, row, views):
    details = []
    for step in result.rounds:
        votes = []
        for i, view in enumerate(views):
            evidence = (view & row[0].isfinite()).nonzero().flatten().tolist()
            spoke = bool(step.speaking[0, i])
            votes.append(dict(expert=i+1, evidence_indices=evidence, spoke=spoke,
                              reason="no readings" if not evidence else
                              "expected payoff supports a claim" if spoke else "passes on current evidence",
                              claim=int(step.proposal[0, i]) if spoke else None,
                              confidence=float(step.confidence[0, i]), drive=float(step.drive[0, i]),
                              witness_rejected=bool(step.witness_rejected[0, i]),
                              vote_rejected=bool(step.vote_rejected[0, i])))
        details.append(dict(experts=votes, agreement=int(step.supporters[0].sum()),
                            answer=int(step.answer[0]), confidence=float(step.answer_confidence[0])))
    return dict(rounds=details, answer=int(result.answer[0]), committed=bool(result.committed[0]))


def replay(original, stream, *, mode="learned", online=False, reviewers=(), keep_trace=False):
    model = deepcopy(original) if online else original
    model.eval()
    gap = GAP[model.task]
    stamps, x, y, *reward = stream
    minutes = x.new_tensor([gap / 60])
    trees = review_all(reviewers, x) if mode == "tree" else None
    optimizer = torch.optim.SGD(model.parameters(), lr=.001) if online else None
    state, previous, last, instances = None, None, None, []
    answers, scores, states, forecasts = [], [], [], []
    verified, updates, rejected, rounds, action_updates = 0, 0, 0, 0, 0
    return_error = []
    before_weights = [p.detach().clone() for p in model.parameters()] if online else []
    for i, stamp in enumerate(stamps.tolist()):
        if last is None or stamp - last != gap:
            state, previous = None, None
        row = x[i:i+1]
        other = None if trees is None else trees[i:i+1]
        with torch.no_grad():
            decision = model(row, minutes, state, mode=mode, counter=other)
            answer = int(decision.answer[0])
            confidence = float(decision.confidence[0])
            answers.append(answer)
            scores.append(confidence if answer else 1 - confidence)
            states.append(bool(decision.committed[0]))
            rounds += len(decision.rounds)
            rejected += sum(int((step.witness_rejected | step.vote_rejected).sum()) for step in decision.rounds)
            if keep_trace and len(instances) < 3 and (i == 0 or not states[-1]):
                instances.append(dict(row=i, **trace(decision, row, model.views)))
        if previous is not None:
            before, prior_row, prior_counter, issued = previous
            present = row[:, :model.sensor_count].isfinite()
            if present.any():
                forecasts.extend((issued.forecast - row[:, :model.sensor_count]).abs()[present].tolist())
            if reward:
                earned = reward[0][i-1:i]
                expected = earned + .95 * decision.action_value.amax(-1)
                logged = issued.action_value.gather(1, y[i-1:i, None]).squeeze(1)
                return_error.append(float((logged - expected).abs()))
            if optimizer is not None and (present.any() or reward):
                optimizer.zero_grad()
                prediction = model(prior_row, minutes, before, mode=mode, counter=prior_counter)
                loss = observation_loss(model, prediction, prior_row, row)
                if reward:
                    loss = loss + reward_loss(prediction, y[i-1:i], earned, decision.action_value,
                                              torch.ones_like(earned))
                    action_updates += int(earned.isfinite().all())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .25, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
        previous = (state, row, other, decision)
        state, last = decision.state.detached(), stamp
    answers = torch.tensor(answers)
    correct = answers.eq(y)
    verified = sum(states)
    outcome = dict(rows=len(y), final_answers=len(answers), quorum=verified, exhausted=len(y)-verified,
                   rounds=rounds, objections=rejected, sensor_updates=updates, reward_updates=action_updates,
                   forecast_mae=sum(forecasts)/len(forecasts) if forecasts else None,
                   parameter_change_l1=sum(float((a - b.detach()).abs().sum())
                                           for a, b in zip(before_weights, model.parameters())) if online else 0.)
    if keep_trace:
        outcome["trace"] = instances
    if reward:
        outcome.update(logged_td_mae=sum(return_error)/len(return_error),
                       on=int(answers.sum()), matches_logged_action=int(correct.sum()))
    else:
        outcome.update(balanced_accuracy=balanced(y, correct), ordinary_accuracy=float(correct.float().mean()),
                       ranking_auc=auc(y, torch.tensor(scores)),
                       vacant=dict(rows=int(y.eq(0).sum()), right=int((y.eq(0) & correct).sum())),
                       occupied=dict(rows=int(y.eq(1).sum()), right=int((y.eq(1) & correct).sum())))
    return outcome


def fit(task, fitting, validation, epochs, max_rounds):
    torch.manual_seed(51)
    model = Board(task, max_rounds=max_rounds)
    target = deepcopy(model).eval() if task == "heat" else None
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    counts = torch.cat([stream[2] for stream in fitting]).bincount(minlength=2).float()
    weights = counts.mean() / counts.clamp_min(1)
    batches = pool(fitting, GAP[task])
    curve, best, saved, selected = [], -float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        values = []
        for ids in torch.randperm(len(batches[0])).split(128):
            opt.zero_grad()
            loss = train_batch(model, tuple(col[ids] for col in batches), weights, target)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            values.append((float(loss.detach()), float(norm)))
        validation_result = replay(model, validation)
        objective = validation_result.get("balanced_accuracy")
        if objective is None:
            objective = -validation_result["logged_td_mae"]
        if objective > best:
            best, selected, saved = objective, epoch, deepcopy(model.state_dict())
        report = dict(epoch=epoch, training_loss=sum(v[0] for v in values)/len(values),
                      largest_gradient=max(v[1] for v in values), validation=validation_result)
        curve.append(report)
        print(json.dumps(report), flush=True)
        if epoch - selected >= 5:
            break
        if target is not None:
            target.load_state_dict(model.state_dict())
    model.load_state_dict(saved)
    return model.eval(), selected, curve


def run(task, source, output, epochs, max_rounds):
    if task == "occupancy":
        streams, archives = [], {}
        for home in range(1, 7):
            stream, hashes = occupancy(source, home, 14 if home <= 3 else 7)
            streams.append(stream)
            archives.update(hashes)
        fitting, validation, test = streams[:3], streams[3], {"H5": streams[4], "H6": streams[5]}
    else:
        fit_data, validation, heldout = heater(source)
        fitting, test = [fit_data], {"held_out_heat": heldout}
        archives = {source.name: sha(source)}
    model, selected, curve = fit(task, fitting, validation, epochs, max_rounds)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "board.pt"
    torch.save(model.state_dict(), checkpoint)
    reviews = tree_review(model, fitting) if task == "occupancy" else []
    reviewer_digests = []
    for i, (_, reviewer) in enumerate(reviews, 1):
        path = output / f"reviewer-{i}.json"
        reviewer.save_model(path)
        reviewer_digests.append(sha(path))
    variants = [("off", False), ("learned", False), ("learned", True)]
    if reviews:
        variants.extend([("tree", False), ("tree", True)])
    results = {}
    for mode, live in variants:
        name = f"{mode}_{'live' if live else 'frozen'}"
        results[name] = {}
        for site, readings in test.items():
            score = replay(model, readings, mode=mode, online=live, reviewers=reviews, keep_trace=True)
            results[name][site] = score
            print(name, site, {k: v for k, v in score.items() if k != "trace"}, flush=True)
    root = Path(__file__).resolve().parent
    code = {name: sha(root / name) for name in ("board.py", "run_board.py")}
    record = dict(task=task, archives=archives, source_sha256=code, checkpoint_sha256=sha(checkpoint),
                  reviewer_sha256=reviewer_digests, max_rounds=max_rounds, expert_count=model.count,
                  quorum=model.quorum, learning_rate=.003, max_epochs=epochs,
                  best_epoch=selected, curve=curve, results=results)
    (output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=GAP)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.rounds) < 1:
        parser.error("Use at least one epoch and one round.")
    torch.set_num_threads(1)
    run(args.task, args.source, args.output, args.epochs, args.rounds)
