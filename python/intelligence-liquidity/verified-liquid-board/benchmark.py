"""Fit and verify the liquid board on one target, human presence in a house."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import torch
import xgboost as xgb

from model import Board, measured_reward, offline_loss
from learn import Feedback
from data import occupancy, sha
from train import balanced, boundaries

STEP = 300


def data(source):
    streams, hashes = [], {}
    for home in range(1, 7):
        stream, files = occupancy(source, home, 14 if home <= 3 else 7)
        streams.append(stream)
        hashes.update(files)
    return streams, hashes


def blocks(streams, length=12):
    x, y = [], []
    for times, inputs, actual in streams:
        edges = boundaries(times, STEP)
        for start, end in zip(edges, edges[1:]):
            for left in range(start, end - length + 1, length):
                x.append(inputs[left:left + length])
                y.append(actual[left:left + length])
    return torch.stack(x), torch.stack(y)


def score(actual, guesses):
    correct = guesses.eq(actual)
    return dict(rows=len(actual), balanced_accuracy=balanced(actual, correct),
                accuracy=correct.float().mean().item(),
                classes=[dict(rows=int(actual.eq(k).sum()), right=int((actual.eq(k) & correct).sum()))
                         for k in (0, 1)])


def replay(original, stream, *, online=False, checks=True, traces=False):
    model = deepcopy(original).eval() if online else original
    learner = Feedback(model) if online else None
    times, features, truth = stream
    minutes = features.new_tensor([5.])
    state, pending, previous = None, None, None
    answers, examples = [], []
    counts = Counter()
    credits = torch.zeros(model.experts)
    participation = torch.zeros(model.experts, dtype=torch.long)
    active_counts = Counter()
    patterns = Counter()
    for index, stamp in enumerate(times.tolist()):
        row = features[index:index + 1]
        if pending is not None:
            past_index, past_stamp, old_row, before, issued = pending
            actual = truth[past_index:past_index + 1]
            if learner:
                reward = learner.observe(past_stamp, stamp, old_row, before, issued, actual)
            else:
                reward = measured_reward(issued.rounds, actual)
            credits += reward[0].detach()
            counts["answers_revealed"] += 1
        if previous is None or stamp - previous != STEP:
            state = model.start(row)
        with torch.no_grad():
            result = model(row, minutes, state, checks=checks)
        adopted = bool(result.committed)
        counts["quorum"] += int(adopted)
        counts["exhausted"] += int(not adopted)
        counts["rounds"] += len(result.rounds)
        for item in result.rounds:
            acts = item.acted[0]
            participation += acts
            active_counts[int(acts.sum())] += 1
            if len(result.rounds) > 1:
                patterns["".join("1" if x else "0" for x in acts.tolist())] += 1
        answer = int(result.answer)
        answers.append(answer)
        if traces and len(examples) < 8 and (index < 3 or not adopted):
            examples.append(dict(row=index, available_sensors=torch.isfinite(row[0]).nonzero().flatten().tolist(),
                                 answer=answer, quorum=adopted, rounds=[
                                     dict(acted=item.acted[0].tolist(), speaks=item.speaks[0].tolist(),
                                          interest=item.interest[0].tolist(), votes=[
                                              int(value) if spoke else None for value, spoke
                                              in zip(item.choices[0], item.speaks[0], strict=True)],
                                          confidence=item.confidence[0].tolist(), challenged=item.challenged[0].tolist())
                                     for item in result.rounds]))
        pending = (index, stamp, row, state, result)
        state, previous = result.state.detached(), stamp
    certificate = {**score(truth, torch.tensor(answers)), **counts,
                   "measured_proposal_rewards": credits.tolist(),
                   "expert_participation": participation.tolist(),
                   "rounds_by_number_of_experts": {str(k): v for k, v in sorted(active_counts.items())},
                   "retry_participation_patterns": dict(patterns)}
    if learner:
        certificate["updates"] = learner.receipts
        certificate["accepted_expert_updates"] = sum(len(receipt["accepted"]) for receipt in learner.receipts)
    if traces:
        certificate["examples"] = examples
    return certificate, model


def fit(fitting, selection, epochs, expert_count, rounds):
    torch.manual_seed(73)
    model = Board(experts=expert_count, rounds=rounds)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.005)
    x, y = blocks(fitting)
    counts = y.flatten().bincount(minlength=2).float()
    weight = counts.mean() / counts
    curve, selected, best, saved = [], 0, -1., None
    for epoch in range(1, epochs + 1):
        model.train()
        entries = []
        for indices in torch.randperm(len(x)).split(32):
            batch, labels = x[indices], y[indices]
            state, costs = None, []
            dt = batch.new_full((len(batch),), 5.)
            for step in range(batch.shape[1]):
                result = model(batch[:, step], dt, state)
                state = result.state
                costs.append(offline_loss(result, labels[:, step], weight))
            optimizer.zero_grad()
            loss = torch.stack(costs).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            entries.append(float(loss.detach()))
        validation, _ = replay(model.eval(), selection)
        value = validation["balanced_accuracy"]
        curve.append(dict(epoch=epoch, training_loss=sum(entries) / len(entries), validation=validation))
        print(json.dumps(curve[-1]), flush=True)
        if value > best:
            best, selected, saved = value, epoch, deepcopy(model.state_dict())
        if epoch - selected >= 4:
            break
    model.load_state_dict(saved)
    return model.eval(), curve, selected


def tree_fit(fitting):
    x, y = (torch.cat([home[k] for home in fitting]) for k in (1, 2))
    counts = y.bincount(minlength=2).float()
    weights = counts.mean() / counts[y]
    train = xgb.DMatrix(x.numpy(), label=y.numpy(), weight=weights.numpy(), nthread=1)
    return xgb.train(dict(tree_method="hist", objective="binary:logistic", eta=.04,
                          max_depth=3, nthread=1, seed=73), train, num_boost_round=150)


def run(source, output, epochs, experts, rounds):
    streams, hashes = data(source)
    model, history, best = fit(streams[:3], streams[3], epochs, experts, rounds)
    tree = tree_fit(streams[:3])
    output.mkdir(exist_ok=True, parents=True)
    checkpoint = output / "board.pt"
    tree_path = output / "tree.json"
    torch.save(model.state_dict(), checkpoint)
    tree.save_model(tree_path)
    results = {}
    for i, stream in enumerate(streams[4:], 5):
        home = f"H{i}"
        x, y = stream[1:]
        tree_answer = torch.tensor(tree.predict(xgb.DMatrix(x.numpy(), nthread=1))) >= .5
        entries = {"xgboost": score(y, tree_answer), "constant": score(y, torch.zeros_like(y))}
        for name, online, checks in (("without_checks", False, False), ("frozen", False, True),
                                     ("delayed_feedback", True, True)):
            entries[name], fitted = replay(model, stream, online=online, checks=checks, traces=True)
            if online:
                path = output / f"{home}-adapted.pt"
                torch.save(fitted.state_dict(), path)
                entries[name]["adapted_sha256"] = sha(path)
            print(home, name, {key: value for key, value in entries[name].items()
                               if key not in ("examples", "updates")}, flush=True)
        results[home] = entries
    root = Path(__file__).resolve().parent
    code = {name: sha(root / name) for name in ("model.py", "learn.py", "benchmark.py")}
    record = dict(dataset="HPDmobile occupancy", train_homes=[1, 2, 3], selection_home=4,
                  replay_homes=[5, 6], archive_sha256=hashes, code_sha256=code, layers=len(model.cells),
                  experts=experts, rounds=rounds, best_epoch=best, fit=history, replay=results,
                  sha256={checkpoint.name: sha(checkpoint), tree_path.name: sha(tree_path)})
    (output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--epochs", type=int, default=16)
    cli.add_argument("--experts", type=int, default=5)
    cli.add_argument("--rounds", type=int, default=4)
    arguments = cli.parse_args()
    if arguments.epochs < 1:
        cli.error("Use at least one training epoch")
    torch.set_num_threads(1)
    run(arguments.source, arguments.output, arguments.epochs, arguments.experts, arguments.rounds)
