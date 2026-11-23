"""Fit and verify the liquid board on one target, human presence in a house."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import platform
from pathlib import Path

import torch
import xgboost as xgb

from model import Board, measured_reward, offline_loss
from learn import Example, Feedback, by_class, paired_margin
from sources import sensor_data, sha, source_hashes

STEP = 300


def add_history(times, features):
    """Append past-only offsets and local time for a causal baseline."""
    center = features.new_full((5,), float("nan"))
    offsets = []
    previous = None
    for stamp, row in zip(times.tolist(), features[:, :5], strict=True):
        if previous is None or stamp - previous != STEP:
            center = torch.full_like(center, float("nan"))
        offsets.append(row - center)
        center = torch.where(row.isfinite(), torch.where(center.isfinite(), .04 * row + .96 * center, row),
                             center)
        previous = stamp
    phase = 2 * math.pi * (times % 86400).float() / 86400
    return torch.cat((features, torch.stack(offsets), torch.stack((phase.sin(), phase.cos()), 1)), 1)


def data(source, history=True):
    streams, hashes = [], {}
    for home in range(1, 7):
        stream, files = sensor_data.occupancy(source, home, 14 if home <= 3 else 7)
        times, features, labels = stream
        streams.append((times, add_history(times, features) if history else features, labels))
        hashes.update(files)
    return streams, hashes


def blocks(streams, length=36):
    x, y = [], []
    for times, inputs, actual in streams:
        edges = [0, *(times.diff().ne(STEP).nonzero().flatten().add(1).tolist()), len(times)]
        for start, end in zip(edges, edges[1:]):
            for left in range(start, end - length + 1, length):
                x.append(inputs[left:left + length])
                y.append(actual[left:left + length])
    if not x:
        raise ValueError("No continuous training block has the required length")
    return torch.stack(x), torch.stack(y)


def score(actual, guesses, probability=None, with_interval=False):
    correct = guesses.eq(actual)
    classes = [dict(rows=int(actual.eq(k).sum()), right=int((actual.eq(k) & correct).sum()))
               for k in (0, 1)]
    result = dict(rows=len(actual), balanced_accuracy=sum(row["right"] / row["rows"] for row in classes
                                                          if row["rows"]) / sum(row["rows"] > 0 for row in classes),
                  accuracy=float(correct.float().mean()), classes=classes)
    if probability is not None:
        error = (probability - actual).square()
        result["brier"] = float(by_class(error, actual)) if all(row["rows"] for row in classes) else None
    if with_interval:
        right = correct.float()
        lower, minus_upper = paired_margin(right, actual), paired_margin(-right, actual)
        result["balanced_interval_90"] = [lower, -minus_upper] if lower is not None and minus_upper is not None else None
    return result


def expert_diagnostics(model, stream):
    """Compare what experts could predict, what they submitted and what counted."""
    times, rows, actual = stream
    summaries = {name: torch.zeros(2, model.experts, 2, dtype=torch.long)
                 for name in ("assessment", "acted", "spoke", "survived")}
    state, previous = None, None
    active = torch.tensor([True])
    with torch.no_grad():
        for stamp, row, truth in zip(times.tolist(), rows, actual.tolist(), strict=True):
            x = row.unsqueeze(0)
            if previous is None or stamp - previous != STEP:
                state = model.start(x)
            dt = x.new_tensor([5.])
            assessment = model.round(x, dt, state, x.new_zeros(1, model.experts, 3),
                                     active, False, assess=True)
            result = model(x, dt, state)
            item = result.rounds[0]
            for name, used in (("assessment", assessment.acted), ("acted", item.acted),
                               ("spoke", item.speaks), ("survived", item.votes.any(-1))):
                choices = assessment.choices if name == "assessment" else item.choices
                summaries[name][truth, :, 0] += used[0]
                summaries[name][truth, :, 1] += (used & choices.eq(truth))[0]
            state, previous = result.state, stamp
    return {key: value.tolist() for key, value in summaries.items()}


def replay(original, stream, *, online=False, checks=True, traces=False):
    model = deepcopy(original).eval() if online else original
    learner = Feedback(model, checks=checks) if online else None
    times, features, truth = stream
    minutes = features.new_tensor([5.])
    state, pending, previous = None, None, None
    answers, probabilities, examples, challenge_examples = [], [], [], []
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
            counts["challenges"] += int(item.challenged.sum())
            counts["challenged_votes"] += int((item.challenged & item.speaks).sum())
            counts["silent_after_acting"] += int((item.acted & ~item.speaks).sum())
            participation += acts
            active_counts[int(acts.sum())] += 1
            if len(result.rounds) > 1:
                patterns["".join("1" if x else "0" for x in acts.tolist())] += 1
        answer = int(result.answer)
        answers.append(answer)
        probabilities.append(float(result.probability))
        challenge = any((item.challenged & item.speaks).any() for item in result.rounds)
        include_example = len(examples) < 8 and (index < 3 or not adopted)
        include_challenge = len(challenge_examples) < 8 and challenge
        if traces and (include_example or include_challenge):
            example = dict(row=index, available_sensors=torch.isfinite(row[0]).nonzero().flatten().tolist(),
                                 answer=answer, quorum=adopted, rounds=[
                                     dict(acted=item.acted[0].tolist(), speaks=item.speaks[0].tolist(),
                                          interest=item.interest[0].tolist(), votes=[
                                              int(value) if spoke else None for value, spoke
                                              in zip(item.choices[0], item.speaks[0], strict=True)],
                                          confidence=item.confidence[0].tolist(), challenged=item.challenged[0].tolist())
                                     for item in result.rounds])
            if include_example:
                examples.append(example)
            if include_challenge:
                challenge_examples.append(example)
        pending = (index, stamp, row, state, result)
        state, previous = result.state.detached(), stamp
    certificate = {**score(truth, torch.tensor(answers), torch.tensor(probabilities), with_interval=traces), **counts,
                   "measured_proposal_rewards": credits.tolist(),
                   "expert_participation": participation.tolist(),
                   "rounds_by_number_of_experts": {str(k): v for k, v in sorted(active_counts.items())},
                   "retry_participation_patterns": dict(patterns)}
    if learner:
        certificate["updates"] = learner.receipts
        certificate["accepted_expert_updates"] = sum(len(receipt["accepted"]) for receipt in learner.receipts)
    if traces:
        certificate["examples"] = examples
        certificate["challenge_examples"] = challenge_examples
    return certificate, model


def fit(fitting, selection, epochs, expert_count, rounds, seed, length, residual):
    torch.manual_seed(seed)
    model = Board(features=fitting[0][1].shape[1], experts=expert_count, rounds=rounds)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.005)
    x, y = blocks(fitting, length)
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
                state = result.state if (step + 1) % 12 else result.state.detached()
                costs.append(offline_loss(result, labels[:, step], weight, residual))
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
    # Calibration and ablations use development Home 4 only.
    model.eval()
    options = []
    for offset, checker, speaking in ((0., .55, .60), (.25, .55, .60), (-.25, .55, .60),
                                     (0., .35, .60), (0., .85, .60), (0., .55, .57), (0., .55, .64)):
        model.vote_offset.fill_(offset)
        model.checker_logit.fill_(checker)
        model.speaking_confidence.fill_(speaking)
        validation, _ = replay(model, selection)
        options.append(dict(offset=offset, checker=checker, speaking=speaking, score=validation["balanced_accuracy"],
                            brier=validation["brier"], challenges=validation["challenged_votes"]))
    best_policy = max(options, key=lambda entry: (entry["score"], -entry["brier"]))
    model.vote_offset.fill_(best_policy["offset"])
    model.checker_logit.fill_(best_policy["checker"])
    model.speaking_confidence.fill_(best_policy["speaking"])
    scale_trials = []
    for scale in (.5, 1., 2.):
        model.probability_scale.fill_(scale)
        report, _ = replay(model, selection)
        scale_trials.append(dict(scale=scale, brier=report["brier"]))
    model.probability_scale.fill_(min(scale_trials, key=lambda entry: entry["brier"])["scale"])
    return model, curve, selected, options, scale_trials


def tree_fit(fitting):
    x, y = (torch.cat([home[k] for home in fitting]) for k in (1, 2))
    counts = y.bincount(minlength=2).float()
    weights = counts.mean() / counts[y]
    train = xgb.DMatrix(x.numpy(), label=y.numpy(), weight=weights.numpy(), nthread=1)
    return xgb.train(dict(tree_method="hist", objective="binary:logistic", eta=.04,
                          max_depth=3, nthread=1, seed=73), train, num_boost_round=150)


def run(source, output, epochs, experts, rounds, seed, length, use_history, residual):
    code = source_hashes()
    streams, hashes = data(source, use_history)
    model, history, best, policies, scales = fit(streams[:3], streams[3], epochs, experts, rounds, seed, length, residual)
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
        tree_probability = torch.tensor(tree.predict(xgb.DMatrix(x.numpy(), nthread=1)))
        entries = {"xgboost": score(y, tree_probability >= .5, tree_probability, with_interval=True),
                   "always_vacant": score(y, torch.zeros_like(y), torch.zeros_like(y), with_interval=True),
                   "always_occupied": score(y, torch.ones_like(y), torch.ones_like(y), with_interval=True)}
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
    if code != source_hashes():
        raise ValueError("Source files changed during the run")
    record = dict(dataset="HPDmobile occupancy", train_homes=[1, 2, 3], selection_home=4,
                  replay_homes=[5, 6], archive_sha256=hashes, code_sha256=code, layers=len(model.cells),
                  experts=experts, rounds=rounds, best_epoch=best, fit=history, replay=results,
                  seed=seed, training_block=length, sensor_history=use_history, residual_weight=residual,
                  features=model.features, development_policies=policies, calibration=scales,
                  development_experts=expert_diagnostics(model, streams[3]),
                  environment=dict(python=platform.python_version(), torch=torch.__version__, xgboost=xgb.__version__,
                                   machine=platform.machine()),
                  sha256={file.name: sha(file) for file in (checkpoint, tree_path, *sorted(output.glob("H*-adapted.pt")))})
    (output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--epochs", type=int, default=16)
    cli.add_argument("--experts", type=int, default=5)
    cli.add_argument("--rounds", type=int, default=4)
    cli.add_argument("--seed", type=int, default=73)
    cli.add_argument("--length", type=int, default=36)
    cli.add_argument("--no-history", action="store_true")
    cli.add_argument("--residual-weight", type=float, default=2.)
    arguments = cli.parse_args()
    if min(arguments.epochs, arguments.length) < 1 or arguments.residual_weight < 0:
        cli.error("Epochs and block length must be positive, and residual weight must be nonnegative")
    torch.set_num_threads(1)
    run(arguments.source, arguments.output, arguments.epochs, arguments.experts, arguments.rounds,
        arguments.seed, arguments.length, not arguments.no_history, arguments.residual_weight)
