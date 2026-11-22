"""Fit and replay an offline heater policy against measured action outcomes."""

from collections import defaultdict
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment import dump, io_data, save, sha
from workflow import Blackboard, DISCOUNT, Observation, Online, bellman_return, verification_loss


def trajectory_batches(continues, length):
    """Keep every transition and never carry recurrent state across a gap."""
    groups = defaultdict(list)
    start = 0
    for i, connected in enumerate(continues.tolist()):
        size = i + 1 - start
        if size == length or not connected:
            groups[size].append(start)
            start = i + 1
    return [torch.tensor(starts)[:, None] + torch.arange(size) for size, starts in groups.items()]


def transitions(rows):
    contiguous = torch.tensor([a[0] + 15 == b[0] for a, b in zip(rows, rows[1:])] + [False])
    return (*io_data.heat_tensors(rows), contiguous)


def target_value(target, inputs, ending, elapsed, state):
    with torch.no_grad():
        current = target.step(*inputs.unbind(-1), elapsed, state, training=True, replay=True)
        follow = target.step(*ending.unbind(-1), elapsed, current.ledger, training=True, replay=True)
        return follow.action_value.max(-1).values, current.ledger


def objective(model, target, data, batch):
    state, target_state, losses = None, None, []
    for ids in batch.unbind(1):
        inputs, ending, actions, rewards, continues = (v[ids] for v in data)
        elapsed = inputs.new_full((len(inputs),), 15.)
        start = model.step(*inputs.unbind(-1), elapsed, state, training=True)
        next_value, target_state = target_value(target, inputs, ending, elapsed, target_state)
        expected = bellman_return(rewards, torch.where(continues, next_value, 0.))
        logged = start.action_value.gather(1, actions[:, None]).squeeze(1)
        losses.append(F.smooth_l1_loss(logged, expected)
                      + .1 * (start.action_value.logsumexp(-1) - logged).mean()
                      + verification_loss(start, inputs[:, 1])
                      + .2 * F.smooth_l1_loss(start.forecast, ending[:, 0])
                      + .02 * start.route_loss)
        state = start.ledger
    return torch.stack(losses).mean()


@torch.no_grad()
def validate(model, target, data):
    x, ending, actions, rewards, continues = data
    state, target_state, td_errors, return_errors = None, None, [], []
    returns = logged_returns(rewards, continues)
    elapsed = x.new_tensor([15.])
    for i in range(len(x)):
        row = x[i:i+1]
        current = model.step(*row.unbind(-1), elapsed, state, training=True, replay=True)
        next_value, target_state = target_value(target, row, ending[i:i+1], elapsed, target_state)
        expected = bellman_return(rewards[i], torch.where(continues[i], next_value, 0.))
        logged = current.action_value[0, actions[i]]
        td_errors.append((logged - expected).abs())
        return_errors.append((logged - returns[i]).abs())
        state = current.ledger if continues[i] else None
        if not continues[i]:
            target_state = None
    return torch.cat(td_errors).mean().item(), torch.stack(return_errors).mean().item()


def logged_returns(rewards, continues):
    """Discount measured rewards up to the next break in the log."""
    returns = rewards.clone()
    for i in range(len(returns) - 2, -1, -1):
        if continues[i]:
            returns[i] += DISCOUNT * returns[i+1]
    return returns


def fit(rows, out, epochs, window=8):
    split = 4 * len(rows) // 5
    if not 0 < split < len(rows) or epochs < 1 or window < 1:
        raise ValueError("Heater fitting needs nonempty fit and validation splits, epochs and a positive window")
    training, valid = transitions(rows[:split]), transitions(rows[split:])
    batches = trajectory_batches(training[-1], window)
    model = Blackboard()
    target = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    curve, td_curve, best = [], [], None
    for epoch in range(1, epochs + 1):
        torch.manual_seed(41 + epoch)
        shuffled = [b[torch.randperm(len(b))].split(64) for b in batches]
        minibatches = [batch for group in shuffled for batch in group]
        for index in torch.randperm(len(minibatches)).tolist():
            optimizer.zero_grad()
            loss = objective(model, target, training, minibatches[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        td_error, error = validate(model, target, valid)
        td_curve.append(td_error)
        target.load_state_dict(model.state_dict())
        curve.append(error)
        if best is None or error < best[0]:
            best = error, epoch
            save(out / "heater.pt", model.state_dict())
        save(out / "heater-resume.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                           validation_logged_return_mae=curve, validation_td_error=td_curve,
                                           best_epoch=best[1]))
        print("heater", epoch, "logged return MAE", round(error, 4), "TD MAE", round(td_error, 4), flush=True)
        if epoch - best[1] == 4:
            break
    model.load_state_dict(torch.load(out / "heater.pt", weights_only=True))
    return model, dict(validation_td_error=td_curve, validation_logged_return_mae=curve, best_epoch=best[1], fit_rows=split,
                       fit_windows=sum(len(b) for b in batches), max_window_transitions=window,
                       validation_rows=len(rows)-split, discount=DISCOUNT, bootstrap_clip=[-5, 0],
                       checkpoint_sha256=sha(out/"heater.pt"), resume_sha256=sha(out/"heater-resume.pt"))


def replay(model, rows, support, live):
    x, _, actions, reward, continues = transitions(rows)
    reference = logged_returns(reward, continues)
    learner = Online(model, max_gap=900) if live else None
    ledger, previous = None, None
    value_errors, examples, selected, updates = [], [], [], 0
    idle_minutes = 0.
    for i, row in enumerate(rows):
        stamp = row[0] * 60
        primary, witness = x[i:i+1].unbind(-1)
        minutes = x.new_tensor([15.])
        if previous is None or stamp - previous.time != 900:
            ledger, idle_minutes = None, 0.
        event = Observation(stamp, primary, witness, minutes, ledger, idle_minutes)
        counts = [int(((cohort - x[i]).abs() <= x.new_tensor([.175, .2])).all(-1).sum())
                  for cohort in support]
        with torch.no_grad():
            proposal = model.step(primary, witness, minutes, ledger, idle_minutes=idle_minutes)
            values = proposal.action_value[0]
            action = int(values.argmax()) if min(counts) >= 8 and bool(proposal.accepted[0]) else None
            selected.append(action)
            if bool(proposal.accepted[0]):
                value_errors.append(abs(float(values[actions[i]]) - float(reference[i])))
        if len(examples) < 4:
            examples.append(dict(row=i, time=stamp, proposed=action, recorded_action=int(actions[i]),
                                 recorded_reward=float(reward[i]), discounted_logged_return=float(reference[i]), energy_kwh=row[4], discomfort_degree_hours=row[5],
                                 support=counts, predicted_values=values.tolist()))
        if learner:
            if ledger is None:
                learner.pending = None
            # Train after committing the current decision. The previous reward
            # came from the logged energy/air temperature at this transition's end.
            if learner.pending is not None:
                hint = float(values.max())
                updates += learner.feedback(event, observed_action=int(actions[i-1]),
                                            observed_reward=float(reward[i-1]),
                                            next_action_return=hint) is not None
            else:
                learner.feedback(event)
        idle_minutes = 0. if proposal.accepted.item() else idle_minutes + minutes.item()
        ledger, previous = proposal.ledger.detach(), event
    return dict(rows=len(x), frozen_or_live="live" if live else "frozen", eligible=sum(a is not None for a in selected),
                on=selected.count(1), off=selected.count(0), discounted_logged_return_mae=(
                    sum(value_errors)/len(value_errors) if value_errors else None),
                compared_logged_rows=len(value_errors), updates=updates, example=examples)


def run(args):
    torch.manual_seed(41)
    archive = args.data / "six-buildings.zip"
    training, test = io_data.heat(archive)
    model, receipt = fit(training, args.out, args.epochs)
    x, _, actions, _ = io_data.heat_tensors(training[:receipt["fit_rows"]])
    support = [x[actions == action] for action in (0, 1)]
    frozen = replay(model, test, support, False)
    initial = {name: value.detach().clone() for name, value in model.named_parameters() if ".expert." in name}
    live = replay(model, test, support, True)
    live["expert_parameter_l1_change"] = sum(float((value.detach()-initial[name]).abs().sum())
        for name, value in model.named_parameters() if name in initial)
    dump(args.out / "control.json", dict(source_sha256=sha(archive), fit=receipt, frozen=frozen, live=live,
         code={p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("workflow.py"))},
         trial="Logged action replay in one building. No actual counterfactual or heater command."))
    print(json.dumps({mode: report["eligible"] for mode, report in [("frozen", frozen), ("live", live)]}))
