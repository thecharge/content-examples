"""Fit and replay an offline heater policy against measured action outcomes."""

import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiment import dump, io_data, save, sha
from workflow import Blackboard, Observation, Online


def objective(model, target, inputs, ending, actions, rewards, continues):
    elapsed = inputs.new_full((len(inputs),), 15.)
    start = model.step(inputs[:, 0], inputs[:, 1], elapsed, training=True)
    with torch.no_grad():
        follow = target.step(ending[:, 0], ending[:, 1], elapsed, start.ledger, training=True, replay=True)
        q = follow.action_value.max(-1).values.clamp(-5, 0)
        target_reward = rewards + .95 * q * continues
        verified = ((start.claims - inputs[:, 1:]).abs() <= .55).float()
    logged = start.action_value.gather(1, actions[:, None]).squeeze(1)
    return (F.smooth_l1_loss(logged, target_reward)
            + .1 * (start.action_value.logsumexp(-1) - logged).mean()
            + .3 * F.smooth_l1_loss(start.claims, inputs[:, 1:].expand_as(start.claims))
            + .12 * F.binary_cross_entropy_with_logits(start.process, verified)
            + .2 * F.smooth_l1_loss(start.forecast, ending[:, 0]))


def fit(rows, out, epochs):
    observed = io_data.heat_tensors(rows)
    contiguous = torch.tensor([a[0] + 15 == b[0] for a, b in zip(rows, rows[1:])] + [False])
    data = (*observed, contiguous)
    split = 4 * len(rows) // 5
    model = Blackboard()
    target = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
    curve, best = [], None
    for epoch in range(1, epochs + 1):
        torch.manual_seed(41 + epoch)
        for ids in torch.randperm(split - 1).split(128):
            optimizer.zero_grad()
            loss = objective(model, target, *(v[ids] for v in data))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        target.load_state_dict(model.state_dict())
        with torch.no_grad():
            x, _, actions, reward, _ = (v[split:] for v in data)
            decisions = model.step(x[:, 0], x[:, 1], x.new_full((len(x),), 15.), training=True, replay=True)
            error = float((decisions.action_value.gather(1, actions[:, None]).squeeze(1) - reward).abs().mean())
        curve.append(error)
        if best is None or error < best[0]:
            best = error, epoch
            save(out / "heater.pt", model.state_dict())
        save(out / "heater-resume.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                           curve=curve, best_epoch=best[1]))
        if epoch - best[1] == 4:
            break
    model.load_state_dict(torch.load(out / "heater.pt", weights_only=True))
    return model, dict(validation_logged_return_error=curve, best_epoch=best[1], fit_rows=split-1,
                       validation_rows=len(rows)-split,
                       checkpoint_sha256=sha(out/"heater.pt"), resume_sha256=sha(out/"heater-resume.pt"))


def replay(model, rows, support, live):
    x, ending, actions, reward = io_data.heat_tensors(rows)
    learner = Online(model, max_gap=900) if live else None
    ledger, previous = None, None
    value_errors, examples, selected, updates = [], [], [], 0
    for i, row in enumerate(rows):
        stamp = row[0] * 60
        primary, witness = x[i:i+1].unbind(-1)
        minutes = x.new_tensor([15.])
        if previous is None or stamp - previous.time != 900:
            ledger = None
        event = Observation(stamp, primary, witness, minutes, ledger)
        counts = [int(((cohort - x[i]).abs() <= x.new_tensor([.175, .2])).all(-1).sum())
                  for cohort in support]
        with torch.no_grad():
            proposal = model.step(primary, witness, minutes, ledger)
            values = proposal.action_value[0]
            action = int(values.argmax()) if min(counts) >= 8 and bool(proposal.accepted[0]) else None
            selected.append(action)
            if bool(proposal.accepted[0]):
                value_errors.append(abs(float(values[actions[i]]) - float(reward[i])))
        if len(examples) < 4:
            examples.append(dict(row=i, time=stamp, proposed=action, recorded_action=int(actions[i]),
                                 recorded_reward=float(reward[i]), energy_kwh=row[4], discomfort_degree_hours=row[5],
                                 support=counts, predicted_values=values.tolist()))
        if learner:
            if ledger is None:
                learner.pending = None
            # Train after committing the current decision. The previous reward
            # came from the logged energy/air temperature at this transition's end.
            if learner.pending is not None:
                hint = float(values[actions[i]]) if bool(proposal.accepted[0]) else None
                updates += learner.feedback(event, observed_action=int(actions[i-1]),
                                            observed_reward=float(reward[i-1]),
                                            next_action_return=hint) is not None
            else:
                learner.feedback(event)
        ledger, previous = proposal.ledger.detach(), event
    return dict(rows=len(x), frozen_or_live="live" if live else "frozen", eligible=sum(a is not None for a in selected),
                on=selected.count(1), off=selected.count(0), logged_reward_error=(
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
