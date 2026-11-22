"""Stream blackboard decisions and their evidence as one JSON object per line."""

import argparse
import json
from pathlib import Path
import sys

import torch

from data import INTERVAL, sha
from blackboard import Blackboard
from online import Event, Online
from train import trace


def stream(lines, model, live=False, trace_updates=False):
    learner = Online(model, INTERVAL[model.task]) if live else None
    state, last = None, None
    for line in lines:
        row = json.loads(line)
        stamp = row["time"]
        if type(stamp) is not int or (last is not None and stamp <= last):
            raise ValueError("time must increase and use integer seconds")
        if last is None or stamp - last != INTERVAL[model.task]:
            state = None
        evidence = torch.tensor([[float(row["evidence"][field]) if row["evidence"].get(field) is not None
                                  else float("nan") for field in model.sensors]])
        minutes = evidence.new_tensor([INTERVAL[model.task] / 60])
        event = Event(stamp, evidence, minutes, state)
        prior_action = row.get("previous_executed_action")
        prior_reward = row.get("previous_measured_reward")
        if (prior_action is None) != (prior_reward is None):
            raise ValueError("Previous action and measured reward must arrive together")
        if prior_action is not None:
            if (model.task != "heat" or type(prior_action) is not int or prior_action not in (0, 1)
                    or not torch.isfinite(torch.tensor(float(prior_reward)))):
                raise ValueError("Heating feedback requires action 0 or 1 and a finite reward")
        with torch.no_grad():
            decision = model(evidence, minutes, state)
            record = trace(decision, evidence, model.sensors)
            answer = int(decision.final.decision[0]) if decision.accepted[0] else None
        yield dict(time=stamp, decision=answer, **record)
        if learner:
            feedback = {} if prior_action is None else dict(action=torch.tensor([prior_action]),
                                                            reward=torch.tensor([float(prior_reward)]))
            update = learner.feedback(event, decision, **feedback)
            if trace_updates and update is not None:
                yield dict(event="weight_update", time=stamp, previous_time=last, **update)
        last, state = stamp, decision.board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=INTERVAL)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--trace-updates", action="store_true")
    parser.add_argument("--save-adapted", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt_path = args.evidence / f"{args.task}.json"
    checkpoint = args.evidence / f"{args.task}.pt"
    receipt = json.loads(receipt_path.read_text())
    if args.trace_updates and not args.live:
        parser.error("--trace-updates requires --live")
    code = Path(__file__).resolve().parent
    if any(sha(code / name) != fingerprint for name, fingerprint in receipt["code"].items()):
        raise ValueError("Code differs from the recorded fit")
    if sha(code.parent / "liquid-workflow-v2" / "workflow.py") != receipt["prior_workflow_sha256"]:
        raise ValueError("Liquid primitives differ from the recorded fit")
    if sha(checkpoint) != receipt["weights_sha256"]:
        raise ValueError("Checkpoint differs from the recorded fit")
    model = Blackboard(args.task)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    if args.save_adapted and (not args.live or args.save_adapted.resolve() == checkpoint.resolve()
                              or args.save_adapted.exists()):
        raise ValueError("Save adapted weights from a live run to a new path")
    for result in stream(sys.stdin, model, args.live, args.trace_updates):
        print(json.dumps(result, allow_nan=False), flush=True)
    if args.save_adapted:
        pending = args.save_adapted.with_suffix(args.save_adapted.suffix + ".pending")
        torch.save(model.state_dict(), pending)
        pending.rename(args.save_adapted)


if __name__ == "__main__":
    main()
