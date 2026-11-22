"""Find a held-out reading where the rewind helps the forecast and hurts occupancy."""
import argparse
from pathlib import Path

import torch

from model import Workflow
from run import dump, io_data, sha


def find_cases(data, checkpoint, cutoff):
    torch.set_num_threads(1)
    model = Workflow().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    first = tradeoff = None
    comparisons = 0
    with torch.no_grad():
        for room, signals, labels, stamps in io_data.test_sites(data)["munich"]:
            ledger = None
            for row in range(len(signals) - 1):
                seconds = int(stamps[row] - stamps[row - 1]) if row else 300
                if seconds > 450:
                    ledger = None
                x = signals[row:row+1]
                elapsed = x.new_tensor([min(seconds, 450) / 60])
                result = model.step(x, elapsed, ledger, trace=True)
                original = ledger
                ledger = result.memory.detach()
                if not any("revisit" in attempt for attempt in result.attempts):
                    continue
                if int(stamps[row+1]-stamps[row]) > 450:
                    continue
                alternate = model.step(x, elapsed, original, trace=True, backtrack=False)
                comparisons += 1
                actual = bool(labels[row])

                def grade(candidate):
                    call = bool(candidate.logit.sigmoid()[0] >= cutoff) if candidate.accepted[0] else None
                    error = float((candidate.memory.forecast - signals[row+1:row+2]).abs().mean())
                    return dict(decision=call, correct=call is actual, next_sensor_error=error,
                                trace=candidate.attempts)

                case = dict(room=room, row=row, timestamp=int(stamps[row]), observed_occupied=actual,
                            with_rewind=grade(result), without_rewind=grade(alternate))
                if first is None:
                    first = case
                amended, unchanged = case["with_rewind"], case["without_rewind"]
                if (amended["next_sensor_error"] < unchanged["next_sensor_error"]
                        and not amended["correct"] and unchanged["correct"]):
                    tradeoff = case
                    return first, tradeoff, comparisons
    return first, tradeoff, comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cutoff", type=float, required=True)
    args = parser.parse_args()
    first, tradeoff, comparisons = find_cases(args.data, args.checkpoint, args.cutoff)
    dump(args.output, dict(checkpoint_sha256=sha(args.checkpoint), cutoff=args.cutoff,
                          compared_rewinds=comparisons, first=first, tradeoff=tradeoff))
