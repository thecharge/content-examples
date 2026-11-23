"""Learn bounded expert bias changes from delayed, recorded occupancy answers."""

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import torch

from benchmark import add_history, score
from learn import Feedback, board_predictions, by_class, deployment_gain, paired_margin
from model import Board
from sources import sensor_data, sha, source_hashes

WINDOW = 288
FIT_HISTORY = 576
MAX_SHIFT = 1.2
SHIFTS = (-.8, -.4, .4, .8)


class BoardFeedback(Feedback):
    def __init__(self, model):
        super().__init__(model, minimum=WINDOW)
        self.origin = model.outputs[-1].bias.detach().clone()
        self.history = tuple(deque(maxlen=FIT_HISTORY) for _ in range(2))

    @torch.no_grad()
    def try_update(self, fit, check):
        if fit[-1].timestamp >= check[0].timestamp:
            raise ValueError("Fit and check intervals overlap")
        # Each class queue can keep an older tail. Merging the latest rows
        # recovers a consecutive suffix of the observed stream.
        fit = fit[-FIT_HISTORY:]
        expected = torch.cat([item.truth for item in fit])
        if int(expected.bincount(minlength=2).min()) < self.minority:
            return dict(status="frozen_recent_history_one_class", accepted=[])

        old_answer, old_probability, _ = board_predictions(self.model, fit)
        old_right = old_answer.eq(expected).float()
        old_error = (old_probability - expected).square()
        candidate = deepcopy(self.model)
        bias = self.model.outputs[-1].bias
        best = None
        for step in SHIFTS:
            moved = self.origin + (bias + step - self.origin).clamp(-MAX_SHIFT, MAX_SHIFT)
            if torch.equal(moved, bias):
                continue
            candidate.outputs[-1].bias.copy_(moved)
            answer, probability, _ = board_predictions(candidate, fit)
            gain = float(by_class(answer.eq(expected).float() - old_right, expected))
            brier = float(by_class(old_error - (probability - expected).square(), expected))
            if gain > 0 and (best is None or (gain, brier) > best[:2]):
                best = gain, brier, step, moved.clone()
        details = dict(fit_first=fit[0].timestamp, fit_last=fit[-1].timestamp, fit_rows=len(fit),
                       check_rows=len(check), accepted=[])
        if best is None:
            return dict(details, status="frozen_no_fit_gain")
        fit_gain, fit_brier, step, moved = best
        candidate.outputs[-1].bias.copy_(moved)
        gain = deployment_gain(self.model, candidate, check)
        accepted = (gain["accuracy_gain"] >= .01 and gain["brier_gain"] >= .002
                    and gain["accuracy_lower"] is not None and gain["accuracy_lower"] > 0
                    and min(gain["class_gain"]) >= -.02 and gain["quorum_gain"] >= -.02)
        if accepted:
            bias.copy_(moved)
            details["accepted"] = list(range(1, self.model.experts + 1))
        return dict(details, status="accepted" if accepted else "frozen_no_board_gain",
                    step=step, fit_gain=fit_gain, fit_brier_gain=fit_brier, board_gain=gain)


def paired_replay(model, stream, evaluation_start):
    times, features, truth = stream
    adaptive = deepcopy(model)
    learner = BoardFeedback(adaptive)
    states = [None, None]
    predictions = [[], []]
    probabilities = [[], []]
    quorum = [[], []]
    last = None
    pending = None
    for index, stamp in enumerate(times.tolist()):
        if pending is not None:
            old_index, old_stamp, old_row, old_state, old_result = pending
            learner.observe(old_stamp, stamp, old_row, old_state, old_result, truth[old_index:old_index + 1])
        row = features[index:index + 1]
        dt = row.new_full((1,), 5.)
        with torch.no_grad():
            for variant, board in enumerate((model, adaptive)):
                if last is None or stamp - last != 300:
                    states[variant] = board.start(row)
                before = states[variant]
                result = board(row, dt, before)
                states[variant] = result.state.detached()
                if variant == 1:
                    pending = (index, stamp, row, before, result)
                predictions[variant].append(int(result.answer))
                probabilities[variant].append(float(result.probability))
                quorum[variant].append(int(result.committed))
        last = stamp

    expected = truth[evaluation_start:]
    guesses = [torch.tensor(values[evaluation_start:]) for values in predictions]
    chances = [torch.tensor(values[evaluation_start:]) for values in probabilities]
    metrics = [score(expected, answers, probs, with_interval=True)
               for answers, probs in zip(guesses, chances, strict=True)]
    gains = guesses[1].eq(expected).float() - guesses[0].eq(expected).float()
    lower, upper = paired_margin(gains, expected), paired_margin(-gains, expected)
    return dict(frozen=metrics[0], adaptive=metrics[1],
                evaluation_from=int(times[evaluation_start]),
                paired_gain_90=None if lower is None or upper is None else [lower, -upper],
                evaluation_quorum=[sum(values[evaluation_start:]) for values in quorum],
                online_updates=learner.receipts,
                accepted_updates=sum(bool(item["accepted"]) for item in learner.receipts),
                final_expert_bias=adaptive.outputs[-1].bias.detach().flatten().tolist())


def run(source, checkpoint, output):
    hashes = {**source_hashes(), "online.py": sha(Path(__file__).resolve())}
    board = Board(features=17).eval()
    board.load_state_dict(torch.load(checkpoint, weights_only=True))
    results, archives = {}, {}
    for home in (5, 6):
        original, _ = sensor_data.occupancy(source, home, 7)
        extended, used = sensor_data.occupancy(source, home, 14)
        times, rows, truth = extended
        split = len(original[0])
        if not torch.equal(original[0], times[:split]) or len(times) <= split:
            raise ValueError("The evaluation must start after the original seven days")
        archives.update(used)
        results[f"H{home}"] = paired_replay(board, (times, add_history(times, rows), truth), split)
        result = results[f"H{home}"]
        print(f"H{home}: {result['frozen']['balanced_accuracy']:.4f} -> "
              f"{result['adaptive']['balanced_accuracy']:.4f}, {result['accepted_updates']} updates", flush=True)
    record = dict(code_sha256=hashes, archive_sha256=archives, initial_checkpoint_sha256=sha(checkpoint),
                  previous_days=7, total_days=14, shift_options=SHIFTS, shift_limit=MAX_SHIFT,
                  fit_capacity=FIT_HISTORY, check_windows=WINDOW, homes=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.archives, args.checkpoint, args.output)
