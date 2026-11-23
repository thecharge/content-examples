"""Test a sensor-dependent vote correction on a separate chronological range."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark import add_history
from learn import board_predictions, by_class, deployment_gain
from online import BoardFeedback, FIT_HISTORY, paired_replay
from model import Board
from sources import sensor_data, sha, source_hashes


class ContextBoard(Board):
    def __init__(self):
        super().__init__(features=17)
        self.register_buffer("correction", torch.zeros(35))

    @staticmethod
    def context(rows):
        valid = rows.isfinite()
        return torch.cat((torch.where(valid, rows, 0.).clamp(-3., 3.),
                          valid.to(rows.dtype), rows.new_ones(len(rows), 1)), dim=1)

    def stages(self, x, *args, **kw):
        cells, margins, *other = super().stages(x, *args, **kw)
        correction = (self.context(x) @ self.correction).clamp(-.8, .8)
        updated = margins[..., -1:] + correction[:, None, None]
        return cells, torch.cat((margins[..., :-1], updated), dim=-1), *other


class ContextFeedback(BoardFeedback):
    @torch.no_grad()
    def try_update(self, fit, check):
        if fit[-1].timestamp >= check[0].timestamp:
            raise ValueError("Fit and check intervals overlap")
        fit = fit[-FIT_HISTORY:]
        expected = torch.cat([item.truth for item in fit])
        if int(expected.bincount(minlength=2).min()) < self.minority:
            return dict(status="frozen_recent_history_one_class", accepted=[])
        old_answer, old_probability, _ = board_predictions(self.model, fit)
        inputs = self.model.context(torch.cat([item.inputs for item in fit]))
        prior = torch.logit(old_probability.clamp(.01, .99))
        # The labels are revealed observations. Fit a regularised residual over
        # the actual probabilities the current board assigned those rows.
        with torch.enable_grad():
            residual = nn.Parameter(torch.zeros_like(self.model.correction))
            optimizer = torch.optim.Adam([residual], lr=.04)
            for _ in range(60):
                optimizer.zero_grad()
                losses = F.binary_cross_entropy_with_logits(prior + inputs @ residual,
                                                            expected.float(), reduction="none")
                loss = by_class(losses, expected) + .08 * residual.square().sum()
                loss.backward()
                optimizer.step()
        candidate = deepcopy(self.model)
        old_right = old_answer.eq(expected).float()
        old_error = (old_probability - expected).square()
        best = None
        for step in (.25, .5, 1.):
            moved = (self.model.correction + step * residual.detach()).clamp(-.8, .8)
            candidate.correction.copy_(moved)
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
        candidate.correction.copy_(moved)
        gain = deployment_gain(self.model, candidate, check)
        accepted = (gain["accuracy_gain"] >= .01 and gain["brier_gain"] >= .002
                    and gain["accuracy_lower"] is not None and gain["accuracy_lower"] > 0
                    and min(gain["class_gain"]) >= -.02 and gain["quorum_gain"] >= -.02)
        if accepted:
            self.model.correction.copy_(moved)
            details["accepted"] = ["contextual readout"]
        return dict(details, status="accepted" if accepted else "frozen_no_board_gain",
                    step=step, fit_gain=fit_gain, fit_brier_gain=fit_brier, board_gain=gain)


def run(source, checkpoint, output):
    board = ContextBoard().eval()
    missing, unexpected = board.load_state_dict(torch.load(checkpoint, weights_only=True), strict=False)
    if missing != ["correction"] or unexpected:
        raise ValueError("Unexpected parameters in base model checkpoint")
    results, archives = {}, {}
    hashes = {**source_hashes(), **{name: sha(Path(__file__).with_name(name))
                                    for name in ("online.py", "contextual.py")}}
    for home in (5, 6):
        original, _ = sensor_data.occupancy(source, home, 14)
        (times, rows, truth), used = sensor_data.occupancy(source, home, 21)
        split = len(original[0])
        if not torch.equal(original[0], times[:split]) or len(times) <= split:
            raise ValueError("The evaluation must start after the first 14 days")
        archives.update(used)
        results[f"H{home}"] = paired_replay(board, (times, add_history(times, rows), truth), split,
                                            learner_type=ContextFeedback)
        result = results[f"H{home}"]
        print(f"H{home}: {result['frozen']['balanced_accuracy']:.4f} -> "
              f"{result['adaptive']['balanced_accuracy']:.4f}, {result['accepted_updates']} updates", flush=True)
    receipt = dict(code_sha256=hashes, archive_sha256=archives, initial_checkpoint_sha256=sha(checkpoint),
                   previous_days=14, total_days=21, homes=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.archives, args.checkpoint, args.output)
