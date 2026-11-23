import unittest
from unittest.mock import patch

import torch

from contextual import ContextBoard
from learn import Example
from model import Board
from online import BoardFeedback


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.model = Board().eval()
        self.learner = BoardFeedback(self.model)
        row = torch.ones(1, 10)
        state = self.model.start(row)
        self.examples = [Example(index * 300, row, state, torch.tensor([index % 2]), torch.zeros(1, 5))
                         for index in range(64)]

    def test_one_selected_change_is_checked_and_other_weights_are_frozen(self):
        origin = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        fit, check = self.examples[:32], self.examples[32:]

        def predictions(model, stream):
            self.assertEqual(stream, fit)
            expected = torch.cat([example.truth for example in stream])
            shift = float((model.outputs[-1].bias.detach() - self.learner.origin).mean())
            answer = expected if .3 < shift < .5 else 1 - expected
            probabilities = answer.float() * .8 + .1
            return answer, probabilities, torch.ones_like(answer, dtype=torch.bool)

        gains = dict(accuracy_gain=.04, accuracy_lower=.02, brier_gain=.01,
                     class_gain=[0., .08], quorum_gain=0.)
        with patch("online.board_predictions", side_effect=predictions), \
                patch("online.deployment_gain", return_value=gains) as grade:
            receipt = self.learner.try_update(fit, check)
        grade.assert_called_once()
        self.assertEqual(grade.call_args.args[2], check)
        self.assertEqual(receipt["status"], "accepted")
        self.assertEqual(receipt["step"], .4)
        for name, weight in self.model.named_parameters():
            torch.testing.assert_close(weight, origin[name] + (.4 if name == "outputs.2.bias" else 0.))

    def test_context_changes_a_vote_correction_in_both_directions(self):
        board = ContextBoard().eval()
        left = torch.zeros(17)
        right = torch.zeros(17)
        left[0], right[0] = -1., 1.
        rows = torch.stack((left, right))
        dt = rows.new_full((2,), 5.)
        settings = dict(checks=False)
        before = board(rows, dt, **settings)
        with torch.no_grad():
            board.correction[0] = 5.
        after = board(rows, dt, **settings)
        early_before = before.rounds[0].margins[..., :-1]
        early_after = after.rounds[0].margins[..., :-1]
        torch.testing.assert_close(early_before, early_after)
        gain = after.rounds[0].margins[..., -1] - before.rounds[0].margins[..., -1]
        torch.testing.assert_close(gain, torch.tensor([[-.8], [.8]]).expand(-1, 5))

    def test_fit_cannot_include_the_check(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.learner.try_update(self.examples[:33], self.examples[32:])


if __name__ == "__main__":
    unittest.main()
