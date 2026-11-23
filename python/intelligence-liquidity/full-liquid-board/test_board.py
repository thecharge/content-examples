from dataclasses import fields
import unittest
from unittest.mock import patch

import torch

from board import Board, State, attempt_reward
from run_board import replay


class BoardTest(unittest.TestCase):
    def model(self, task="occupancy", **options):
        model = Board(task, **options)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        return model

    def test_three_claims_commit(self):
        model = self.model()
        x = torch.ones(1, 10)
        with torch.no_grad():
            model.instant[:3, :10, 1].fill_(1.)
        decision = model(x, x.new_tensor([5.]), mode="off")
        self.assertEqual(len(decision.rounds), 1)
        self.assertEqual(int(decision.rounds[0].supporters.sum()), 3)
        self.assertTrue(decision.committed.item())
        self.assertEqual(decision.answer.item(), 1)
        self.assertGreater(decision.confidence.item(), .5)

    def test_split_exhausts_but_returns_a_decision_without_mutating_board(self):
        model = self.model(max_rounds=3)
        x = torch.ones(1, 10)
        before = model.start(x)
        with torch.no_grad():
            model.instant[:2, :10, 1].fill_(1.)
            model.instant[2:4, :10, 0].fill_(1.)
        result = model(x, x.new_tensor([5.]), before, mode="off")
        self.assertEqual(len(result.rounds), 3)
        self.assertFalse(result.committed.item())
        self.assertIn(result.answer.item(), (0, 1))
        self.assertEqual(int(result.rounds[-1].supporters.sum()), 2)
        for part in fields(State):
            torch.testing.assert_close(getattr(before, part.name), getattr(result.state, part.name))

    def test_challenger_reopens_decision_and_retry_can_commit(self):
        model = self.model(max_rounds=3)
        x = torch.ones(1, 10)
        minutes = x.new_tensor([5.])
        counter = x.new_full((1, model.count), .02)
        with torch.no_grad():
            model.instant[:, :10, 1].fill_(1.)
        original = model.round
        feedback_history = []

        def exchange(*args):
            feedback_history.append(args[5].clone())
            if len(feedback_history) == 1:
                return original(*args)
            revised = list(args)
            revised[6] = 1 - args[6]
            return original(*revised)

        with patch.object(model, "round", side_effect=exchange):
            result = model(x, minutes, mode="tree", counter=counter)
        self.assertEqual(len(result.rounds), 2)
        self.assertTrue(result.rounds[0].vote_rejected.all())
        self.assertTrue(feedback_history[1][..., 0].lt(0).all())
        self.assertTrue(result.committed.item())
        self.assertEqual(result.answer.item(), 1)

    def test_repeat_failure_increases_verified_penalty(self):
        model = self.model()
        x = torch.ones(1, 10)
        item = model(x, x.new_tensor([5.]), mode="off").rounds[0]
        failed = torch.zeros_like(item.speaking)
        first = attempt_reward(item, failed, 0)
        third = attempt_reward(item, failed, 2)
        torch.testing.assert_close(first - third, torch.full_like(first, 1.10 * (1.25**2 - 1)))

    def test_online_updates_use_logged_rewards_and_skip_gaps(self):
        model = self.model("heat")
        x = torch.tensor([[.1, .2, 0, 0, 0], [.2, .4, 0, 0, 0], [.5, .2, 0, 0, 0]])
        stream = (torch.tensor([0, 900, 2700]), x, torch.tensor([1, 0, 1]),
                  torch.tensor([-.1, -.28, -.3]))
        frozen = replay(model, stream)
        adapted = replay(model, stream, online=True)
        self.assertEqual(frozen["final_answers"], 3)
        self.assertEqual(frozen["reward_updates"], 0)
        self.assertEqual(adapted["reward_updates"], 1)
        self.assertEqual(adapted["sensor_updates"], 1)
        self.assertGreater(adapted["parameter_change_l1"], 0.)

    def test_configurable_majority_and_missing_evidence(self):
        model = self.model(views=((0,),) * 7)
        self.assertEqual(model.quorum, 4)
        x = torch.full((1, 10), float("nan"))
        result = model(x, x.new_tensor([5.]))
        self.assertFalse(result.rounds[0].speaking.any())
        self.assertFalse(result.committed.item())
        self.assertEqual(result.answer.item(), model.fallback.item())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
