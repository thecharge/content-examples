"""Run directly with python test_model.py."""
import unittest

import torch
from torch import nn

from model import DeltaBoard, EXPERTS, LiquidCell, build, review, training_loss


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.x = torch.randn(4, 5, 2)
        self.clock = torch.full((4, 5), 5.0)
        self.clock[:, 0] = 0

    def test_shapes_and_training(self):
        y = torch.tensor([0, 1, 1, 0])
        for name in ("current", "gru", "liquid", "moe", "moe_delta"):
            with self.subTest(name=name):
                model = build(name)
                self.assertEqual(model(self.x, self.clock)["logits"].shape, (4,))
                loss = training_loss(model, self.x, self.clock, y, 1.5)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertIsNotNone(model.readout.weight.grad)
                if name.startswith("moe"):
                    self.assertIsNotNone(model.router.weight.grad)
                if name == "moe_delta":
                    self.assertTrue(any(p.grad is not None for p in model.checkers.parameters()))

    def test_clock(self):
        cell = LiquidCell(2, 1)
        with torch.no_grad():
            cell.drive.weight.zero_()
            cell.drive.bias.copy_(torch.tensor([1.0, 0.0]))
        start = torch.zeros(2, 1)
        inputs = torch.ones(2, 2)
        short = cell(inputs, start, torch.tensor([0.0, 5.0]))
        long = cell(inputs, start, torch.tensor([0.0, 10.0]))
        self.assertEqual(short[0].item(), 0)
        self.assertGreater(long[1].item(), short[1].item())
        torch.testing.assert_close(
            cell(inputs, short, torch.tensor([0.0, 5.0])), long)

    def test_sparse_and_dormant_clock(self):
        model = build("moe")
        desired = iter([0, 1, 0, 1, 0])
        clocks = [[] for _ in range(EXPERTS)]

        def route(_module, _args, scores):
            forced = torch.zeros_like(scores)
            forced[:, next(desired)] = 5
            return forced

        def record(expert):
            def before(_module, args):
                clocks[expert].append(args[2].tolist())
            return before

        model.router.register_forward_hook(route)
        for expert, cell in enumerate(model.experts):
            cell.register_forward_pre_hook(record(expert))
        output = model(self.x, self.clock)
        self.assertEqual(output["route"][0].tolist(), [0, 1, 0, 1, 0])
        self.assertEqual(output["expert_counts"].tolist(), [12, 8, 0])
        self.assertEqual(clocks, [[[0] * 4, [10] * 4, [10] * 4],
                                  [[5] * 4, [10] * 4], []])

    def test_delta_write_decay_and_query(self):
        board = DeltaBoard(2, size=2)
        with torch.no_grad():
            for layer in (board.key, board.query, board.value, board.gate):
                layer.weight.zero_()
            board.key.bias.copy_(torch.tensor([4.0, 0.0]))
            board.query.bias.copy_(torch.tensor([0.0, 4.0]))
            board.value.bias.copy_(torch.tensor([1.0, -1.0]))
            board.gate.bias.zero_()
            board.output.weight.copy_(torch.eye(2))
        old = torch.ones(1, 2, 2)
        features = torch.zeros(1, 2)
        unchanged, _ = board(features, old, torch.zeros(1))
        torch.testing.assert_close(unchanged, old)
        updated, read = board(features, old, torch.full((1,), 5.0))
        decay = torch.exp(-(nn.functional.softplus(board.decay) + 1e-4))
        fraction = 0.5 * -torch.expm1(torch.tensor(-1.0))
        target = torch.tensor([1.0, -1.0]).tanh()
        torch.testing.assert_close(updated[0, 0], decay + fraction * (target - decay))
        torch.testing.assert_close(updated[0, 1], decay.expand(2))
        torch.testing.assert_close(read[0], updated[0, 1])

    def test_prefix_causality_and_latest_reading(self):
        changed = self.x.clone()
        changed[:, 3:] += 20
        x = torch.cat((self.x, changed))
        clock = self.clock.repeat(2, 1)
        for name in ("moe", "moe_delta"):
            model = build(name)
            observed = []

            def record(_module, args):
                observed.append(args[0].detach())

            model.readout.register_forward_pre_hook(record)
            out = model(x, clock)
            torch.testing.assert_close(out["route"][:4, :3], out["route"][4:, :3])
            for step in range(3):
                torch.testing.assert_close(observed[step][:4], observed[step][4:])
        current = build("current")
        changed[:, -1] = self.x[:, -1]
        torch.testing.assert_close(current(changed, self.clock)["logits"],
                                   current(self.x, self.clock)["logits"])

    @staticmethod
    def force_review(model, gain):
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))
            for checker in model.checkers:
                checker.correct.weight.zero_()
                checker.correct.bias.zero_()
                checker.gain[-1].weight.zero_()
                checker.gain[-1].bias.fill_(gain)

    def test_review_commit_budget_and_trace(self):
        model = build("moe_delta")
        self.force_review(model, gain=10)
        ordinary, _, initial, context, choices = model._run(self.x, self.clock, branch=True)
        self.assertEqual(choices.tolist(), [1] * 4)
        output = review(model, self.x, self.clock, 0.5)
        self.assertTrue(output["attempted"].all())
        self.assertTrue(output["revision_accepted"].all())
        self.assertTrue((output["revision_count"] == 1).all())
        self.assertTrue((output["budget_remaining"] == 0).all())
        self.assertEqual(output["initial_expert"].tolist(), [0] * 4)
        self.assertEqual(output["challenger_expert"].tolist(), [1] * 4)
        self.assertEqual(output["returned_expert"].tolist(), [1] * 4)
        self.assertEqual(output["expert_calls"].item(), self.clock.numel() + len(self.x))
        torch.testing.assert_close(output["ordinary_logits"], ordinary["logits"])
        torch.testing.assert_close(output["logits"], output["challenger_logits"])
        torch.testing.assert_close(output["initial_check"], torch.full((4,), 0.5))
        self.assertTrue(torch.isfinite(output["process_gain"]).all())
        self.assertTrue(torch.isfinite(output["revision_gain"]).all())
        self.assertTrue((output["committed_timestamps"][:, 1] == self.clock.sum(1)).all())
        torch.testing.assert_close(
            output["committed_timestamps"][:, 0], context[6][:, 0])
        torch.testing.assert_close(initial.timestamps[:, 0], self.clock.sum(1))

    def test_review_rollback_and_invalid_clock(self):
        model = build("moe_delta")
        self.force_review(model, gain=-10)
        _, _, initial, _, _ = model._run(self.x, self.clock, branch=True)
        output = review(model, self.x, self.clock, 0.5)
        self.assertTrue(output["revision_rejected"].all())
        self.assertFalse(output["revision_accepted"].any())
        torch.testing.assert_close(output["logits"], initial.logits)
        torch.testing.assert_close(output["committed_board"], initial.board)
        torch.testing.assert_close(output["committed_states"], initial.states)
        torch.testing.assert_close(output["committed_timestamps"], initial.timestamps)
        self.clock[0, 1] = -1
        invalid = review(model, self.x, self.clock, 0.5)
        self.assertFalse(invalid["time_valid"][0])
        self.assertFalse(invalid["accepted"][0])
        self.assertFalse(invalid["attempted"][0])
        self.assertEqual(invalid["challenger_expert"][0].item(), -1)

    def test_invalid_tentative_proposal_rolls_back(self):
        model = build("moe_delta")
        self.force_review(model, gain=10)
        with torch.no_grad():
            model.experts[1].drive.bias.fill_(torch.nan)
        ordinary = model(self.x, self.clock)
        output = review(model, self.x, self.clock, 0.5)
        self.assertTrue(output["attempted"].all())
        self.assertFalse(output["challenger_valid"].any())
        self.assertTrue(output["revision_rejected"].all())
        torch.testing.assert_close(output["logits"], ordinary["logits"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
