"""Regression checks for recurrent fitting and online adaptation."""

from dataclasses import replace
import unittest

import torch

from control import logged_returns, objective, trajectory_batches, validate
from experiment import WARMUP_ROWS, WINDOW_ROWS, sequence_blocks
from workflow import Blackboard, Observation, Online, bellman_return, routing_balance


class RecordingBlackboard(Blackboard):
    def __init__(self):
        super().__init__(use_checks=False)
        self.input_ledgers = []
        self.output_ledgers = []

    def step(self, primary, witness, minutes, ledger=None, **kwargs):
        self.input_ledgers.append(ledger)
        result = super().step(primary, witness, minutes, ledger, **kwargs)
        self.output_ledgers.append(result.ledger)
        return result


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_heater_windows_cover_all_rows_and_stop_at_gaps(self):
        continues = torch.tensor([True, True, False, True, False, False])
        batches = trajectory_batches(continues, length=2)
        windows = sorted(window for batch in batches for window in batch.tolist())
        self.assertEqual(windows, [[0, 1], [2], [3, 4], [5]])
        torch.testing.assert_close(logged_returns(-torch.ones(6), continues),
                                   -torch.tensor([2.8525, 1.95, 1., 1.95, 1., 1.]))

    def test_heater_target_and_model_keep_independent_ledgers(self):
        model, target = RecordingBlackboard(), RecordingBlackboard()
        x = torch.tensor([[.1, .2], [.2, .3]])
        data = x, x.flip(0), torch.zeros(2, dtype=torch.long), -torch.ones(2), torch.tensor([True, False])
        objective(model, target, data, torch.tensor([[0, 1]])).backward()
        self.assertIs(model.input_ledgers[1], model.output_ledgers[0])
        self.assertIs(target.input_ledgers[1], target.output_ledgers[0])
        self.assertIs(target.input_ledgers[2], target.output_ledgers[0])
        self.assertIsNone(target.layers[0].expert.strength.grad)

    def test_heater_validation_uses_discount_and_resets_after_a_terminal_row(self):
        model, target = RecordingBlackboard(), RecordingBlackboard()
        for network, value in ((model, -1.), (target, -2.)):
            with torch.no_grad():
                network.action_value.weight.zero_()
                network.action_value.bias.fill_(value)
        x = torch.zeros(3, 2)
        data = x, x, torch.zeros(3, dtype=torch.long), -torch.ones(3), torch.tensor([True, False, False])
        td, recorded = validate(model, target, data)
        self.assertAlmostEqual(td, 1.9 / 3, places=6)
        self.assertAlmostEqual(recorded, .95 / 3, places=6)
        self.assertIsNone(model.input_ledgers[2])
        torch.testing.assert_close(bellman_return(torch.tensor(-1.), torch.tensor(-2.)), torch.tensor(-2.9))

    def test_online_witness_updates_checker_only_when_a_label_exists(self):
        for witness in (.1, float("nan")):
            model = Blackboard()
            learner = Online(model)
            heads = [model.layers[0].shadow.weight, model.layers[0].process.weight]
            original = [p.detach().clone() for p in heads]
            previous = Observation(0, torch.tensor([.5]), torch.tensor([witness]), torch.tensor([5.]), None)
            self.assertIsNone(learner.feedback(previous))
            self.assertIsNotNone(learner.feedback(replace(previous, time=300, primary=torch.tensor([1.2]))))
            self.assertEqual([not torch.equal(a, b) for a, b in zip(original, heads, strict=True)],
                             [witness == witness] * 2)

    def test_idle_expert_ages_during_rejected_intervals_without_mutating_ledger(self):
        model = Blackboard(fixed_expert=True)
        primary = torch.tensor([.5])
        base = model.start(primary)
        ledger = replace(base, experts=tuple(torch.ones_like(bank) for bank in base.experts))
        rejected = model.step(primary, torch.tensor([100.]), torch.tensor([55.]), ledger)
        self.assertFalse(rejected.accepted.item())
        torch.testing.assert_close(rejected.ledger.experts[0], ledger.experts[0])
        accepted = model.step(primary, torch.tensor([.1]), torch.tensor([5.]), rejected.ledger,
                              tolerance=10., idle_minutes=55.)
        self.assertTrue(accepted.accepted.item())
        torch.testing.assert_close(accepted.ledger.experts[0][0, 1:], torch.full((2, 8), .5))
        torch.testing.assert_close(ledger.experts[0], torch.ones_like(ledger.experts[0]))

    def test_router_balance_penalizes_a_collapsed_greedy_route(self):
        uniform = torch.full((4, 3), 1/3)
        logits = torch.tensor([[6., 0., 0.]] * 4, requires_grad=True)
        greedy = logits.argmax(-1)
        collapsed = routing_balance(logits.softmax(-1), greedy)
        self.assertGreater(collapsed, routing_balance(uniform, greedy))
        collapsed.backward()
        self.assertTrue((logits.grad[:, 0] > 0).all())

    def test_occupancy_windows_reuse_only_context_and_never_cross_a_gap(self):
        stamps = torch.cat((torch.arange(60) * 300, 30_000 + torch.arange(24) * 300))
        readings, labels = torch.zeros(len(stamps), 2), torch.zeros(len(stamps))
        blocks = list(sequence_blocks([("room", readings, labels, stamps)]))
        self.assertEqual(len(blocks), 3)
        for _, _, t in blocks:
            self.assertEqual(len(t), WINDOW_ROWS)
            self.assertTrue((t.diff() == 300).all())
        self.assertEqual(blocks[0][2][-1].item(), blocks[1][2][WARMUP_ROWS].item())


if __name__ == "__main__":
    unittest.main()
