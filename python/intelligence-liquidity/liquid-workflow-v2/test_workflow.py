import unittest

import torch
from torch import nn

from workflow import Blackboard, DeltaAttention, LiquidODE, Observation, Online, WIDTH

torch.set_num_threads(1)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.net = Blackboard()
        self.reading = torch.tensor([.8])
        self.minutes = torch.tensor([5.])

    def test_ode_identity_at_zero_and_numerical_refinement(self):
        fast = LiquidODE(1)
        reference = LiquidODE(1, substeps=600)
        reference.load_state_dict(fast.state_dict())
        coarse = LiquidODE(1, substeps=1)
        coarse.load_state_dict(fast.state_dict())
        old, index = torch.full((1, WIDTH), .7), torch.zeros(1, dtype=torch.long)
        x = self.reading[:, None]
        torch.testing.assert_close(fast(x, old, self.minutes * 0, index), old, rtol=0, atol=0)
        expected = reference(x, old, self.minutes, index)
        self.assertLess((fast(x, old, self.minutes, index)-expected).abs().mean(),
                        (coarse(x, old, self.minutes, index)-expected).abs().mean())

    def test_delta_matches_independent_matrix_arithmetic(self):
        board = torch.tensor([[[[.8, .2], [.4, -.1]]]])
        key = torch.tensor([[[1., 0.]]])
        target = torch.tensor([[[.6, .35]]])
        alpha = torch.tensor([[[.5, .25]]])
        rate = torch.tensor([[.25]])
        updated = DeltaAttention.update(board, key, target, alpha, rate)
        torch.testing.assert_close(updated, torch.tensor([[[[.45, .1625], [.1, -.025]]]]))

    def test_failed_reading_restores_entire_ledger_even_after_retry(self):
        first = self.net.step(self.reading, torch.tensor([.1]), self.minutes, tolerance=10)
        rejected = self.net.step(self.reading, torch.tensor([90.]), self.minutes, first.ledger)
        self.assertFalse(rejected.accepted.item())
        self.assertTrue(any(item["retry_from"] is not None for item in rejected.attempts))
        torch.testing.assert_close(first.ledger.board, rejected.ledger.board, atol=0, rtol=0)
        for a, b in zip((*first.ledger.experts, *first.ledger.reviewers),
                        (*rejected.ledger.experts, *rejected.ledger.reviewers), strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_middle_and_final_objections_are_sent_back_one_layer(self):
        for layer, expected in [(1, 0), (2, 1)]:
            with torch.no_grad():
                self.net.layers[layer].shadow.weight.zero_()
                self.net.layers[layer].shadow.bias.fill_(3)
            result = self.net.step(self.reading, torch.tensor([0.]), self.minutes, tolerance=2)
            self.assertEqual({e["retry_from"] for e in result.attempts}, {None, expected})
            with torch.no_grad():
                self.net.layers[layer].shadow.bias.zero_()

    def test_retry_can_use_a_distinct_expert(self):
        result = self.net.step(self.reading, torch.tensor([1000.]), self.minutes)
        choices = [row["expert"].item() for row in result.attempts if row["layer"] == 0]
        self.assertEqual(len(choices), 3)
        self.assertEqual(len(set(choices)), 3)

    def test_two_valid_repairs_use_the_learned_score(self):
        class ControlledExpert(nn.Module):
            def forward(self, inputs, state, minutes, expert):
                values = inputs.new_tensor([1., .3, -.3])[expert]
                return values[:, None].expand(-1, WIDTH)

        class ControlledReviewer(nn.Module):
            def forward(self, inputs, state, minutes, expert):
                return inputs[:, WIDTH:WIDTH+1].expand(-1, WIDTH)

        layer = self.net.layers[0]
        layer.expert = ControlledExpert()
        layer.reviewer = ControlledReviewer()
        with torch.no_grad():
            layer.route.weight.zero_()
            layer.route.bias.copy_(torch.tensor([2., 1., 0.]))
            layer.draft.weight.zero_()
            layer.draft.bias.zero_()
            layer.draft.weight[0, WIDTH] = 1.
            for index, stage in enumerate(self.net.layers):
                stage.process.weight.zero_()
                stage.process.bias.zero_()
                stage.shadow.weight.zero_()
                stage.shadow.bias.zero_()
                if index == 0:
                    stage.shadow.weight[0, 0] = 1.
                    stage.process.weight[0, 0] = -20.
        answer = self.net.step(self.reading, torch.tensor([0.]), self.minutes)
        self.assertTrue(answer.accepted.item())
        torch.testing.assert_close(answer.ledger.experts[0][0, 2], torch.full((WIDTH,), -.3))
        torch.testing.assert_close(answer.ledger.experts[0][0, 1], torch.zeros(WIDTH))

    def test_witness_is_independent_of_specialist_forward(self):
        a = self.net.step(self.reading, torch.tensor([0.]), self.minutes, tolerance=100)
        b = self.net.step(self.reading, torch.tensor([1.]), self.minutes, tolerance=100)
        torch.testing.assert_close(a.claims, b.claims, atol=0, rtol=0)
        self.assertFalse(torch.equal(a.process, b.process))
        missing = self.net.step(self.reading, torch.tensor([float("nan")]), self.minutes)
        self.assertFalse(missing.accepted.item())

    def test_later_observation_trains_ode_and_updates_only_in_time_order(self):
        prior = self.net.layers[0].expert.target.detach().clone()
        learn = Online(self.net, lr=.01)
        a = Observation(0, self.reading, torch.tensor([.1]), self.minutes, None)
        b = Observation(300, torch.tensor([1.3]), torch.tensor([.2]), self.minutes, None)
        self.assertIsNone(learn.feedback(a))
        torch.testing.assert_close(prior, self.net.layers[0].expert.target, atol=0, rtol=0)
        self.assertIsNotNone(learn.feedback(b, observed_action=1, observed_reward=-1., next_action_return=-.2))
        self.assertFalse(torch.equal(prior, self.net.layers[0].expert.target))
        with self.assertRaisesRegex(ValueError, "time order"):
            learn.feedback(a)


class LocalTemperatureTests(unittest.TestCase):
    def test_local_reference_has_no_current_reading_and_ignores_spikes(self):
        import math
        from local_reference import LocalTemperature

        sensor = LocalTemperature()
        self.assertTrue(math.isnan(sensor.aligned(27.)))
        sensor.observe(27.)
        self.assertAlmostEqual(sensor.aligned(27.6), .2)
        self.assertEqual(sensor.center, 27.)
        sensor.observe(38.)
        self.assertEqual(sensor.center, 27.)
        sensor.observe(27.6)
        self.assertAlmostEqual(sensor.center, 27.15)
        self.assertLess(sensor.aligned(27.), 0.)


if __name__ == "__main__":
    unittest.main()
