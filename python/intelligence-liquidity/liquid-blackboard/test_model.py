import unittest

import torch

from experiment import build, stream
from model import DEPTH, EXPERTS, logged_q_loss


class BlackboardTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.x = torch.tensor([[.1, -.3], [.2, -.1], [.8, .4], [.9, -.5]])
        self.t = torch.arange(len(self.x)) * 300

    def test_each_layer_routes_and_can_reject_repairs(self):
        model = build("full")
        for layer in model.layers:
            layer.process.weight.data.zero_()
            layer.process.bias.data.fill_(-1)
        step = model.step(self.x, torch.full((len(self.x),), 5.), None)
        self.assertEqual(step.routes.shape, (len(self.x), DEPTH, EXPERTS))
        torch.testing.assert_close(step.routes.sum(-1), torch.ones(len(self.x), DEPTH))
        self.assertEqual(len(step.memory.falsifiers), DEPTH)
        self.assertTrue(torch.all(step.corrections == 0))
        for layer in model.layers:
            layer.process.bias.data.fill_(1)
        repaired = model.step(self.x, torch.full((len(self.x),), 5.), None)
        self.assertTrue(torch.all(repaired.corrections > 0))
        self.assertFalse(torch.allclose(step.occupied, repaired.occupied))

    def test_online_predictions_do_not_see_future_inputs(self):
        model = build("full")
        weights = model.state_dict()
        other = build("full")
        other.load_state_dict(weights)
        output, details = stream(model, self.x, self.t, learn=True)
        altered = self.x.clone()
        altered[-1] += 3
        repeated, _ = stream(other, altered, self.t, learn=True)
        torch.testing.assert_close(output[:-1], repeated[:-1], atol=0, rtol=0)
        self.assertEqual(details["arrival_updates"], len(self.x) - 1)
        self.assertNotEqual(details["adapter_offset"], [0, 0])

    def test_outcome_reinforces_logged_action(self):
        q = torch.zeros(1, 2, requires_grad=True)
        loss = logged_q_loss(q, q, torch.tensor([1]), torch.tensor([2.]),
                             torch.tensor([0.]), torch.tensor([0.]))
        loss.backward()
        self.assertLess(q.grad[0, 1].item(), 0)
        self.assertGreater(q.grad[0, 0].item(), 0)


if __name__ == "__main__":
    unittest.main()
