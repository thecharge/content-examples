import unittest
from unittest.mock import patch

import torch

from learn import Example, Feedback
from model import Board, measured_reward, offline_loss


class BoardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.model = Board().eval()
        self.row = torch.linspace(-1, 1, 10).reshape(1, 10)
        self.dt = torch.tensor([5.])
        self.before = self.model.start(self.row)
        with torch.no_grad():
            self.model.subscription.bias.fill_(6.)

    def test_every_expert_connects_all_inputs_to_a_three_layer_prediction(self):
        row = self.row.requires_grad_(True)
        result = self.model(row, self.dt, self.before)
        scores = result.rounds[0].margins
        self.assertEqual(scores.shape, (1, 5, 3))
        for expert in range(5):
            (gradient,) = torch.autograd.grad(scores[0, expert, -1], row, retain_graph=True)
            self.assertTrue(gradient.ne(0).all())
        offline_loss(result, torch.tensor([1]), torch.ones(2)).backward()
        self.assertTrue(all(cell.strength.grad.abs().sum() > 0 for cell in self.model.cells))

    def test_experts_decide_locally_and_uninterested_paths_do_not_run(self):
        with torch.no_grad():
            self.model.subscription.weight.zero_()
            self.model.subscription.bias.copy_(torch.tensor([[6.], [-6.], [6.], [-6.], [6.]]))
            self.model.outputs[-1].bias.fill_(8.)
        calls = []
        hook = self.model.cells[0].register_forward_pre_hook(lambda _, args: calls.append(args[-1].tolist()))
        try:
            result = self.model(self.row, self.dt, self.before, checks=False)
        finally:
            hook.remove()
        self.assertEqual(calls, [[0, 2, 4]])
        self.assertEqual(result.rounds[0].acted.tolist(), [[True, False, True, False, True]])
        self.assertTrue(result.committed.item())
        self.assertEqual(result.state.elapsed.tolist(), [[0., 5., 0., 5., 0.]])

    def test_each_expert_can_choose_a_different_path_for_the_same_evidence(self):
        with torch.no_grad():
            self.model.specialty.weight.zero_()
            self.model.specialty.bias.zero_()
            self.model.subscription.weight.zero_()
            self.model.subscription.bias.fill_(5.)
            self.model.subscription.bias[:2].fill_(-.1)
            self.model.specialty.weight[0, 0, 0] = 2.
            self.model.specialty.weight[1, 0, 0] = -2.
            self.model.subscription.weight[:2, 0, 0] = 4.
            self.model.outputs[-1].bias.fill_(9.)
        inputs = torch.stack((torch.ones(10), -torch.ones(10)))
        decision = self.model(inputs, torch.full((2,), 5.), checks=False)
        self.assertEqual(decision.rounds[0].acted.tolist(),
                         [[True, False, True, True, True], [False, True, True, True, True]])

    def test_unresolved_feedback_can_draw_in_another_specialist(self):
        with torch.no_grad():
            self.model.specialty.weight.zero_()
            self.model.specialty.bias.zero_()
            self.model.subscription.weight.zero_()
            self.model.subscription.bias.copy_(torch.tensor([[5.], [5.], [-.1], [-5.], [-5.]]))
            self.model.outputs[-1].bias.fill_(9.)
            last_feedback = self.model.input.weight.shape[1] - 1
            self.model.specialty.weight[2, last_feedback, 0] = 8.
            self.model.subscription.weight[2, 0, 0] = 4.
        decision = self.model(self.row, self.dt, checks=False)
        self.assertEqual(len(decision.rounds), 2)
        self.assertFalse(decision.rounds[0].acted[0, 2].item())
        self.assertTrue(decision.rounds[1].acted[0, 2].item())
        self.assertTrue(decision.committed.item())

    def test_quorum_commits_and_no_evidence_exhausts_without_changing_memory(self):
        with torch.no_grad():
            self.model.outputs[-1].bias.fill_(7)
            agreed = self.model(self.row, self.dt, self.before, checks=False)
            self.assertTrue(agreed.committed.item())
            self.assertEqual(agreed.answer.item(), 1)
            failed = self.model(torch.full_like(self.row, float("nan")), self.dt,
                                agreed.state, checks=False)
            self.assertFalse(failed.committed.item())
            self.assertEqual(len(failed.rounds), 4)
            self.assertEqual(failed.answer.item(), 1)
            self.assertTrue(torch.equal(failed.state.memory, agreed.state.memory))
            for prior, after in zip(agreed.state.cells, failed.state.cells, strict=True):
                self.assertTrue(torch.equal(prior, after))

    def test_only_an_actual_later_answer_sets_reward(self):
        with torch.no_grad():
            self.model.outputs[-1].bias.fill_(7)
            issued = self.model(self.row, self.dt, self.before, checks=False)
        feedback = Feedback(self.model)
        params = [p.clone() for p in self.model.parameters()]
        # An outcome timestamp equal to the current decision would expose its answer.
        with self.assertRaises(ValueError):
            feedback.observe(0, 0, self.row, self.before, issued, torch.tensor([0]))
        reward = feedback.observe(0, 300, self.row, self.before, issued, torch.tensor([0]))
        self.assertTrue(reward.lt(0).all())
        self.assertTrue(measured_reward(issued.rounds, torch.tensor([1])).gt(0).all())
        self.assertTrue(all(torch.equal(p, current) for p, current in zip(params, self.model.parameters())))

    def test_update_changes_only_experts_with_verified_gain(self):
        feedback = Feedback(self.model)
        example = Example(100, self.row, self.before, torch.tensor([1]), torch.zeros(1, 5))
        later = Example(200, self.row, self.before, torch.tensor([0]), torch.zeros(1, 5))
        prior = self.model.input.weight.detach().clone()
        def fixture(model, examples, train=False):
            if train:
                return model.input.weight.mean((1, 2)), torch.zeros(5)
            candidates = torch.tensor([.7, .99, .8, 1.5, 1.])
            accuracy = torch.tensor([.7, .8, .5, .9, .8])
            return (torch.ones(5), torch.full((5,), .7)) if model is self.model else (candidates, accuracy)
        with patch("learn.measurements", side_effect=fixture):
            receipt = feedback.try_update([example], [later])
        self.assertEqual(receipt["accepted"], [1, 2])
        self.assertTrue(torch.equal(prior[2:], self.model.input.weight[2:]))
        self.assertFalse(torch.equal(prior[:2], self.model.input.weight[:2]))

    def test_buffer_freezes_if_there_is_only_one_class(self):
        feedback = Feedback(self.model, minimum=4)
        with torch.no_grad():
            issued = self.model(self.row, self.dt, self.before)
        for index in range(4):
            feedback.observe(index, index + 1, self.row, self.before, issued, torch.tensor([1]))
        self.assertEqual(feedback.receipts[0]["status"], "frozen_insufficient_classes")
        self.assertEqual(feedback.receipts[0]["accepted"], [])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
