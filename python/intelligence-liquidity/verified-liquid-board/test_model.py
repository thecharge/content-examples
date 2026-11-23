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

    def force_full_participation(self):
        with torch.no_grad():
            self.model.subscription.weight.zero_()
            self.model.subscription.bias.fill_(6.)

    def test_every_expert_connects_all_inputs_to_a_three_layer_prediction(self):
        self.force_full_participation()
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
        self.force_full_participation()
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
        self.force_full_participation()
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

    @torch.no_grad()
    def test_checker_changes_quorum_and_rolls_back_the_failed_round(self):
        self.force_full_participation()
        for output, checker in zip(self.model.outputs, self.model.falsifiers, strict=True):
            output.weight.zero_()
            output.bias.fill_(8.)
            checker[-1].weight.zero_()
            checker[-1].bias.fill_(-8.)
        self.model.process.weight.zero_()
        self.model.process.bias.fill_(-8.)
        unchecked = self.model(self.row, self.dt, self.before, checks=False)
        checked = self.model(self.row, self.dt, self.before)
        self.assertTrue(unchecked.committed.item())
        self.assertEqual(unchecked.answer.item(), 1)
        self.assertFalse(checked.committed.item())
        self.assertTrue(checked.rounds[0].challenged.all())
        self.assertEqual(checked.answer.item(), self.before.last.item())
        self.assertEqual(checked.probability.item(), .5)
        self.assertTrue(torch.equal(checked.state.memory, self.before.memory))

    @torch.no_grad()
    def test_a_committed_round_does_not_save_a_challenged_expert_state(self):
        self.force_full_participation()
        for output, checker in zip(self.model.outputs, self.model.falsifiers, strict=True):
            output.weight.zero_()
            output.bias[:3].fill_(8.)
            output.bias[3:].fill_(-8.)
            checker[-1].weight.zero_()
            checker[-1].bias.fill_(8.)
        self.model.process.weight.zero_()
        self.model.process.bias.fill_(-8.)
        decision = self.model(self.row, self.dt, self.before)
        self.assertTrue(decision.committed.item())
        self.assertEqual(decision.answer.item(), 1)
        self.assertEqual(decision.rounds[0].challenged.tolist(), [[False, False, False, True, True]])
        self.assertEqual(decision.state.elapsed.tolist(), [[0., 0., 0., 5., 5.]])
        for prior, after in zip(self.before.cells, decision.state.cells, strict=True):
            self.assertTrue(torch.equal(prior[:, 3:], after[:, 3:]))
            self.assertFalse(torch.equal(prior[:, :3], after[:, :3]))

    @torch.no_grad()
    def test_acting_experts_can_withhold_a_vote(self):
        self.force_full_participation()
        for output in self.model.outputs:
            output.weight.zero_()
            output.bias.zero_()
        self.model.drive.weight.zero_()
        self.model.drive.bias.fill_(-8.)
        result = self.model(self.row, self.dt, self.before, checks=False)
        self.assertTrue(result.rounds[0].acted.all())
        self.assertFalse(result.rounds[0].speaks.any())
        self.assertFalse(result.committed.item())

    def test_real_gradient_changes_only_the_expert_with_verified_gain(self):
        feedback = Feedback(self.model)
        fit = [Example(index * 300, self.row + index * .2, self.before,
                       torch.tensor([index % 2]), torch.zeros(1, 5)) for index in range(6)]
        check = [Example((index + 6) * 300, example.inputs, self.before, example.truth, example.reward)
                 for index, example in enumerate(fit)]
        prior = self.model.input.weight.detach().clone()
        approved = dict(accuracy_gain=.03, brier_gain=.01, accuracy_lower=.01,
                        class_gain=[.03, .03], quorum_gain=.01)
        with patch("learn.deployment_gain", return_value=approved) as grade:
            receipt = feedback.try_update(fit, check)
        grade.assert_called_once()
        self.assertEqual(len(receipt["accepted"]), 1)
        chosen = receipt["accepted"][0] - 1
        changed = (prior != self.model.input.weight).flatten(1).any(1)
        self.assertEqual(changed.nonzero().flatten().tolist(), [chosen])

    def test_real_board_replay_freezes_when_an_update_does_not_improve_accuracy(self):
        self.force_full_participation()
        with torch.no_grad():
            self.model.outputs[-1].bias.fill_(8.)
        feedback = Feedback(self.model, rate=0., checks=False)
        examples = [Example(index * 300, self.row, self.before,
                            torch.tensor([index % 2]), torch.zeros(1, 5)) for index in range(8)]
        receipt = feedback.try_update(examples[:4], examples[4:])
        self.assertEqual(receipt["status"], "frozen_no_board_gain")
        self.assertEqual(receipt["board_gain"]["accuracy_gain"], 0.)
        self.assertEqual(receipt["accepted"], [])

    def test_buffer_freezes_if_there_is_only_one_class(self):
        feedback = Feedback(self.model, minimum=4, minority=2)
        with torch.no_grad():
            issued = self.model(self.row, self.dt, self.before)
        for index in range(4):
            feedback.observe(index, index + 1, self.row, self.before, issued, torch.tensor([1]))
        self.assertEqual(feedback.receipts[0]["status"], "frozen_insufficient_classes")
        self.assertEqual(feedback.receipts[0]["accepted"], [])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
