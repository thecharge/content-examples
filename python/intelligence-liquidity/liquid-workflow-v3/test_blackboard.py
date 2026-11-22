import json
import unittest

import torch
from torch import nn

from blackboard import Blackboard, OCCUPANCY, VIEWS
from online import Event, Online
from serve import stream


class FixedEvidence(nn.Module):
    def forward(self, values):
        return values.new_tensor([-5., 5.]).expand(*values.shape[:-1], 2)


class DirectionalExperts(nn.Module):
    def forward(self, features, state, minutes, expert):
        sign = -1 + 2 * features[:, -2]
        return sign[:, None].expand_as(state)


class BlackboardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(1)

    def test_parallel_proposals_only_change_after_their_own_evidence_changes(self):
        model = Blackboard("occupancy")
        x = torch.ones(1, len(OCCUPANCY))
        a = model(x, torch.tensor([5.]))
        x[:, 0] += 2
        b = model(x, torch.tensor([5.]))
        # Memory is empty at this decision. Only the air expert sees CO2.
        torch.testing.assert_close(a.first.proposals.values[:, 1:], b.first.proposals.values[:, 1:])
        self.assertFalse(torch.equal(a.first.proposals.values[:, 0], b.first.proposals.values[:, 0]))

    def test_objection_rewinds_disputed_proposal_and_preserves_other_first_proposals(self):
        model = Blackboard("occupancy")
        model.independent = FixedEvidence()
        model.propose = DirectionalExperts()
        with torch.no_grad():
            model.value.zero_()
            model.value[:, 0, 1] = 1.
            model.bias[1:].copy_(torch.tensor([-2., 2.]))
        result = model(torch.zeros(1, len(OCCUPANCY)), torch.tensor([5.]))
        self.assertEqual(result.first.votes.tolist(), [[0, 1, 1]])
        self.assertEqual(result.revised.tolist(), [[True, False, False]])
        self.assertEqual(result.final.votes.tolist(), [[1, 1, 1]])
        torch.testing.assert_close(result.first.proposals.values[:, 1:], result.final.proposals.values[:, 1:])
        self.assertTrue(result.accepted.item())
        self.assertEqual(result.board.commits.item(), 1)

    def test_missing_quorum_keeps_board_bit_exact(self):
        model = Blackboard("occupancy")
        x = torch.full((1, len(OCCUPANCY)), float("nan"))
        x[0, VIEWS["occupancy"][0][0]] = .5
        before = model.start(x)
        before.memory.fill_(.2)
        before.states.fill_(.5)
        result = model(x, torch.tensor([5.]), before)
        self.assertFalse(result.accepted.item())
        for prior, actual in zip(before.__dict__.values(), result.board.__dict__.values(), strict=True):
            torch.testing.assert_close(prior, actual, equal_nan=True, rtol=0., atol=0.)

    def test_heater_guard_challenges_off_below_comfort_floor(self):
        model = Blackboard("heat")
        with torch.no_grad():
            model.value.zero_()
            model.bias.copy_(torch.tensor([[10., -10.]] * 3))
        result = model(torch.tensor([[-.6, .0, .0, .0, .0]]), torch.tensor([15.]))
        self.assertEqual(result.revised.tolist(), [[True, True, True]])
        self.assertFalse(result.accepted.item())

    def test_live_reward_and_self_supervised_updates_change_parameters(self):
        model = Blackboard("heat")
        learner = Online(model, 900)
        values = torch.zeros(1, 5)
        elapsed = torch.tensor([15.])
        first = model(values, elapsed)
        self.assertIsNone(learner.feedback(Event(0, values, elapsed, None), first))
        second = model(values+.2, elapsed, first.board)
        before = model.value.detach().clone()
        update = learner.feedback(Event(900, values+.2, elapsed, first.board.detach()), second,
                                  action=torch.tensor([1]), reward=torch.tensor([-.5]))
        self.assertGreater(update["sensor_loss"], 0.)
        self.assertGreater(update["reward_loss"], 0.)
        self.assertFalse(torch.equal(model.value, before))

    def test_stream_emits_decision_before_applying_previous_reward(self):
        model = Blackboard("heat")
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        sample = dict(indoor=.1, outdoor=.2, previous_action=1., previous_energy=.4, previous_discomfort=.1)
        lines = [json.dumps(dict(time=0, evidence=sample)),
                 json.dumps(dict(time=900, evidence=sample, previous_executed_action=1,
                                 previous_measured_reward=-.4))]
        events = stream(lines, model, True)
        self.assertIn("proposals", next(events)["first"])
        next(events)
        self.assertTrue(all(torch.equal(value, initial[key]) for key, value in model.state_dict().items()))
        with self.assertRaises(StopIteration):
            next(events)
        self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
