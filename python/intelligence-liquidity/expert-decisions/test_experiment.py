import unittest

import torch

from experiment import choose, combine, verify


class ExpertPolicyTests(unittest.TestCase):
    def test_driver_can_attempt_a_low_confidence_claim_but_not_without_evidence(self):
        p = torch.full((1, 5), .52)
        vote, confidence, acting, payoff, curious = choose(p, torch.tensor([[1., 0., 1., 1., 1.]]),
                                                         torch.tensor([0]))
        self.assertEqual(acting.tolist(), [[True, False, False, False, False]])
        self.assertTrue(curious[0, 0])
        self.assertLess(payoff[0, 0], 0)
        self.assertAlmostEqual(confidence[0, 0].item(), .52)

    def test_review_needs_dissent_and_evidence(self):
        p = torch.tensor([[.8, .6, .3, .5, .7]])
        action = torch.ones_like(p, dtype=torch.bool)
        counter = torch.tensor([[.2, .2, .6, .9, .95]])
        known = torch.tensor([[True, False, True, True, True]])
        kept, veto = verify(p, action, counter, known, .75)
        self.assertEqual(veto.tolist(), [[True, False, False, False, False]])
        self.assertEqual(int(kept.sum()), 4)

    def test_empty_pool_withholds(self):
        p, called = combine(torch.full((1, 5), .9), torch.zeros((1, 5), dtype=torch.bool))
        self.assertEqual(p.item(), .5)
        self.assertFalse(called.item())
