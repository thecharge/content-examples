from pathlib import Path
import tempfile
import unittest

import torch

from model import Workflow
from serve import Stream


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        torch.set_num_threads(1)

    def test_checker_returns_to_previous_stage_and_calls_a_different_expert(self):
        model = Workflow()
        x = torch.tensor([[.5,-.2]])
        result = model.step(x,torch.tensor([5.]),force_objection=True,trace=True)
        self.assertEqual(result.attempts[1]["revisit"],1)
        self.assertEqual(result.attempts[1]["objection_from"],2)
        self.assertNotEqual(result.attempts[0]["first"],result.attempts[2]["first"])
        if result.attempts[2]["retry"]:
            self.assertNotEqual(result.attempts[2]["first"],result.attempts[2]["retry"])
        self.assertLessEqual(sum("revisit" in x for x in result.attempts),1)

    def test_required_check_withholds_a_guess_after_retries(self):
        model = Workflow()
        result = model.step(torch.tensor([[4.,3.]]),torch.tensor([5.]),tolerance=1e-6,trace=True)
        self.assertFalse(bool(result.accepted[0]))
        self.assertTrue(any("revisit" in x for x in result.attempts))

    def test_later_feedback_updates_experts_and_snapshot_restores_next_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.pt"
            saved = Path(folder)/"live.pt"
            torch.save(Workflow().state_dict(),path)
            a=Stream(path,.5,live=True)
            read=lambda k,x: dict(timestamp=300*k,co2_ppm=x,temperature_c=22)
            first=a.observe(read(1,560))
            old=[p.detach().clone() for n,p in a.model.named_parameters() if ".experts." in n]
            second=a.observe(read(2,610))
            self.assertNotIn("feedback_loss",first)
            self.assertIn("feedback_loss",second)
            new=[p for n,p in a.model.named_parameters() if ".experts." in n]
            self.assertTrue(any(not torch.equal(x,y) for x,y in zip(old,new,strict=True)))
            a.snapshot(saved)
            b=Stream(path,.5,live=True)
            b.restore(saved)
            left,right=a.observe(read(3,605)),b.observe(read(3,605))
            for key in ("timestamp","occupied","probability","trace","feedback_loss"):
                self.assertEqual(left[key],right[key])
            with self.assertRaises(ValueError):
                b.observe(read(3,605))


if __name__=="__main__":
    unittest.main()
