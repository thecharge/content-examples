"""Check the causal stream, baseline arithmetic, and replay integrity."""

from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import torch

from benchmark import add_history, blocks, replay, score
from check_replay import check, match
from model import Board
from sources import sensor_data, source_hashes


class ReplayTests(unittest.TestCase):
    def test_features_use_only_past_readings_and_reset_at_gaps(self):
        times = torch.tensor([300, 600, 900, 1500, 1800])
        features = torch.arange(50, dtype=torch.float).reshape(5, 10)
        complete = add_history(times, features)
        torch.testing.assert_close(complete[:3], add_history(times[:3], features[:3]), equal_nan=True)
        self.assertTrue(complete[[0, 3], 10:15].isnan().all())
        self.assertEqual(complete[1, 10:15].tolist(), [10.] * 5)
        x, y = blocks([(times, features, torch.arange(5))], length=2)
        self.assertEqual(y.tolist(), [[0, 1], [3, 4]])
        torch.testing.assert_close(x[1], features[3:])

    def test_online_replay_cannot_see_the_current_answer(self):
        times = torch.arange(1, 5) * 300
        rows = torch.ones(4, 10)
        first = torch.tensor([0, 0, 1, 1])
        second = torch.tensor([1, 0, 1, 1])
        board = Board().eval()
        a, _ = replay(board, (times, rows, first), online=True, traces=True)
        b, _ = replay(board, (times, rows, second), online=True, traces=True)
        self.assertEqual(a["examples"], b["examples"])
        self.assertEqual(a["answers_revealed"], 3)

    def test_baselines_expose_majority_class_effect(self):
        actual = torch.tensor([0, 1, 1, 1])
        occupied = score(actual, torch.ones_like(actual), torch.ones_like(actual))
        self.assertEqual(occupied["accuracy"], .75)
        self.assertEqual(occupied["balanced_accuracy"], .5)
        self.assertEqual(occupied["classes"], [{"rows": 1, "right": 0}, {"rows": 3, "right": 3}])

    def test_receipt_checks_hashes_before_loading_archives(self):
        match({"values": [1., 5]}, {"values": [1.0000001, 5]})
        with self.assertRaisesRegex(ValueError, "values.1"):
            match({"values": [1., 4]}, {"values": [1., 5]})
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "board.pt").write_bytes(b"tampered")
            (folder / "results.json").write_text(json.dumps(
                {"code_sha256": source_hashes(), "sha256": {"board.pt": "changed"}}))
            with self.assertRaisesRegex(ValueError, "Hash mismatch for board.pt"):
                check(folder / "missing-archives", folder)

    def test_parser_uses_last_valid_label_and_latest_sensor_in_the_bucket(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            start = datetime(2020, 1, 2)
            labels = ["timestamp,occupied"]
            for i in range(8640):
                at = start + timedelta(seconds=10 * i)
                label = "?" if i == 29 else "1" if i == 28 else "0"
                labels.append(f"{at.isoformat()},{label}")
            with ZipFile(folder / "H1_GROUNDTRUTH.zip", "w") as archive:
                archive.writestr("2020-01-01.csv", "timestamp,occupied\n2020-01-01T00:00:00,1")
                archive.writestr("2020-01-02.csv", "\n".join(labels))
            with ZipFile(folder / "H1_ENVIRONMENTAL.zip", "w") as archive:
                archive.writestr("hub/2020-01-02.csv", "\n".join((
                    ",co2eq_ppm,tvoc_ppb,temp_c,rh_percent,light_lux",
                    "2020-01-02T00:04:50,900,0,24,40,0",
                    "2020-01-02T00:00:10,400,0,24,40,0",
                    "2020-01-02T00:05:10,400,0,24,40,0",
                )))
            (times, rows, actual), hashes = sensor_data.occupancy(folder, 1, 1)
            self.assertEqual(times.diff().tolist(), [300])
            self.assertEqual(actual.tolist(), [1, 0])
            self.assertEqual(rows[:, 0].tolist(), [1., 0.])
            self.assertEqual(rows[1, 5].item(), -1.)
            self.assertEqual(len(hashes), 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
