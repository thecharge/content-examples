"""Lifecycle and schema integration checks beyond the numerical replay tests."""

from copy import deepcopy
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from multistream import MultiRuntime
from adaptation import StreamExample, create, pack, restore
from graph import GraphState
from runtime import pack_state
import torch
import test_adaptive
from test_multistream import equal, make, values


class Lifecycle(unittest.TestCase):
    def test_room_calibration_checks_out_of_order_stream_labels(self):
        runtime = make()
        runtime.settings = replace(
            runtime.settings, contextual=True, window=4, minority=1
        )
        for step in range(8):
            for stream, offset, truth in (("a", 900, 1), ("b", 300, 0)):
                stamp = offset + step * 300
                issued = runtime.predict(
                    stamp,
                    values(step),
                    stream_id=stream,
                    room="H1",
                    evidence_id=f"{stream}/{step}",
                )
                runtime.observe(issued["token"], truth, stamp + 1)
        learner = runtime.rooms["H1"]["learner"]
        self.assertEqual(learner.latest, 16)
        self.assertEqual(len(learner.receipts), 4)
        for receipt in learner.receipts:
            if receipt.get("accepted"):
                self.assertLess(
                    receipt["fit_revealed_through"], receipt["check_prediction_first"]
                )

    def test_calibration_never_fits_labels_unavailable_at_check_prediction(self):
        runtime = make()
        learner = create(runtime.graph, runtime.settings, stream=True)

        def row(sequence, prediction, revealed):
            return StreamExample(
                sequence,
                torch.zeros(1, len(runtime.graph.fields)),
                GraphState(),
                torch.tensor([sequence % 2]),
                torch.tensor([0.0]),
                sequence * 300,
                "a",
                prediction,
                revealed,
            )

        fit = [row(1, 1, 20), row(2, 2, 21)]
        check = [row(3, 10, 30), row(4, 11, 31)]
        receipt = learner.try_update(fit, check)
        self.assertEqual(receipt["status"], "frozen_no_available_fit_history")
        learner.changed_at = 12
        receipt = learner.try_update(fit, check)
        self.assertEqual(receipt["status"], "frozen_check_predates_calibration")
        learner.check = check
        learner.history[0].append(fit[1])
        learner.history[1].append(fit[0])
        recovered = restore(runtime.graph, runtime.settings, pack(learner))
        self.assertTrue(equal(pack(learner), pack(recovered)))

    def test_all_36_migrate_adopt_snapshot_and_rollback(self):
        r = make()
        r.settings = replace(r.settings, window=48, minority=16)
        for i in range(36):
            out = r.predict(300, values(i), stream_id=f"s{i}", room=f"H{i // 6 + 1}")
            r.discard(out["token"])
        before = {
            sid: deepcopy(pack_state(slot["state"])) for sid, slot in r.streams.items()
        }
        fixture = test_adaptive.Learning().make_trial("retire")
        r.shadow = r.graph.fork()
        r.shadow.retire("b0", 0)
        r.trial = fixture.trial
        r.protected_rooms = [f"H{i}" for i in range(1, 7)]
        r.check = []
        for i, room in enumerate(r.protected_rooms):
            r.check.extend(
                dict(row, room=room, stamp=300 * (i * 48 + j + 1))
                for j, row in enumerate(fixture.check)
            )
        for slot in r.streams.values():
            slot["shadow_state"] = r.shadow.align(slot["state"], r.graph)
        pending = r.predict(600, values(), stream_id="s0", room="H1")
        r.finish()
        self.assertTrue(r.receipts[-1]["accepted"])
        self.assertEqual(r.graph.size(), 4)
        self.assertEqual(r.version, 1)
        for slot in r.streams.values():
            self.assertEqual(slot["state"].local["b0"].cells[0].shape[1], 4)
            self.assertEqual(slot["fallback_state"].local["b0"].cells[0].shape[1], 5)
        buffer = io.BytesIO()
        r.save(buffer)
        buffer.seek(0)
        r = MultiRuntime.load(buffer)
        self.assertIn(pending["token"], r.pending)
        r.monitor = []
        for i, room in enumerate(r.monitor_rooms):
            r.monitor.extend(
                dict(
                    row,
                    room=room,
                    stamp=300 * (i * 48 + j + 1),
                    candidate_answer=1 - row["truth"] if room == "H6" else row["truth"],
                    candidate=0.9
                    if room == "H6" and not row["truth"]
                    else 0.1
                    if room == "H6"
                    else row["candidate"],
                )
                for j, row in enumerate(fixture.check)
            )
        r.finish_monitor()
        self.assertTrue(r.monitor_audit[-1]["rollback"])
        self.assertFalse(r.monitor_audit[-1]["per_room"]["H6"]["passed"])
        self.assertEqual(r.graph.size(), 5)
        for sid, slot in r.streams.items():
            self.assertEqual(slot["state"].local["b0"].cells[0].shape[1], 5)
            if sid != "s0":
                self.assertTrue(equal(before[sid], pack_state(slot["state"])))

    def test_schema_boundary_saves_mapped_private_contexts_and_old_pending(self):
        r = make()
        r.settings = replace(
            r.settings,
            window=4,
            minority=1,
            full_weight_learning=True,
            contextual=True,
            policy_updates=1,
            group_updates=1,
        )
        old = r.predict(100, values(), stream_id="slow", room="H1")
        for i in range(4):
            out = r.predict(
                (i + 1) * 300,
                dict(values(i), new_sensor=float(i)),
                stream_id="fast",
                room="H1",
            )
            r.observe(out["token"], i % 2, (i + 1) * 300 + 1)
        self.assertIn("new_sensor", r.graph.fields)
        self.assertIn(old["token"], r.pending)
        saved = MultiRuntime.load(r.stage_directory / "boundary.pt")
        self.assertIn(old["token"], saved.pending)
        for slot in saved.streams.values():
            self.assertEqual(len(slot["correction"]), 2 * len(saved.graph.fields) + 1)
        # The old token supplies data without replaying its incompatible state.
        r.observe(old["token"], 1, 1300)
        self.assertEqual(r.late_feedback, 1)


if __name__ == "__main__":
    unittest.main()
