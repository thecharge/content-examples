"""Independent streams, real batched execution, versioning and durable recovery."""

from copy import deepcopy
from dataclasses import replace
import io
import tempfile
from pathlib import Path
import unittest

import torch

from batching import Backend
from graph import Graph
from ledger import Ledger
from multistream import MultiRuntime, StreamLimits
from provenance import Sensor
from runtime import Runtime, Settings, pack_state
from session import Session, pack_runtime, unpack_runtime
from test_adaptive import base, decisive


torch.set_num_threads(1)


def make(**kwargs):
    torch.manual_seed(73)
    return MultiRuntime(
        Graph(base()).eval(),
        settings=Settings(contextual=False, full_weight_learning=False),
        **kwargs,
    )


def values(i=0):
    return {name: float((i + j) % 5) / 5 for j, name in enumerate(make_fields)}


make_fields = Graph(base()).fields


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class Streams(unittest.TestCase):
    def test_batched_36_streams_match_independent_states_and_clocks(self):
        runtime = make()
        # Exercise both descendant boards and a higher-arity tensor message.
        runtime.graph.spawn("b0", 0)
        runtime.graph.connect("b0", ["e0", "e1"], ["e2"], runtime.graph.fields[:2])
        independent = [
            Runtime(runtime.graph.fork(), settings=runtime.settings) for _ in range(36)
        ]
        for i in range(36):
            runtime.open(f"s{i}", f"H{i // 6 + 1}")
        for step in range(4):
            for left in range(0, 36, 8):
                items = [
                    dict(
                        stream_id=f"s{i}",
                        room=f"H{i // 6 + 1}",
                        stamp=(step + 1) * 300 + i * 10000,
                        values=values(step + i),
                        evidence_id=f"{i}:{step}",
                    )
                    for i in range(left, min(left + 8, 36))
                ]
                outputs = runtime.predict_batch(items)
                for item, out, i in zip(items, outputs, range(left, min(left + 8, 36))):
                    expected = independent[i].predict(item["stamp"], item["values"])
                    self.assertEqual(out["answer"], expected["answer"])
                    self.assertAlmostEqual(
                        out["probability"], expected["probability"], places=6
                    )
                    runtime.discard(out["token"])
                    independent[i].discard(expected["token"])
                    # Batch GEMMs can round differently; state tensors must agree numerically.
                    a, b = (
                        pack_state(runtime.streams[f"s{i}"]["state"]),
                        pack_state(independent[i].state),
                    )
                    for path in a["local"]:
                        for x, y in zip(
                            a["local"][path]["cells"], b["local"][path]["cells"]
                        ):
                            torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)
                        torch.testing.assert_close(a["stamps"][path], b["stamps"][path])
        self.assertEqual(len(runtime.streams), 36)

    def test_absent_child_context_does_not_become_a_phantom_clock(self):
        r = make()
        r.graph.spawn("b0", 0)
        r.graph.route_floor = 0.0
        for core in r.graph.cores.values():
            decisive(core)
        runs = [Runtime(r.graph.fork(), settings=r.settings) for _ in range(2)]
        for sid in ("a", "b"):
            r.open(sid, "H1")
        for step in range(5):
            readings = [
                values(step),
                {name: None for name in make_fields} if step < 4 else values(step),
            ]
            batch = [
                dict(
                    stream_id=sid,
                    room="H1",
                    stamp=(step + 1) * 300,
                    values=x,
                    evidence_id=f"{sid}:{step}",
                )
                for sid, x in zip(("a", "b"), readings)
            ]
            outputs = r.predict_batch(batch)
            for i, (out, x) in enumerate(zip(outputs, readings)):
                expected = runs[i].predict((step + 1) * 300, x)
                self.assertEqual(out["answer"], expected["answer"])
                self.assertAlmostEqual(
                    out["probability"], expected["probability"], places=6
                )
                self.assertEqual(
                    set(r.streams[("a", "b")[i]]["state"].local),
                    set(runs[i].state.local),
                )
                r.discard(out["token"])
                runs[i].discard(expected["token"])

    def test_atomic_batch_commits_feedback_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "batch.sqlite")
            try:
                session = Session(ledger, make())
                sensors = {
                    name: Sensor(name, "p", "first", "H1", "v1", "c1", "n1")
                    for name in values()
                }
                items = [
                    dict(
                        event_id=f"o{i}",
                        event_time=300,
                        arrival_time=301,
                        values=values(i),
                        sensors=sensors,
                        stream_id=f"s{i}",
                        context="H1",
                        evidence_id=f"unique{i}",
                    )
                    for i in range(6)
                ]
                result = session.predict_batch("batch", items)
                self.assertEqual(session.predict_batch("batch", items), result)
                self.assertEqual(len(session.runtime.pending), 6)
                recovered = Session(ledger)
                for out in reversed(result["predictions"]):
                    recovered.reward(
                        f"r{out['token']}",
                        "batch",
                        1.0,
                        302,
                        prediction_token=out["token"],
                    )
                    recovered.label(
                        f"l{out['token']}",
                        "batch",
                        1,
                        303,
                        authority="truth",
                        prediction_token=out["token"],
                    )
                self.assertFalse(recovered.runtime.pending)
            finally:
                ledger.close()

    def test_memory_reservation_and_expired_deduplication(self):
        with self.assertRaises(ValueError):
            r = make(stream_limits=replace(StreamLimits(), maximum_resident_bytes=1))
            r.open("a", "H1")
        r = make(stream_limits=replace(StreamLimits(), maximum_deduplication=1))
        first = r.predict(300, values(), stream_id="a", room="H1", evidence_id="old")
        late = r.predict(300, values(), stream_id="b", room="H1", evidence_id="old")
        r.observe(first["token"], 1, 301)
        second = r.predict(600, values(), stream_id="a", room="H1", evidence_id="new")
        r.observe(second["token"], 1, 601)
        with self.assertRaises(ValueError):
            r.observe(late["token"], 1, 601)
        self.assertIn(late["token"], r.pending)

    def test_feedback_only_orders_within_stream_and_replicas_deduplicate(self):
        r = make()
        a = r.predict(300, values(), stream_id="a", room="H1", evidence_id="same")
        b = r.predict(300, values(), stream_id="b", room="H1", evidence_id="same")
        c = r.predict(300, values(2), stream_id="c", room="H2", evidence_id="other")
        self.assertTrue(r.observe(c["token"], 1, 301)["shared"])
        self.assertTrue(r.observe(b["token"], 0, 301)["shared"])
        self.assertFalse(r.observe(a["token"], 0, 301)["shared"])
        self.assertEqual(len(r.history), 2)
        self.assertEqual(len(r.pending), 0)

    def test_same_stream_out_of_order_feedback_rejected_without_changes(self):
        r = make()
        a = r.predict(300, values())
        b = r.predict(600, values())
        with self.assertRaises(ValueError):
            r.observe(b["token"], 1, 601)
        self.assertEqual(len(r.pending), 2)
        r.observe(a["token"], 0, 601)
        r.observe(b["token"], 1, 602)

    def test_stream_state_cannot_leak_or_alias(self):
        r = make()
        a = r.predict(300, values(), stream_id="a", room="H1")
        b = r.predict(300, values(1), stream_id="b", room="H2")
        before = deepcopy(pack_state(r.streams["a"]["state"]))
        r.predict(600, values(3), stream_id="b", room="H2")
        self.assertTrue(equal(before, pack_state(r.streams["a"]["state"])))
        self.assertNotEqual(
            r.streams["a"]["state"].local["b0"].memory.data_ptr(),
            r.streams["b"]["state"].local["b0"].memory.data_ptr(),
        )
        r.observe(a["token"], 0, 301)
        r.observe(b["token"], 1, 601)

    def test_snapshot_resumes_all_pending_feedback_queue_and_rng(self):
        r = make()
        for i in range(6):
            r.predict(300, values(i), stream_id=f"s{i}", room=f"H{i + 1}")
            r.enqueue(
                stream_id=f"s{i}",
                room=f"H{i + 1}",
                stamp=600,
                values=values(i + 1),
                evidence_id=f"next{i}",
            )
        buffer = io.BytesIO()
        r.save(buffer)
        buffer.seek(0)
        loaded = MultiRuntime.load(buffer)
        self.assertTrue(equal(pack_runtime(r), pack_runtime(loaded)))
        for token in list(r.pending):
            r.observe(token, token % 2, 301)
            loaded.observe(token, token % 2, 301)
        self.assertEqual(r.drain(), loaded.drain())

    def test_scheduler_does_not_block_healthy_room_behind_full_stream(self):
        r = make(stream_limits=replace(StreamLimits(), maximum_pending_per_stream=1))
        r.predict(300, values(), stream_id="a", room="H1")
        r.open("b", "H2")
        r.enqueue(
            stream_id="a", room="H1", stamp=600, values=values(), evidence_id="a2"
        )
        r.enqueue(
            stream_id="b", room="H2", stamp=300, values=values(), evidence_id="b1"
        )
        self.assertEqual([o["stream_id"] for o in r.drain()], ["b"])
        self.assertEqual(len(r.queue), 1)

    def test_per_room_veto_rejects_hidden_harm(self):
        r = make()
        r.protected_rooms = ["H1", "H2"]
        r.shadow = r.graph.fork()
        rows = []
        for room in ("H1", "H2"):
            for i in range(32):
                truth = i % 2
                answer = truth if room == "H1" else 1 - truth
                rows.append(
                    dict(
                        room=room,
                        truth=truth,
                        active=float(truth),
                        active_answer=truth,
                        candidate=float(answer),
                        candidate_answer=answer,
                        active_quorum=True,
                        candidate_quorum=True,
                    )
                )
        allowed, report = r.quality_veto(rows)
        self.assertFalse(allowed)
        self.assertTrue(report["H1"]["passed"])
        self.assertFalse(report["H2"]["passed"])

    def test_pending_predictions_do_not_block_shared_stage(self):
        r = make()
        r.settings = replace(
            r.settings,
            window=4,
            history=576,
            minority=1,
            full_weight_learning=True,
            policy_updates=1,
            group_updates=1,
        )
        r.open("slow", "H1")
        stale = r.predict(100, values(), stream_id="slow", room="H1")
        for i in range(4):
            out = r.predict(300 * (i + 1), values(i), stream_id="fast", room="H1")
            r.observe(out["token"], i % 2, 300 * (i + 1) + 1)
        self.assertIn(stale["token"], r.pending)
        self.assertTrue(r.shadow is not None or len(r.receipts) > 0)
        boundary = r.stage_directory / "boundary.pt"
        if boundary.exists():
            saved = MultiRuntime.load(boundary)
            self.assertIn(stale["token"], saved.pending)

    def test_private_contextual_feedback_and_snapshot(self):
        r = make()
        r.settings = replace(r.settings, contextual=True, window=4, minority=1)
        r.open("a", "H1")
        r.open("b", "H2")
        for i in range(10):
            out = r.predict((i + 1) * 300, values(i), stream_id="a", room="H1")
            r.observe(out["token"], i % 2, (i + 1) * 300 + 1)
        self.assertEqual(len(r.streams["b"]["learner"].history[0]), 0)
        self.assertEqual(len(r.streams["b"]["learner"].history[1]), 0)
        loaded = unpack_runtime(pack_runtime(r), None)
        self.assertTrue(equal(pack_runtime(r), pack_runtime(loaded)))

    def test_same_room_unique_labels_may_arrive_in_other_stream_order(self):
        r = make()
        r.settings = replace(r.settings, contextual=True, window=4, minority=1)
        a = r.predict(900, values(), stream_id="a", room="H1", evidence_id="newer")
        b = r.predict(300, values(1), stream_id="b", room="H1", evidence_id="older")
        r.observe(a["token"], 1, 901)
        r.observe(b["token"], 0, 301)
        self.assertEqual(r.rooms["H1"]["learner"].latest, 2)
        self.assertIs(r.streams["a"]["learner"], r.streams["b"]["learner"])
        self.assertEqual(len(r.history), 2)

    def test_stale_version_label_is_retained_without_state_replay(self):
        r = make()
        out = r.predict(300, values())
        r.version += 1
        r.observe(out["token"], 1, 301)
        self.assertEqual(r.late_feedback, 1)
        self.assertEqual(len(r.history), 1)

    def test_bounds_and_unavailable_accelerator_are_explicit(self):
        r = make(stream_limits=replace(StreamLimits(), maximum_streams=6))
        for i in range(6):
            r.open(str(i), str(i))
        with self.assertRaises(ValueError):
            r.open("extra", "0")
        with self.assertRaises(ValueError):
            r.predict_batch(
                [
                    dict(
                        stream_id="0",
                        room="0",
                        stamp=300,
                        values=values(),
                        evidence_id="a",
                    )
                ]
                * 2
            )
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                Backend("cuda")

    def test_transactional_recovery_replica_feedback_and_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "test.sqlite")
            try:
                session = Session(ledger, make())
                sensors = {
                    name: Sensor(name, "p", "first", "H1", "v1", "c1", "n1")
                    for name in values()
                }
                for sid in ("a", "b", "c"):
                    session.predict(
                        "o" + sid,
                        300,
                        300,
                        values(),
                        sensors,
                        stream_id=sid,
                        context="H1",
                        evidence_id="same",
                    )
                recovered = Session(ledger)
                for sid in ("c", "b", "a"):
                    recovered.label("l" + sid, "o" + sid, 1, 301, authority="truth")
                self.assertEqual(len(recovered.runtime.history), 1)
                self.assertEqual(len(recovered.runtime.streams), 3)
                recovered.correct("bad", "oc", "bad sensor")
                with self.assertRaises(ValueError):
                    recovered.predict(
                        "next", 600, 600, values(), sensors, stream_id="a", context="H1"
                    )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
