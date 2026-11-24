"""Exercise real liquid messages and journal failure boundaries."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from graph import Graph, GraphState
from hypergraph import HyperLimits, correlation, evidence
from ledger import Ledger, StorageLimits
from provenance import Sensor
from runtime import Runtime, Settings
from session import Session, pack_runtime
from test_adaptive import base, decisive

torch.set_num_threads(1)


def sensors(values, incarnation="first"):
    return {
        n: Sensor(n, "publisher", incarnation, "room-a", "v1", "c1", "n1")
        for n in values
    }


def rows(count=32):
    result = []
    for i in range(count):
        context = "left" if i < count // 2 else "right"
        a = 0.1 + 0.01 * i
        b = a if context == "left" else a + 0.3
        result.append(
            dict(
                stamp=300 * (i + 1),
                revealed_at=300 * (i + 1) + 1,
                truth=i % 2,
                observation_id=f"o{i}",
                label_id=f"l{i}",
                context=context,
                proposals={
                    "b0/e0": dict(probability=a, hidden=torch.arange(8).float() * a),
                    "b0/e1": dict(probability=b, hidden=torch.arange(8).float() * b),
                },
            )
        )
    return result


class Messages(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_zero_gate_preserves_sequence_and_nonzero_gate_has_gradients(self):
        original = Graph(base()).eval()
        grown = original.fork()
        edge = grown.connect("b0", ["e0", "e1"], ["e2", "e3"], grown.fields[:2])[
            "hyperedge"
        ]
        left, right = GraphState(), GraphState()
        for i in range(8):
            x = torch.randn(1, 17)
            a = original(x, torch.tensor([5.0]), left, stamp=300 * (i + 1))
            b = grown(x, torch.tensor([5.0]), right, stamp=300 * (i + 1))
            self.assertTrue(torch.equal(a.probability, b.probability))
            left, right = a.state.detached(), b.state.detached()
        decisive(grown.cores["b0"])
        grown.messages[edge].gain.data.fill_(0.3)
        result = grown(torch.ones(1, 17), torch.tensor([5.0]), stamp=300)
        result.probability.sum().backward()
        self.assertGreater(float(grown.messages[edge].readout.weight.grad.norm()), 0)
        self.assertIsNotNone(grown.cores["b0"].cells[-1].strength.grad)

    def test_neuron_is_one_vertex_in_multiple_hyperedges_and_restores(self):
        graph = Graph(base()).eval()
        vertices = [
            dict(expert="e0", layer="l0", unit=2),
            dict(expert="e1", layer="l1"),
        ]
        a = graph.connect(
            "b0", ["e0", "e1"], ["e2"], graph.fields[:1], vertices=vertices
        )["hyperedge"]
        b = graph.connect(
            "b0", ["e0", "e1"], ["e3"], graph.fields[:1], vertices=vertices
        )["hyperedge"]
        self.assertEqual(
            graph.hyperedges[a]["vertices"][0], graph.hyperedges[b]["vertices"][0]
        )
        graph.messages[a].gain.data.fill_(0.2)
        restored = Graph.restore(graph.pack()).eval()
        x = torch.randn(1, 17)
        self.assertTrue(
            torch.equal(
                graph(x, torch.tensor([5.0])).probability,
                restored(x, torch.tensor([5.0])).probability,
            )
        )
        graph.change_liquid("b0", 0, False, unit=2)
        self.assertEqual(set(graph.clean_incidence()), {a, b})

    def test_delta_branch_routes_higher_arity_message(self):
        graph = Graph(base()).eval()
        branch = graph.branch_attention("b0")["attention"]
        edge = graph.connect("b0", ["e0", "e1"], ["e2"], graph.fields[:2], branch)[
            "hyperedge"
        ]
        decisive(graph.cores["b0"])
        graph.messages[edge].gain.data.fill_(0.5)
        before = graph(
            torch.ones(1, 17), torch.tensor([5.0]), stamp=300
        ).state.detached()
        result = graph(torch.ones(1, 17), torch.tensor([5.0]), before, stamp=600)
        result.probability.sum().backward()
        self.assertIsNotNone(graph.cores["b0"].delta.node(branch).query.weight.grad)
        graph.prune_attention("b0", branch)
        self.assertEqual(graph.clean_incidence(), [edge])

    def test_split_merge_and_retirement_lineage_and_limits(self):
        graph = Graph(base(), hyper_limits=dict(max_edges=2)).eval()
        parent = graph.connect(
            "b0", ["e0", "e1", "e2", "e3"], ["e4"], graph.fields[:1]
        )["hyperedge"]
        result = graph.split_edge(parent)
        self.assertEqual(len(graph.hyperedges), 2)
        for child in result["children"]:
            self.assertEqual(graph.hyperedges[child]["parents"], [parent])
        with self.assertRaises(ValueError):
            graph.connect("b0", ["e0", "e1"], ["e2"], graph.fields[:1])
        merged = graph.merge_edge(*result["children"])["hyperedge"]
        self.assertEqual(graph.hyperedges[merged]["parents"], result["children"])
        graph.drop_edge(merged)
        self.assertEqual(len(graph.hyperedges), 0)

    def test_work_and_membership_rejected_before_module_allocation(self):
        graph = Graph(base()).eval()
        graph.max_work = graph.profile()["work_units"]
        with patch("graph.Message", side_effect=AssertionError("allocated")):
            with self.assertRaises(ValueError):
                graph.connect("b0", ["e0", "e1"], ["e2"], graph.fields[:1])
        self.assertEqual(len(graph.hyperedges), 0)

    def test_partial_layer_sharing_preserves_private_state_and_detaches_on_growth(self):
        graph = Graph(base(), max_work=2_000_000).eval()
        graph.spawn("b0", 0)
        a = graph.deepen("b0", 0, graph.fields[:2])["board"]
        b = graph.deepen("b0", 5, graph.fields[:2])["board"]
        graph.tie_layer(a, "l0", b, "l0")
        self.assertIs(graph.cores[a].cells[0], graph.cores[b].cells[0])
        restored = Graph.restore(graph.pack())
        self.assertIs(restored.cores[a].cells[0], restored.cores[b].cells[0])
        left = graph.cores[a].start(torch.ones(1, graph.cores[a].features))
        right = graph.cores[b].start(torch.ones(1, graph.cores[b].features))
        left.cells[0].add_(1)
        self.assertEqual(float(right.cells[0].sum()), 0)
        graph.change_liquid(b, 0, True, 0)
        self.assertIsNot(graph.cores[a].cells[0], graph.cores[b].cells[0])
        self.assertEqual(len(graph.cores[a].cells[0].units), 8)
        self.assertEqual(len(graph.cores[b].cells[0].units), 9)

    def test_expanded_policy_masks_and_selects_hyperedge_actions(self):
        from policy import ACTIONS

        graph = Graph(base()).eval()
        runtime = Runtime(
            graph,
            settings=Settings(contextual=False, exploration=0),
            guards=__import__("control").Guards(minimum_life=1),
        )
        runtime.history.extend(rows())
        runtime.controller.last = dict(
            growth_pressure=True,
            cooled=True,
            brier=0.5,
            signals={
                f"b0/{i}": dict(gradient=1.0, neuron=0) for i in graph.cores["b0"].ids
            },
        )
        for policy in graph.cores["b0"].policies:
            policy.action[-1].bias.data[ACTIONS.index("connect")] = 10
        self.assertEqual(ACTIONS[runtime.choose()[2]], "connect")
        graph.hyper_limits = replace(graph.hyper_limits, max_edges=0)
        self.assertNotEqual(ACTIONS[runtime.choose()[2]], "connect")


class Similarity(unittest.TestCase):
    def test_pearson_alone_cannot_nominate_wrong_calibration_merge(self):
        self.assertAlmostEqual(correlation([0.1, 0.2, 0.3], [0.6, 0.7, 0.8]), 1)
        sample = rows()
        for row in sample:
            row["proposals"]["b0/e1"]["probability"] = (
                row["proposals"]["b0/e0"]["probability"] + 0.5
            )
        measured = evidence(sample, "b0", HyperLimits())[0]
        self.assertAlmostEqual(measured["pearson"], 1)
        self.assertFalse(measured["merge_nomination"])

    def test_distinct_context_residuals_nominate_not_execute_split(self):
        measured = evidence(rows(), "b0", HyperLimits())[0]
        self.assertTrue(measured["split_nomination"])
        self.assertEqual(len(measured["support"]), 32)
        self.assertEqual(measured["metric"], "prediction-residual-linear-cka-v1")

    def test_repeated_support_and_invalid_data_cannot_create_evidence(self):
        sample = rows(8) * 4
        self.assertEqual(evidence(sample, "b0", HyperLimits()), [])
        sample = rows()
        for row in sample:
            row["invalidated"] = True
        self.assertEqual(evidence(sample, "b0", HyperLimits()), [])
        self.assertIsNone(correlation([1, 1, 1], [2, 2, 2]))


class Transactions(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "run.db"
        self.ledger = Ledger(self.path)
        runtime = Runtime(
            Graph(base()),
            settings=Settings(contextual=False, full_weight_learning=False),
        )
        self.session = Session(self.ledger, runtime)
        self.values = {runtime.graph.fields[0]: 1.0}
        self.sensors = sensors(self.values)

    def tearDown(self):
        self.ledger.close()
        self.folder.cleanup()

    def predict(self, identity="observation", stamp=300, **kwargs):
        return self.session.predict(
            identity, stamp, stamp, self.values, self.sensors, **kwargs
        )

    def test_prediction_retry_exactly_once_and_crash_recovery(self):
        a = self.predict()
        b = self.predict()
        self.assertEqual(a, b)
        self.assertEqual(self.session.runtime.serial, 1)
        self.assertEqual(self.session.revision, 2)
        restarted = Session(self.ledger)
        self.assertEqual(restarted.runtime.serial, 1)
        self.assertEqual(restarted.runtime.pending[0]["observation_id"], "observation")
        with self.assertRaises(ValueError):
            self.predict(stamp=600)

    def test_failed_commit_does_not_change_live_runtime_or_events(self):
        before = pack_runtime(self.session.runtime)
        with patch.object(
            self.ledger, "commit", side_effect=OSError("disk write failed")
        ):
            with self.assertRaises(OSError):
                self.predict()
        self.assertEqual(self.session.runtime.serial, before["serial"])
        self.assertEqual(self.ledger.revision, 1)
        self.assertEqual(self.session.runtime.state.local, {})

    def test_stale_writer_cannot_overwrite_and_can_refresh(self):
        other_ledger = Ledger(self.path)
        try:
            other = Session(other_ledger)
            self.predict()
            with self.assertRaises(ValueError):
                other.predict("other", 600, 600, self.values, self.sensors)
            other.refresh()
            issued = other.predict("other", 600, 600, self.values, self.sensors)
            self.assertEqual(issued["token"], 1)
        finally:
            other_ledger.close()

    def test_delayed_label_reward_and_replacement_provenance(self):
        self.predict(quality={next(iter(self.values)): "broken"})
        self.session.reward("reward", "observation", 0.7, 301)
        self.session.label("label", "observation", 1, 302, authority="occupancy-meter")
        row = self.session.runtime.history[-1]
        self.assertIsNone(next(iter(row["values"].values())))
        self.assertEqual(row["label_id"], "label")
        self.assertEqual(row["reward"], 0.7)
        self.session.predict(
            "replacement", 600, 600, self.values, sensors(self.values, "replacement")
        )
        record = json.loads(
            self.ledger.db.execute(
                "SELECT payload FROM events WHERE id='replacement'"
            ).fetchone()[0]
        )
        self.assertEqual(
            next(iter(record["request"]["sensors"].values()))["incarnation"],
            "replacement",
        )

    def test_correction_invalidates_descendants_and_requires_clean_restore(self):
        self.predict()
        self.session.label("label", "observation", 1, 301, authority="meter")
        self.predict("next", 600)
        self.session.correct("correction", "label", "label was wrong")
        invalid = {
            r[0] for r in self.ledger.db.execute("SELECT id FROM events WHERE valid=0")
        }
        self.assertEqual(invalid, {"label", "next"})
        with self.assertRaises(ValueError):
            self.predict("blocked", 900)
        with self.assertRaises(ValueError):
            self.session.restore("badrestore", 3)
        self.session.restore("restore", 2)
        self.assertIsNone(self.session.blocked)
        self.assertEqual(self.session.runtime.serial, 1)

    def test_tensor_dedup_does_not_alias_mutable_tensors(self):
        with self.ledger.db:
            encoded = self.ledger.encode([torch.ones(2), torch.ones(2)])
        self.assertEqual(encoded["list"][0], encoded["list"][1])
        decoded = self.ledger.decode(encoded)
        decoded[0][0] = 7
        self.assertEqual(float(decoded[1][0]), 1)

    def test_corruption_is_detected_before_recovery(self):
        self.ledger.db.execute(
            "UPDATE objects SET data=? WHERE kind='tensor'", (b"corrupt",)
        )
        with self.assertRaises(ValueError):
            Session(self.ledger)

    def test_missing_evidence_cannot_define_a_feature(self):
        with self.assertRaises(ValueError):
            self.session.derive("derived", "mean", list(self.values), "invented")
        self.assertEqual(self.session.features.definitions, {})

    def test_derived_feature_keeps_parents_and_does_not_create_labels(self):
        self.predict()
        self.session.derive("derived", "mean", list(self.values), "observation")
        self.predict("next", 600)
        row = self.session.runtime.pending[1]
        self.assertEqual(row["values"]["mean"], 1)
        self.assertEqual(len(self.session.runtime.history), 0)
        self.assertIn("mean", self.session.runtime.queued_fields)


class Retention(unittest.TestCase):
    def test_roots_keep_shared_payload_and_old_unpinned_payload_expires(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(
                Path(folder) / "run.db", StorageLimits(retained_revisions=1)
            )
            try:
                ledger.commit("a", "test", {}, [torch.tensor([1.0])], 0)
                ledger.pin("audit", 1)
                ledger.commit("b", "test", {}, [torch.tensor([2.0])], 1)
                self.assertEqual(float(ledger.recover(1)[0][0]), 1)
                ledger.unpin("audit")
                ledger.collect()
                with self.assertRaises(ValueError):
                    ledger.recover(1)
                self.assertEqual(float(ledger.recover(2)[0][0]), 2)
            finally:
                ledger.close()

    def test_budget_rejection_rolls_back_objects_and_event(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder) / "run.db", StorageLimits(max_objects=1))
            try:
                with self.assertRaises(ValueError):
                    ledger.commit("a", "test", {}, [torch.ones(2)], 0)
                self.assertEqual(ledger.revision, 0)
                self.assertEqual(
                    ledger.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0], 0
                )
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
