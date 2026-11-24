"""New behavior beyond the copied adaptive board, including failure paths."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from control import Guards
from graph import Graph, GraphState
from policy import ACTIONS
from reference import FIELDS, WIDTH
from runtime import Runtime, Settings, train_candidate
from test_adaptive import base, decisive, state_equal
from workflow import Journal, recover, train

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


class Scaling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_layer_insert_every_location_preserves_recurrent_sequence(self):
        for location in range(4):
            original = Graph(base()).eval()
            evolved = original.fork()
            evolved.insert_layer("b0", location)
            old, new = GraphState(), GraphState()
            for step in range(6):
                x = torch.randn(2, 17)
                if step == 2:
                    x[:] = float("nan")
                a = original(x, torch.tensor([5.0, 7.0]), old, stamp=(step + 1) * 420)
                b = evolved(x, torch.tensor([5.0, 7.0]), new, stamp=(step + 1) * 420)
                self.assertTrue(torch.equal(a.answer, b.answer))
                self.assertTrue(torch.equal(a.committed, b.committed))
                torch.testing.assert_close(
                    a.probability, b.probability, atol=1e-6, rtol=0
                )
                for identity in original.cores["b0"].layer_ids:
                    i = original.cores["b0"].layer_ids.index(identity)
                    j = evolved.cores["b0"].layer_ids.index(identity)
                    torch.testing.assert_close(
                        a.decision.state.cells[i],
                        b.decision.state.cells[j],
                        atol=1e-6,
                        rtol=0,
                    )
                old, new = a.state.detached(), b.state.detached()

    def test_layer_alignment_retains_existing_states_and_removes_bypass(self):
        original = Graph(base()).eval()
        decisive(original.cores["b0"])
        before = original(torch.ones(1, 17), torch.tensor([5.0]), stamp=300).state
        grown = original.fork()
        grown.insert_layer("b0", 1)
        aligned = grown.align(before, original)
        self.assertEqual(len(aligned.local["b0"].cells), 4)
        torch.testing.assert_close(
            aligned.local["b0"].cells[2], before.local["b0"].cells[1]
        )
        grown.remove_layer("b0", 1)
        a = original(torch.ones(1, 17), torch.tensor([5.0]), before, stamp=600)
        b = grown(torch.ones(1, 17), torch.tensor([5.0]), before, stamp=600)
        self.assertTrue(state_equal(a.decision.state, b.decision.state))
        self.assertEqual(float(a.probability), float(b.probability))

    def test_attention_tree_vertical_horizontal_and_memory_alignment(self):
        original = Graph(base()).eval()
        decisive(original.cores["b0"])
        before = original(torch.ones(1, 17), torch.tensor([5.0]), stamp=300).state
        grown = original.fork()
        a = grown.branch_attention("b0")["attention"]
        grown.branch_attention("b0")["attention"]
        c = grown.branch_attention("b0", a)["attention"]
        grown.change_heads("b0", True, attention=c)
        aligned = grown.align(before, original)
        self.assertEqual(grown.cores["b0"].delta.total_heads, 6)
        self.assertEqual(aligned.local["b0"].memory.shape[1], 6)
        old, new = before, aligned
        for step in range(4):
            x = torch.randn(1, 17)
            original_result = original(
                x, torch.tensor([5.0]), old, stamp=600 + step * 300
            )
            r = grown(x, torch.tensor([5.0]), new, stamp=600 + step * 300)
            torch.testing.assert_close(
                original_result.probability, r.probability, atol=1e-6, rtol=0
            )
            self.assertTrue(torch.equal(original_result.answer, r.answer))
            old, new = original_result.state.detached(), r.state.detached()
        restored = Graph.restore(grown.pack()).eval()
        self.assertEqual(grown.layout(), restored.layout())
        r = restored(x, torch.tensor([5.0]), new, stamp=1800)
        g = grown(x, torch.tensor([5.0]), new, stamp=1800)
        self.assertTrue(torch.equal(r.probability, g.probability))
        grown.prune_attention("b0", a)
        self.assertEqual(len(list(grown.cores["b0"].delta.nodes())), 2)
        with self.assertRaises(ValueError):
            grown.prune_attention("b0", "root")

    def test_attention_routes_a_liquid_child_and_backpropagates(self):
        graph = Graph(base()).eval()
        attention = graph.branch_attention("b0")["attention"]
        child = graph.deepen("b0", 0, FIELDS[:3], attention=attention)["board"]
        for core in graph.cores.values():
            decisive(core)
        graph.cores["b0"].child_gain.data[0] = 0.5
        x = torch.ones(1, 17)
        result = graph(x, torch.tensor([5.0]), stamp=300)
        self.assertIn(child, [b for b, _ in result.trace.values()])
        result.probability.sum().backward()
        self.assertIsNotNone(graph.cores["b0"].child_gain.grad)
        self.assertIsNotNone(graph.cores["b0"].delta.branches["a0"].query.weight.grad)
        self.assertTrue(
            torch.isfinite(
                graph.cores["b0"].delta.branches["a0"].query.weight.grad
            ).all()
        )

    def test_global_bound_rejects_before_allocating_and_leaves_graph_unchanged(self):
        graph = Graph(
            base(), max_attention_nodes=1, max_total_heads=2, max_layers=3
        ).eval()
        before = deepcopy(graph.layout())
        with patch("graph.deepcopy", side_effect=AssertionError("allocated")):
            with self.assertRaises(ValueError):
                graph.branch_attention("b0")
            with self.assertRaises(ValueError):
                graph.insert_layer("b0", 0)
            with self.assertRaises(ValueError):
                graph.change_heads("b0", True)
        self.assertEqual(before, graph.layout())
        graph.max_work = graph.profile()["work_units"]
        with patch("graph.deepcopy", side_effect=AssertionError("allocated")):
            with self.assertRaises(ValueError):
                graph.spawn("b0", 0)
            with self.assertRaises(ValueError):
                graph.deepen("b0", 0, FIELDS[:2])
        with self.assertRaises(ValueError):
            graph.add_fields(["new"])
        self.assertEqual(before, graph.layout() | {"max_work": before["max_work"]})

    def test_minimums_and_recursive_work_are_global(self):
        graph = Graph(base(), min_experts=5, min_encoding=16, max_boards=1).eval()
        with self.assertRaises(ValueError):
            graph.retire("b0", 0)
        with self.assertRaises(ValueError):
            graph.prune_neuron("b0", 0)
        with self.assertRaises(ValueError):
            graph.remove_layer("b0", 0)
        with self.assertRaises(ValueError):
            graph.deepen("b0", 0, FIELDS[:2])
        graph = Graph(base(), max_attention_depth=1).eval()
        a = graph.branch_attention("b0")["attention"]
        with self.assertRaises(ValueError):
            graph.branch_attention("b0", a)
        with self.assertRaises(ValueError):
            graph(torch.ones(9, 17), torch.ones(9))

    def test_encoding_contraction_checkpoint_roundtrip(self):
        graph = Graph(base()).eval()
        for _ in range(3):
            graph.prune_neuron("b0", 2)
        restored = Graph.restore(graph.pack()).eval()
        x = torch.randn(1, 17)
        self.assertEqual(restored.cores["b0"].encoding, 13)
        self.assertTrue(
            torch.equal(
                graph(x, torch.tensor([5.0])).probability,
                restored(x, torch.tensor([5.0])).probability,
            )
        )

    def test_new_actions_are_selected_by_learned_values_and_masked(self):
        graph = Graph(base()).eval()
        runtime = Runtime(
            graph,
            settings=Settings(contextual=False, exploration=0),
            guards=Guards(minimum_life=1),
        )
        runtime.controller.last = dict(
            growth_pressure=True,
            cooled=True,
            brier=0.5,
            signals={
                f"b0/{i}": dict(gradient=1.0, neuron=0) for i in graph.cores["b0"].ids
            },
        )
        for action in ("add_layer", "branch_attention"):
            for policy in graph.cores["b0"].policies:
                policy.learn_action(
                    torch.zeros(WIDTH + 8), ACTIONS.index(action), 1.0, updates=20
                )
                with torch.no_grad():
                    policy.action[-1].bias.zero_()
                    policy.action[-1].bias[ACTIONS.index(action)] = 5
            self.assertEqual(ACTIONS[runtime.choose()[2]], action)
        graph.max_attention_nodes = 1
        self.assertNotEqual(ACTIONS[runtime.choose()[2]], "branch_attention")

    def test_explicit_reward_is_delayed_unique_and_recoverable(self):
        runtime = Runtime(
            Graph(base()),
            settings=Settings(contextual=False, full_weight_learning=False),
        )
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            journal = Journal(runtime, folder / "events.jsonl")
            first = journal.predict(300, dict.fromkeys(FIELDS, 1.0))
            journal.snapshot(folder / "snapshot.pt")
            with self.assertRaises(ValueError):
                journal.reward(first["token"], 2, 301)
            journal.reward(first["token"], 0.7, 301)
            with self.assertRaises(ValueError):
                journal.reward(first["token"], 0.8, 301)
            restored, cursor = recover(
                folder / "snapshot.pt", folder / "events.jsonl", 301
            )
            self.assertEqual(restored.pending[first["token"]]["reward"], 0.7)
            self.assertEqual(cursor, 2)
            restored.observe(first["token"], 1, 302)
            self.assertEqual(restored.history[-1]["reward"], 0.7)

    def test_reward_alone_cannot_certify_accuracy_or_create_a_stage(self):
        runtime = Runtime(Graph(base()), settings=Settings(contextual=False))
        first = runtime.predict(300, dict.fromkeys(FIELDS, 1.0))
        runtime.reward(first["token"], 1, 301)
        self.assertEqual(len(runtime.history), 0)
        self.assertIsNone(runtime.shadow)
        self.assertIn(first["token"], runtime.pending)

    def test_post_adoption_monitor_rolls_back_and_survives_snapshot(self):
        runtime = Runtime(
            Graph(base()), settings=Settings(window=4, minority=2, contextual=False)
        )
        runtime.fallback = runtime.graph.fork()
        runtime.graph.insert_layer("b0", 1)
        runtime.monitor = [
            dict(
                stamp=i * 300,
                truth=i % 2,
                active=0.1 if i % 2 == 0 else 0.9,
                active_answer=i % 2,
                active_quorum=True,
                candidate=0.9 if i % 2 == 0 else 0.1,
                candidate_answer=1 - i % 2,
                candidate_quorum=True,
            )
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "x.pt"
            runtime.save(file)
            restored = Runtime.load(file)
        self.assertIsNotNone(restored.fallback)
        restored.finish_monitor()
        self.assertEqual(len(restored.graph.cores["b0"].cells), 3)
        self.assertTrue(restored.monitor_audit[-1]["rollback"])
        self.assertIsNone(restored.fallback)

    def test_inference_without_feedback_does_not_mutate_topology(self):
        runtime = Runtime(
            Graph(base()), settings=Settings(contextual=False, maximum_pending=4)
        )
        layout = deepcopy(runtime.graph.layout())
        for step in range(4):
            runtime.predict((step + 1) * 300, dict.fromkeys(FIELDS, 0.2))
        self.assertEqual(runtime.graph.layout(), layout)
        with self.assertRaises(ValueError):
            runtime.predict(1500, dict.fromkeys(FIELDS, 0.2))

    def test_training_opens_inserted_stage_and_attention_gate(self):
        ancestor = Graph(base()).eval()
        graph = ancestor.fork()
        mutation = graph.insert_layer("b0", 1)
        graph.branch_attention("b0")
        rows = [
            dict(
                stamp=300 * (i + 1),
                truth=(i // 3) % 2,
                values=dict.fromkeys(FIELDS, float(2 * ((i // 3) % 2) - 1)),
            )
            for i in range(24)
        ]
        train_candidate(
            graph,
            rows,
            Settings(fit_passes=1, gradient_window=4),
            ancestor=ancestor,
            mutation=mutation,
        )
        self.assertNotEqual(float(graph.cores["b0"].layer_gain[1]), 0.0)
        self.assertNotEqual(float(graph.cores["b0"].delta.gains["a0"]), 0.0)

    def test_network_fork_retains_private_context_and_can_retire(self):
        graph = Graph(base(), max_work=2_000_000).eval()
        decisive(graph.cores["b0"])
        graph.spawn("b0", 0)
        child = graph.deepen("b0", 0, FIELDS[:3])["board"]
        graph.edges["b0/" + graph.cores["b0"].ids[-1]] = deepcopy(graph.edges["b0/e0"])
        decisive(graph.cores[child])
        graph.cores["b0"].child_gain.data.fill_(0.5)
        before = graph(torch.ones(1, 17), torch.tensor([5.0]), stamp=300).state
        old = graph.fork()
        fork = graph.fork_network("b0", 5)["board"]
        self.assertIsNot(graph.cores[fork].cells[0], graph.cores[child].cells[0])
        aligned = graph.align(before, old)
        a = old(torch.ones(1, 17), torch.tensor([5.0]), before, stamp=600)
        b = graph(torch.ones(1, 17), torch.tensor([5.0]), aligned, stamp=600)
        torch.testing.assert_close(a.probability, b.probability, atol=1e-6, rtol=0)
        self.assertTrue(any(path.endswith("/" + fork) for path in b.state.local))
        graph.retire_network("b0", 5)
        self.assertNotIn(fork, graph.cores)
        self.assertIn(child, graph.cores)

    def test_unsupported_monitor_rolls_back_without_nonfinite_receipt(self):
        runtime = Runtime(
            Graph(base()), settings=Settings(window=4, minority=2, contextual=False)
        )
        runtime.fallback = runtime.graph.fork()
        runtime.monitor = [
            dict(
                stamp=i * 300,
                truth=1,
                active=0.9,
                active_answer=1,
                active_quorum=True,
                candidate=0.9,
                candidate_answer=1,
                candidate_quorum=True,
            )
            for i in range(4)
        ]
        runtime.finish_monitor()
        self.assertTrue(runtime.monitor_audit[-1]["rollback"])
        json.dumps(list(runtime.monitor_audit), allow_nan=False)

    def test_offline_finetune_reuses_adaptive_stage_loop(self):
        configuration = json.loads((ROOT / "configuration.json").read_text())
        configuration["learning"].update(
            window=8,
            history=16,
            minority=2,
            fit_passes=1,
            contextual=False,
            policy_updates=1,
            group_updates=1,
        )
        configuration["guards"].update(minimum_life=8, probe_rows=8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            cfg = folder / "config.json"
            cfg.write_text(json.dumps(configuration))
            source = Runtime(Graph(base()), settings=Settings(contextual=False))
            source.graph.insert_layer("b0", 1)
            source.graph.branch_attention("b0")
            source.save(folder / "source.pt")
            rows = [
                dict(
                    stamp=300 * (i + 1),
                    truth=(i // 2) % 2,
                    values=dict.fromkeys(FIELDS, float(2 * ((i // 2) % 2) - 1)),
                )
                for i in range(32)
            ]
            dataset = folder / "train.jsonl"
            dataset.write_text("".join(json.dumps(r) + "\n" for r in rows))
            result, receipt = train(
                dataset,
                folder / "fine.pt",
                "finetune",
                checkpoint=folder / "source.pt",
                configuration=cfg,
            )
            self.assertTrue(receipt["structural_receipts"])
            for trial in receipt["structural_receipts"]:
                if "check_first" in trial:
                    self.assertLess(trial["fit_revealed_through"], trial["check_first"])
            self.assertEqual(receipt["mode"], "finetune")
            self.assertEqual(
                Runtime.load(folder / "fine.pt").graph.layout(), result.graph.layout()
            )

    def test_recurrent_neuron_split_preserves_liquid_sequence_and_checkpoint(self):
        graph = Graph(base()).eval()
        decisive(graph.cores["b0"])
        before = graph(torch.randn(1, 17), torch.tensor([5.0]), stamp=300).state
        grown = graph.fork()
        grown.change_liquid("b0", 2, True, unit=3)
        aligned = grown.align(before, graph)
        self.assertEqual(aligned.local["b0"].cells[-1].shape[-1], 9)
        torch.testing.assert_close(
            aligned.local["b0"].cells[-1][..., -1], before.local["b0"].cells[-1][..., 3]
        )
        left, right = before, aligned
        for step in range(8):
            x = torch.randn(1, 17)
            a = graph(x, torch.tensor([5.0]), left, stamp=600 + step * 300)
            b = grown(x, torch.tensor([5.0]), right, stamp=600 + step * 300)
            self.assertTrue(torch.equal(a.answer, b.answer))
            self.assertTrue(torch.equal(a.committed, b.committed))
            torch.testing.assert_close(a.probability, b.probability, atol=1e-6, rtol=0)
            torch.testing.assert_close(
                graph.cores["b0"].hidden(a.decision.state.cells),
                grown.cores["b0"].hidden(b.decision.state.cells),
                atol=1e-6,
                rtol=0,
            )
            left, right = a.state.detached(), b.state.detached()
        restored = Graph.restore(grown.pack()).eval()
        a = grown(x, torch.tensor([5.0]), right, stamp=3300)
        b = restored(x, torch.tensor([5.0]), right, stamp=3300)
        self.assertTrue(torch.equal(a.probability, b.probability))
        self.assertEqual(grown.layout(), restored.layout())

    def test_recurrent_neuron_contraction_and_limits(self):
        graph = Graph(base(), min_liquid=6, max_liquid=9).eval()
        graph.change_liquid("b0", 0, True, 2)
        with self.assertRaises(ValueError):
            graph.change_liquid("b0", 0, True, 2)
        for _ in range(3):
            graph.change_liquid("b0", 0, False, 0)
        with self.assertRaises(ValueError):
            graph.change_liquid("b0", 0, False, 0)
        restored = Graph.restore(graph.pack()).eval()
        x = torch.randn(1, 17)
        self.assertEqual(len(restored.cores["b0"].cells[0].units), 6)
        self.assertTrue(
            torch.equal(
                graph(x, torch.tensor([5.0])).probability,
                restored(x, torch.tensor([5.0])).probability,
            )
        )

    def test_detach_one_shared_kernel_does_not_clone_the_remaining_group(self):
        graph = Graph(base(), max_work=2_000_000)
        graph.spawn("b0", 0)
        graph.spawn("b0", 0)
        children = [graph.deepen("b0", i, FIELDS[:2])["board"] for i in (0, 5, 6)]
        for i, name in enumerate(children):
            graph.cores[name].outputs[-1].bias.data.add_(i)
        graph.share_equal()
        graph.share_equal()
        first, second, third = children
        self.assertIs(graph.cores[first].cells, graph.cores[second].cells)
        shared = graph.cores[second].cells
        with patch("graph.deepcopy", wraps=deepcopy) as copies:
            graph.untie(first)
        self.assertEqual(copies.call_count, 1)
        self.assertIsNot(graph.cores[first].cells, shared)
        self.assertIs(graph.cores[second].cells, shared)
        self.assertIs(graph.cores[third].cells, shared)
        restored = Graph.restore(graph.pack())
        self.assertIs(restored.cores[second].cells, restored.cores[third].cells)
        self.assertIsNot(restored.cores[first].cells, restored.cores[second].cells)

    def test_worker_cancellation_terminates_and_reaps_its_child(self):
        from worker import launch
        from unittest.mock import Mock

        process = Mock(pid=12345)
        process.wait.side_effect = [KeyboardInterrupt(), -15]
        process.poll.return_value = None
        config = dict(memory_mib=128, cpu_seconds=10, wall_seconds=15, file_mib=16)
        with (
            patch("worker.subprocess.Popen", return_value=process),
            patch("worker.os.killpg") as terminate,
        ):
            with self.assertRaises(KeyboardInterrupt):
                launch(config, "allocation-test", [])
        terminate.assert_called_once()
        self.assertEqual(terminate.call_args.args[0], 12345)
        self.assertEqual(process.wait.call_count, 2)

    def test_worker_enforces_memory_without_touching_parent(self):
        configuration = dict(
            memory_mib=128, cpu_seconds=10, wall_seconds=15, file_mib=16
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "limit.json"
            path.write_text(json.dumps(configuration))
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "worker.py"),
                    "--configuration",
                    str(path),
                    "allocation-test",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OS rejected allocation", result.stdout)
        self.assertEqual(sum(range(10)), 45)


if __name__ == "__main__":
    unittest.main()
