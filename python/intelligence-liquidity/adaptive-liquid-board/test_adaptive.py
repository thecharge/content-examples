"""Behavioral checks; controlled outcomes test mechanisms, not field accuracy."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from graph import Graph, GraphState
from policy import ACTIONS, Policy
from reference import Board, FIELDS, REFERENCE, WIDTH, primitives
from runtime import Runtime, Settings, train_candidate
from control import Guards

torch.set_num_threads(1)


def base():
    board = Board(features=17)
    board.load_state_dict(
        torch.load(REFERENCE / "evidence/seed-73/board.pt", weights_only=True)
    )
    return board.eval()


def decisive(core):
    with torch.no_grad():
        core.subscription.weight.zero_()
        core.subscription.bias.fill_(10)
        for i, output in enumerate(core.outputs):
            output.weight.zero_()
            output.bias.fill_(2.0 if i == 0 else 0.0)
        for checker in core.falsifiers:
            for p in checker.parameters():
                p.zero_()
        core.process.weight.zero_()
        core.process.bias.fill_(10)


def state_equal(a, b):
    return all(
        torch.equal(x, y)
        for x, y in zip(
            (*a.cells, a.memory, a.last, a.elapsed),
            (*b.cells, b.memory, b.last, b.elapsed),
        )
    )


class Fidelity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_exact_native_sequence_including_missing_and_retries(self):
        original = base()
        graph = Graph(original).eval()
        old, new = None, GraphState()
        for step in range(8):
            x = torch.randn(2, 17)
            if step == 3:
                x[:] = float("nan")
            if step == 5:
                x[:, ::2] = float("nan")
            dt = torch.tensor([5.0, 7.0])
            a, b = original(x, dt, old), graph(x, dt, new, stamp=(step + 1) * 420)
            self.assertTrue(torch.equal(a.probability, b.probability))
            self.assertTrue(torch.equal(a.answer, b.answer))
            self.assertTrue(state_equal(a.state, b.decision.state))
            self.assertEqual(len(a.rounds), len(b.decision.rounds))
            for left, right in zip(a.rounds, b.decision.rounds):
                for name in ("margins", "challenged", "votes", "acted", "accepted"):
                    self.assertTrue(
                        torch.equal(getattr(left, name), getattr(right, name)), name
                    )
            old, new = a.state.detached(), b.state.detached()

    def test_schema_expansion_preserves_every_native_projection(self):
        original = base()
        graph = Graph(original).eval()
        x = torch.randn(1, 17)
        before = graph(x, torch.tensor([5.0]))
        graph.add_fields(["humidity:new", "sound"])
        values = dict(zip(FIELDS, x[0].tolist()))
        values.update(sound=24.0, **{"humidity:new": 0.7})
        after = graph(graph.tensor(values), torch.tensor([5.0]))
        self.assertTrue(torch.equal(before.probability, after.probability))
        self.assertEqual(graph.fields[-2:], ["humidity:new", "sound"])
        self.assertTrue(
            torch.equal(
                graph.tensor(values), graph.tensor(dict(reversed(list(values.items()))))
            )
        )
        with self.assertRaises(ValueError):
            graph.tensor({"sound": float("inf")})
        with self.assertRaises(ValueError):
            graph.tensor({"parent:0": 1})

    def test_untrained_new_input_cannot_fabricate_evidence(self):
        graph = Graph(base()).eval()
        before = graph(torch.full((1, 17), float("nan")), torch.tensor([5.0]))
        values = {"new:cue": 1.0}
        after = graph(graph.tensor(values), torch.tensor([5.0]))
        self.assertTrue(torch.equal(before.probability, after.probability))
        self.assertTrue(torch.equal(before.answer, after.answer))
        self.assertFalse(bool(after.committed))
        self.assertEqual(graph.unsupported_fields(), ["new:cue"])

    def test_new_field_receives_gradient_and_changes_fitted_weights(self):
        graph = Graph(base())
        graph.add_fields(["sound"])
        decisive(graph.cores["b0"])
        rows = []
        for i in range(24):
            values = dict(zip(FIELDS, torch.randn(17).tolist()))
            values["sound"] = float(2 * (i % 2) - 1)
            rows.append(dict(values=values, stamp=300 * (i + 1), truth=i % 2))
        before = graph.cores["b0"].input.weight[:, 17].clone()
        train_candidate(graph, rows, Settings(window=24, history=24, minority=1))
        self.assertGreater(
            float((graph.cores["b0"].input.weight[:, 17] - before).abs().sum()), 0
        )
        self.assertTrue(torch.isfinite(graph.cores["b0"].input.weight).all())

    def test_peer_birth_identity_quorum_alignment_and_retirement(self):
        graph = Graph(base()).eval()
        x = torch.randn(1, 17)
        state = graph(x, torch.tensor([5.0]), stamp=300).state.detached()
        new = graph.fork()
        new.spawn("b0", 2)
        self.assertEqual(new.size(), 6)
        self.assertEqual(new.cores["b0"].majority, 3)
        aligned = new.align(state, graph).local["b0"]
        self.assertTrue(
            torch.equal(aligned.cells[0][:, :5], state.local["b0"].cells[0])
        )
        self.assertTrue(
            torch.equal(aligned.cells[0][:, 5], state.local["b0"].cells[0][:, 2])
        )
        new.retire("b0", 2)
        self.assertEqual(new.size(), 5)
        self.assertEqual(len(set(new.cores["b0"].ids)), 5)
        while new.size() > 1:
            new.retire("b0", 0)
        with self.assertRaises(ValueError):
            new.retire("b0", 0)

    def test_child_is_full_board_and_parent_rollback_is_atomic(self):
        graph = Graph(base()).eval()
        decisive(graph.cores["b0"])
        child = graph.deepen("b0", 0, FIELDS[:5])["board"]
        decisive(graph.cores[child])
        x = torch.randn(1, 17)
        dt = torch.tensor([5.0])
        result = graph(x, dt, stamp=300)
        self.assertTrue(bool(result.committed))
        self.assertEqual(len(result.trace), 2)
        path = "b0/e0/" + child
        self.assertIn(path, result.state.local)
        self.assertEqual(result.state.local[path].memory.shape, (1, 2, 4, 4))
        self.assertEqual(len(graph.cores[child].falsifiers), 3)
        self.assertEqual(graph.cores[child].round_limit, 4)
        before = result.state.detached()
        with torch.no_grad():
            for checker in graph.cores["b0"].falsifiers:
                checker[-1].bias.fill_(-10)
            graph.cores["b0"].process.bias.fill_(-10)
        failed = graph(x, dt, before, stamp=600)
        self.assertFalse(bool(failed.committed))
        self.assertEqual(len(failed.decision.rounds), 4)
        self.assertTrue(state_equal(failed.state.local[path], before.local[path]))
        self.assertTrue(torch.equal(failed.state.stamps[path], before.stamps[path]))
        self.assertTrue(
            torch.equal(failed.decision.state.memory, before.local["b0"].memory)
        )

    def test_nested_depth_and_resource_limits(self):
        graph = Graph(base(), max_depth=2, max_experts=8).eval()
        a = graph.deepen("b0", 0, FIELDS[:3])["board"]
        b = graph.deepen(a, 0, FIELDS[:2])["board"]
        self.assertEqual(graph.depth(b), 2)
        with self.assertRaises(ValueError):
            graph.deepen(b, 0, FIELDS[:1])
        graph.spawn(a, 0)
        with self.assertRaises(ValueError):
            graph.spawn(a, 0)
        for core in graph.cores.values():
            decisive(core)
            with torch.no_grad():
                core.child_gain.fill_(0.5)
        result = graph(torch.ones(1, 17), torch.tensor([5.0]), stamp=300)
        self.assertEqual(len(result.trace), 4)
        graph.retire("b0", 0)
        self.assertEqual(list(graph.cores), ["b0"])

    def test_identical_children_share_parameters_but_keep_context_state(self):
        graph = Graph(base()).eval()
        decisive(graph.cores["b0"])
        graph.spawn("b0", 0)
        a = graph.deepen("b0", 0, FIELDS[:4])["board"]
        b = graph.deepen("b0", 5, FIELDS[:4])["board"]
        # Same selected peer, copied liquid/readout/checker/policy parameters.
        self.assertIn((a, b), graph.matches())
        result = graph(torch.randn(1, 17), torch.tensor([5.0]), stamp=300)
        old = graph.fork()
        initial = result.state.detached()
        self.assertIn("b0/e5/" + b, initial.local)
        initial.local["b0/e5/" + b].memory.add_(1)
        detail = graph.share_equal()
        self.assertEqual(detail["removed"], b)
        aligned = graph.align(initial, old)
        self.assertIn("b0/e5/" + a, aligned.local)
        self.assertTrue(
            state_equal(initial.local["b0/e5/" + b], aligned.local["b0/e5/" + a])
        )
        self.assertNotEqual(
            aligned.local["b0/e0/" + a].memory.data_ptr(),
            aligned.local["b0/e5/" + a].memory.data_ptr(),
        )
        next_x = torch.randn(1, 17)
        before = old(next_x, torch.tensor([5.0]), initial, stamp=600)
        after = graph(next_x, torch.tensor([5.0]), aligned, stamp=600)
        self.assertTrue(torch.equal(before.probability, after.probability))
        self.assertEqual(len(graph.cores), 2)

    def test_full_subtree_sharing_retains_descendant_contexts(self):
        graph = Graph(base()).eval()
        graph.spawn("b0", 0)
        a = graph.deepen("b0", 0, FIELDS[:3])["board"]
        b = graph.deepen("b0", 5, FIELDS[:3])["board"]
        ca = graph.deepen(a, 0, FIELDS[:2])["board"]
        cb = graph.deepen(b, 0, FIELDS[:2])["board"]
        for core in graph.cores.values():
            decisive(core)
            with torch.no_grad():
                core.child_gain.fill_(0.5)
        result = graph(torch.randn(1, 17), torch.tensor([5.0]), stamp=300)
        old = graph.fork()
        initial = result.state.detached()
        initial.local["b0/e5/" + b + "/e5/" + cb].memory.add_(1)
        self.assertIn((a, b), graph.matches())
        graph.share_equal()
        aligned = graph.align(initial, old)
        self.assertIn("b0/e5/" + a + "/e0/" + ca, aligned.local)
        self.assertTrue(
            state_equal(
                initial.local["b0/e5/" + b + "/e5/" + cb],
                aligned.local["b0/e5/" + a + "/e0/" + ca],
            )
        )
        x = torch.randn(1, 17)
        self.assertTrue(
            torch.equal(
                old(x, torch.tensor([5.0]), initial, stamp=600).probability,
                graph(x, torch.tensor([5.0]), aligned, stamp=600).probability,
            )
        )

    def test_child_elapsed_time_keeps_unix_timestamp_precision(self):
        from unittest.mock import patch

        graph = Graph(base()).eval()
        decisive(graph.cores["b0"])
        child = graph.deepen("b0", 0, FIELDS[:3])["board"]
        decisive(graph.cores[child])
        start = 1_700_000_000.0
        x = torch.ones(1, 17)
        state = graph(x, torch.tensor([5.0]), stamp=start).state.detached()
        self.assertEqual(state.stamps["b0/e0/" + child].dtype, torch.float64)
        with patch.object(
            graph.cores[child], "forward", wraps=graph.cores[child].forward
        ) as called:
            graph(x, torch.tensor([5.0]), state, stamp=start + 300)
        self.assertTrue(
            torch.equal(called.call_args_list[0].args[1], torch.tensor([5.0]))
        )

    def test_different_weights_cannot_share(self):
        graph = Graph(base())
        graph.spawn("b0", 0)
        a = graph.deepen("b0", 0, FIELDS[:3])["board"]
        b = graph.deepen("b0", 5, FIELDS[:3])["board"]
        with torch.no_grad():
            next(graph.cores[b].parameters()).add_(0.01)
            next(graph.cores[b].cells.parameters()).add_(0.01)
        self.assertFalse(graph.matches())
        with self.assertRaises(ValueError):
            graph.share_equal()

    def test_equal_liquid_weights_share_even_when_readouts_differ(self):
        graph = Graph(base()).eval()
        graph.spawn("b0", 0)
        a = graph.deepen("b0", 0, FIELDS[:3])["board"]
        b = graph.deepen("b0", 5, FIELDS[:3])["board"]
        with torch.no_grad():
            graph.cores[b].outputs[-1].bias.add_(0.2)
        self.assertFalse(graph.matches())
        self.assertIn((a, b), graph.kernel_matches())
        before = sum(p.numel() for p in graph.parameters())
        detail = graph.share_equal()
        self.assertEqual(detail["kind"], "liquid_kernel")
        self.assertIs(graph.cores[a].cells, graph.cores[b].cells)
        self.assertLess(sum(p.numel() for p in graph.parameters()), before)
        restored = Graph.restore(graph.pack())
        self.assertIs(restored.cores[a].cells, restored.cores[b].cells)
        graph.spawn(b, 0)
        self.assertIsNot(graph.cores[a].cells, graph.cores[b].cells)
        self.assertEqual(graph.cores[a].experts, 1)
        self.assertEqual(graph.cores[b].experts, 2)

    def test_two_head_delta_forgets_with_elapsed_time(self):
        memory = primitives.DeltaAttention()
        with torch.no_grad():
            memory.write.weight.zero_()
            memory.write.bias.zero_()
            memory.write.bias[-2:].fill_(-100)  # isolate forgetting from new writes
        board = torch.ones(1, 2, 4, 4)
        draft = torch.zeros(1, WIDTH)
        short = memory(board, draft, torch.tensor([5.0]))
        long = memory(board, draft, torch.tensor([50.0]))
        self.assertTrue((long < short).all())
        self.assertEqual(long.shape, (1, 2, 4, 4))


class Learning(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_neural_action_values_learn_external_payoff(self):
        p = Policy(17)
        context = torch.ones(16)
        generator = torch.Generator().manual_seed(9)
        allowed = [True] * len(ACTIONS)
        self.assertEqual(p.choose(context, allowed, generator, 0)[0], 0)
        p.learn_action(context, ACTIONS.index("deepen"), 0.9)
        self.assertEqual(
            p.choose(context, allowed, generator, 0)[0], ACTIONS.index("deepen")
        )
        p.learn_action(context, ACTIONS.index("deepen"), -1.0)
        self.assertNotEqual(
            p.choose(context, allowed, generator, 0)[0], ACTIONS.index("deepen")
        )
        self.assertNotEqual(float(p.action(context)[2]), float(p.action(-context)[2]))

    def test_feature_group_is_learned_from_errors(self):
        p = Policy(17)
        hidden = torch.ones(WIDTH)
        relevance = torch.zeros(17)
        relevance[-5:] = 1
        for _ in range(5):
            p.learn_group(hidden, relevance)
        selected = p.select_group(hidden, FIELDS)
        self.assertTrue(set(FIELDS[-5:]) <= set(selected))
        self.assertNotEqual(selected, list(FIELDS[:9]))

    def test_runtime_neural_policy_drives_peer_and_depth_candidates(self):
        for action in ("spawn", "deepen", "retire"):
            runtime = Runtime(
                Graph(base()),
                settings=Settings(window=24, history=24, minority=1, exploration=0),
                guards=Guards(minimum_life=24),
            )
            for core in runtime.graph.cores.values():
                for p in core.policies:
                    with torch.no_grad():
                        p.action[-1].bias[ACTIONS.index(action)] = 10
            for i in range(24):
                values = dict(zip(FIELDS, torch.randn(17).tolist()))
                values["new:cue"] = float(i % 2)
                issued = runtime.predict(300 * (i + 1), values)
                runtime.observe(issued["token"], i % 2, 300 * (i + 1) + 1)
            self.assertEqual(runtime.trial["action"], action)
            self.assertIsNotNone(runtime.shadow)
            wanted = {"spawn": 6, "deepen": 6, "retire": 4}[action]
            self.assertEqual(runtime.shadow.size(), wanted)
            self.assertEqual(
                runtime.graph.size(), 5
            )  # prospective trial, no premature adoption

    def make_trial(self, action, bad=False):
        runtime = Runtime(
            Graph(base()), settings=Settings(window=48, history=48, minority=16)
        )
        runtime.shadow = runtime.graph.fork()
        if action == "retire":
            runtime.shadow.retire("b0", 0)
        runtime.trial = dict(
            action=action,
            action_index=ACTIONS.index(action),
            context=[0.0] * 16,
            before_parameters=sum(p.numel() for p in runtime.graph.parameters()),
            candidate_parameters=sum(p.numel() for p in runtime.shadow.parameters()),
            fit_revealed_through=1,
        )
        runtime.check = []
        for i in range(48):
            truth = i % 2
            new = 1 - truth if bad else truth
            old = truth if action == "retire" else 1 - truth
            runtime.check.append(
                dict(
                    truth=truth,
                    stamp=300 * (i + 1),
                    revealed_at=300 * (i + 1) + 1,
                    active=0.9 if old else 0.1,
                    candidate=0.9 if new else 0.1,
                    active_answer=old,
                    candidate_answer=new,
                    active_quorum=True,
                    candidate_quorum=True,
                )
            )
        return runtime

    def test_objective_lifecycle_retires_only_without_measured_harm(self):
        good = self.make_trial("retire")
        good.finish()
        self.assertTrue(good.receipts[-1]["accepted"])
        self.assertEqual(good.graph.size(), 4)
        bad = self.make_trial("retire", bad=True)
        bad.finish()
        self.assertFalse(bad.receipts[-1]["accepted"])
        self.assertEqual(bad.graph.size(), 5)
        self.assertLess(bad.receipts[-1]["measured_return"], 0)

    def test_strict_gain_and_insufficient_evidence_gates(self):
        good = self.make_trial("learn")
        good.finish()
        self.assertTrue(good.receipts[-1]["accepted"])
        bad = self.make_trial("learn", bad=True)
        bad.finish()
        self.assertFalse(bad.receipts[-1]["accepted"])
        scarce = self.make_trial("learn")
        scarce.check = scarce.check[:8]
        scarce.finish()
        self.assertEqual(scarce.receipts[-1]["status"], "insufficient_check_classes")

    def test_delayed_labels_order_and_exact_pending_checkpoint_resume(self):
        run = Runtime(
            Graph(base()), settings=Settings(window=24, history=24, minority=1)
        )
        for i in range(25):
            issued = run.predict(
                300 * (i + 1), dict(zip(FIELDS, torch.randn(17).tolist()))
            )
            if i < 24:
                run.observe(issued["token"], i % 2, 300 * (i + 1) + 1)
        self.assertIsNotNone(run.shadow)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.pt"
            run.save(path)
            restored = Runtime.load(path)
        self.assertFalse(restored.shadow.training)
        for item in (run, restored):
            item.observe(issued["token"], 0, 7501)
        values = dict(zip(FIELDS, torch.randn(17).tolist()))
        values["new:sound"] = 1.0
        a = run.predict(7800, values)
        b = restored.predict(7800, values)
        self.assertEqual(a, b)
        self.assertTrue(state_equal(run.state.local["b0"], restored.state.local["b0"]))
        with self.assertRaises(ValueError):
            run.observe(a["token"], 1, 7800)
        run.observe(a["token"], 1, 7801)
        with self.assertRaises(ValueError):
            run.observe(a["token"], 1, 7802)
        with self.assertRaises(ValueError):
            run.predict(7800, values)

    def test_new_input_causes_learned_child_growth_and_prospective_acceptance(self):
        from demonstrate import demonstrate

        runtime, record = demonstrate()
        receipt = record["receipts"][0]
        self.assertEqual(receipt["action"], "deepen")
        self.assertTrue(receipt["accepted"])
        self.assertEqual(runtime.graph.size(), 6)
        self.assertGreater(receipt["balanced_gain"], 0.01)
        self.assertGreater(receipt["brier_gain"], 0.002)
        self.assertGreater(receipt["paired_lower"], 0)
        self.assertLess(receipt["fit_revealed_through"], receipt["check_first"])
        self.assertIn("new:cue", receipt["mutation"]["group"])
        self.assertEqual(len(runtime.graph.edges), 1)

    def test_redundant_experts_retire_from_actual_prospective_predictions(self):
        from demonstrate import demonstrate

        runtime, record = demonstrate(seed=97, windows=193)
        self.assertEqual(runtime.graph.size(), 4)
        first, second = record["receipts"]
        self.assertTrue(first["action"] == "retire" and first["accepted"])
        self.assertGreater(first["parameter_saving"], 0)
        self.assertGreaterEqual(min(first["class_gain"]), 0)
        self.assertEqual(second["action"], "learn")
        self.assertFalse(second["control"]["cooled"])

    def test_missing_and_bounded_inputs(self):
        run = Runtime(
            Graph(base(), max_fields=18), settings=Settings(maximum_pending=1)
        )
        issued = run.predict(300, {})
        self.assertFalse(issued["quorum"])
        with self.assertRaises(ValueError):
            run.predict(600, {})
        run.discard(issued["token"])
        run.predict(600, {"new": 1})
        run.discard(1)
        with self.assertRaises(ValueError):
            run.predict(900, {"another": 1})


class Evolution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)

    def test_growth_preserves_recurrent_function_before_fitting(self):
        for kind in ("spawn", "deepen", "split"):
            graph = Graph(base()).eval()
            decisive(graph.cores["b0"])
            child = graph.deepen("b0", 2, FIELDS[:5])["board"]
            decisive(graph.cores[child])
            with torch.no_grad():
                graph.cores["b0"].child_gain[2] = 0.5
            state = graph(
                torch.randn(1, 17), torch.tensor([5.0]), stamp=300
            ).state.detached()
            candidate = graph.fork()
            if kind == "spawn":
                candidate.spawn("b0", 2)
            elif kind == "deepen":
                candidate.deepen("b0", 1, FIELDS[:4])
            else:
                candidate.split("b0", 2, 3)
            aligned = candidate.align(state, graph)
            for i in range(6):
                x = torch.randn(1, 17)
                a = graph(x, torch.tensor([5.0]), state, stamp=600 + 300 * i)
                b = candidate(x, torch.tensor([5.0]), aligned, stamp=600 + 300 * i)
                self.assertTrue(torch.equal(a.answer, b.answer), kind)
                self.assertTrue(torch.equal(a.committed, b.committed), kind)
                self.assertTrue(
                    torch.allclose(a.probability, b.probability, atol=1e-6, rtol=0),
                    kind,
                )
                self.assertTrue(
                    torch.allclose(
                        a.decision.state.memory,
                        b.decision.state.memory,
                        atol=1e-6,
                        rtol=0,
                    ),
                    kind,
                )
                state, aligned = a.state.detached(), b.state.detached()
            restored = Graph.restore(candidate.pack()).eval()
            self.assertTrue(
                torch.equal(
                    restored(x, torch.tensor([5.0]), aligned, stamp=2400).probability,
                    candidate(x, torch.tensor([5.0]), aligned, stamp=2400).probability,
                )
            )

    def test_controller_hysteresis_cooldown_lifetime_and_budget(self):
        from control import Controller

        graph = Graph(base()).eval()
        controller = Controller(Guards(minimum_life=48))

        def rows(chance):
            return [
                dict(
                    values=dict(zip(FIELDS, [0.0] * 17)),
                    truth=i % 2,
                    probability=chance if i % 2 else 1 - chance,
                )
                for i in range(48)
            ]

        high = controller.assess(graph, rows(0.5), 0)
        self.assertTrue(high["growth_latched"])
        self.assertTrue(high["plateau"])
        key = next(iter(high["signals"]))
        self.assertTrue(controller.allowed(key, 48, 0.5)[0])
        self.assertFalse(controller.allowed(key, 1, 0.01)[1])
        controller.attempted("split")
        controller.assess(graph, rows(0.5), 0)
        self.assertFalse(controller.allowed(key, 48, 0.5)[0])
        controller.assess(graph, rows(0.99), 0)
        self.assertFalse(controller.growth_latched)
        tiny = Controller(Guards(max_parameters=1))
        self.assertFalse(tiny.budget(graph)["allowed"])
        child = graph.fork()
        child.deepen("b0", 0, FIELDS[:3])
        bounded = Controller(Guards(max_context_experts=5))
        self.assertTrue(bounded.budget(graph)["allowed"])
        report = bounded.budget(
            graph,
            sum(p.numel() for p in child.parameters()),
            candidate_graph=child,
        )
        self.assertEqual(report["context_experts"], 6)
        self.assertFalse(report["allowed"])

    def test_schema_waits_for_stage_and_boundary_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Runtime(
                Graph(base()),
                settings=Settings(window=24, history=24, minority=1),
                stage_directory=directory,
            )
            for i in range(24):
                values = dict(zip(FIELDS, [0.2] * 17))
                values["new:cue"] = float(i % 2)
                p = run.predict(300 * (i + 1), values)
                self.assertEqual(len(run.graph.fields), 17)
                run.observe(p["token"], i % 2, 300 * (i + 1) + 1)
            self.assertEqual(len(run.graph.fields), 18)
            restored = Runtime.load(Path(directory) / "boundary.pt")
            self.assertTrue(restored.stage_ready)
            values = dict(zip(FIELDS, [0.2] * 17))
            values["new:cue"] = 1.0
            a = run.predict(7500, values)
            b = restored.predict(7500, values)
            self.assertEqual(a, b)
            self.assertEqual(run.trial, restored.trial)
            for name, value in run.shadow.state_dict().items():
                self.assertTrue(
                    torch.equal(value, restored.shadow.state_dict()[name]), name
                )
            self.assertTrue(run.trial["fit"]["static_parameter_shapes"])
            self.assertLess(
                run.trial["fit"]["old_learning_rate"],
                run.trial["fit"]["new_learning_rate"],
            )

    def test_invalid_magnitude_cannot_poison_runtime(self):
        runtime = Runtime(Graph(base()))
        with self.assertRaises(ValueError):
            runtime.predict(300, {FIELDS[0]: 1e308})
        result = runtime.predict(300, {FIELDS[0]: 0.1})
        self.assertEqual(result["token"], 0)

    def test_actual_neuron_split_learns_and_survives_future_checks(self):
        from demonstrate import demonstrate

        runtime, record = demonstrate(seed=83)
        first = record["receipts"][0]
        self.assertEqual(first["action"], "split")
        self.assertTrue(first["accepted"])
        self.assertEqual(runtime.graph.cores["b0"].encoding, 17)
        self.assertTrue(first["preservation"]["passed"])
        self.assertGreater(first["balanced_gain"], 0.01)


class Continuation(unittest.TestCase):
    def test_shared_equal_boards_keep_memory_when_head_identities_differ(self):
        graph = Graph(base()).eval()
        graph.deepen("b0", 0, FIELDS[:3])
        graph.deepen("b0", 1, FIELDS[:3])
        a, b = graph.cores["b1"], graph.cores["b2"]
        b.load_state_dict(a.state_dict())
        b.delta.identities = [10, 11]
        b.delta.next_identity = 12
        x = torch.zeros(1, b.features)
        native = b.start(x)
        memory = torch.randn_like(native.memory)
        from dataclasses import replace

        before = GraphState(
            local={"b0/e1/b2": replace(native, memory=memory)},
            stamps={"b0/e1/b2": torch.tensor([300.0], dtype=torch.float64)},
        )
        old = graph.fork()
        graph.share_equal()
        aligned = graph.align(before, old)
        self.assertTrue(torch.equal(aligned.local["b0/e1/b1"].memory, memory))

    def test_neural_policy_proposes_attention_growth_and_pruning(self):
        for action in ("grow_heads", "prune_heads"):
            graph = Graph(base()).eval()
            decisive(graph.cores["b0"])
            for policy in graph.cores["b0"].policies:
                with torch.no_grad():
                    policy.action[-1].bias.fill_(-10.0)
                    policy.action[-1].bias[ACTIONS.index(action)] = 10.0
            runtime = Runtime(
                graph,
                settings=Settings(
                    window=24, history=48, minority=4, exploration=0.0, contextual=False
                ),
                guards=Guards(minimum_life=24),
            )
            for i in range(24):
                issued = runtime.predict(300.0 * (i + 1), dict(zip(FIELDS, [0.0] * 17)))
                runtime.observe(issued["token"], i % 2, 300.0 * (i + 1) + 1.0)
            self.assertEqual(runtime.trial["action"], action)
            self.assertEqual(
                runtime.shadow.cores["b0"].delta.count,
                3 if action == "grow_heads" else 1,
            )
            self.assertTrue(runtime.trial["budget"]["allowed"])
            if action == "grow_heads":
                self.assertTrue(runtime.trial["preservation"]["passed"])
            with tempfile.TemporaryDirectory() as directory:
                runtime.save(Path(directory) / "heads.pt")
                restored = Runtime.load(Path(directory) / "heads.pt")
                self.assertEqual(restored.shadow.layout(), runtime.shadow.layout())

    def test_reinforcement_replay_is_bounded_and_restored(self):
        graph = Graph(base(), policy_capacity=2)
        policy = graph.cores["b0"].policies[0]
        for i in range(3):
            policy.learn_action(torch.full((WIDTH + 8,), float(i)), i, 0.1 * i)
        self.assertEqual(int(policy.replay_count), 2)
        self.assertEqual(int(policy.replay_cursor), 1)
        restored = Graph.restore(graph.pack()).cores["b0"].policies[0]
        self.assertTrue(torch.equal(policy.replay_context, restored.replay_context))
        self.assertTrue(torch.equal(policy.replay_reward, restored.replay_reward))

    def test_multiple_new_fields_keep_old_row_update_rate(self):
        from dataclasses import replace

        original = Graph(base()).eval()
        original.add_fields(["cue:a", "cue:b"])
        slow, full = original.fork(), original.fork()
        rows = []
        for i in range(32):
            values = dict(zip(FIELDS, [0.0] * 17))
            values.update({"cue:a": float(i % 2), "cue:b": 1.0})
            rows.append(
                dict(stamp=300.0 * (i + 1), values=values, truth=i % 2 if i < 16 else 1)
            )
        settings = Settings(gradient_window=64)
        torch.manual_seed(17)
        train_candidate(
            slow,
            rows,
            settings,
            ancestor=original,
            guards=Guards(old_gradient_scale=0.1),
        )
        torch.manual_seed(17)
        train_candidate(
            full,
            rows,
            settings,
            ancestor=original,
            guards=Guards(old_gradient_scale=1.0),
        )
        before = original.cores["b0"].input.weight.detach()
        a = slow.cores["b0"].input.weight.detach() - before
        b = full.cores["b0"].input.weight.detach() - before
        # Original feature columns keep the slow rate, while both new fields
        # receive the full update after Adam, including its weight decay.
        self.assertTrue(
            torch.allclose(a[:, :17], 0.1 * b[:, :17], atol=5e-8, rtol=1e-3)
        )
        self.assertTrue(torch.allclose(a[:, 17:19], b[:, 17:19], atol=5e-8))
        self.assertGreater(float(b[:, 17:19].abs().sum()), 0.0)

    def test_attention_split_preserves_sequence_and_prunes_to_minimum(self):
        original = Graph(base()).eval()
        changed = original.fork()
        changed.change_heads("b0", True, 0)
        old_state, new_state = GraphState(), GraphState()
        for i in range(12):
            x = torch.randn(1, 17)
            a = original(x, torch.tensor([5.0]), old_state, stamp=300.0 * (i + 1))
            b = changed(x, torch.tensor([5.0]), new_state, stamp=300.0 * (i + 1))
            self.assertTrue(torch.equal(a.answer, b.answer))
            self.assertTrue(torch.allclose(a.probability, b.probability, atol=1e-6))
            old_state, new_state = a.state, b.state
            self.assertTrue(
                torch.allclose(
                    b.state.local["b0"].memory[:, 2],
                    b.state.local["b0"].memory[:, 0],
                    atol=1e-7,
                )
            )
        restored = Graph.restore(changed.pack()).eval()
        self.assertEqual(restored.cores["b0"].delta.count, 3)
        aligned = changed.align(old_state, original)
        self.assertTrue(
            torch.equal(
                aligned.local["b0"].memory[:, 2], old_state.local["b0"].memory[:, 0]
            )
        )
        changed.change_heads("b0", False, 2)
        changed.change_heads("b0", False, 1)
        self.assertEqual(changed.cores["b0"].delta.count, 1)
        with self.assertRaises(ValueError):
            changed.change_heads("b0", False, 0)
        limited = Graph(base(), max_heads=2)
        with self.assertRaises(ValueError):
            limited.change_heads("b0", True)

    def test_contextual_continuation_matches_native_delayed_feedback(self):
        from contextual import ContextBoard, ContextFeedback

        native = ContextBoard().eval()
        native.load_state_dict(base().state_dict(), strict=False)
        feedback = ContextFeedback(native)
        feedback.minimum, feedback.minority = 24, 4
        runtime = Runtime(
            Graph(base()),
            structural=False,
            settings=Settings(window=24, minority=4, full_weight_learning=False),
        )
        previous, state = None, None
        torch.manual_seed(11)
        for i in range(97):
            stamp = 300.0 * (i + 1)
            if previous is not None:
                oldstamp, x, before, result, token, truth = previous
                feedback.observe(
                    oldstamp, stamp, x, before, result, torch.tensor([truth])
                )
                runtime.observe(token, truth, stamp)
            x = torch.randn(1, 17)
            before = native.start(x) if state is None else state
            with torch.no_grad():
                result = native(x, torch.tensor([5.0]), before)
            state = result.state.detached()
            issued = runtime.predict(stamp + 0.001, dict(zip(FIELDS, x[0].tolist())))
            self.assertEqual(issued["answer"], int(result.answer))
            self.assertAlmostEqual(
                issued["probability"], float(result.probability), places=6
            )
            self.assertTrue(
                torch.equal(runtime.graph.cores["b0"].correction, native.correction)
            )
            previous = stamp, x, before, result, issued["token"], i % 2
        self.assertEqual(len(runtime.contextual.receipts), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.pt"
            runtime.save(path)
            restored = Runtime.load(path)
            self.assertEqual(
                len(restored.contextual.check), len(runtime.contextual.check)
            )
            self.assertEqual(restored.contextual.latest, runtime.contextual.latest)
            for a, b in zip(
                restored.contextual.history[0], runtime.contextual.history[0]
            ):
                self.assertTrue(torch.equal(a.inputs, b.inputs))

    def test_early_stop_restores_best_earlier_validation_weights(self):
        graph = Graph(base()).eval()
        weights = deepcopy(graph.state_dict())
        rows = [
            dict(
                stamp=300.0 * (i + 1), values=dict(zip(FIELDS, [0.0] * 17)), truth=i % 2
            )
            for i in range(40)
        ]
        fit = train_candidate(
            graph,
            rows,
            Settings(fit_passes=5, early_stop_patience=2, early_stop_min_delta=0.9),
        )
        self.assertTrue(fit["stopped_early"])
        self.assertEqual(fit["completed_passes"], 2)
        self.assertEqual(fit["best_epoch"], 0)
        self.assertEqual(fit["validation_rows"], 8)
        for name, value in graph.state_dict().items():
            self.assertTrue(torch.equal(value, weights[name]), name)

    def test_journal_point_in_time_restores_pending_and_rejects_tampering(self):
        from workflow import Journal, recover

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(Graph(base()))
            journal = Journal(runtime, root / "events.jsonl")
            values = dict(zip(FIELDS, [0.0] * 17))
            first = journal.predict(300.0, values)
            journal.snapshot(root / "snapshot.pt")
            journal.observe(first["token"], 1, 301.0)
            second = journal.predict(600.0, values)
            journal.observe(second["token"], 0, 601.0)
            expected = journal.predict(900.0, values)
            recovered, cursor = recover(
                root / "snapshot.pt", root / "events.jsonl", 600.0
            )
            self.assertEqual(cursor, 3)
            self.assertEqual(list(recovered.pending), [second["token"]])
            recovered.observe(second["token"], 0, 601.0)
            self.assertEqual(recovered.predict(900.0, values), expected)
            with self.assertRaises(ValueError):
                recover(root / "snapshot.pt", root / "events.jsonl", 299.0)
            path = root / "events.jsonl"
            path.write_text(path.read_text().replace("300.0", "299.0", 1))
            with self.assertRaises(ValueError):
                recover(root / "snapshot.pt", path, 900.0)

    def test_scratch_and_finetune_write_reusable_snapshot_receipts(self):
        import json
        from workflow import train, load_configuration
        from dataclasses import asdict

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, guards, limits = load_configuration()
            config = dict(
                learning=dict(asdict(settings), fit_passes=2, early_stop_min_delta=0.9),
                guards=asdict(guards),
                limits=limits,
            )
            (root / "config.json").write_text(json.dumps(config))
            rows = [
                dict(
                    stamp=300.0 * (i + 1),
                    values=dict(zip(FIELDS, [0.0] * 17)),
                    truth=i % 2,
                )
                for i in range(30)
            ]
            (root / "data.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            scratch, receipt = train(
                root / "data.jsonl",
                root / "scratch.pt",
                "scratch",
                configuration=root / "config.json",
            )
            self.assertIsNone(receipt["initial_checkpoint_sha256"])
            tuned, receipt = train(
                root / "data.jsonl",
                root / "tuned.pt",
                "finetune",
                root / "scratch.pt",
                root / "config.json",
                until=6000.0,
            )
            self.assertEqual(receipt["rows"], 20)
            self.assertIsNotNone(receipt["initial_checkpoint_sha256"])
            self.assertEqual(
                Runtime.load(root / "tuned.pt").graph.layout(), tuned.graph.layout()
            )


if __name__ == "__main__":
    unittest.main()
