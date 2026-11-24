"""Bounded feedback control at isolated architecture stages, with gradient probes."""

from dataclasses import asdict, dataclass
import math

import torch

from reference import offline_loss


@dataclass(frozen=True)
class Guards:
    max_abs_input: float = 10000.0
    target_brier: float = 0.10
    error_deadband: float = 0.02
    drift_threshold: float = 0.02
    plateau_tolerance: float = 0.002
    minimum_gradient: float = 1e-7
    minimum_life: int = 288
    cooldown_stages: int = 2
    max_parameters: int = 200_000
    max_training_bytes: int = 128 * 1024 * 1024
    probe_rows: int = 96
    probe_batch: int = 8
    max_context_experts: int = 256
    old_gradient_scale: float = 0.1
    warmup_steps: int = 4
    max_batch: int = 8
    replay_states: int = 4096

    def __post_init__(self):
        if not math.isfinite(self.max_abs_input) or self.max_abs_input <= 0:
            raise ValueError("Input magnitude bound must be finite and positive")
        if (
            not 0 < self.target_brier < 1
            or not 0 < self.error_deadband < self.target_brier
        ):
            raise ValueError("Invalid target or hysteresis deadband")
        for value in (
            self.drift_threshold,
            self.plateau_tolerance,
            self.minimum_gradient,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid signal threshold")
        if (
            min(
                self.minimum_life,
                self.cooldown_stages,
                self.max_parameters,
                self.max_training_bytes,
                self.probe_rows,
                self.probe_batch,
                self.max_context_experts,
                self.warmup_steps,
                self.max_batch,
                self.replay_states,
            )
            < 1
        ):
            raise ValueError("Guard counts and budgets must be positive")
        if not 0 < self.old_gradient_scale <= 1:
            raise ValueError("Invalid old-weight gradient scale")


def gradient_probe(graph, rows, limit, batch=8):
    """Accumulate bounded microbatch gradients without changing any weights."""
    graph = graph.fork()
    graph.enable_fields(graph.unsupported_fields())
    samples = rows[-limit:]
    labels = torch.tensor([row["truth"] for row in samples])
    counts = labels.bincount(minlength=2).float()
    if not counts.min():
        return {}
    graph.eval()
    names = list(graph.cores)
    parameters = [graph.cores[name].input.weight for name in names]
    gradients = [None for _ in parameters]
    for start in range(0, len(samples), batch):
        part = samples[start : start + batch]
        x = torch.cat(
            [
                graph.tensor(
                    {k: v for k, v in row["values"].items() if k in graph.fields}
                )
                for row in part
            ]
        )
        truth = labels[start : start + batch]
        result = graph(x, x.new_full((len(x),), 5.0), stamp=300.0)
        loss = torch.stack(
            [
                offline_loss(decision, truth, counts.mean() / counts)
                for _, decision in result.trace.values()
            ]
        ).mean()
        chunk = torch.autograd.grad(loss, parameters, allow_unused=True)
        for i, value in enumerate(chunk):
            if value is not None:
                contribution = value.detach() * (len(part) / len(samples))
                gradients[i] = (
                    contribution
                    if gradients[i] is None
                    else gradients[i] + contribution
                )
        del result, loss, chunk
    signals = {}
    for board, gradient in zip(names, gradients):
        if gradient is None:
            continue
        for index, identity in enumerate(graph.cores[board].ids):
            matrix = gradient[index].detach()
            # A diagnostic of where encoding gradients concentrate. This is not
            # the GradMax new-neuron optimization for a feedforward network.
            _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
            signals[board + "/" + identity] = dict(
                gradient=float(matrix.norm()),
                singular=float(singular[0]),
                neuron=int(right[0].abs().argmax()),
            )
    return signals


class Controller:
    def __init__(self, guards=None):
        self.guards = guards or Guards()
        self.stage = 0
        self.last_mutation = -1000
        self.failures = 0
        self.integral = 0.0
        self.growth_latched = False
        self.last = {}

    def assess(self, graph, rows, new_fields):
        self.stage += 1
        half = max(1, len(rows) // 2)

        def brier(part):
            values = [
                sum(
                    (r["probability"] - r["truth"]) ** 2
                    for r in part
                    if r["truth"] == c
                )
                / sum(r["truth"] == c for r in part)
                for c in (0, 1)
                if any(r["truth"] == c for r in part)
            ]
            return sum(values) / len(values)

        older, recent = brier(rows[:half]), brier(rows[half:])
        error = brier(rows)
        g = self.guards
        self.integral = max(
            0.0, min(1.0, 0.9 * self.integral + 0.1 * (error - g.target_brier))
        )
        drift = recent - older
        plateau = older - recent <= g.plateau_tolerance
        on = g.target_brier + g.error_deadband * (1 + min(self.failures, 4))
        if error < g.target_brier - g.error_deadband:
            self.growth_latched = False
        elif error + self.integral > on and (plateau or drift > g.drift_threshold):
            self.growth_latched = True
        pressure = self.growth_latched or new_fields > 0 or drift > g.drift_threshold
        cooled = self.stage - self.last_mutation >= g.cooldown_stages
        signals = (
            gradient_probe(graph, rows, g.probe_rows, g.probe_batch)
            if pressure and cooled and self.budget(graph)["allowed"]
            else {}
        )
        self.last = dict(
            stage=self.stage,
            brier=error,
            older_brier=older,
            recent_brier=recent,
            drift=drift,
            plateau=plateau,
            integral=self.integral,
            effective_error=error + self.integral,
            growth_latched=self.growth_latched,
            growth_pressure=pressure,
            cooled=cooled,
            signals=signals,
            guards=asdict(g),
        )
        return self.last

    def allowed(self, key, age, activity):
        g = self.guards
        s = self.last
        signal = s.get("signals", {}).get(key, {})
        growth = (
            s.get("growth_pressure", False)
            and s.get("cooled", False)
            and signal.get("gradient", 0.0) >= g.minimum_gradient
        )
        # Pruning is considered only after a measured lifetime and under either
        # low utilization or errors above target. It still needs no-harm checks.
        prune = (
            s.get("cooled", False)
            and age >= g.minimum_life
            and (activity < 0.25 or s.get("brier", 1.0) > g.target_brier)
        )
        return growth, prune, signal.get("neuron", 0)

    def budget(self, graph, candidate_parameters=None, *, candidate_graph=None):
        active = sum(p.numel() for p in graph.parameters())
        candidate = active if candidate_parameters is None else candidate_parameters
        topology = graph if candidate_graph is None else candidate_graph

        profile = topology.profile()
        contexts = profile["context_experts"]
        layers = max(len(c.cells) for c in topology.cores.values())
        heads = profile["heads"]
        # Reservation includes three simultaneous models (active, trial and
        # contextual replay), Adam/gradient/update copies, bounded replay states,
        # and a conservative autograd allowance. OS address-space is separate.
        activations = (
            2
            * contexts
            * (topology.max_encoding + topology.max_liquid * layers + heads * 16)
            * max(12, self.guards.probe_batch)
            * 4
            * 512
        )
        state_bytes = (contexts * (layers * topology.max_liquid + 1) + heads * 16) * 8
        estimate = (
            active * 8
            + candidate * 28
            + topology.profile()["buffer_bytes"] * 3
            + activations
            + state_bytes * self.guards.replay_states
        )
        try:
            topology.validate_topology()
            valid = True
        except ValueError:
            valid = False
        return dict(
            parameters=candidate,
            context_experts=contexts,
            work_units=profile["work_units"],
            estimated_training_bytes=estimate,
            allowed=valid
            and self.guards.probe_batch
            <= min(topology.max_batch, self.guards.max_batch)
            and candidate <= self.guards.max_parameters
            and contexts <= self.guards.max_context_experts
            and estimate <= self.guards.max_training_bytes,
        )

    def attempted(self, action):
        if action in (
            "spawn",
            "deepen",
            "retire",
            "share",
            "split",
            "grow_heads",
            "prune_heads",
            "add_layer",
            "remove_layer",
            "branch_attention",
            "prune_attention",
            "prune_neuron",
            "fork_network",
            "retire_network",
            "split_liquid",
            "prune_liquid",
            "connect",
            "split_edge",
            "merge_edge",
            "retire_edge",
        ):
            self.last_mutation = self.stage

    def outcome(self, action, accepted):
        if action in (
            "spawn",
            "deepen",
            "split",
            "grow_heads",
            "add_layer",
            "branch_attention",
            "fork_network",
            "split_liquid",
            "connect",
            "split_edge",
        ):
            self.failures = 0 if accepted else self.failures + 1

    def pack(self):
        return dict(
            guards=asdict(self.guards),
            stage=self.stage,
            last_mutation=self.last_mutation,
            failures=self.failures,
            integral=self.integral,
            growth_latched=self.growth_latched,
            last=self.last,
        )

    @classmethod
    def restore(cls, payload):
        result = cls(Guards(**payload["guards"]))
        for key, value in payload.items():
            if key != "guards":
                setattr(result, key, value)
        return result
