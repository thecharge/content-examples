"""Causal learning: neural structural decisions, prospective tests, and recovery."""

from collections import Counter, deque
from dataclasses import asdict, dataclass
import math
import tempfile
from pathlib import Path

import torch
from copy import deepcopy

from control import Controller, Guards
from graph import Graph, GraphState
from policy import ACTIONS
from reference import State, WIDTH, by_class, offline_loss, paired_margin


@dataclass(frozen=True)
class Settings:
    window: int = 288
    history: int = 576
    minority: int = 16
    exploration: float = 0.35
    fit_rate: float = 0.002
    fit_passes: int = 1
    complexity_rate: float = 0.002
    maximum_pending: int = 64
    maximum_receipts: int = 128
    contextual: bool = True
    full_weight_learning: bool = True
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.0001
    validation_fraction: float = 0.2
    gradient_window: int = 12
    gradient_clip: float = 1.0
    statistic_retention: float = 0.95
    checkpoint_retention: int = 2
    minimum_accuracy_gain: float = 0.01
    minimum_brier_gain: float = 0.002
    maximum_class_loss: float = 0.0
    maximum_quorum_loss: float = 0.0
    policy_rate: float = 0.02
    policy_updates: int = 8
    group_rate: float = 0.02
    group_updates: int = 12
    group_fraction: float = 0.5
    minimum_policy_reward: float = 0.01
    external_reward_scale: float = 0.1
    compute_rate: float = 0.002

    def __post_init__(self):
        if (
            min(
                self.window,
                self.history,
                self.minority,
                self.fit_passes,
                self.maximum_pending,
                self.maximum_receipts,
            )
            < 1
        ):
            raise ValueError("All learning counts must be positive")
        if self.window < 2 * self.minority or not 0 <= self.exploration <= 1:
            raise ValueError("Check capacity or exploration probability is invalid")
        if (
            not 0 < self.validation_fraction < 0.5
            or not 0 <= self.statistic_retention < 1
        ):
            raise ValueError("Invalid validation fraction or statistic retention")
        if (
            min(
                self.early_stop_patience,
                self.gradient_window,
                self.checkpoint_retention,
            )
            < 1
        ):
            raise ValueError("Training and checkpoint counts must be positive")
        if self.gradient_clip <= 0 or self.early_stop_min_delta < 0:
            raise ValueError("Invalid gradient clip or early-stop improvement")
        if (
            not 0 < self.group_fraction <= 1
            or min(self.policy_updates, self.group_updates) < 1
        ):
            raise ValueError("Invalid policy update or grouping settings")
        if any(
            not math.isfinite(x) or x < 0
            for x in (
                self.minimum_accuracy_gain,
                self.minimum_brier_gain,
                self.maximum_class_loss,
                self.maximum_quorum_loss,
                self.policy_rate,
                self.group_rate,
                self.minimum_policy_reward,
                self.external_reward_scale,
                self.compute_rate,
            )
        ):
            raise ValueError("Invalid quality threshold or policy learning rate")
        if (
            not math.isfinite(self.fit_rate)
            or self.fit_rate <= 0
            or not math.isfinite(self.complexity_rate)
            or self.complexity_rate < 0
        ):
            raise ValueError(
                "Learning rate must be positive and complexity cost nonnegative"
            )


def train_candidate(
    model,
    samples,
    settings,
    *,
    ancestor=None,
    guards=None,
    mutation=None,
    fresh_fields=(),
):
    model.validate_topology()
    if not Controller(guards).budget(model)["allowed"]:
        raise ValueError("Candidate training reservation exceeds its envelope")
    fresh_fields = sorted(set(fresh_fields) | set(model.unsupported_fields()))
    model.enable_fields(fresh_fields)
    split = max(1, int(len(samples) * (1 - settings.validation_fraction)))
    training, validation = samples[:split], samples[split:]
    can_validate = all(
        {r["truth"] for r in part} == {0, 1} for part in (training, validation)
    )
    if not can_validate:
        training, validation = samples, []
    labels = torch.tensor([row["truth"] for row in training])
    counts = labels.bincount(minlength=2).float()
    if not counts.min():
        raise ValueError("Candidate fitting needs both classes")
    weight = counts.mean() / counts
    guards = guards or Guards()
    old = {} if ancestor is None else dict(ancestor.named_parameters())
    groups = {"old": [], "new": []}
    masks = {}
    for name, parameter in model.named_parameters():
        previous = old.get(name)
        if (
            ancestor is not None
            and mutation
            and mutation.get("kind") in ("liquid_layer_insert", "liquid_layer_remove")
        ):
            board = mutation["board"]
            prefix = "cores." + board + "."
            local = name.removeprefix(prefix).split(".")
            if name.startswith(prefix) and local[0] in (
                "cells",
                "mixers",
                "outputs",
                "falsifiers",
            ):
                identity = model.cores[board].layer_ids[int(local[1])]
                if identity in ancestor.cores[board].layer_ids:
                    local[1] = str(ancestor.cores[board].layer_ids.index(identity))
                    previous = old.get(prefix + ".".join(local))
                else:
                    previous = None
        feature_columns = []
        for board, core in model.cores.items():
            prefix = "cores." + board + "."
            local = name.removeprefix(prefix)
            projection = local in (
                "input.weight",
                "specialty.weight",
                "query.weight",
            ) or (local.startswith("falsifiers.") and local.endswith(".0.weight"))
            if name.startswith(prefix) and projection:
                feature_columns = [
                    i
                    for field in fresh_fields
                    if field in core.names
                    for i in (
                        core.names.index(field),
                        core.features + core.names.index(field),
                    )
                ]
        if previous is None:
            groups["new"].append(parameter)
        elif (
            parameter.shape == previous.shape
            and not feature_columns
            and not (
                mutation and mutation.get("parent") and name.endswith("child_gain")
            )
        ):
            groups["old"].append(parameter)
        else:
            groups["new"].append(parameter)
            mask = torch.full_like(parameter, guards.old_gradient_scale)
            if parameter.shape == previous.shape:
                mask.fill_(guards.old_gradient_scale)
                if feature_columns:
                    mask.index_fill_(1, torch.tensor(feature_columns), 1.0)
                if mutation and mutation.get("parent"):
                    parent, expert = mutation["parent"].split("/")
                    if name == "cores." + parent + ".child_gain":
                        mask[model.cores[parent].ids.index(expert)] = 1.0
            else:
                axes = [
                    i
                    for i, (a, b) in enumerate(zip(previous.shape, parameter.shape))
                    if a != b
                ]
                if (
                    len(axes) == 1
                    and parameter.shape[axes[0]] == previous.shape[axes[0]] + 1
                ):
                    axis = axes[0]
                    insertion = previous.shape[axis]
                    if (
                        mutation
                        and mutation.get("kind") == "net2wider_silu"
                        and axis == 1
                        and not name.endswith("input.bias")
                    ):
                        insertion = ancestor.cores[mutation["board"]].encoding
                    indices = torch.arange(previous.shape[axis])
                    indices += indices.ge(insertion)
                    mask.select(axis, insertion).fill_(1.0)
                if mutation and mutation.get("kind") == "liquid_neuron_split":
                    core = model.cores[mutation["board"]]
                    layer = core.layer_ids.index(mutation["layer"])
                    prefix = "cores." + mutation["board"] + f".cells.{layer}."
                    if name.startswith(prefix):
                        mask.select(1, parameter.shape[1] - 1).fill_(1.0)
                        if name.rsplit(".", 1)[-1] in (
                            "strength",
                            "slope",
                            "center",
                            "target",
                        ):
                            mask.select(2, parameter.shape[2] - 1).fill_(1.0)
                if (
                    mutation
                    and mutation.get("kind") == "liquid_layer_insert"
                    and name == "cores." + mutation["board"] + ".layer_gain"
                ):
                    mask.fill_(guards.old_gradient_scale)
                    mask[mutation["index"]] = 1.0
                if mutation and mutation.get("kind") == "attention_split":
                    path = mutation.get("attention", "root").split(":")[1:]
                    prefix = (
                        "cores."
                        + mutation["board"]
                        + ".delta."
                        + "".join("branches." + p + "." for p in path)
                    )
                    if name.startswith(prefix):
                        delta = model.cores[mutation["board"]].delta.node(
                            mutation.get("attention", "root")
                        )
                        count, dimension = delta.count, delta.dimension
                        if name.startswith(prefix + "write."):
                            rows = [
                                block * count * dimension + j
                                for block in range(3)
                                for j in range(
                                    (count - 1) * dimension, count * dimension
                                )
                            ]
                            rows += [3 * count * dimension + count - 1]
                        else:
                            rows = list(
                                range((count - 1) * dimension, count * dimension)
                            )
                        mask.index_fill_(0, torch.tensor(rows), 1.0)
            if feature_columns:
                mask.index_fill_(1, torch.tensor(feature_columns), 1.0)
            masks[name] = mask
    optimizer = torch.optim.AdamW(
        [
            dict(
                params=groups[k],
                lr=settings.fit_rate,
                scale=(guards.old_gradient_scale if k == "old" else 1.0),
            )
            for k in ("old", "new")
            if groups[k]
        ],
        weight_decay=0.005,
    )
    updates = 0
    static = [(name, id(p), tuple(p.shape)) for name, p in model.named_parameters()]

    def validation_error():
        if not validation:
            return None
        model.eval()
        state, previous, rows = GraphState(), None, []
        with torch.no_grad():
            for row in validation:
                if previous is None or row["stamp"] - previous != 300:
                    state = GraphState()
                x = model.tensor(row["values"])
                result = model(x, x.new_tensor([5.0]), state, stamp=row["stamp"])
                state, previous = result.state.detached(), row["stamp"]
                rows.append(
                    dict(
                        row,
                        validation=float(result.probability),
                        validation_answer=int(result.answer),
                    )
                )
        return metrics(rows, "validation")["brier"]

    best_error, best_epoch, stale, completed = validation_error(), 0, 0, 0
    best_weights = deepcopy(model.state_dict()) if validation else None
    validation_history = []
    for epoch in range(settings.fit_passes):
        model.train()
        state, previous, losses = GraphState(), None, []
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(training):
            if previous is None or row["stamp"] - previous != 300:
                state = GraphState()
            x = model.tensor(row["values"])
            result = model(x, x.new_tensor([5.0]), state, stamp=row["stamp"])
            state = result.state
            target = torch.tensor([row["truth"]])
            costs = [
                offline_loss(decision, target, weight)
                for _, decision in result.trace.values()
            ]
            losses.append(torch.stack(costs).mean())
            if len(losses) == settings.gradient_window or index == len(training) - 1:
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Candidate has nonfinite fitting loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip, error_if_nonfinite=True
                )
                if static != [
                    (name, id(p), tuple(p.shape))
                    for name, p in model.named_parameters()
                ]:
                    raise ValueError(
                        "Parameter objects or shapes changed within a fitting stage"
                    )
                updates += 1
                for group in optimizer.param_groups:
                    group["lr"] = (
                        settings.fit_rate
                        * group["scale"]
                        * min(1.0, updates / guards.warmup_steps)
                    )
                before = {
                    name: p.detach().clone()
                    for name, p in model.named_parameters()
                    if name in masks
                }
                optimizer.step()
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in masks:
                            parameter.copy_(
                                before[name] + masks[name] * (parameter - before[name])
                            )
                optimizer.zero_grad(set_to_none=True)
                state, losses = state.detached(), []
            previous = row["stamp"]
        completed += 1
        error = validation_error()
        if error is not None:
            validation_history.append(error)
            if error < best_error - settings.early_stop_min_delta:
                best_error, best_epoch, stale = error, epoch + 1, 0
                best_weights = deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= settings.early_stop_patience:
                    break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return dict(
        optimizer="fresh_AdamW",
        updates=updates,
        warmup_steps=guards.warmup_steps,
        old_learning_rate=settings.fit_rate * guards.old_gradient_scale,
        new_learning_rate=settings.fit_rate,
        static_parameter_shapes=True,
        requested_passes=settings.fit_passes,
        completed_passes=completed,
        best_epoch=best_epoch,
        stopped_early=completed < settings.fit_passes,
        validation_status=(
            "chronological_holdout" if validation else "insufficient_validation_classes"
        ),
        validation_rows=len(validation),
        training_rows=len(training),
        validation_brier=validation_history,
        best_validation_brier=best_error,
    )


def metrics(rows, key):
    actual = torch.tensor([row["truth"] for row in rows])
    probability = torch.tensor([row[key] for row in rows])
    correct = torch.tensor([row[key + "_answer"] for row in rows]).eq(actual)
    return dict(
        accuracy=float(correct.float().mean()),
        balanced_accuracy=float(by_class(correct.float(), actual)),
        brier=float(by_class((probability - actual).square(), actual)),
        recall=[float(correct[actual == c].float().mean()) for c in (0, 1)],
    )


def pack_state(state):
    return dict(
        local={
            key: dict(
                cells=list(value.cells),
                memory=value.memory,
                last=value.last,
                elapsed=value.elapsed,
            )
            for key, value in state.local.items()
        },
        stamps=state.stamps,
    )


def unpack_state(payload):
    return GraphState(
        {
            key: State(
                tuple(value["cells"]), value["memory"], value["last"], value["elapsed"]
            )
            for key, value in payload["local"].items()
        },
        payload["stamps"],
    )


class Runtime:
    def __init__(
        self,
        graph,
        *,
        settings=None,
        seed=73,
        structural=True,
        guards=None,
        stage_directory=None,
    ):
        if any(p.device.type != "cpu" for p in graph.parameters()):
            raise ValueError(
                "This runtime is CPU-only; GPU/distributed wrappers need a separate stage runner"
            )
        self.graph = graph.eval()
        self.controller = Controller(guards)
        if not self.controller.budget(graph)["allowed"]:
            raise ValueError("Initial graph exceeds its resource envelope")
        self.stage_directory = (
            Path(stage_directory)
            if stage_directory is not None
            else Path(tempfile.mkdtemp(prefix="adaptive-board-stages-"))
        )
        self.stage_directory.mkdir(parents=True, exist_ok=True)
        self.queued_fields = []
        self.stage_ready = False
        self.fresh_fields = []
        self.settings = settings or Settings()
        replay_slots = (
            3 * self.settings.history
            + 2 * self.settings.window
            + self.settings.maximum_pending
        )
        if replay_slots > self.controller.guards.replay_states:
            raise ValueError(
                "History, monitoring and pending states exceed replay reservation"
            )
        self.generator = torch.Generator().manual_seed(seed)
        self.seed, self.structural = seed, structural
        self.state, self.shadow_state = GraphState(), GraphState()
        self.shadow, self.trial = None, None
        self.fallback, self.fallback_state, self.monitor = None, GraphState(), []
        self.monitor_action = None
        self.monitor_audit = deque(maxlen=(settings or Settings()).maximum_receipts)
        self.history = deque(maxlen=self.settings.history)
        self.pending, self.check = {}, []
        self.receipts = deque(maxlen=self.settings.maximum_receipts)
        self.stats = {}
        self.last_prediction = self.last_reveal = -math.inf
        self.fresh_stream = True
        self.serial = self.since_trial = self.trial_number = self.new_fields = 0
        from adaptation import create

        self.contextual = (
            create(self.graph, self.settings) if self.settings.contextual else None
        )
        self.contextual_audit = deque(maxlen=self.settings.maximum_receipts)

    def reset_state(self):
        self.fresh_stream = True
        self.state, self.shadow_state = GraphState(), GraphState()
        self.fallback_state = GraphState()

    @torch.no_grad()
    def predict(self, stamp, values):
        if self.stage_ready:
            with torch.enable_grad():
                self.propose()
        if not math.isfinite(stamp) or stamp <= max(
            self.last_prediction, self.last_reveal
        ):
            raise ValueError("Prediction time must follow all previous events")
        if len(self.pending) >= self.settings.maximum_pending:
            raise ValueError("Pending-label budget exhausted")
        for name, value in values.items():
            if not isinstance(name, str) or not name or name.startswith("parent:"):
                raise ValueError("Invalid input name")
            if value is not None and (
                math.isinf(float(value))
                or abs(float(value)) > self.controller.guards.max_abs_input
            ):
                raise ValueError("Input exceeds the finite magnitude guard")
        unknown = sorted(set(values) - set(self.graph.fields) - set(self.queued_fields))
        if (
            len(self.graph.fields) + len(self.queued_fields) + len(unknown)
            > self.graph.max_fields
        ):
            raise ValueError("Input budget exceeded")
        self.queued_fields.extend(unknown)
        # Schema is immutable during both fitting and prospective testing.
        x = self.graph.tensor(
            {k: v for k, v in values.items() if k in self.graph.fields}
        )
        minutes = 5.0 if self.fresh_stream else (stamp - self.last_prediction) / 60.0
        context_before = pack_state(self.state)
        result = self.graph(x, x.new_tensor([minutes]), self.state, stamp=stamp)
        self.state = result.state.detached()
        proposals = {}
        for path, (board, decision) in result.trace.items():
            core = self.graph.cores[board]
            first = decision.rounds[0]
            for index, identity in enumerate(core.ids):
                proposals[board + "/" + identity] = dict(
                    hidden=core.hidden(first.cells)[0, index].detach(),
                    probability=float(first.margins[0, index, -1].sigmoid()),
                    acted=bool(first.acted[0, index]),
                    confidence=float(first.confidence[0, index]),
                    speaks=bool(first.speaks[0, index]),
                )
        candidate = None
        if self.shadow is not None:
            candidate_x = self.shadow.tensor(
                {k: v for k, v in values.items() if k in self.shadow.fields}
            )
            candidate = self.shadow(
                candidate_x,
                candidate_x.new_tensor([minutes]),
                self.shadow_state,
                stamp=stamp,
            )
            self.shadow_state = candidate.state.detached()
        fallback = None
        if self.fallback is not None:
            fx = self.fallback.tensor(
                {k: v for k, v in values.items() if k in self.fallback.fields}
            )
            fallback = self.fallback(
                fx, fx.new_tensor([minutes]), self.fallback_state, stamp=stamp
            )
            self.fallback_state = fallback.state.detached()
        token = self.serial
        self.serial += 1
        row = dict(
            stamp=stamp,
            values=dict(values),
            proposals=proposals,
            probability=float(result.probability),
            answer=int(result.answer),
            quorum=bool(result.committed),
            shadow=None if candidate is None else float(candidate.probability),
            shadow_answer=None if candidate is None else int(candidate.answer),
            shadow_quorum=None if candidate is None else bool(candidate.committed),
            trial=self.trial_number if candidate is not None else None,
            context_before=context_before,
            fallback=None if fallback is None else float(fallback.probability),
            fallback_answer=None if fallback is None else int(fallback.answer),
            fallback_quorum=None if fallback is None else bool(fallback.committed),
        )
        self.pending[token] = row
        self.last_prediction = stamp
        self.fresh_stream = False
        return dict(
            token=token,
            probability=row["probability"],
            answer=row["answer"],
            quorum=row["quorum"],
            experts=self.graph.size(),
            depth=max(self.graph.depth(board) for board in self.graph.cores),
            fields=len(self.graph.fields),
        )

    def discard(self, token):
        if token not in self.pending:
            raise ValueError("Unknown prediction")
        del self.pending[token]

    def reward(self, token, value, revealed_at):
        if (
            token not in self.pending
            or not math.isfinite(value)
            or not -1 <= value <= 1
        ):
            raise ValueError(
                "Reward requires a pending token and a finite value in [-1, 1]"
            )
        row = self.pending[token]
        if (
            "reward" in row
            or not math.isfinite(revealed_at)
            or revealed_at <= row["stamp"]
            or revealed_at < max(self.last_reveal, self.last_prediction)
        ):
            raise ValueError("Reward must be unique, delayed and chronological")
        row["reward"] = float(value)
        row["reward_at"] = revealed_at
        self.last_reveal = revealed_at

    def observe(self, token, truth, revealed_at):
        if token not in self.pending or token != min(self.pending):
            raise ValueError("Reveal each pending answer once and in prediction order")
        row = self.pending[token]
        if truth not in (0, 1) or not math.isfinite(revealed_at):
            raise ValueError(
                "A binary measured answer and finite reveal time are required"
            )
        if revealed_at <= row["stamp"] or revealed_at < max(
            self.last_reveal, self.last_prediction
        ):
            raise ValueError(
                "An answer must arrive after prediction, without moving time backward"
            )
        del self.pending[token]
        self.last_reveal = revealed_at
        row = dict(row, truth=int(truth), revealed_at=revealed_at)
        if self.contextual is not None:
            # Keep the original causal contextual update independent of whether
            # a structural candidate is currently under prospective evaluation.
            from adaptation import ReplayBoard

            self.contextual.model = ReplayBoard(self.graph)
            before = unpack_state(row["context_before"])
            # The issued rounds are reconstructed only for measured expert reward;
            # correctness and context updates still use the external answer.
            with torch.no_grad():
                inputs = self.graph.tensor(
                    {k: v for k, v in row["values"].items() if k in self.graph.fields}
                )
                issued = self.graph(
                    inputs, inputs.new_tensor([5.0]), before, stamp=row["stamp"]
                )
            self.contextual.observe(
                row["stamp"],
                revealed_at,
                inputs,
                before,
                issued.decision,
                torch.tensor([truth]),
            )
            self.contextual.receipts = self.contextual.receipts[
                -self.settings.maximum_receipts :
            ]
            if not self.contextual.check:
                self.contextual_audit.append(deepcopy(self.contextual.receipts[-1]))
            if self.shadow is not None:
                # Both paired learners get the same measured readout update.
                self.shadow.cores["b0"].correction.copy_(
                    self.graph.cores["b0"].correction
                )
        if self.fallback is not None:
            self.fallback.cores["b0"].correction.copy_(
                self.graph.cores["b0"].correction
            )
        retention = self.settings.statistic_retention
        for key, proposal in row["proposals"].items():
            old = self.stats.get(
                key, dict(loss=0.25, credit=0.0, age=0, activity=0.0, confidence=0.5)
            )
            right = int(proposal["probability"] >= 0.5) == truth
            credit = (0.9 if right else -1.1) if proposal["speaks"] else 0.0
            self.stats[key] = dict(
                loss=retention * old["loss"]
                + (1 - retention) * (proposal["probability"] - truth) ** 2,
                credit=retention * old["credit"] + (1 - retention) * credit,
                age=old["age"] + 1,
                activity=retention * old["activity"]
                + (1 - retention) * proposal["acted"],
                confidence=proposal["confidence"],
                hidden=proposal["hidden"],
            )
        self.history.append(row)
        if self.shadow is not None:
            if row["trial"] == self.trial_number:
                self.check.append(
                    dict(
                        stamp=row["stamp"],
                        revealed_at=revealed_at,
                        truth=int(truth),
                        reward=row.get("reward"),
                        active=row["probability"],
                        candidate=row["shadow"],
                        active_answer=row["answer"],
                        candidate_answer=row["shadow_answer"],
                        active_quorum=row["quorum"],
                        candidate_quorum=row["shadow_quorum"],
                    )
                )
            if len(self.check) >= self.settings.window:
                self.finish()
            return
        if self.fallback is not None and row.get("fallback") is not None:
            self.monitor.append(
                dict(
                    stamp=row["stamp"],
                    truth=truth,
                    active=row["fallback"],
                    active_answer=row["fallback_answer"],
                    active_quorum=row["fallback_quorum"],
                    candidate=row["probability"],
                    candidate_answer=row["answer"],
                    candidate_quorum=row["quorum"],
                )
            )
            if len(self.monitor) >= self.settings.window:
                self.finish_monitor()
            return
        self.since_trial += 1
        if not self.settings.full_weight_learning:
            return
        if self.since_trial >= self.settings.window and not self.pending:
            self.propose()

    def context(self, board, index):
        core = self.graph.cores[board]
        key = board + "/" + core.ids[index]
        stats = self.stats.get(key, {})
        hidden = stats.get("hidden", torch.zeros(WIDTH))
        values = [
            stats.get("loss", 0.25),
            stats.get("credit", 0.0),
            min(stats.get("age", 0) / 576.0, 4.0),
            self.new_fields / max(1, len(self.graph.fields)),
            float(key in self.graph.edges),
            self.graph.depth(board) / max(1, self.graph.max_depth),
            stats.get("activity", 0.0),
            stats.get("confidence", 0.5),
        ]
        return torch.cat((hidden, hidden.new_tensor(values)))

    def teach_groups(self, rows):
        for board, core in self.graph.cores.items():
            for index, identity in enumerate(core.ids):
                key = board + "/" + identity
                hidden = self.context(board, index)[:WIDTH]
                relevance = []
                for name in core.names:
                    pairs = [
                        (
                            float(row["values"][name]),
                            row["truth"]
                            - row["proposals"].get(key, {}).get("probability", 0.5),
                        )
                        for row in rows
                        if row["values"].get(name) is not None
                        and math.isfinite(float(row["values"][name]))
                    ]
                    if len(pairs) < 8:
                        relevance.append(0.0)
                        continue
                    x, y = torch.tensor(pairs).T
                    x, y = x - x.mean(), y - y.mean()
                    correlation = (x @ y).abs() / (
                        (x.square().sum() * y.square().sum()).sqrt().clamp_min(1e-8)
                    )
                    relevance.append(float(correlation))
                core.policies[index].learn_group(
                    hidden,
                    torch.tensor(relevance),
                    rate=self.settings.group_rate,
                    updates=self.settings.group_updates,
                )

    def choose(self):
        choices = []
        for board, core in self.graph.cores.items():
            for index, identity in enumerate(core.ids):
                context = self.context(board, index)
                key = board + "/" + identity
                stats = self.stats.get(key, {})
                growth, prune, neuron = self.controller.allowed(
                    key, stats.get("age", 0), stats.get("activity", 0.0)
                )
                # Conservative preflight before allocating any shadow or new parameters.
                ceiling = sum(p.numel() for p in self.graph.parameters()) + max(
                    sum(p.numel() for p in c.parameters())
                    for c in self.graph.cores.values()
                )
                room = self.controller.budget(self.graph, ceiling)["allowed"]
                allowed = [
                    True,
                    growth and room and self.graph.size() < self.graph.max_experts,
                    growth
                    and room
                    and self.graph.depth(board) < self.graph.max_depth
                    and key not in self.graph.edges
                    and self.graph.size() < self.graph.max_experts,
                    prune and (board != "b0" or core.experts > self.graph.min_experts),
                    bool(self.graph.matches() or self.graph.kernel_matches()),
                    growth and room and core.encoding < self.graph.max_encoding,
                    growth
                    and room
                    and any(
                        n.count < self.graph.max_heads for _, n in core.delta.nodes()
                    ),
                    prune
                    and any(
                        n.count > self.graph.min_heads for _, n in core.delta.nodes()
                    ),
                    growth and room and len(core.cells) < self.graph.max_layers,
                    prune and len(core.cells) > self.graph.min_layers,
                    growth
                    and room
                    and any(
                        path.count(":") < self.graph.max_attention_depth
                        for path, _ in core.delta.nodes()
                    ),
                    prune and len(list(core.delta.nodes())) > 1,
                    prune and core.encoding > self.graph.min_encoding,
                    growth
                    and room
                    and key in self.graph.edges
                    and sum(
                        e["child"] == self.graph.edges[key]["child"]
                        for e in self.graph.edges.values()
                    )
                    > 1,
                    prune and key in self.graph.edges,
                    growth
                    and room
                    and any(len(c.units) < self.graph.max_liquid for c in core.cells),
                    prune
                    and any(len(c.units) > self.graph.min_liquid for c in core.cells),
                ]
                allowed = [
                    flag and self.graph.can_reserve(name, board)
                    for name, flag in zip(ACTIONS, allowed)
                ]
                if not self.structural:
                    allowed = [True] + [False] * (len(ACTIONS) - 1)
                with torch.no_grad():
                    utility = (
                        core.policies[index]
                        .action(context)
                        .masked_fill(~torch.tensor(allowed), -torch.inf)
                        .max()
                    )
                choices.append((board, index, context, allowed, utility))
        probabilities = torch.stack([c[-1] for c in choices]).softmax(0)
        chosen = int(torch.multinomial(probabilities, 1, generator=self.generator))
        board, index, context, allowed, _ = choices[chosen]
        action, utility = (
            self.graph.cores[board]
            .policies[index]
            .choose(
                context,
                allowed,
                self.generator,
                self.settings.exploration if self.structural else 0.0,
            )
        )
        return board, index, action, context, utility

    def propose(self):
        rows = list(self.history)
        counts = Counter(row["truth"] for row in rows)
        if min(counts.get(0, 0), counts.get(1, 0)) < self.settings.minority:
            self.since_trial = 0
            self.receipts.append(
                dict(
                    status="waiting_for_fitting_classes",
                    accepted=False,
                    revealed_at=self.last_reveal,
                    classes=dict(counts),
                )
            )
            return
        if self.pending or self.shadow is not None or self.fallback is not None:
            raise ValueError("A mutation stage requires no pending labels or candidate")
        if not self.stage_ready:
            current = sum(p.numel() for p in self.graph.parameters())
            per_field = sum(
                2 * c.experts * (c.encoding + 2 * WIDTH)
                + 2 * WIDTH
                + 12 * WIDTH
                + c.experts * (WIDTH + 1)
                for c in self.graph.cores.values()
            )
            schema_budget = self.controller.budget(
                self.graph, current + 2 * per_field * len(self.queued_fields)
            )
            if not schema_budget["allowed"]:
                self.receipts.append(
                    dict(
                        status="schema_resource_guard",
                        accepted=False,
                        budget=schema_budget,
                    )
                )
                self.since_trial = 0
                return
            self.fresh_fields = list(self.queued_fields)
            self.new_fields += len(self.graph.add_fields(self.queued_fields))
            if self.queued_fields and self.contextual is not None:
                from adaptation import create

                self.contextual = create(self.graph, self.settings)
            self.queued_fields = []
            self.teach_groups(rows)
            self.controller.assess(
                self.graph,
                rows,
                max(self.new_fields, len(self.graph.unsupported_fields())),
            )
            budget = self.controller.budget(self.graph)
            if not budget["allowed"]:
                self.receipts.append(
                    dict(status="resource_guard", accepted=False, budget=budget)
                )
                self.since_trial = 0
                return
            self.stage_ready = True
            # Durable, resumable boundary before a mutation and fresh optimizer.
            self.save(self.stage_directory / "boundary.pt")
            archived = self.stage_directory / f"boundary-{self.serial:09d}.pt"
            self.save(archived)
            for stale in sorted(self.stage_directory.glob("boundary-*.pt"))[
                : -self.settings.checkpoint_retention
            ]:
                stale.unlink()
        self.stage_ready = False
        board, index, action, context, utility = self.choose()
        core = self.graph.cores[board]
        expert = core.ids[index]
        reservation = self.graph.reserve(ACTIONS[action], board)
        reserve_budget = self.controller.budget(self.graph, reservation["parameters"])
        if not reserve_budget["allowed"]:
            self.receipts.append(
                dict(
                    status="allocation_reservation_rejected",
                    accepted=False,
                    action=ACTIONS[action],
                    budget=reserve_budget,
                )
            )
            self.since_trial = 0
            return
        candidate = self.graph.fork()
        detail = {}
        regrouped = {}
        if ACTIONS[action] == "learn":
            for key, edge in candidate.edges.items():
                parent, expert_id = key.split("/")
                owner = candidate.cores[parent]
                i = owner.ids.index(expert_id)
                group = owner.policies[i].select_group(
                    self.context(parent, i)[:WIDTH],
                    owner.names,
                    self.settings.group_fraction,
                )
                edge["group"] = sorted(group)
                regrouped[key] = sorted(group)
            detail["regrouped"] = regrouped
        if ACTIONS[action] == "spawn":
            detail = candidate.spawn(board, index)
        elif ACTIONS[action] == "deepen":
            group = core.policies[index].select_group(
                context[:WIDTH], core.names, self.settings.group_fraction
            )
            paths = [path for path, _ in core.delta.nodes()]
            attention = paths[
                int(torch.randint(len(paths), (), generator=self.generator))
            ]
            detail = candidate.deepen(board, index, group, attention=attention)
        elif ACTIONS[action] == "retire":
            detail = dict(removed=candidate.retire(board, index))
        elif ACTIONS[action] == "share":
            detail = candidate.share_equal()
        elif ACTIONS[action] == "split":
            key = board + "/" + expert
            neuron = self.controller.last["signals"][key]["neuron"]
            detail = candidate.split(board, index, neuron)
        elif ACTIONS[action] in ("grow_heads", "prune_heads"):
            growing = ACTIONS[action] == "grow_heads"
            nodes = [
                (path, node)
                for path, node in candidate.cores[board].delta.nodes()
                if (
                    node.count < candidate.max_heads
                    if growing
                    else node.count > candidate.min_heads
                )
            ]
            attention, delta = nodes[
                int(torch.randint(len(nodes), (), generator=self.generator))
            ]
            mass = (
                delta.readout.detach()
                .reshape(delta.count, delta.dimension, WIDTH)
                .square()
                .sum((1, 2))
            )
            head = int(mass.argmax() if growing else mass.argmin())
            detail = candidate.change_heads(board, growing, head, attention=attention)
        elif ACTIONS[action] == "add_layer":
            location = int(
                torch.randint(len(core.cells) + 1, (), generator=self.generator)
            )
            detail = candidate.insert_layer(board, location)
        elif ACTIONS[action] == "remove_layer":
            detail = candidate.remove_layer(
                board, int(core.layer_gain.detach().abs().argmin())
            )
        elif ACTIONS[action] == "branch_attention":
            paths = [
                path
                for path, _ in core.delta.nodes()
                if path.count(":") < candidate.max_attention_depth
            ]
            path = paths[int(torch.randint(len(paths), (), generator=self.generator))]
            detail = candidate.branch_attention(board, path)
        elif ACTIONS[action] == "prune_attention":
            paths = [path for path, _ in core.delta.nodes() if path != "root"]
            detail = candidate.prune_attention(board, paths[-1])
        elif ACTIONS[action] == "prune_neuron":
            detail = candidate.prune_neuron(
                board, int(core.input.weight.detach().square().sum((0, 1)).argmin())
            )
        elif ACTIONS[action] == "fork_network":
            detail = candidate.fork_network(board, index)
        elif ACTIONS[action] == "retire_network":
            detail = candidate.retire_network(board, index)
        elif ACTIONS[action] in ("split_liquid", "prune_liquid"):
            growing = ACTIONS[action] == "split_liquid"
            layers = [
                i
                for i, c in enumerate(core.cells)
                if (
                    len(c.units) < candidate.max_liquid
                    if growing
                    else len(c.units) > candidate.min_liquid
                )
            ]
            layer = layers[
                int(torch.randint(len(layers), (), generator=self.generator))
            ]
            cell = candidate.cores[board].cells[layer]
            mass = cell.readout.detach().square().sum((0, 2))
            detail = candidate.change_liquid(
                board, layer, growing, int(mass.argmax() if growing else mass.argmin())
            )
        self.controller.attempted(ACTIONS[action])
        candidate_budget = self.controller.budget(
            self.graph,
            sum(p.numel() for p in candidate.parameters()),
            candidate_graph=candidate,
        )
        if not candidate_budget["allowed"]:
            self.controller.outcome(ACTIONS[action], False)
            self.receipts.append(
                dict(
                    status="resource_guard",
                    action=ACTIONS[action],
                    accepted=False,
                    budget=candidate_budget,
                )
            )
            self.since_trial = 0
            return
        preservation = None
        if ACTIONS[action] in (
            "spawn",
            "deepen",
            "split",
            "grow_heads",
            "add_layer",
            "branch_attention",
            "fork_network",
            "split_liquid",
        ):
            preservation = self.verify_preservation(candidate, rows[-12:])
            if not preservation["passed"]:
                self.controller.outcome(ACTIONS[action], False)
                self.receipts.append(
                    dict(
                        status="function_guard",
                        action=ACTIONS[action],
                        accepted=False,
                        preservation=preservation,
                    )
                )
                self.since_trial = 0
                return
        self.trial_number += 1
        torch.manual_seed(self.seed + self.trial_number)
        # Exact sharing must remain exact: it changes storage, not learned predictions.
        fit_report = None
        if ACTIONS[action] != "share":
            fit_report = train_candidate(
                candidate,
                rows,
                self.settings,
                ancestor=self.graph,
                guards=self.controller.guards,
                mutation=detail,
                fresh_fields=self.fresh_fields,
            )
        self.shadow, self.shadow_state = candidate, candidate.align(
            self.state, self.graph
        )
        self.trial = dict(
            number=self.trial_number,
            action=ACTIONS[action],
            board=board,
            expert=expert,
            action_index=action,
            context=context.tolist(),
            neural_utility=utility,
            mutation=detail,
            fit=fit_report,
            preservation=preservation,
            control=self.controller.last.copy(),
            budget=candidate_budget,
            boundary_checkpoint="boundary.pt",
            fit_first=rows[0]["stamp"],
            fit_last=rows[-1]["stamp"],
            fit_revealed_through=max(row["revealed_at"] for row in rows),
            fit_rows=len(rows),
            before_experts=self.graph.size(),
            candidate_experts=candidate.size(),
            before_profile=self.graph.profile(),
            candidate_profile=candidate.profile(),
            before_parameters=sum(p.numel() for p in self.graph.parameters()),
            candidate_parameters=sum(p.numel() for p in candidate.parameters()),
            compute_rate=self.settings.compute_rate,
            acceptance={
                name: getattr(self.settings, name)
                for name in (
                    "minimum_accuracy_gain",
                    "minimum_brier_gain",
                    "maximum_class_loss",
                    "maximum_quorum_loss",
                )
            },
        )
        self.check, self.since_trial, self.new_fields = [], 0, 0

    @torch.no_grad()
    def verify_preservation(self, candidate, rows):
        active = self.state.detached()
        shadow = candidate.align(active, self.graph)
        maximum = 0.0
        passed = True
        for index, row in enumerate(rows):
            stamp = self.last_prediction + 300 * (index + 1)
            x = self.graph.tensor(
                {k: v for k, v in row["values"].items() if k in self.graph.fields}
            )
            a = self.graph(x, x.new_tensor([5.0]), active, stamp=stamp)
            b = candidate(x, x.new_tensor([5.0]), shadow, stamp=stamp)
            maximum = max(maximum, float((a.probability - b.probability).abs().max()))
            passed = (
                passed
                and torch.equal(a.answer, b.answer)
                and torch.equal(a.committed, b.committed)
            )
            active, shadow = a.state.detached(), b.state.detached()
        return dict(
            passed=bool(passed and maximum <= 1e-6),
            frames=len(rows),
            maximum_probability_difference=maximum,
        )

    def finish(self):
        counts = Counter(row["truth"] for row in self.check)
        accepted, gain, reason, active, candidate = (
            False,
            None,
            "insufficient_check_classes",
            None,
            None,
        )
        if min(counts.get(0, 0), counts.get(1, 0)) >= self.settings.minority:
            expected = torch.tensor([r["truth"] for r in self.check])
            old_right = torch.tensor(
                [r["active_answer"] == r["truth"] for r in self.check]
            ).float()
            new_right = torch.tensor(
                [r["candidate_answer"] == r["truth"] for r in self.check]
            ).float()
            active, candidate = metrics(self.check, "active"), metrics(
                self.check, "candidate"
            )
            balanced_gain = float(by_class(new_right - old_right, expected))
            class_gain = [
                float((new_right - old_right)[expected == c].mean()) for c in (0, 1)
            ]
            quorum_gain = sum(
                int(r["candidate_quorum"]) - int(r["active_quorum"]) for r in self.check
            ) / len(self.check)
            lower = paired_margin(new_right - old_right, expected)
            brier_gain = active["brier"] - candidate["brier"]
            saving = (
                self.trial["before_parameters"] - self.trial["candidate_parameters"]
            ) / self.trial["before_parameters"]
            old_work = self.trial.get("before_profile", {}).get("work_units", 1)
            new_work = self.trial.get("candidate_profile", {}).get(
                "work_units", old_work
            )
            work_saving = (old_work - new_work) / old_work
            gain = (
                brier_gain
                + self.settings.complexity_rate * saving
                + self.settings.compute_rate * work_saving
            )
            if self.trial["action"] in (
                "retire",
                "share",
                "prune_heads",
                "remove_layer",
                "prune_attention",
                "prune_neuron",
                "retire_network",
                "prune_liquid",
            ):
                accepted = (
                    max(saving, work_saving) > 0
                    and brier_gain >= 0
                    and min(class_gain) >= 0
                    and quorum_gain >= 0
                )
            else:
                # Keep the original deployment requirements for weight changes and growth.
                accepted = (
                    balanced_gain >= self.settings.minimum_accuracy_gain
                    and brier_gain >= self.settings.minimum_brier_gain
                    and lower is not None
                    and lower > 0
                    and min(class_gain) >= -self.settings.maximum_class_loss
                    and quorum_gain >= -self.settings.maximum_quorum_loss
                    and gain > 0
                )
            reason = "accepted_measured_gain" if accepted else "rejected_measured_gain"
            extra = dict(
                balanced_gain=balanced_gain,
                class_gain=class_gain,
                quorum_gain=quorum_gain,
                paired_lower=lower,
                brier_gain=brier_gain,
                parameter_saving=saving,
                work_saving=work_saving,
            )
        else:
            extra = {}
        # A failed, unsupported or harmful attempt has a negative training return.
        measured_return = (
            max(self.settings.minimum_policy_reward, gain or 0.0)
            if accepted
            else min(
                -self.settings.minimum_policy_reward,
                gain or -self.settings.minimum_policy_reward,
            )
        )
        external = [r["reward"] for r in self.check if r.get("reward") is not None]
        external_mean = sum(external) / len(external) if external else 0.0
        # An explicit environment reward shapes credit, never bypasses quality.
        measured_return += self.settings.external_reward_scale * external_mean
        context = torch.tensor(self.trial["context"])
        for model in (self.graph, self.shadow):
            for core in model.cores.values():
                for policy in core.policies:
                    policy.learn_action(
                        context,
                        self.trial["action_index"],
                        measured_return,
                        rate=self.settings.policy_rate,
                        updates=self.settings.policy_updates,
                    )
        receipt = dict(
            self.trial,
            status=reason,
            accepted=accepted,
            measured_return=measured_return,
            external_reward_mean=external_mean,
            external_reward_count=len(external),
            check_first=self.check[0]["stamp"],
            check_last=self.check[-1]["stamp"],
            classes=dict(counts),
            active=active,
            candidate=candidate,
            penalized_gain=gain,
            check_predictions=self.check.copy(),
            **extra,
        )
        if receipt["fit_revealed_through"] >= receipt["check_first"]:
            raise AssertionError("Fitting answers must precede every check prediction")
        self.controller.outcome(self.trial["action"], accepted)
        if accepted:
            self.fallback, self.fallback_state = self.graph, self.state
            self.monitor = []
            self.monitor_action = dict(
                action=self.trial["action"],
                action_index=self.trial["action_index"],
                context=self.trial["context"],
            )
            self.graph, self.state = self.shadow, self.shadow_state
            if self.contextual is not None:
                from adaptation import create

                # Keep learned correction, restart incompatible replay states.
                self.contextual = create(self.graph, self.settings)
            live = {b + "/" + e for b, c in self.graph.cores.items() for e in c.ids}
            self.stats = {k: v for k, v in self.stats.items() if k in live}
        self.receipts.append(receipt)
        self.shadow, self.trial, self.shadow_state, self.check = (
            None,
            None,
            GraphState(),
            [],
        )

    def finish_monitor(self):
        counts = Counter(r["truth"] for r in self.monitor)
        supported = min(counts.get(0, 0), counts.get(1, 0)) >= self.settings.minority
        old, new = (
            (metrics(self.monitor, "active"), metrics(self.monitor, "candidate"))
            if supported
            else (None, None)
        )
        # Strict finite-block no-harm: unsupported blocks roll back rather than
        # retaining an ungraded architecture indefinitely.
        class_gain = [
            sum(
                int(r["candidate_answer"] == c) - int(r["active_answer"] == c)
                for r in self.monitor
                if r["truth"] == c
            )
            / counts[c]
            for c in (0, 1)
            if counts[c]
        ]
        quorum_gain = sum(
            int(r["candidate_quorum"]) - int(r["active_quorum"]) for r in self.monitor
        ) / len(self.monitor)
        rollback = (
            not supported
            or new["brier"] > old["brier"]
            or min(class_gain) < 0
            or quorum_gain < 0
        )
        self.monitor_audit.append(
            dict(
                rollback=rollback,
                supported=supported,
                active=old,
                candidate=new,
                class_gain=class_gain,
                quorum_gain=quorum_gain,
                predictions=deepcopy(self.monitor),
            )
        )
        if rollback:
            if self.monitor_action is not None:
                self.controller.outcome(self.monitor_action["action"], False)
                for core in self.fallback.cores.values():
                    for policy in core.policies:
                        policy.learn_action(
                            torch.tensor(self.monitor_action["context"]),
                            self.monitor_action["action_index"],
                            -self.settings.minimum_policy_reward,
                            rate=self.settings.policy_rate,
                            updates=self.settings.policy_updates,
                        )
            self.graph, self.state = self.fallback, self.fallback_state
            if self.contextual is not None:
                from adaptation import create

                self.contextual = create(self.graph, self.settings)
            live = {b + "/" + e for b, c in self.graph.cores.items() for e in c.ids}
            self.stats = {k: v for k, v in self.stats.items() if k in live}
        self.fallback, self.fallback_state, self.monitor = None, GraphState(), []
        self.since_trial = 0
        self.monitor_action = None

    def save(self, path):
        from adaptation import pack

        payload = dict(
            schema="liquid-adaptable-net-v1",
            graph=self.graph.pack(),
            state=pack_state(self.state),
            shadow=None if self.shadow is None else self.shadow.pack(),
            shadow_state=pack_state(self.shadow_state),
            fallback=None if self.fallback is None else self.fallback.pack(),
            fallback_state=pack_state(self.fallback_state),
            monitor=self.monitor,
            monitor_action=self.monitor_action,
            monitor_audit=list(self.monitor_audit),
            settings=asdict(self.settings),
            seed=self.seed,
            structural=self.structural,
            controller=self.controller.pack(),
            stage_directory=None,
            queued_fields=self.queued_fields,
            stage_ready=self.stage_ready,
            fresh_fields=self.fresh_fields,
            generator=self.generator.get_state(),
            rng=torch.get_rng_state(),
            history=list(self.history),
            pending=self.pending,
            check=self.check,
            receipts=list(self.receipts),
            stats=self.stats,
            trial=self.trial,
            last_prediction=self.last_prediction,
            last_reveal=self.last_reveal,
            serial=self.serial,
            since_trial=self.since_trial,
            trial_number=self.trial_number,
            new_fields=self.new_fields,
            fresh_stream=self.fresh_stream,
            contextual=None if self.contextual is None else pack(self.contextual),
            contextual_audit=list(self.contextual_audit),
        )
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, stage_directory=None):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if payload.pop("schema") != "liquid-adaptable-net-v1":
            raise ValueError("Unknown runtime schema")
        payload.pop("stage_directory")
        result = cls(
            Graph.restore(payload.pop("graph")),
            settings=Settings(**payload.pop("settings")),
            seed=payload.pop("seed"),
            structural=payload.pop("structural"),
            stage_directory=stage_directory,
        )
        result.controller = Controller.restore(payload.pop("controller"))
        result.state = unpack_state(payload.pop("state"))
        result.shadow_state = unpack_state(payload.pop("shadow_state"))
        shadow = payload.pop("shadow")
        result.shadow = None if shadow is None else Graph.restore(shadow).eval()
        fallback = payload.pop("fallback")
        result.fallback = None if fallback is None else Graph.restore(fallback).eval()
        result.fallback_state = unpack_state(payload.pop("fallback_state"))
        result.monitor_audit = deque(
            payload.pop("monitor_audit"), maxlen=result.settings.maximum_receipts
        )
        from adaptation import restore

        contextual = payload.pop("contextual")
        result.contextual = (
            None
            if contextual is None
            else restore(result.graph, result.settings, contextual)
        )
        result.generator.set_state(payload.pop("generator"))
        torch.set_rng_state(payload.pop("rng"))
        result.history = deque(payload.pop("history"), maxlen=result.settings.history)
        result.receipts = deque(
            payload.pop("receipts"), maxlen=result.settings.maximum_receipts
        )
        result.contextual_audit = deque(
            payload.pop("contextual_audit"), maxlen=result.settings.maximum_receipts
        )
        for key, value in payload.items():
            setattr(result, key, value)
        return result
