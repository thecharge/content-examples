"""The original board, extended in place with schema growth and child boards."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import weakref

import torch
from torch.nn import functional as F
from torch import nn

from policy import Policy
from hierarchy import DeltaTree
from liquid import LiquidNeurons
from reference import Board, Decision, ExpertLinear, FIELDS, LiquidODE, State, WIDTH


def merge_state(mask, new, old):
    return State(
        tuple(
            torch.where(mask[:, None, None], n, o) for n, o in zip(new.cells, old.cells)
        ),
        torch.where(mask[:, None, None, None], new.memory, old.memory),
        torch.where(mask, new.last, old.last),
        torch.where(mask[:, None], new.elapsed, old.elapsed),
    )


@dataclass
class GraphState:
    local: dict = field(default_factory=dict)
    stamps: dict = field(default_factory=dict)

    def detached(self):
        return GraphState(
            {k: v.detached() for k, v in self.local.items()}, self.stamps.copy()
        )


@dataclass
class Result:
    decision: Decision
    state: GraphState
    trace: dict

    @property
    def answer(self):
        return self.decision.answer

    @property
    def probability(self):
        return self.decision.probability

    @property
    def committed(self):
        return self.decision.committed


class Core(Board):
    """Native voting/checking/retries; children amend individual expert margins."""

    def __init__(self, names, experts=5, rounds=4, depth=3, policy_capacity=64):
        super().__init__(
            features=len(names), experts=max(5, experts), rounds=rounds, depth=depth
        )
        self.cells = nn.ModuleList(LiquidNeurons(cell) for cell in self.cells)
        self.names = list(names)
        self.ids = [f"e{i}" for i in range(self.experts)]
        self.next_expert = self.experts
        self.child_gain = nn.Parameter(torch.zeros(self.experts))
        self.register_buffer("vote_mass", torch.ones(self.experts))
        self.register_buffer(
            "input_support", torch.ones(self.features, dtype=torch.bool)
        )
        self.encoding = 2 * WIDTH
        self.delta = DeltaTree()
        self.register_buffer("correction", torch.zeros(2 * self.features + 1))
        self.births = {}
        self.policies = nn.ModuleList(
            Policy(self.features, policy_capacity) for _ in range(self.experts)
        )
        self.layer_ids = [f"l{i}" for i in range(len(self.cells))]
        self.next_layer = len(self.cells)
        self.layer_gain = nn.Parameter(torch.ones(len(self.cells)))
        self._owner = None
        self._call = None
        if experts != self.experts:
            self.reindex(list(range(experts)))

    def bind(self, graph, identity):
        self._owner = weakref.ref(graph)
        self.identity = identity

    @staticmethod
    def context(rows):
        valid = rows.isfinite()
        return torch.cat(
            (
                torch.where(valid, rows, 0.0).clamp(-3.0, 3.0),
                valid.to(rows.dtype),
                rows.new_ones(len(rows), 1),
            ),
            dim=1,
        )

    def start(self, x):
        native = super().start(x)
        return replace(
            native,
            cells=tuple(
                x.new_zeros(len(x), self.experts, len(c.units)) for c in self.cells
            ),
            memory=x.new_zeros(
                len(x),
                self.delta.total_heads,
                self.delta.dimension,
                self.delta.dimension,
            ),
        )

    def expand(self, names):
        if names == self.names:
            return
        if not set(self.names) <= set(names):
            raise ValueError("Schema changes cannot discard existing fields")
        old_count, new_count = self.features, len(names)
        supported = torch.tensor(
            [
                name.startswith("parent:")
                or (
                    name in self.names
                    and bool(self.input_support[self.names.index(name)])
                )
                for name in names
            ],
            dtype=torch.bool,
        )
        locations = [names.index(name) for name in self.names]
        correction = self.correction.new_zeros(2 * new_count + 1)
        correction[locations] = self.correction[:old_count]
        correction[[new_count + i for i in locations]] = self.correction[
            old_count : 2 * old_count
        ]
        correction[-1] = self.correction[-1]
        self.correction = correction

        def expand_parameter(module, attribute, axis):
            old = getattr(module, attribute)
            shape = list(old.shape)
            shape[axis] += 2 * (new_count - old_count)
            new = old.new_zeros(shape)
            mapping = locations + [new_count + i for i in locations]
            mapping += list(range(2 * new_count, shape[axis]))
            new.index_copy_(
                axis, torch.tensor(mapping, device=old.device), old.detach()
            )
            setattr(module, attribute, nn.Parameter(new))

        for module in (self.input, self.specialty):
            expand_parameter(module, "weight", 1)
        expand_parameter(self.query, "weight", 1)
        self.query.in_features = 2 * new_count
        for checker in self.falsifiers:
            expand_parameter(checker[0], "weight", 1)
            checker[0].in_features = 2 * new_count
        for policy in self.policies:
            policy.extend(self.names, names)
        self.names, self.features = list(names), new_count
        self.input_support = supported

    def reindex(self, indices):
        if not indices or min(indices) < 0 or max(indices) >= self.experts:
            raise ValueError("A board must retain at least one expert")
        for module in self.modules():
            if isinstance(module, (ExpertLinear, LiquidODE)):
                for name, param in list(module.named_parameters(recurse=False)):
                    setattr(module, name, nn.Parameter(param.detach()[indices].clone()))
        self.child_gain = nn.Parameter(self.child_gain.detach()[indices].clone())
        self.vote_mass = self.vote_mass[indices].clone()
        self.policies = nn.ModuleList(deepcopy(self.policies[i]) for i in indices)
        old_ids = self.ids
        ids, used = [], set()
        for i in indices:
            identity = old_ids[i]
            if identity in used:
                identity = f"e{self.next_expert}"
                self.next_expert += 1
            ids.append(identity)
            used.add(identity)
        self.ids = ids
        self.births = {k: v for k, v in self.births.items() if k in ids}
        self.experts, self.majority = (
            len(indices),
            math.floor(float(self.vote_mass.sum()) / 2) + 1,
        )

    def stages(self, x, minutes, before, feedback, active, *, assess=False):
        x = torch.where(self.input_support[None, :], x, float("nan"))
        values = self.liquid_stages(x, minutes, before, feedback, active, assess=assess)
        cells, margins, *rest = values
        correction = (self.context(x) @ self.correction).clamp(-0.8, 0.8)
        margins = torch.cat(
            (margins[..., :-1], margins[..., -1:] + correction[:, None, None]), dim=-1
        )
        values = cells, margins, *rest
        if self._call is None:
            return values
        cells, margins, *rest = values
        graph = self._owner()
        path, stamp, initial = self._call
        acted = rest[-2]
        additions = []
        for index, identity in enumerate(self.ids):
            edge = graph.edges.get(self.identity + "/" + identity)
            if edge is None or not acted[:, index].any():
                continue
            child_id, group = edge["child"], edge["group"]
            child = graph.cores[child_id]
            source = {name: x[:, i] for i, name in enumerate(self.names)}
            available = (
                torch.stack([source[name].isfinite() for name in group]).any(0)
                & acted[:, index]
            )
            packed_query = torch.cat((torch.nan_to_num(x), x.isfinite().to(x)), -1)
            routing = self.delta.route(
                edge.get("attention", "root"), before.memory, self.query(packed_query)
            )
            if not self.training:
                available = available & routing.ge(graph.route_floor)
                if not available.any():
                    continue
            packed = []
            for name in child.names:
                if name.startswith("parent:"):
                    column = self.hidden(cells)[:, index, int(name.split(":")[1])]
                else:
                    column = (
                        source[name]
                        if name in group
                        else x.new_full((len(x),), float("nan"))
                    )
                packed.append(torch.where(available, column, float("nan")))
            child_x = torch.stack(packed, -1)
            child_path = path + "/" + identity + "/" + child_id
            child_result = graph._run(
                child_id, child_x, minutes, initial, stamp, child_path
            )
            adjustment = torch.logit(child_result.probability.clamp(0.01, 0.99))
            adjustment = (
                adjustment
                * routing
                * self.child_gain[index].tanh()
                * child_result.committed
                * available
            )
            # Extend the expert's last liquid layer; retain native vote and checker logic.
            update = torch.zeros_like(margins)
            update[:, index, -1] = adjustment
            margins = margins + update
            additions.append((index, child_result))
        self._round_children.append(additions)
        return cells, margins, *rest

    def liquid_stages(self, x, minutes, before, feedback, active, *, assess=False):
        mask = x.isfinite()
        values = torch.where(mask, x, 0.0)
        packed = torch.cat((values, mask.to(x.dtype)), -1)
        recall = self.delta.read(before.memory, self.query(packed))
        full_input = torch.cat((packed, recall), -1)[:, None].expand(
            -1, self.experts, -1
        )
        context = torch.cat((full_input, feedback), -1)
        local = self.specialty(torch.cat((context, self.hidden(before.cells)), -1))
        interest = self.subscription(F.silu(local)).squeeze(-1)
        if assess:
            acts = torch.ones_like(interest, dtype=torch.bool)
        elif self.training:
            # Each expert explores independently. A row can attract zero, some or all.
            acts = torch.rand_like(interest).lt(
                0.12 + 0.76 * interest.detach().sigmoid()
            )
        else:
            acts = interest.ge(0)
        available = mask.any(-1) & active
        acts = acts & available[:, None]
        rows, experts = acts.nonzero(as_tuple=True)
        encoded = F.silu(self.input(context[rows, experts], experts))
        previous = x.new_zeros(len(rows), WIDTH)
        times = minutes[rows] + before.elapsed[rows, experts]
        cells, margins, falsifiers = [], [], []
        running = x.new_zeros(len(rows))
        dense = x.new_zeros(len(x), self.experts)
        for index, cell in enumerate(self.cells):
            incoming = F.silu(
                self.mixers[index](torch.cat((encoded, previous), -1), experts)
            )
            updated = cell(incoming, before.cells[index][rows, experts], times, experts)
            gain = self.layer_gain[index]
            previous = gain * cell.decode(updated, experts) + (1 - gain) * previous
            features = torch.cat((encoded, previous), -1)
            # The next liquid layer adjusts the last prediction.
            running = running + gain * self.outputs[index](features, experts).squeeze(
                -1
            )
            margins.append(dense.index_put((rows, experts), running))
            cells.append(before.cells[index].index_put((rows, experts), updated))
            falsifiers.append(gain * self.falsifiers[index](packed).squeeze(-1))
        process = dense.index_put(
            (rows, experts), self.process(features, experts).squeeze(-1)
        )
        drive = dense.index_put(
            (rows, experts), self.drive(features, experts).squeeze(-1)
        )
        return (
            tuple(cells),
            torch.stack(margins, -1),
            torch.stack(falsifiers, -1),
            process,
            drive,
            available,
            mask.float().mean(-1),
            acts,
            interest,
        )

    def split_neuron(self, expert, neuron):
        """Net2Wider on a SiLU encoding unit, leaving liquid state width intact."""
        if not 0 <= neuron < self.encoding:
            raise ValueError("Invalid encoding neuron")
        width = self.encoding
        with torch.no_grad():
            weight, bias = self.input.weight, self.input.bias
            new_weight = weight.new_zeros(self.experts, weight.shape[1], width + 1)
            new_bias = bias.new_zeros(self.experts, width + 1)
            new_weight[..., :width].copy_(weight)
            new_bias[..., :width].copy_(bias)
            new_weight[expert, :, width].copy_(weight[expert, :, neuron])
            new_bias[expert, width].copy_(bias[expert, neuron])
            self.input.weight, self.input.bias = nn.Parameter(new_weight), nn.Parameter(
                new_bias
            )
            for module in (*self.mixers, *self.outputs, self.process, self.drive):
                old = module.weight
                new = old.new_zeros(self.experts, old.shape[1] + 1, old.shape[2])
                new[:, :width].copy_(old[:, :width])
                new[:, width + 1 :].copy_(old[:, width:])
                new[expert, neuron].mul_(0.5)
                new[expert, width].copy_(old[expert, neuron] * 0.5)
                module.weight = nn.Parameter(new)
        self.encoding += 1

    def round(self, x, minutes, before, feedback, active, checks, *, assess=False):
        result = super().round(
            x, minutes, before, feedback, active, checks, assess=assess
        )
        if torch.equal(self.vote_mass, torch.ones_like(self.vote_mass)):
            return result
        votes = result.votes * self.vote_mass[None, :, None]
        count = votes.sum(1)
        strength = (votes * (result.confidence - 0.5).unsqueeze(-1)).sum(1)
        choice = (count + strength / self.vote_mass.sum()).argmax(-1)
        tied = count[:, 0].eq(count[:, 1]) & strength[:, 0].eq(strength[:, 1])
        remaining = result.acted & ~result.challenged
        final = result.margins[..., -1] - self.vote_offset
        fallback = torch.where(
            remaining.any(-1),
            (final * remaining * self.vote_mass).sum(-1).ge(0).long(),
            before.last,
        )
        return replace(
            result,
            choice=torch.where(tied, fallback, choice),
            accepted=count.max(-1).values.gt(self.vote_mass.sum() / 2),
        )

    def decode(self, index, state):
        return self.cells[index].decode(state)

    def hidden(self, cells):
        result = cells[0].new_zeros(*cells[0].shape[:2], WIDTH)
        for gain, cell, state in zip(self.layer_gain, self.cells, cells, strict=True):
            result = gain * cell.decode(state) + (1 - gain) * result
        return result

    def forward(self, *args, **kw):
        self._round_children = []
        if (
            not self.training
            and bool(self.layer_gain.eq(1).all())
            and torch.equal(self.vote_mass, torch.ones_like(self.vote_mass))
            and all(
                len(c.units) == WIDTH
                and torch.equal(c.readout, torch.eye(WIDTH).expand_as(c.readout))
                for c in self.cells
            )
        ):
            return super().forward(*args, **kw)
        return self.weighted_forward(*args, **kw)

    def weighted_forward(self, x, minutes, before=None, *, checks=True):
        before = self.start(x) if before is None else before
        active = torch.ones(len(x), device=x.device, dtype=torch.bool)
        feedback = x.new_zeros(len(x), self.experts, 3)
        answer, last = before.last, before.last
        probability = x.new_full((len(x),), 0.5)
        cells, memory = before.cells, before.memory
        elapsed = before.elapsed + minutes[:, None]
        rounds = []
        for number in range(self.round_limit):
            result = self.round(x, minutes, before, feedback, active, checks)
            rounds.append(result)
            answer = torch.where(active, result.choice, answer)
            accepted_votes = result.votes.any(-1)
            usable = torch.where(
                accepted_votes.any(1, keepdim=True),
                accepted_votes,
                result.acted & ~result.challenged,
            )
            margins = (
                result.margins[..., -1] - self.vote_offset
            ).abs() * self.probability_scale
            strength = (margins.sigmoid() - 0.5) * usable
            agrees = result.choices.eq(result.choice[:, None])
            signed = torch.where(agrees, strength, -strength)
            confidence = (
                0.5
                + (signed * self.vote_mass).sum(1).clamp_min(0.0) / self.vote_mass.sum()
            )
            current = torch.where(result.choice.eq(1), confidence, 1 - confidence)
            probability = torch.where(active, current, probability)
            accepts = active & result.accepted
            support = result.votes.gather(
                2, result.choice[:, None, None].expand(-1, self.experts, 1)
            ).squeeze(-1)
            last = torch.where(accepts, result.choice, last)
            changed = accepts[:, None] & result.acted & ~result.challenged
            cells = tuple(
                torch.where(changed.unsqueeze(-1), trial, old)
                for trial, old in zip(result.cells, cells, strict=True)
            )
            elapsed = torch.where(changed, 0.0, elapsed)
            weighted_support = support * self.vote_mass
            summary = (self.hidden(result.cells) * weighted_support.unsqueeze(-1)).sum(
                1
            ) / weighted_support.sum(1, keepdim=True).clamp_min(1e-8)
            memory = torch.where(
                accepts[:, None, None, None],
                self.delta(before.memory, summary, minutes),
                memory,
            )
            active = active & ~accepts
            if not active.any():
                break
            direction = (2 * result.choice - 1).to(x.dtype)
            feedback = torch.stack(
                (
                    result.challenged.to(x.dtype) - 0.5 * result.acted.to(x.dtype),
                    direction[:, None].expand_as(result.confidence),
                    torch.full_like(result.confidence, (number + 1) / self.round_limit),
                ),
                -1,
            )
        return Decision(
            State(cells, memory, last, elapsed),
            answer,
            tuple(rounds),
            ~active,
            probability,
        )


class Graph(nn.Module):
    def __init__(
        self,
        base=None,
        names=FIELDS,
        *,
        max_experts=32,
        max_depth=3,
        max_fields=64,
        max_encoding=32,
        min_heads=1,
        max_heads=8,
        policy_capacity=64,
        min_layers=3,
        max_layers=8,
        min_encoding=8,
        min_experts=1,
        max_boards=16,
        max_attention_nodes=16,
        max_attention_depth=3,
        max_total_heads=64,
        max_work=1000000,
        max_batch=8,
        route_floor=0.05,
        min_liquid=4,
        max_liquid=16,
    ):
        super().__init__()
        if policy_capacity < 1:
            raise ValueError("Policy replay capacity must be positive")
        root = Core(names, policy_capacity=policy_capacity)
        if base is not None:
            missing, extra = root.load_state_dict(base.state_dict(), strict=False)
            if extra or any(
                not (
                    k
                    in (
                        "child_gain",
                        "vote_mass",
                        "input_support",
                        "correction",
                        "delta.readout",
                        "layer_gain",
                    )
                    or k.startswith("policies.")
                    or (k.startswith("cells.") and k.endswith(".readout"))
                )
                for k in missing
            ):
                raise ValueError(
                    "The base checkpoint does not match the reference architecture"
                )
        self.cores = nn.ModuleDict({"b0": root})
        self.edges = {}
        self.kernel_ties = {}
        self.fields = list(names)
        self.next_board = 1
        self.max_experts, self.max_depth, self.max_fields = (
            max_experts,
            max_depth,
            max_fields,
        )
        self.max_encoding = max_encoding
        self.min_liquid, self.max_liquid = min_liquid, max_liquid
        if not 1 <= min_liquid <= WIDTH <= max_liquid:
            raise ValueError(
                "Liquid width limits must contain the initial eight neurons"
            )
        self.route_floor = route_floor
        if not 0 <= route_floor < 1:
            raise ValueError("Routing floor must be in [0, 1)")
        if not 1 <= min_heads <= 2 <= max_heads:
            raise ValueError("Head bounds must contain the two-head reference")
        self.min_heads, self.max_heads = min_heads, max_heads
        self.policy_capacity = policy_capacity
        self.min_layers, self.max_layers = min_layers, max_layers
        self.min_encoding, self.min_experts = min_encoding, min_experts
        self.max_boards = max_boards
        self.max_attention_nodes, self.max_attention_depth = (
            max_attention_nodes,
            max_attention_depth,
        )
        self.max_total_heads, self.max_work, self.max_batch = (
            max_total_heads,
            max_work,
            max_batch,
        )
        if (
            not 3 <= min_layers <= 3 <= max_layers
            or not 1 <= min_encoding <= 16 <= max_encoding
            or not 1 <= min_experts <= 5 <= max_experts
        ):
            raise ValueError("Bounds must contain the initial network")
        if (
            min(max_boards, max_attention_nodes, max_total_heads, max_work, max_batch)
            < 1
            or max_attention_depth < 0
        ):
            raise ValueError("Global graph limits must be positive")
        self.bind()
        self.validate_topology()

    def bind(self):
        for name, board in self.cores.items():
            board.bind(self, name)
            board._call = None
            board._round_children = []

    def fork(self):
        result = deepcopy(self)
        result.bind()
        return result

    def add_fields(self, names):
        names = list(names)
        if any(
            not isinstance(n, str) or not n or n.startswith("parent:") for n in names
        ):
            raise ValueError(
                "Input names must be nonempty strings outside the parent namespace"
            )
        new = sorted(set(names) - set(self.fields))
        if len(self.fields) + len(new) > self.max_fields:
            raise ValueError("Input budget exceeded")
        if new:
            # Each context can add work at every retry, including nested calls.
            def added_work(board):
                core = self.cores[board]
                return core.round_limit * (
                    2 * len(new) * core.experts * core.encoding
                    + sum(
                        added_work(e["child"])
                        for k, e in self.edges.items()
                        if k.split("/")[0] == board
                    )
                )

            if self.profile()["work_units"] + added_work("b0") > self.max_work:
                raise ValueError("Schema expansion exceeds the work envelope")
            self.fields.extend(new)
            for name, core in self.cores.items():
                core.expand(
                    self.fields
                    + ([] if name == "b0" else [f"parent:{i}" for i in range(WIDTH)])
                )
        return new

    def unsupported_fields(self):
        core = self.cores["b0"]
        return [
            name for i, name in enumerate(core.names) if not bool(core.input_support[i])
        ]

    def enable_fields(self, names):
        for core in self.cores.values():
            for name in names:
                if name in core.names:
                    core.input_support[core.names.index(name)] = True

    def tensor(self, values):
        self.add_fields(values)
        row = []
        for name in self.fields:
            value = values.get(name)
            value = float("nan") if value is None else float(value)
            if math.isinf(value):
                raise ValueError("Infinite inputs are invalid")
            row.append(value)
        return torch.tensor([row], dtype=next(self.parameters()).dtype)

    def forward(self, x, minutes, state=None, *, stamp=None):
        self.validate_topology()
        if len(x) > self.max_batch:
            raise ValueError("Batch ceiling exceeded; split larger batches")
        if (
            x.shape[1] != len(self.fields)
            or not torch.isfinite(minutes).all()
            or not (minutes > 0).all()
        ):
            raise ValueError(
                "Input shape and positive finite elapsed minutes are required"
            )
        if torch.isinf(x).any():
            raise ValueError("Infinite inputs are invalid")
        state = GraphState() if state is None else state
        stamp = (
            float(state.stamps.get("b0", torch.tensor(0.0)).max())
            + float(minutes.max()) * 60.0
            if stamp is None
            else float(stamp)
        )
        if not math.isfinite(stamp):
            raise ValueError("A finite stream time is required")
        return self._run("b0", x, minutes, state, stamp, "b0")

    def _run(self, identity, x, minutes, initial, stamp, path):
        core = self.cores[identity]
        before = initial.local.get(path)
        if before is None:
            before = core.start(x)
        elif before.elapsed.shape[1] != core.experts:
            raise ValueError("State must be aligned to the current expert identities")
        dt = minutes
        if path != "b0" and path in initial.stamps:
            dt = ((stamp - initial.stamps[path]) / 60.0).to(x)
            if not (dt > 0).all():
                raise ValueError("Child stream time must increase")
        core._call = (path, stamp, initial)
        try:
            decision = core(x, dt, before)
            rounds = core._round_children
        finally:
            core._call = None
            core._round_children = []
        local, stamps = initial.local.copy(), initial.stamps.copy()
        local[path], stamps[path] = decision.state, torch.full(
            (len(x),), stamp, dtype=torch.float64, device=x.device
        )
        trace = {path: (identity, decision)}
        for result, children in zip(decision.rounds, rounds):
            for index, child in children:
                changed = (
                    result.active
                    & result.accepted
                    & result.acted[:, index]
                    & ~result.challenged[:, index]
                )
                trace.update(child.trace)
                if not changed.any():
                    continue
                prefix = path + "/" + core.ids[index] + "/"
                for key, new in child.state.local.items():
                    if not key.startswith(prefix):
                        continue
                    old = local.get(key)
                    if old is None:
                        child_identity = key.split("/")[-1]
                        dummy = x.new_zeros(len(x), self.cores[child_identity].features)
                        old = self.cores[child_identity].start(dummy)
                    local[key] = merge_state(changed, new, old)
                    prior_stamp = stamps.get(
                        key,
                        torch.full(
                            (len(x),), stamp, dtype=torch.float64, device=x.device
                        )
                        - minutes.double() * 60.0,
                    )
                    stamps[key] = torch.where(
                        changed,
                        torch.full(
                            (len(x),), stamp, dtype=torch.float64, device=x.device
                        ),
                        prior_stamp,
                    )
        return Result(decision, GraphState(local, stamps), trace)

    @staticmethod
    def extra_limits():
        return (
            "min_layers",
            "max_layers",
            "min_encoding",
            "min_experts",
            "max_boards",
            "max_attention_nodes",
            "max_attention_depth",
            "max_total_heads",
            "max_work",
            "max_batch",
            "route_floor",
            "min_liquid",
            "max_liquid",
        )

    def profile(self):
        def walk(board):
            core = self.cores[board]
            # A deterministic operation proxy, evaluated at worst-case participation
            # and retry count. Shared weights still incur work per context.
            own = core.round_limit * (
                core.experts
                * (
                    sum(
                        (core.encoding + WIDTH) * 3 * WIDTH
                        + len(c.units) * (2 * WIDTH + len(c.units)) * c.substeps
                        for c in core.cells
                    )
                    + (2 * core.features + WIDTH + 3) * core.encoding
                )
                + core.delta.total_heads * WIDTH * 13
            )
            contexts = core.experts
            for key, edge in self.edges.items():
                if key.split("/")[0] == board:
                    work, count = walk(edge["child"])
                    # Children can be called on every parent retry.
                    own += core.round_limit * work
                    contexts += count
            return own, contexts

        work, contexts = walk("b0")
        return dict(
            parameters=sum(p.numel() for p in self.parameters()),
            buffer_bytes=sum(b.numel() * b.element_size() for b in self.buffers()),
            boards=len(self.cores),
            experts=self.size(),
            attention_nodes=sum(
                len(list(c.delta.nodes())) for c in self.cores.values()
            ),
            heads=sum(c.delta.total_heads for c in self.cores.values()),
            context_experts=contexts,
            work_units=work,
        )

    def validate_topology(self):
        p = self.profile()
        if (
            p["boards"] > self.max_boards
            or p["experts"] > self.max_experts
            or p["heads"] > self.max_total_heads
            or p["attention_nodes"] > self.max_attention_nodes
            or p["work_units"] > self.max_work
        ):
            raise ValueError("Global topology/work budget exceeded")
        for name, core in self.cores.items():
            if (
                not self.min_layers <= len(core.cells) <= self.max_layers
                or not self.min_encoding <= core.encoding <= self.max_encoding
            ):
                raise ValueError("Liquid-layer or encoding boundary exceeded")
            if self.depth(name) > self.max_depth:
                raise ValueError("Network depth boundary exceeded")
            if any(
                not self.min_liquid <= len(c.units) <= self.max_liquid
                for c in core.cells
            ):
                raise ValueError("Liquid neuron boundary exceeded")
            for path, node in core.delta.nodes():
                if (
                    not self.min_heads <= node.count <= self.max_heads
                    or path.count(":") > self.max_attention_depth
                ):
                    raise ValueError("Attention boundary exceeded")
        return p

    def reserve(self, action, board="b0"):
        p = self.profile()
        core = self.cores[board]
        growth = action in (
            "spawn",
            "deepen",
            "split",
            "grow_heads",
            "add_layer",
            "branch_attention",
            "fork_network",
            "split_liquid",
        )
        result = dict(p)
        if growth:
            result["parameters"] += 2 * max(
                sum(v.numel() for v in c.parameters()) for c in self.cores.values()
            )

            def visits(node, multiplier=1):
                count = multiplier if node == board else 0
                for key, edge in self.edges.items():
                    if key.split("/")[0] == node:
                        count += visits(
                            edge["child"], multiplier * self.cores[node].round_limit
                        )
                return count

            calls = visits("b0")

            def local(c, experts=None, features=None):
                e = c.experts if experts is None else experts
                f = c.features if features is None else features
                return c.round_limit * (
                    e
                    * (
                        sum(
                            (c.encoding + WIDTH) * 3 * WIDTH
                            + len(cell.units)
                            * (2 * WIDTH + len(cell.units))
                            * cell.substeps
                            for cell in c.cells
                        )
                        + (2 * f + WIDTH + 3) * c.encoding
                    )
                    + c.delta.total_heads * WIDTH * 13
                )

            def subtree(node):
                return local(self.cores[node]) + self.cores[node].round_limit * sum(
                    subtree(e["child"])
                    for k, e in self.edges.items()
                    if k.split("/")[0] == node
                )

            if action == "deepen":
                extra = core.round_limit * local(core, 1, len(self.fields) + WIDTH)
            elif action == "spawn":
                extra = core.round_limit * (
                    len(core.cells) * (core.encoding + WIDTH) * 3 * WIDTH
                    + (2 * core.features + WIDTH + 3) * core.encoding
                )
                extra += core.round_limit * max(
                    [
                        subtree(e["child"])
                        for k, e in self.edges.items()
                        if k.split("/")[0] == board
                    ]
                    or [0]
                )
            elif action == "split_liquid":
                extra = (
                    core.round_limit
                    * core.experts
                    * (2 * WIDTH + 2 * self.max_liquid + 1)
                    * max(c.substeps for c in core.cells)
                )
            elif action == "add_layer":
                extra = (
                    core.round_limit
                    * core.experts
                    * (
                        (core.encoding + WIDTH) * 3 * WIDTH
                        + self.max_liquid
                        * (2 * WIDTH + self.max_liquid)
                        * max(c.substeps for c in core.cells)
                    )
                )
            elif action == "split":
                extra = (
                    core.round_limit
                    * core.experts
                    * (len(core.cells) * 3 * WIDTH + 2 * core.features + WIDTH + 3)
                )
            else:
                extra = core.round_limit * self.min_heads * WIDTH * 13
            result["work_units"] += calls * extra
            result["context_experts"] *= 2
        if action == "fork_network":
            targets = [
                self.cores[e["child"]]
                for k, e in self.edges.items()
                if k.startswith(board + "/")
            ]
            if not targets:
                raise ValueError("No network exists to fork")
            result["boards"] += 1
            result["experts"] += max(c.experts for c in targets)
            result["attention_nodes"] += max(
                len(list(c.delta.nodes())) for c in targets
            )
            result["heads"] += max(c.delta.total_heads for c in targets)
        if action == "spawn":
            result["experts"] += 1
        elif action == "deepen":
            result["experts"] += 1
            result["boards"] += 1
            result["attention_nodes"] += len(list(core.delta.nodes()))
            result["heads"] += core.delta.total_heads
        elif action == "grow_heads":
            result["heads"] += 1
        elif action == "branch_attention":
            result["attention_nodes"] += 1
            result["heads"] += self.min_heads
        if (
            result["experts"] > self.max_experts
            or result["boards"] > self.max_boards
            or result["attention_nodes"] > self.max_attention_nodes
            or result["heads"] > self.max_total_heads
            or result["work_units"] > self.max_work
        ):
            raise ValueError("Mutation reservation exceeds the global envelope")
        return result

    def can_reserve(self, action, board="b0"):
        try:
            self.reserve(action, board)
            return True
        except ValueError:
            return False

    def change_liquid(self, board, layer, grow, unit=0):
        self.reserve("split_liquid" if grow else "prune_liquid", board)
        core = self.cores[board]
        cell = core.cells[layer]
        if (
            grow
            and len(cell.units) >= self.max_liquid
            or not grow
            and len(cell.units) <= self.min_liquid
        ):
            raise ValueError("Liquid neuron floor or ceiling reached")
        self.untie(board)
        cell = core.cells[layer]
        cell.split(unit) if grow else cell.prune(unit)
        return dict(
            board=board,
            layer=core.layer_ids[layer],
            unit=unit,
            width=len(cell.units),
            kind="liquid_neuron_split" if grow else "liquid_neuron_prune",
        )

    def fork_network(self, board, expert):
        self.reserve("fork_network", board)
        parent = self.cores[board]
        edge = self.edges[board + "/" + parent.ids[expert]]
        previous = edge["child"]
        if sum(e["child"] == previous for e in self.edges.values()) < 2:
            raise ValueError("Only a shared child network needs an independent fork")
        clone = deepcopy(self.cores[previous])
        clone._call, clone._round_children = None, []
        identity = f"b{self.next_board}"
        self.next_board += 1
        self.cores[identity] = clone
        edge["child"] = identity
        for key, child in list(self.edges.items()):
            if key.startswith(previous + "/"):
                self.edges[identity + key[len(previous) :]] = deepcopy(child)
        clone.bind(self, identity)
        return dict(
            board=identity,
            donor=previous,
            parent=board + "/" + parent.ids[expert],
            kind="network_fork",
        )

    def retire_network(self, board, expert):
        key = board + "/" + self.cores[board].ids[expert]
        if key not in self.edges:
            raise ValueError("No child network to retire")
        del self.edges[key]
        return dict(parent=key, removed=self.collect(), kind="network_retire")

    def insert_layer(self, board, index):
        self.reserve("add_layer", board)
        core = self.cores[board]
        if len(core.cells) >= self.max_layers or not 0 <= index <= len(core.cells):
            raise ValueError("Liquid-layer ceiling or invalid location")
        self.untie(board)
        donor = min(index, len(core.cells) - 1)
        for attribute in ("cells", "mixers", "outputs", "falsifiers"):
            modules = getattr(core, attribute)
            modules.insert(index, deepcopy(modules[donor]))
        identity = f"l{core.next_layer}"
        core.next_layer += 1
        core.layer_ids.insert(index, identity)
        core.layer_gain = nn.Parameter(
            torch.cat(
                (
                    core.layer_gain.detach()[:index],
                    torch.zeros(1),
                    core.layer_gain.detach()[index:],
                )
            )
        )
        return dict(
            board=board, layer=identity, index=index, kind="liquid_layer_insert"
        )

    def remove_layer(self, board, index):
        core = self.cores[board]
        if len(core.cells) <= self.min_layers or not 0 <= index < len(core.cells):
            raise ValueError("Liquid-layer floor or invalid location")
        self.untie(board)
        layer = core.layer_ids.pop(index)
        for attribute in ("cells", "mixers", "outputs", "falsifiers"):
            del getattr(core, attribute)[index]
        core.layer_gain = nn.Parameter(
            torch.cat(
                (
                    core.layer_gain.detach()[:index],
                    core.layer_gain.detach()[index + 1 :],
                )
            )
        )
        return dict(board=board, layer=layer, index=index, kind="liquid_layer_remove")

    def prune_neuron(self, board, index):
        core = self.cores[board]
        if core.encoding <= self.min_encoding or not 0 <= index < core.encoding:
            raise ValueError("Encoding floor or invalid location")
        keep = [i for i in range(core.encoding) if i != index]
        core.input.weight = nn.Parameter(core.input.weight.detach()[..., keep].clone())
        core.input.bias = nn.Parameter(core.input.bias.detach()[..., keep].clone())
        for module in (*core.mixers, *core.outputs, core.process, core.drive):
            rows = keep + list(range(core.encoding, module.weight.shape[1]))
            module.weight = nn.Parameter(module.weight.detach()[:, rows].clone())
        core.encoding -= 1
        return dict(board=board, neuron=index, kind="encoding_prune")

    def branch_attention(self, board, parent="root"):
        self.reserve("branch_attention", board)
        tree = self.cores[board].delta
        p = self.profile()
        if (
            p["attention_nodes"] >= self.max_attention_nodes
            or p["heads"] + self.min_heads > self.max_total_heads
            or parent.count(":") >= self.max_attention_depth
        ):
            raise ValueError("Attention tree boundary reached")
        path = tree.add(parent)
        node = tree.node(path)
        while node.count < self.min_heads:
            node.resize([*range(node.count), 0])
        return dict(board=board, attention=path, kind="delta_attention_branch")

    def prune_attention(self, board, path):
        if path == "root":
            raise ValueError("Cannot remove the root attention")
        core = self.cores[board]
        core.delta.remove(path)
        # Redirect surviving networks to the closest surviving ancestor. This
        # behavioral change must earn a no-harm check before adoption.
        parent = path.rsplit(":", 1)[0]
        for key, edge in self.edges.items():
            if key.startswith(board + "/") and (
                edge.get("attention", "root") == path
                or edge.get("attention", "root").startswith(path + ":")
            ):
                edge["attention"] = parent
        return dict(board=board, attention=path, kind="delta_attention_prune")

    def size(self):
        return sum(core.experts for core in self.cores.values())

    def depth(self, board):
        def walk(node, depth):
            result = depth if node == board else -1
            for key, edge in self.edges.items():
                if key.split("/")[0] == node:
                    result = max(result, walk(edge["child"], depth + 1))
            return result

        return walk("b0", 0)

    def spawn(self, board, expert):
        self.reserve("spawn", board)
        if self.size() >= self.max_experts:
            raise ValueError("Expert budget exhausted")
        self.untie(board)
        core = self.cores[board]
        donor = core.ids[expert]
        core.reindex([*range(core.experts), expert])
        core.vote_mass[expert] *= 0.5
        core.vote_mass[-1] = core.vote_mass[expert]
        core.majority = math.floor(float(core.vote_mass.sum()) / 2) + 1
        core.births[core.ids[-1]] = donor
        edge = self.edges.get(board + "/" + donor)
        if edge:
            self.edges[board + "/" + core.ids[-1]] = deepcopy(edge)
        return dict(
            board=board, expert=core.ids[-1], donor=donor, kind="vote_mass_split"
        )

    def split(self, board, expert, neuron):
        self.reserve("split", board)
        core = self.cores[board]
        if core.encoding >= self.max_encoding:
            raise ValueError("Encoding neuron budget exhausted")
        core.split_neuron(expert, neuron)
        return dict(
            board=board,
            expert=core.ids[expert],
            neuron=neuron,
            encoding=core.encoding,
            kind="net2wider_silu",
        )

    def change_heads(self, board, grow, head=0, attention="root"):
        self.reserve("grow_heads" if grow else "prune_heads", board)
        delta = self.cores[board].delta.node(attention)
        if grow:
            if (
                delta.count >= self.max_heads
                or self.profile()["heads"] >= self.max_total_heads
            ):
                raise ValueError("Attention head ceiling reached")
            delta.resize([*range(delta.count), head])
        else:
            if delta.count <= self.min_heads:
                raise ValueError("Attention head minimum reached")
            delta.resize([i for i in range(delta.count) if i != head])
        return dict(
            board=board,
            head=head,
            attention=attention,
            heads=delta.count,
            kind="attention_split" if grow else "attention_prune",
        )

    def deepen(self, board, expert, group, attention="root"):
        self.reserve("deepen", board)
        core = self.cores[board]
        key = board + "/" + core.ids[expert]
        if (
            key in self.edges
            or self.depth(board) >= self.max_depth
            or self.size() >= self.max_experts
            or len(self.cores) >= self.max_boards
        ):
            raise ValueError(
                "A child already exists, or a depth/expert limit was reached"
            )
        if not group or not set(group) <= set(self.fields):
            raise ValueError("A child group must use registered fields")
        core.delta.node(attention)
        child = deepcopy(core)
        child._call, child._round_children = None, []
        child.reindex([expert])
        child.expand(self.fields + [f"parent:{i}" for i in range(WIDTH)])
        identity = f"b{self.next_board}"
        self.next_board += 1
        self.cores[identity] = child
        self.edges[key] = dict(child=identity, group=sorted(group), attention=attention)
        with torch.no_grad():
            core.child_gain[expert] = 0.0
        child.bind(self, identity)
        return dict(board=identity, parent=key, group=sorted(group))

    def collect(self):
        live = {"b0"}
        while True:
            expanded = live | {
                edge["child"]
                for key, edge in self.edges.items()
                if key.split("/")[0] in live
            }
            if expanded == live:
                break
            live = expanded
        removed = []
        for name in list(self.cores):
            if name not in live:
                self.untie(name)
                del self.cores[name]
                removed.append(name)
        self.edges = {
            key: edge for key, edge in self.edges.items() if key.split("/")[0] in live
        }
        return removed

    def retire(self, board, expert):
        self.untie(board)
        core = self.cores[board]
        key = board + "/" + core.ids[expert]
        if board == "b0" and core.experts <= self.min_experts:
            raise ValueError("Cannot retire the last root")
        if core.experts == 1:
            self.edges = {k: v for k, v in self.edges.items() if v["child"] != board}
        else:
            self.edges.pop(key, None)
            core.reindex([i for i in range(core.experts) if i != expert])
        return self.collect()

    def signature(self, board):
        core = self.cores[board]
        digest = hashlib.sha256(json.dumps(core.names).encode())
        for name, value in core.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        for index, identity in enumerate(core.ids):
            edge = self.edges.get(board + "/" + identity)
            if edge:
                digest.update(
                    json.dumps(
                        [
                            index,
                            edge["group"],
                            edge.get("attention", "root"),
                            self.signature(edge["child"]),
                        ]
                    ).encode()
                )
        return digest.hexdigest()

    def matches(self):
        pairs, prior = [], {}
        for board in self.cores:
            if board == "b0":
                continue
            signature = self.signature(board)
            if signature in prior:
                pairs.append((prior[signature], board))
            else:
                prior[signature] = board
        return pairs

    def kernel_matches(self):
        pairs = []
        names = [name for name in self.cores if name != "b0"]
        for i, kept in enumerate(names):
            for other in names[i + 1 :]:
                a, b = self.cores[kept].cells, self.cores[other].cells
                if a is b:
                    continue
                left, right = a.state_dict(), b.state_dict()
                if left.keys() == right.keys() and all(
                    torch.equal(left[k], right[k]) for k in left
                ):
                    pairs.append((kept, other))
        return pairs

    def untie(self, board):
        # A resized expert axis must not resize another parent's liquid kernel.
        component = {board}
        while True:
            bigger = component | {
                n
                for pair in self.kernel_ties.items()
                if set(pair) & component
                for n in pair
            }
            if bigger == component:
                break
            component = bigger
        if len(component) > 1:
            self.cores[board].cells = deepcopy(self.cores[board].cells)
            self.kernel_ties = {
                k: v for k, v in self.kernel_ties.items() if not ({k, v} & component)
            }
            remaining = sorted(n for n in component if n != board and n in self.cores)
            if len(remaining) > 1:
                kept = remaining[0]
                for other in remaining[1:]:
                    self.cores[other].cells = self.cores[kept].cells
                    self.kernel_ties[other] = kept

    def share_equal(self):
        pairs = self.matches()
        if not pairs:
            kernels = self.kernel_matches()
            if not kernels:
                raise ValueError(
                    "No equal child liquid weights or complete child boards exist"
                )
            kept, other = kernels[0]
            while kept in self.kernel_ties:
                kept = self.kernel_ties[kept]
            # Keep all readouts, checkers, policies and every path's state separate.
            self.cores[other].cells = self.cores[kept].cells
            self.kernel_ties[other] = kept
            return dict(kind="liquid_kernel", kept=kept, tied=other)
        kept, removed = pairs[0]
        # Hashes find candidates; exact equality is required before aliasing weights.
        a, b = self.cores[kept].state_dict(), self.cores[removed].state_dict()
        if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
            raise ValueError("Child weights differ")
        for edge in self.edges.values():
            if edge["child"] == removed:
                edge["child"] = kept
        self.collect()
        return dict(kind="complete_board", kept=kept, removed=removed)

    def align(self, before, old):
        """Retain states by stable identity; births get zero state."""
        state = GraphState()
        for path, native in before.local.items():
            parts = path.split("/")
            old_board, board = "b0", "b0"
            mapped, valid = ["b0"], True
            for offset in range(1, len(parts), 2):
                previous_core, new_core = old.cores[old_board], self.cores[board]
                previous_expert = parts[offset]
                if old_board == board:
                    if previous_expert not in new_core.ids:
                        valid = False
                        break
                    expert = previous_expert
                else:
                    # Full-board sharing maps equal expert rows by position,
                    # including every corresponding descendant context.
                    expert = new_core.ids[previous_core.ids.index(previous_expert)]
                edge = self.edges.get(board + "/" + expert)
                if edge is None:
                    valid = False
                    break
                old_board, board = parts[offset + 1], edge["child"]
                mapped.extend((expert, board))
            if not valid:
                continue
            new_path = "/".join(mapped)
            new, previous = self.cores[board], old.cores[old_board]
            dummy = native.memory.new_zeros(len(native.last), new.features)
            target = new.start(dummy)
            cells = [c.clone() for c in target.cells]
            elapsed = target.elapsed.clone()
            for dest, name in enumerate(new.ids):
                source_id = name if name in previous.ids else new.births.get(name)
                if old_board != board or source_id in previous.ids:
                    source = (
                        dest if old_board != board else previous.ids.index(source_id)
                    )
                    for layer, c in zip(new.layer_ids, cells):
                        if old_board != board or layer in previous.layer_ids:
                            location = (
                                new.layer_ids.index(layer)
                                if old_board != board
                                else previous.layer_ids.index(layer)
                            )
                            current, prior = (
                                new.cells[new.layer_ids.index(layer)],
                                previous.cells[location],
                            )
                            for unit, identity in enumerate(current.units):
                                origin = (
                                    identity
                                    if identity in prior.units
                                    else current.births.get(identity)
                                )
                                if old_board != board or origin in prior.units:
                                    source_unit = (
                                        unit
                                        if old_board != board
                                        else prior.units.index(origin)
                                    )
                                    c[:, dest, unit] = native.cells[location][
                                        :, source, source_unit
                                    ]
                    elapsed[:, dest] = native.elapsed[:, source]
            state.local[new_path] = State(
                tuple(cells),
                # Exact whole-board sharing preserves head positions, even
                # when equal heads have different lifecycle identities.
                (
                    native.memory.clone()
                    if old_board != board
                    else new.delta.align_memory(native.memory, previous.delta)
                ),
                native.last,
                elapsed,
            )
            state.stamps[new_path] = before.stamps[path]
        # A split peer inherits its donor's complete child context, not just cells.
        for board, core in self.cores.items():
            if board not in old.cores:
                continue
            for born, donor in core.births.items():
                if born in old.cores[board].ids:
                    continue
                for parent_path in list(state.local):
                    if parent_path.split("/")[-1] != board:
                        continue
                    prefix = parent_path + "/" + donor + "/"
                    for key in list(state.local):
                        if key.startswith(prefix):
                            destination = (
                                parent_path + "/" + born + "/" + key[len(prefix) :]
                            )
                            native = state.local[key]
                            state.local[destination] = State(
                                tuple(c.clone() for c in native.cells),
                                native.memory.clone(),
                                native.last.clone(),
                                native.elapsed.clone(),
                            )
                            state.stamps[destination] = state.stamps[key].clone()
        return state.detached()

    def layout(self):
        return dict(
            fields=self.fields,
            edges=self.edges,
            kernel_ties=self.kernel_ties,
            next_board=self.next_board,
            max_experts=self.max_experts,
            max_depth=self.max_depth,
            max_fields=self.max_fields,
            max_encoding=self.max_encoding,
            min_heads=self.min_heads,
            max_heads=self.max_heads,
            policy_capacity=self.policy_capacity,
            **{k: getattr(self, k) for k in self.extra_limits()},
            cores={
                name: dict(
                    names=core.names,
                    experts=core.experts,
                    rounds=core.round_limit,
                    depth=len(core.cells),
                    ids=core.ids,
                    next_expert=core.next_expert,
                    encoding=core.encoding,
                    births=core.births,
                    attention=core.delta.metadata(),
                    layer_ids=core.layer_ids,
                    next_layer=core.next_layer,
                    liquid=[c.metadata() for c in core.cells],
                )
                for name, core in self.cores.items()
            },
        )

    def pack(self):
        return dict(layout=deepcopy(self.layout()), weights=deepcopy(self.state_dict()))

    @classmethod
    def restore(cls, payload):
        layout = payload["layout"]
        graph = cls(
            names=layout["fields"],
            max_experts=layout["max_experts"],
            max_depth=layout["max_depth"],
            max_fields=layout["max_fields"],
            max_encoding=layout["max_encoding"],
            min_heads=layout["min_heads"],
            max_heads=layout["max_heads"],
            policy_capacity=layout["policy_capacity"],
            **{k: layout[k] for k in cls.extra_limits()},
        )
        graph.cores = nn.ModuleDict()
        for name, item in layout["cores"].items():
            core = Core(
                item["names"],
                item["experts"],
                item["rounds"],
                item["depth"],
                layout["policy_capacity"],
            )
            while core.encoding < item["encoding"]:
                core.split_neuron(0, 0)
            while core.encoding > item["encoding"]:
                width = core.encoding
                core.input.weight = nn.Parameter(
                    core.input.weight.detach()[..., : width - 1].clone()
                )
                core.input.bias = nn.Parameter(
                    core.input.bias.detach()[..., : width - 1].clone()
                )
                for module in (*core.mixers, *core.outputs, core.process, core.drive):
                    rows = list(range(width - 1)) + list(
                        range(width, module.weight.shape[1])
                    )
                    module.weight = nn.Parameter(
                        module.weight.detach()[:, rows].clone()
                    )
                core.encoding -= 1
            for cell, shape in zip(core.cells, item["liquid"], strict=True):
                cell.restore_shape(shape)
            core.delta.restore_shape(item["attention"])
            core.layer_ids, core.next_layer = item["layer_ids"], item["next_layer"]
            core.ids, core.next_expert = item["ids"], item["next_expert"]
            core.births = deepcopy(item["births"])
            graph.cores[name] = core
        graph.edges, graph.next_board = deepcopy(layout["edges"]), layout["next_board"]
        graph.load_state_dict(payload["weights"])
        graph.kernel_ties = deepcopy(layout["kernel_ties"])
        for other, kept in graph.kernel_ties.items():
            graph.cores[other].cells = graph.cores[kept].cells
        graph.bind()
        graph.validate_topology()
        return graph
