"""Explicit stream-state gather/scatter and interchangeable tensor execution."""

from dataclasses import fields, is_dataclass
from copy import deepcopy
import hashlib

import torch

from graph import Graph, GraphState
from reference import State


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(
            **{f.name: move(getattr(value, f.name), device) for f in fields(value)}
        )
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    if isinstance(value, list):
        return [move(v, device) for v in value]
    return value


def gather(graph, states, stamps, minutes):
    """Missing child paths start privately, even when a different row used them."""
    keys = set().union(*(s.local for s in states))
    local, clocks = {}, {}
    for path in keys:
        core = graph.cores[path.split("/")[-1]]
        pieces, times = [], []
        for i, state in enumerate(states):
            pieces.append(
                state.local.get(path) or core.start(torch.zeros(1, core.features))
            )
            times.append(
                state.stamps.get(
                    path,
                    torch.tensor([stamps[i] - 60 * minutes[i]], dtype=torch.float64),
                )
            )
        local[path] = State(
            tuple(
                torch.cat([s.cells[j] for s in pieces]) for j in range(len(core.cells))
            ),
            torch.cat([s.memory for s in pieces]),
            torch.cat([s.last for s in pieces]),
            torch.cat([s.elapsed for s in pieces]),
        )
        clocks[path] = torch.cat(times)
    present = {
        path: torch.tensor([path in s.local for s in states], dtype=torch.bool)
        for path in keys
    }
    return GraphState(local, clocks, present)


def scatter(state, count):
    return [
        GraphState(
            {
                p: State(
                    tuple(c[i : i + 1].detach().clone() for c in s.cells),
                    s.memory[i : i + 1].detach().clone(),
                    s.last[i : i + 1].detach().clone(),
                    s.elapsed[i : i + 1].detach().clone(),
                )
                for p, s in state.local.items()
                if p not in state.present or bool(state.present[p][i])
            },
            {
                p: t[i : i + 1].detach().clone()
                for p, t in state.stamps.items()
                if p not in state.present or bool(state.present[p][i])
            },
        )
        for i in range(count)
    ]


def fingerprint(graph):
    h = hashlib.sha256()
    for name, t in sorted(graph.state_dict().items()):
        t = t.detach().cpu().contiguous()
        h.update(str((name, str(t.dtype), tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class Backend:
    """The controller owns CPU weights; accelerator replicas are read-only per batch."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "Choose cpu or cuda (the PyTorch namespace also used by ROCm)"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Requested accelerator is not available; no CPU substitution"
            )
        self.mirror = None
        self.key = None

    @torch.no_grad()
    def run(self, graph, rows, states, stamps, minutes, corrections):
        state = gather(graph, states, stamps, minutes)
        if self.device.type == "cpu":
            model = graph
        else:
            key = fingerprint(graph)
            if self.key != key:
                self.mirror = (
                    Graph.restore(deepcopy(graph.pack())).eval().to(self.device)
                )
                self.key = key
            model = self.mirror
        x = torch.cat([graph.tensor(row) for row in rows]).to(self.device)
        core = model.cores["b0"]
        core.stream_correction = torch.stack(corrections).to(x)
        try:
            result = model(
                x,
                x.new_tensor(minutes),
                move(state, self.device),
                stamp=torch.tensor(stamps, dtype=torch.float64, device=self.device),
            )
        finally:
            del core.stream_correction
        result = move(result, "cpu")
        return result, scatter(result.state, len(rows))
