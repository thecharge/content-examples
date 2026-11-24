"""Bounded, higher-arity messages between liquid experts, with measured nominations."""

from dataclasses import asdict, dataclass
from itertools import combinations
import math

import torch
from torch import nn

from reference import WIDTH


@dataclass(frozen=True)
class HyperLimits:
    min_edges: int = 0
    max_edges: int = 16
    max_members: int = 16
    max_mass: float = 16.0
    minimum_samples: int = 16
    window: int = 96
    minimum_context: int = 8
    merge_correlation: float = 0.95
    maximum_disagreement: float = 0.03
    split_residual_gap: float = 0.10

    def __post_init__(self):
        if not 0 <= self.min_edges <= self.max_edges or self.max_members < 3:
            raise ValueError("Invalid hyperedge bounds")
        if not 2 <= self.minimum_context <= self.minimum_samples <= self.window:
            raise ValueError("Invalid evidence window bounds")
        if (
            not 0 <= self.merge_correlation <= 1
            or not 0 <= self.maximum_disagreement <= 1
        ):
            raise ValueError("Invalid similarity thresholds")
        if not 0 < self.split_residual_gap <= 1:
            raise ValueError("Invalid residual threshold")
        if not math.isfinite(self.max_mass) or self.max_mass < 1:
            raise ValueError("Invalid message mass ceiling")


class Message(nn.Module):
    """The source states are liquid. Delta memory gates this bounded readout."""

    def __init__(self):
        super().__init__()
        self.readout = nn.Linear(WIDTH, 1, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden):
        return self.gain.tanh() * self.readout(hidden).squeeze(-1).tanh()


def correlation(left, right):
    """Undefined constant signals stay undefined; do not invent perfect agreement."""
    a, b = (
        torch.tensor(left, dtype=torch.float64),
        torch.tensor(right, dtype=torch.float64),
    )
    a, b = a - a.mean(), b - b.mean()
    denominator = float(a.norm() * b.norm())
    return (
        None if denominator <= torch.finfo(a.dtype).eps else float(a @ b / denominator)
    )


def evidence(rows, board, limits):
    """Provenance-bearing pair measurements; predictions must predate their labels."""
    rows = rows[-limits.window :]
    pairs = []
    keys = (
        sorted(set.intersection(*(set(r.get("proposals", {})) for r in rows)))
        if rows
        else []
    )
    keys = [k for k in keys if k.startswith(board + "/")]
    for left, right in combinations(keys, 2):
        # A support identity is an observation, not a repeated window containing it.
        unique = {}
        for row in rows:
            identity = row.get("observation_id", str(row["stamp"]))
            if row.get("invalidated") or row["revealed_at"] <= row["stamp"]:
                continue
            unique[identity] = row
        sample = list(unique.values())
        if len(sample) < limits.minimum_samples or {r["truth"] for r in sample} != {
            0,
            1,
        }:
            continue
        a = [r["proposals"][left]["probability"] for r in sample]
        b = [r["proposals"][right]["probability"] for r in sample]
        gap = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
        residuals = []
        contexts = {}
        for r, x, y in zip(sample, a, b):
            context = str(r.get("context", "default"))
            contexts.setdefault(context, []).append((r["truth"] - x) - (r["truth"] - y))
        for context, values in sorted(contexts.items()):
            if len(values) >= limits.minimum_context:
                residuals.append(
                    dict(
                        context=context,
                        count=len(values),
                        gap=sum(values) / len(values),
                    )
                )
        r = correlation(a, b)
        split = (
            len(residuals) >= 2
            and max(x["gap"] for x in residuals) - min(x["gap"] for x in residuals)
            >= limits.split_residual_gap
        )
        hidden_a = torch.stack(
            [x["proposals"][left]["hidden"] for x in sample]
        ).double()
        hidden_b = torch.stack(
            [x["proposals"][right]["hidden"] for x in sample]
        ).double()
        hidden_a -= hidden_a.mean(0)
        hidden_b -= hidden_b.mean(0)
        cross = (hidden_a.T @ hidden_b).square().sum()
        scale = (hidden_a.T @ hidden_a).norm() * (hidden_b.T @ hidden_b).norm()
        cka = None if float(scale) == 0 else float(cross / scale)
        pairs.append(
            dict(
                metric="prediction-residual-linear-cka-v1",
                left=left,
                right=right,
                support=sorted(unique),
                labels=[r.get("label_id", str(r["revealed_at"])) for r in sample],
                revisions=sorted(
                    set(
                        str(r.get("model_revision", "unversioned-direct-runtime"))
                        for r in sample
                    )
                ),
                first=min(r["stamp"] for r in sample),
                last=max(r["stamp"] for r in sample),
                count=len(sample),
                classes={str(c): sum(r["truth"] == c for r in sample) for c in (0, 1)},
                pearson=r,
                mean_absolute_disagreement=gap,
                linear_cka=cka,
                contexts=residuals,
                merge_nomination=r is not None
                and r >= limits.merge_correlation
                and gap <= limits.maximum_disagreement,
                split_nomination=split,
                thresholds=asdict(limits),
            )
        )
    return pairs
