"""A tree of delta memories. Each node has heads, forgetting and a learned gate."""

from copy import deepcopy
import math

import torch
from torch import nn

from attention import AdaptiveDelta
from reference import WIDTH


class DeltaTree(AdaptiveDelta):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict()
        self.gains = nn.ParameterDict()
        self.next_branch = 0

    @property
    def total_heads(self):
        return self.count + sum(c.total_heads for c in self.branches.values())

    def nodes(self, path="root"):
        yield path, self
        for name, child in self.branches.items():
            yield from child.nodes(path + ":" + name)

    def node(self, path):
        result = self
        for name in path.split(":")[1:]:
            result = result.branches[name]
        return result

    def add(self, parent):
        owner = self.node(parent)
        name = f"a{owner.next_branch}"
        owner.next_branch += 1
        child = DeltaTree()
        child.resize([0])
        owner.branches[name] = child
        owner.gains[name] = nn.Parameter(torch.tensor(0.0))
        return parent + ":" + name

    def remove(self, path):
        parent, name = path.rsplit(":", 1)
        owner = self.node(parent)
        del owner.branches[name]
        del owner.gains[name]

    def read(self, memory, query):
        result = super().read(memory[:, : self.count], query)
        offset = self.count
        for name, child in self.branches.items():
            size = child.total_heads
            recalled = child.read(memory[:, offset : offset + size], query + result)
            result = result + self.gains[name].tanh() * recalled
            offset += size
        return result

    def forward(self, memory, draft, minutes):
        results = [super().forward(memory[:, : self.count], draft, minutes)]
        offset = self.count
        # Every node receives the old parent recall, never a future write.
        parent = super().read(memory[:, : self.count], draft)
        for child in self.branches.values():
            size = child.total_heads
            results.append(
                child(memory[:, offset : offset + size], draft + parent, minutes)
            )
            offset += size
        return torch.cat(results, dim=1)

    def route(self, path, memory, query):
        if path == "root":
            return query.new_ones(len(query))
        offsets = {}
        offset = 0
        for name, node in self.nodes():
            offsets[name] = offset
            offset += node.count
        node = self.node(path)
        start = offsets[path]
        recalled = AdaptiveDelta.read(
            node, memory[:, start : start + node.count], query
        )
        return ((recalled * query).sum(-1) / math.sqrt(WIDTH)).sigmoid()

    def metadata(self):
        return dict(
            super().metadata(),
            branches={k: v.metadata() for k, v in self.branches.items()},
            next_branch=self.next_branch,
        )

    def restore_shape(self, data):
        super().restore_shape(data)
        self.branches = nn.ModuleDict()
        self.gains = nn.ParameterDict()
        for name, shape in data.get("branches", {}).items():
            child = DeltaTree()
            child.restore_shape(shape)
            self.branches[name] = child
            self.gains[name] = nn.Parameter(torch.tensor(0.0))
        self.next_branch = data.get("next_branch", 0)

    def align_memory(self, memory, previous):
        old = {}
        offset = 0
        for name, node in previous.nodes():
            old[name] = (node, memory[:, offset : offset + node.count])
            offset += node.count
        parts = []
        for name, node in self.nodes():
            if name in old:
                source, values = old[name]
                parts.append(AdaptiveDelta.align_memory(node, values, source))
            else:
                parts.append(
                    memory.new_zeros(
                        len(memory), node.count, self.dimension, self.dimension
                    )
                )
        return torch.cat(parts, dim=1)
