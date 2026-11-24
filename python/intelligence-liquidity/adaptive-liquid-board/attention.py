"""Delta memory with a variable number of four-coordinate heads."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from reference import WIDTH, primitives


class AdaptiveDelta(primitives.DeltaAttention):
    dimension = WIDTH // primitives.HEADS

    def __init__(self):
        super().__init__()
        self.identities = [0, 1]
        self.next_identity = 2
        self.births = {}
        self.readout = nn.Parameter(torch.eye(WIDTH))

    @property
    def count(self):
        return len(self.identities)

    def heads(self, value):
        return value.reshape(-1, self.count, self.dimension)

    def read(self, board, query):
        query = F.normalize(self.heads(self.query(query)), dim=-1)
        recalled = torch.einsum("bhi,bhij->bhj", query, board).flatten(1)
        if (
            not self.training
            and self.count == 2
            and torch.equal(self.readout, torch.eye(WIDTH))
        ):
            return recalled
        return recalled @ self.readout

    def forward(self, board, draft, minutes):
        size = self.count * self.dimension
        key, target, forget, rate = self.write(draft).split(
            (size, size, size, self.count), -1
        )
        alpha = (-F.softplus(self.heads(forget)) * minutes[:, None, None] / 15).exp()
        return self.update(
            board,
            F.normalize(self.heads(key), dim=-1),
            self.heads(target).tanh(),
            alpha,
            rate.sigmoid(),
        )

    def resize(self, indices):
        """Copy heads and their write/read maps; split a donor's read mass."""
        old_count, d = self.count, self.dimension
        old_query, old_write = self.query, self.write
        query = nn.Linear(WIDTH, len(indices) * d).to(old_query.weight)
        write = nn.Linear(WIDTH, len(indices) * (3 * d + 1)).to(old_write.weight)
        coordinates = [i * d + j for i in indices for j in range(d)]
        write_rows = [
            block * old_count * d + j for block in range(3) for j in coordinates
        ]
        write_rows += [3 * old_count * d + i for i in indices]
        readout = self.readout.detach()[coordinates].clone()
        with torch.no_grad():
            query.weight.copy_(old_query.weight[coordinates])
            query.bias.copy_(old_query.bias[coordinates])
            write.weight.copy_(old_write.weight[write_rows])
            write.bias.copy_(old_write.bias[write_rows])
            for pos, source in enumerate(indices):
                readout[pos * d : (pos + 1) * d] /= indices.count(source)
        old = self.identities
        identities, used = [], set()
        for source in indices:
            identity = old[source]
            if identity in used:
                identity = self.next_identity
                self.next_identity += 1
                self.births[identity] = old[source]
            identities.append(identity)
            used.add(identity)
        self.identities = identities
        self.births = {k: v for k, v in self.births.items() if k in identities}
        self.query, self.write = query, write
        self.readout = nn.Parameter(readout)

    def metadata(self):
        return dict(
            identities=self.identities,
            next_identity=self.next_identity,
            births=self.births,
        )

    def restore_shape(self, data):
        while self.count < len(data["identities"]):
            self.resize([*range(self.count), 0])
        if self.count > len(data["identities"]):
            self.resize(list(range(len(data["identities"]))))
        self.identities = list(data["identities"])
        self.next_identity = data["next_identity"]
        self.births = deepcopy(data["births"])

    def align_memory(self, memory, previous):
        result = memory.new_zeros(
            len(memory), self.count, self.dimension, self.dimension
        )
        for dest, identity in enumerate(self.identities):
            source = (
                identity
                if identity in previous.identities
                else self.births.get(identity)
            )
            if source in previous.identities:
                result[:, dest] = memory[:, previous.identities.index(source)]
        return result
