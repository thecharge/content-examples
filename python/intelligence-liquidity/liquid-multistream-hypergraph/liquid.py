"""Variable-width conductance LTC with an eight-coordinate public readout."""

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from reference import LiquidODE, WIDTH


class LiquidNeurons(LiquidODE):
    def __init__(self, source):
        super().__init__(
            source.strength.shape[1] - WIDTH, source.leak.shape[0], source.substeps
        )
        self.load_state_dict(source.state_dict())
        self.readout = nn.Parameter(
            torch.eye(WIDTH).expand(len(self.leak), -1, -1).clone()
        )
        self.units = list(range(WIDTH))
        self.next_unit = WIDTH
        self.births = {}

    def decode(self, state, experts=None):
        if (
            not self.training
            and len(self.units) == WIDTH
            and torch.equal(self.readout, torch.eye(WIDTH).expand_as(self.readout))
        ):
            return state
        if experts is None:
            return torch.einsum("bei,eio->beo", state, self.readout)
        return torch.einsum("bi,bio->bo", state, self.readout[experts])

    def split(self, unit):
        count = len(self.units)
        if not 0 <= unit < count:
            raise ValueError("Invalid liquid neuron")
        inputs = self.strength.shape[1] - count
        destinations = [*range(count), unit]
        rows = [*range(inputs + count), inputs + unit]
        with torch.no_grad():
            for name in ("strength", "slope", "center", "target"):
                previous = getattr(self, name).detach()
                new = previous[:, rows][:, :, destinations].clone()
                if name == "strength":
                    half = F.softplus(new[:, inputs + unit]) * 0.5
                    split = half + torch.log(-torch.expm1(-half))
                    new[:, inputs + unit] = split
                    new[:, -1] = split
                setattr(self, name, nn.Parameter(new))
            for name in ("leak", "rest"):
                setattr(
                    self,
                    name,
                    nn.Parameter(getattr(self, name).detach()[:, destinations].clone()),
                )
            readout = self.readout.detach()[:, destinations].clone()
            readout[:, unit] *= 0.5
            readout[:, -1] *= 0.5
            self.readout = nn.Parameter(readout)
        born = self.next_unit
        self.next_unit += 1
        self.births[born] = self.units[unit]
        self.units.append(born)

    def prune(self, unit):
        count = len(self.units)
        if not 0 <= unit < count or count == 1:
            raise ValueError("Invalid liquid neuron retirement")
        inputs = self.strength.shape[1] - count
        keep = [i for i in range(count) if i != unit]
        rows = [*range(inputs), *[inputs + i for i in keep]]
        for name in ("strength", "slope", "center", "target"):
            setattr(
                self,
                name,
                nn.Parameter(getattr(self, name).detach()[:, rows][:, :, keep].clone()),
            )
        for name in ("leak", "rest", "readout"):
            setattr(
                self, name, nn.Parameter(getattr(self, name).detach()[:, keep].clone())
            )
        self.units.pop(unit)
        self.births = {k: v for k, v in self.births.items() if k in self.units}

    def metadata(self):
        return dict(units=self.units, next_unit=self.next_unit, births=self.births)

    def restore_shape(self, item):
        while len(self.units) < len(item["units"]):
            self.split(0)
        while len(self.units) > len(item["units"]):
            self.prune(len(self.units) - 1)
        self.units = list(item["units"])
        self.next_unit = item["next_unit"]
        self.births = deepcopy(item["births"])
