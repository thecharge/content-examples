"""Sequence encoders and task-specific readouts."""
import torch
from torch import nn
from torch.nn import functional as F
from ncps.torch import LTC

UNITS = 8
SUBSTEPS = 6
OCC_MODELS = ("current", "gru", "library", "liquid", "attention", "attention_aux")
HEAT_MODELS = ("mean", "liquid", "attention")


class Conductance(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.capacitance = nn.Parameter(torch.zeros(UNITS))
        self.leak = nn.Parameter(torch.zeros(UNITS))
        self.bias = nn.Parameter(torch.zeros(UNITS))
        self.sensory = nn.Parameter(self.connections(features))
        self.recurrent = nn.Parameter(self.connections(UNITS))

    @staticmethod
    def connections(sources):
        shape = (sources, UNITS)
        return torch.stack((torch.full(shape, -2.), torch.ones(shape),
                            torch.zeros(shape), torch.randn(shape)), dim=-1)

    @staticmethod
    def conductance(source, connections):
        strength, slope, midpoint, reversal = connections.unbind(-1)
        active = F.softplus(strength) * torch.sigmoid(slope * (source.unsqueeze(-1) - midpoint))
        return active.sum(dim=-2), (active * reversal).sum(dim=-2)

    def forward(self, x, gaps):
        capacitance = F.softplus(self.capacitance) + 1e-6
        leak = F.softplus(self.leak) + 1e-6
        sensory, sensory_pull = self.conductance(x, self.sensory)
        state = x.new_zeros((len(x), UNITS))
        states = []
        for index, elapsed in enumerate(gaps.unbind(1)):
            dt = elapsed[:, None] / SUBSTEPS
            for _ in range(SUBSTEPS):
                recurrent, pull = self.conductance(state, self.recurrent)
                state = ((capacitance * state + dt * (leak * self.bias + sensory_pull[:, index] + pull))
                         / (capacitance + dt * (leak + sensory[:, index] + recurrent)))
            states.append(state)
        return torch.stack(states, dim=1)


class Network(nn.Module):
    def __init__(self, task, name):
        super().__init__()
        self.name = name
        self.task = task
        features = 3 if task == "occupancy" else 2
        if name == "current":
            self.encoder = nn.Sequential(nn.Linear(features + 1, UNITS), nn.ReLU())
        elif name == "gru":
            self.encoder = nn.GRU(features + 1, UNITS, batch_first=True)
        elif name == "library":
            self.encoder = LTC(features, UNITS, batch_first=True, ode_unfolds=SUBSTEPS)
        else:
            self.encoder = Conductance(features)
        self.attention = nn.Linear(UNITS, 1) if name.startswith("attention") else None
        self.auxiliary = nn.Linear(UNITS, features) if name == "attention_aux" else None
        self.readout = (nn.Linear(UNITS, 1) if task == "occupancy" else
                        nn.Sequential(nn.Linear(UNITS + 2, UNITS), nn.Tanh(), nn.Linear(UNITS, 3)))

    def forward(self, x, gaps, action=None):
        if self.name in ("current", "gru"):
            inputs = torch.cat((x, gaps.log().unsqueeze(-1)), dim=-1)
            states = self.encoder(inputs) if self.name == "current" else self.encoder(inputs)[0]
        elif self.name == "library":
            # The package sequence wrapper squeezes elapsed time across the batch.
            state = x.new_zeros((len(x), UNITS))
            sequence = []
            for reading, elapsed in zip(x.unbind(1), gaps.unbind(1)):
                output, state = self.encoder.rnn_cell(reading, state, elapsed[:, None])
                sequence.append(output)
            states = torch.stack(sequence, dim=1)
        else:
            states = self.encoder(x, gaps)
        weights = None
        context = states[:, -1]
        if self.attention is not None:
            weights = self.attention(states).squeeze(-1).softmax(dim=1)
            context = (weights.unsqueeze(-1) * states).sum(dim=1)
        next_reading = None if self.auxiliary is None else self.auxiliary(context)
        if self.task == "heat":
            context = torch.cat((context, F.one_hot(action, 2).to(x.dtype)), dim=1)
        return self.readout(context), weights, next_reading
