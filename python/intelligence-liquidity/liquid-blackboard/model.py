"""Liquid experts, delta blackboard, liquid objections, and outcome rewards."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 8
DEPTH = 3
EXPERTS = 3


class Conductance(nn.Module):
    """One bounded, semi-implicit liquid update with positive conductances."""

    def __init__(self, inputs, width, experts=1):
        super().__init__()
        def connections(sources):
            shape = (experts, sources, width)
            return nn.Parameter(torch.stack((
                torch.full(shape, -2.), torch.ones(shape),
                torch.zeros(shape), torch.randn(shape),
            ), -1))
        self.sensor = connections(inputs)
        self.recurrent = connections(width)
        self.leak = nn.Parameter(torch.full((experts, width), -2.))
        self.rest = nn.Parameter(torch.zeros(experts, width))
        self.capacitance = nn.Parameter(torch.zeros(experts, width))

    @staticmethod
    def pull(source, edges):
        strength, slope, threshold, destination = edges.unbind(-1)
        active = F.softplus(strength) * torch.sigmoid(slope * (source.unsqueeze(-1) - threshold))
        return active.sum(-2), (active * destination.tanh()).sum(-2)

    def forward(self, x, previous, minutes):
        sensory, pull = self.pull(x, self.sensor)
        leak = F.softplus(self.leak) + 1e-5
        capacitance = F.softplus(self.capacitance) + 1e-5
        dt = minutes[:, None, None] / 10
        state = previous
        for _ in range(2):
            recurring, recurring_pull = self.pull(state, self.recurrent)
            state = ((capacitance * state + dt * (leak * self.rest.tanh() + pull + recurring_pull))
                     / (capacitance + dt * (leak + sensory + recurring)))
        return state


class DeltaAttention(nn.Module):
    """Each key channel learns its own decay. Writes subtract the old prediction."""

    def __init__(self, width):
        super().__init__()
        self.signals = nn.Linear(width, width * 4 + 1)
        self.minimum_decay = nn.Parameter(torch.full((width,), -3.0))

    def forward(self, x, board, elapsed):
        key, query, value, decay, write = self.signals(x).split((WIDTH,) * 4 + (1,), -1)
        key = F.normalize(key, dim=-1)
        query = F.normalize(query, dim=-1)
        minutes = elapsed[:, None] / 5
        keep = (-F.softplus(self.minimum_decay + decay) * minutes).exp()
        prior = board * keep.unsqueeze(-1)
        prediction = torch.bmm(key.unsqueeze(1), prior).squeeze(1)
        correction = (value.tanh() - prediction) * write.sigmoid()
        stored = prior + key.unsqueeze(-1) * correction.unsqueeze(1)
        return stored, torch.bmm(query.unsqueeze(1), stored).squeeze(1)


@dataclass
class Memory:
    experts: tuple
    falsifiers: tuple
    board: torch.Tensor
    last: torch.Tensor
    previous_forecast: torch.Tensor

    def detach(self):
        return Memory(tuple(x.detach() for x in self.experts),
                      tuple(x.detach() for x in self.falsifiers), self.board.detach(),
                      self.last.detach(), self.previous_forecast.detach())


@dataclass
class Step:
    memory: Memory
    occupied: torch.Tensor
    q: torch.Tensor
    routes: torch.Tensor
    objections: torch.Tensor
    process: torch.Tensor
    candidate_forecasts: torch.Tensor
    corrections: torch.Tensor


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = Conductance(4 + WIDTH * 2, WIDTH, EXPERTS)
        self.router = nn.Linear(4 + WIDTH * 2, EXPERTS)
        self.falsifier = Conductance(4 + WIDTH * 3, WIDTH)
        self.correct = nn.Linear(WIDTH, WIDTH)
        self.gate = nn.Linear(WIDTH, 1)
        self.process = nn.Linear(WIDTH * 2, 1)
        self.forecast = nn.Linear(WIDTH, 2)


class Blackboard(nn.Module):
    def __init__(self, full=True, use_delta=True, use_falsifier=True):
        super().__init__()
        self.full = full
        self.use_delta = use_delta
        self.use_falsifier = use_falsifier
        self.adapter_scale = nn.Parameter(torch.ones(2))
        self.adapter_offset = nn.Parameter(torch.zeros(2))
        self.input = nn.Linear(4, WIDTH)
        self.layers = nn.ModuleList(Layer() for _ in range(DEPTH)) if full else nn.ModuleList()
        self.plain = Conductance(4, WIDTH) if not full else None
        self.attention = DeltaAttention(WIDTH)
        self.occupancy = nn.Linear(WIDTH, 1)
        self.actions = nn.Linear(WIDTH, 2)
        self.forecast = nn.Linear(WIDTH, 2)

    def empty(self, x):
        n = len(x)
        return Memory(tuple(x.new_zeros(n, EXPERTS if self.full else 1, WIDTH)
                            for _ in range(len(self.layers) or 1)),
                      tuple(x.new_zeros(n, 1, WIDTH) for _ in self.layers),
                      x.new_zeros(n, WIDTH, WIDTH), x, x)

    def step(self, x, minutes, memory):
        first = memory is None
        old = self.empty(x) if first else memory
        normalized = x * self.adapter_scale + self.adapter_offset
        features = torch.cat((normalized, (x - old.last).clamp(-3, 3)), -1)
        incumbent = self.input(features).tanh()
        board = old.board
        read = board.mean(1)
        contradiction = (x - old.previous_forecast).clamp(-3, 3)
        states, reviews, routes, objections, process, alternatives, corrections = [], [], [], [], [], [], []
        if self.plain is not None:
            states.append(self.plain(features.unsqueeze(1), old.experts[0], minutes))
            incumbent = states[0].squeeze(1)
        for index, layer in enumerate(self.layers):
            signal = torch.cat((features, incumbent, read), -1)
            weights = .85 * layer.router(signal).softmax(-1) + .15 / EXPERTS
            # All experts train. Supply different views to establish distinct roles.
            expert_input = signal.unsqueeze(1).expand(-1, EXPERTS, -1)
            mask = signal.new_ones(EXPERTS, signal.shape[-1])
            mask[0, 2:4] = 0  # levels
            mask[1, :2] = 0   # change
            state = layer.expert(expert_input * mask, old.experts[index], minutes)
            proposal = (state * weights.unsqueeze(-1)).sum(1)
            evidence = torch.cat((contradiction, features[:, 2:], incumbent, proposal, read), -1)
            checked = layer.falsifier(evidence.unsqueeze(1), old.falsifiers[index], minutes)
            objection = layer.gate(checked.squeeze(1)).sigmoid()
            repaired = (incumbent + layer.correct(checked.squeeze(1)).tanh()) / 2
            predicted_gain = layer.process(torch.cat((checked.squeeze(1), incumbent), -1)).squeeze(-1)
            accepts = (predicted_gain > 0).unsqueeze(-1)
            gate = objection * accepts if self.use_falsifier else torch.zeros_like(objection)
            # This layer can revise the incoming proposal before forwarding it.
            incumbent = proposal.lerp(repaired, gate)
            if self.use_delta:
                board, read = self.attention(incumbent, board, minutes / len(self.layers))
                incumbent = (incumbent + read).tanh()
            states.append(state)
            reviews.append(checked)
            routes.append(weights)
            objections.append(objection.squeeze(-1))
            process.append(predicted_gain)
            alternatives.append(torch.stack((layer.forecast(proposal), layer.forecast(repaired)), 1))
            corrections.append(gate.squeeze(-1))
        forecast = x + self.forecast(incumbent)
        memory = Memory(tuple(states), tuple(reviews), board, x, forecast)
        return Step(memory, self.occupancy(incumbent).squeeze(-1), self.actions(incumbent),
                    torch.stack(routes, 1) if routes else x.new_empty(len(x), 0, EXPERTS),
                    torch.stack(objections, 1) if objections else x.new_empty(len(x), 0),
                    torch.stack(process, 1) if process else x.new_empty(len(x), 0),
                    torch.stack(alternatives, 1) if alternatives else x.new_empty(len(x), 0, 2, 2),
                    torch.stack(corrections, 1) if corrections else x.new_empty(len(x), 0))


def sensor_loss(current, next_reading, step):
    factual = F.smooth_l1_loss(step.memory.previous_forecast, next_reading)
    if not step.process.numel():
        return factual
    observed_delta = (next_reading - current)[:, None, None, :]
    errors = (step.candidate_forecasts - observed_delta).abs().mean(-1)
    verified_gain = (errors[:, :, 0] - errors[:, :, 1]).detach()
    learning = F.smooth_l1_loss(step.process, verified_gain)
    objection = F.binary_cross_entropy(step.objections, (verified_gain > 0).float())
    specialization = ((step.routes.mean(0) - 1 / EXPERTS) ** 2).sum(-1).mean()
    return factual + .1 * errors.mean() + .1 * learning + .05 * objection + .1 * specialization


def logged_q_loss(q, next_q, action, outcome, continuation, process_gain):
    """Conservative Q learning. Only the logged action gets an outcome target."""
    taken = q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    target = outcome + .05 * process_gain.clamp(-1, 1) + .90 * continuation * next_q.detach().max(-1).values
    return F.smooth_l1_loss(taken, target) + .1 * (q.logsumexp(-1) - taken).mean()
