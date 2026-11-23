"""Liquid expert consensus with delayed answers and verified updates."""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liquid-workflow-v3"))
from blackboard import DeltaAttention, HEADS, LiquidODE, WIDTH


class ExpertLinear(nn.Module):
    """Separate parameters for each expert, evaluated in one batch."""

    def __init__(self, experts, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(experts, outputs))
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, x, experts=None):
        if experts is None:
            return torch.einsum("bei,eio->beo", x, self.weight) + self.bias
        return torch.einsum("ni,nio->no", x, self.weight[experts]) + self.bias[experts]


@dataclass(frozen=True)
class State:
    cells: tuple
    memory: torch.Tensor
    last: torch.Tensor
    elapsed: torch.Tensor

    def detached(self):
        return State(tuple(cell.detach() for cell in self.cells), self.memory.detach(), self.last, self.elapsed)


@dataclass(frozen=True)
class Round:
    cells: tuple
    margins: torch.Tensor
    falsifiers: torch.Tensor
    process: torch.Tensor
    drive: torch.Tensor
    available: torch.Tensor
    acted: torch.Tensor
    interest: torch.Tensor
    evidence: torch.Tensor
    confidence: torch.Tensor
    choices: torch.Tensor
    speaks: torch.Tensor
    challenged: torch.Tensor
    votes: torch.Tensor
    choice: torch.Tensor
    accepted: torch.Tensor
    active: torch.Tensor


@dataclass(frozen=True)
class Decision:
    state: State
    answer: torch.Tensor
    rounds: tuple
    committed: torch.Tensor


class Board(nn.Module):
    def __init__(self, features=10, experts=5, rounds=4, depth=3):
        super().__init__()
        if experts < 5 or depth < 3 or rounds < 1:
            raise ValueError("At least five experts, three liquid layers and one round are required")
        self.features, self.experts, self.round_limit = features, experts, rounds
        self.majority = experts // 2 + 1
        inputs = 2 * features + WIDTH + 3
        self.specialty = ExpertLinear(experts, inputs + WIDTH, 2 * WIDTH)
        self.subscription = ExpertLinear(experts, 2 * WIDTH, 1)
        self.input = ExpertLinear(experts, inputs, 2 * WIDTH)
        self.mixers = nn.ModuleList(ExpertLinear(experts, 3 * WIDTH, 2 * WIDTH) for _ in range(depth))
        self.cells = nn.ModuleList(LiquidODE(2 * WIDTH, experts, substeps=2) for _ in range(depth))
        self.outputs = nn.ModuleList(ExpertLinear(experts, 3 * WIDTH, 1) for _ in range(depth))
        self.falsifiers = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * features, 2 * WIDTH), nn.SiLU(), nn.Linear(2 * WIDTH, 1))
            for _ in range(depth)
        )
        self.process = ExpertLinear(experts, 3 * WIDTH, 1)
        self.drive = ExpertLinear(experts, 3 * WIDTH, 1)
        self.query = nn.Linear(2 * features, WIDTH)
        self.delta = DeltaAttention()

    def start(self, x):
        return State(
            tuple(x.new_zeros(len(x), self.experts, WIDTH) for _ in self.cells),
            x.new_zeros(len(x), HEADS, WIDTH // HEADS, WIDTH // HEADS),
            torch.zeros(len(x), dtype=torch.long, device=x.device),
            x.new_zeros(len(x), self.experts),
        )

    def stages(self, x, minutes, before, feedback, active, *, assess=False):
        mask = x.isfinite()
        values = torch.where(mask, x, 0.)
        packed = torch.cat((values, mask.to(x.dtype)), -1)
        recall = self.delta.read(before.memory, self.query(packed))
        full_input = torch.cat((packed, recall), -1)[:, None].expand(-1, self.experts, -1)
        context = torch.cat((full_input, feedback), -1)
        local = self.specialty(torch.cat((context, before.cells[-1]), -1))
        interest = self.subscription(F.silu(local)).squeeze(-1)
        if assess:
            acts = torch.ones_like(interest, dtype=torch.bool)
        elif self.training:
            # Each expert explores independently. A row can attract zero, some or all.
            acts = torch.rand_like(interest).lt(.12 + .76 * interest.detach().sigmoid())
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
            incoming = F.silu(self.mixers[index](torch.cat((encoded, previous), -1), experts))
            previous = cell(incoming, before.cells[index][rows, experts], times, experts)
            features = torch.cat((encoded, previous), -1)
            # The next liquid layer adjusts the last prediction.
            running = running + self.outputs[index](features, experts).squeeze(-1)
            margins.append(dense.index_put((rows, experts), running))
            cells.append(before.cells[index].index_put((rows, experts), previous))
            falsifiers.append(self.falsifiers[index](packed).squeeze(-1))
        process = dense.index_put((rows, experts), self.process(features, experts).squeeze(-1))
        drive = dense.index_put((rows, experts), self.drive(features, experts).squeeze(-1))
        return (tuple(cells), torch.stack(margins, -1), torch.stack(falsifiers, -1),
                process, drive, available, mask.float().mean(-1), acts, interest)

    def round(self, x, minutes, before, feedback, active, checks, *, assess=False):
        cells, margins, falsifiers, process, drive, available, evidence, acted, interest = self.stages(
            x, minutes, before, feedback, active, assess=assess
        )
        final = margins[..., -1]
        choices = final.ge(0)
        confidence = final.abs().sigmoid()
        speaks = acted & (confidence + .15 * drive.sigmoid()).ge(.60)
        counter = falsifiers[:, None, :]
        strong = counter.abs().ge(1.4)
        contradicts = margins.ge(0) != counter.ge(0)
        challenged_layers = (strong & contradicts).sum(-1)
        challenged = (challenged_layers.ge(2) & process.sigmoid().lt(.8)) if checks else torch.zeros_like(speaks)
        challenged = challenged & acted
        eligible = speaks & ~challenged
        votes = eligible.unsqueeze(-1) & F.one_hot(choices.long(), 2).bool()
        count = votes.sum(1)
        confidence_sum = (votes * (confidence - .5).unsqueeze(-1)).sum(1)
        choice = (count + confidence_sum / self.experts).argmax(-1)
        tied = count[:, 0].eq(count[:, 1]) & confidence_sum[:, 0].eq(confidence_sum[:, 1])
        # Exhaustion still has an answer. With observations, use the strongest aggregate evidence.
        fallback = torch.where(acted.any(-1), final.sum(-1).ge(0).long(), before.last)
        choice = torch.where(tied, fallback, choice)
        accepted = count.max(-1).values.ge(self.majority)
        return Round(cells, margins, falsifiers, process, drive, available, acted, interest, evidence, confidence,
                     choices, speaks, challenged, votes, choice, accepted, active)

    def forward(self, x, minutes, before=None, *, checks=True):
        before = self.start(x) if before is None else before
        active = torch.ones(len(x), device=x.device, dtype=torch.bool)
        feedback = x.new_zeros(len(x), self.experts, 3)
        answer, last = before.last, before.last
        cells, memory = before.cells, before.memory
        elapsed = before.elapsed + minutes[:, None]
        rounds = []
        for number in range(self.round_limit):
            result = self.round(x, minutes, before, feedback, active, checks)
            rounds.append(result)
            answer = torch.where(active, result.choice, answer)
            accepts = active & result.accepted
            support = result.votes.gather(2, result.choice[:, None, None].expand(-1, self.experts, 1)).squeeze(-1)
            last = torch.where(accepts, result.choice, last)
            changed = accepts[:, None] & result.acted
            cells = tuple(torch.where(changed.unsqueeze(-1), trial, old)
                          for trial, old in zip(result.cells, cells, strict=True))
            elapsed = torch.where(changed, 0., elapsed)
            summary = (result.cells[-1] * support.unsqueeze(-1)).sum(1) / support.sum(1, keepdim=True).clamp_min(1)
            memory = torch.where(accepts[:, None, None, None], self.delta(before.memory, summary, minutes), memory)
            active = active & ~accepts
            if not active.any():
                break
            direction = (2 * result.choice - 1).to(x.dtype)
            feedback = torch.stack((result.challenged.to(x.dtype) - .5 * result.acted.to(x.dtype),
                                    direction[:, None].expand_as(result.confidence),
                                    torch.full_like(result.confidence, (number + 1) / self.round_limit)), -1)
        return Decision(State(cells, memory, last, elapsed), answer, tuple(rounds), ~active)

    def expert_parameters(self):
        for group in (self.specialty, self.subscription, self.input, self.mixers, self.cells, self.outputs, self.process, self.drive):
            yield from group.parameters()


def measured_reward(rounds, truth):
    """The measured answer, never the board or verifier, grades each proposal."""
    scores = []
    for index, item in enumerate(rounds):
        correct = item.choices.eq(truth[:, None])
        reward = torch.where(correct, .90, -1.10 * 1.25**index)
        clean = item.accepted & ~(item.challenged & item.speaks).any(-1)
        reward = reward + .15 * (clean[:, None] & correct)
        scores.append(torch.where(item.active[:, None] & item.speaks, reward, 0.))
    return torch.stack(scores).sum(0)


def offline_loss(result, target, class_weight):
    losses = []
    for index, item in enumerate(result.rounds):
        truth = target[:, None, None].expand_as(item.margins).float()
        by_stage = F.binary_cross_entropy_with_logits(item.margins, truth, reduction="none")
        error = (truth - item.margins.sigmoid()).abs().detach()
        difficulty = F.pad(error[..., :-1], (1, 0))
        # Boosting principle: later stages spend more gradient on remaining errors.
        boosted = (by_stage * (1 + 2 * difficulty)).mean(-1)
        correct = item.choices.eq(target[:, None]).float()
        process = F.binary_cross_entropy_with_logits(item.process, correct, reduction="none")
        payoff = torch.where(correct.bool(), .90, -1.10 * 1.25**index)
        attempt = ((item.confidence.detach() + .15 * item.drive.sigmoid() - .60) * 12).sigmoid()
        valid = (item.active & item.available) * class_weight[target]
        independent = F.binary_cross_entropy_with_logits(item.falsifiers, target[:, None].expand_as(item.falsifiers).float(), reduction="none")
        issued = item.acted * valid[:, None]
        expertise = boosted + .12 * process - .10 * attempt * payoff
        subscription = F.binary_cross_entropy_with_logits(item.interest, correct, reduction="none")
        acting_loss = ((expertise + .25 * subscription) * issued).sum() / issued.sum().clamp_min(1)
        checker_loss = (independent.mean(1) * valid).sum() / valid.sum().clamp_min(1)
        # A cost for frequent activation, and a small floor for learning participation.
        probability = item.interest.sigmoid()
        floor = F.relu(.35 - (probability * valid[:, None]).sum(0) / valid.sum().clamp_min(1))
        activity_cost = (probability.mean(1) * valid).sum() / valid.sum().clamp_min(1)
        losses.append(acting_loss + .18 * checker_loss + .06 * activity_cost + .15 * floor.square().mean())
    return torch.stack(losses).mean()
