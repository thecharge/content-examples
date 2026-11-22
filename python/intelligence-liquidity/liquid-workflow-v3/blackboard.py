"""Parallel liquid proposals, explicit objections and transactional commits."""

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

PRIOR = Path(__file__).resolve().parents[1] / "liquid-workflow-v2"
spec = spec_from_file_location("liquid_primitives", PRIOR / "workflow.py")
primitives = module_from_spec(spec)
sys.modules[spec.name] = primitives
spec.loader.exec_module(primitives)
DeltaAttention = primitives.DeltaAttention
LiquidODE = primitives.LiquidODE
WIDTH, HEADS = primitives.WIDTH, primitives.HEADS

EXPERTS = 3
DISCOUNT = .95
OCCUPANCY = ("co2eq", "voc", "temperature", "humidity", "light",
             "co2eq_change", "voc_change", "temperature_change", "humidity_change", "light_change")
HEATING = ("indoor", "outdoor", "previous_action", "previous_energy", "previous_discomfort")
VIEWS = {
    "occupancy": ((0, 1, 5, 6), (2, 3, 7, 8), (4, 8, 9)),
    "heat": ((0, 1, 4), (0, 2, 3), (1, 2, 3, 4)),
}


@dataclass(frozen=True)
class Board:
    states: torch.Tensor
    review_states: torch.Tensor
    memory: torch.Tensor
    last_decision: torch.Tensor
    commits: torch.Tensor
    evidence: torch.Tensor
    votes: torch.Tensor

    def detach(self):
        return Board(self.states.detach(), self.review_states.detach(), self.memory.detach(),
                     self.last_decision, self.commits,
                     self.evidence.detach(), self.votes)


@dataclass(frozen=True)
class Proposals:
    states: torch.Tensor
    values: torch.Tensor
    forecasts: torch.Tensor


@dataclass(frozen=True)
class Round:
    proposals: Proposals
    votes: torch.Tensor
    validity: torch.Tensor
    counter_values: torch.Tensor
    contradicts: torch.Tensor
    available: torch.Tensor
    decision: torch.Tensor
    aggregate: torch.Tensor
    consensus_objection: torch.Tensor
    quorum: torch.Tensor


@dataclass(frozen=True)
class Decision:
    board: Board
    first: Round
    final: Round
    revised: torch.Tensor
    accepted: torch.Tensor
    independent: torch.Tensor
    evidence: torch.Tensor
    forecast: torch.Tensor


class Blackboard(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.sensors = OCCUPANCY if task == "occupancy" else HEATING
        views = VIEWS[task]
        self.register_buffer("views", torch.tensor([[i in view for i in range(len(self.sensors))]
                                                    for view in views]))
        self.propose = LiquidODE(2 * len(self.sensors) + WIDTH + 2, EXPERTS)
        self.value = nn.Parameter(torch.empty(EXPERTS, WIDTH, 2))
        nn.init.xavier_uniform_(self.value)
        self.bias = nn.Parameter(torch.zeros(EXPERTS, 2))
        self.instant = nn.Parameter(torch.zeros(EXPERTS, 2 * len(self.sensors), 2))
        self.forecast = nn.Linear(WIDTH, len(self.sensors))
        nn.init.zeros_(self.forecast.weight)
        nn.init.zeros_(self.forecast.bias)
        self.query = nn.Linear(2 * len(self.sensors), WIDTH)
        self.attention = DeltaAttention()
        self.evidence_filter = LiquidODE(2 * len(self.sensors), EXPERTS + 1)
        self.independent = nn.Linear(WIDTH, 2)
        self.reviewer = nn.Sequential(nn.Linear(2 * len(self.sensors) + 4, 12), nn.Tanh(), nn.Linear(12, 1))
        self.register_buffer("proposal_offset", torch.zeros(EXPERTS))
        self.register_buffer("review_offset", torch.zeros(EXPERTS + 1))
        self.no_evidence = .05

    def start(self, x):
        batch = len(x)
        return Board(x.new_zeros(batch, EXPERTS, WIDTH), x.new_zeros(batch, EXPERTS + 1, WIDTH),
                     x.new_zeros(batch, HEADS, WIDTH // HEADS, WIDTH // HEADS),
                     torch.full((batch,), -1, device=x.device, dtype=torch.long),
                     torch.zeros(batch, device=x.device, dtype=torch.long),
                     torch.full_like(x, float("nan")),
                     torch.full((batch, EXPERTS), -1, device=x.device, dtype=torch.long))

    def candidates(self, values, mask, recall, before, minutes, rows, experts, objection, signal):
        available = mask[rows] & self.views[experts]
        features = torch.cat((values[rows] * available, available.to(values.dtype),
                              recall[rows], objection[:, None], signal[:, None]), -1)
        state = self.propose(features, before.states[rows, experts], minutes[rows], experts)
        votes = (torch.einsum("bw,bwk->bk", state, self.value[experts]) + self.bias[experts]
                 + torch.einsum("bw,bwk->bk", features[:, :2 * len(self.sensors)], self.instant[experts]))
        votes = votes - F.pad(self.proposal_offset[experts, None], (1, 0))
        return Proposals(state, votes, self.forecast(state))

    def review(self, proposals, values, mask, independent, counter_values):
        # Each proposal's critic receives raw evidence outside that expert's view.
        other = mask[:, None] & ~self.views
        counter_input = torch.cat((values[:, None] * other, other.to(values.dtype)), -1)
        context = torch.cat((counter_input, proposals.values, counter_values), -1)
        validity = self.reviewer(context).squeeze(-1)
        proposal_vote = proposals.values.argmax(-1)
        independent_vote = independent.argmax(-1)
        scale = .25 if self.task == "heat" else 1.
        proof_strength = (independent / scale).softmax(-1).amax(-1)
        available = (mask[:, None] & self.views).any(-1)
        counter_strength = (counter_values / scale).softmax(-1).amax(-1)
        contrary = ((proposal_vote != counter_values.argmax(-1)) & (counter_strength >= .85)
                    & other.any(-1))
        eligible = available & (~contrary | (validity.sigmoid() >= .8))
        weights = validity.sigmoid().clamp_min(self.no_evidence) * eligible
        aggregate = (proposals.values * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(.01)
        decision = aggregate.argmax(-1)
        agrees = eligible & (proposal_vote == decision[:, None])
        quorum = agrees.sum(-1) >= 2
        # A separate full-evidence prediction can challenge the collective decision.
        # A supermajority of strongly supported proposals can answer that objection.
        consensus_objection = ((decision != independent_vote) & (proof_strength >= .9)
                               & ~((agrees & (validity.sigmoid() >= .8)).sum(-1) >= 3))
        if self.task == "heat":
            # Below the comfort floor, an off proposal requires reconsideration.
            below_floor = mask[:, 0] & (values[:, 0] < -.5)
            contrary = contrary | (below_floor[:, None] & (proposal_vote == 0))
            consensus_objection = consensus_objection | (below_floor & (decision == 0))
        return Round(proposals, proposal_vote, validity, counter_values, contrary, available, decision, aggregate,
                     consensus_objection, quorum)

    def forward(self, evidence, minutes, before=None, *, force_commit=False):
        mask = evidence.isfinite()
        values = torch.where(mask, evidence, 0.)
        before = self.start(values) if before is None else before
        inputs = torch.cat((values, mask.to(values.dtype)), -1)
        recall = self.attention.read(before.memory, self.query(inputs))
        review_mask = torch.cat((mask[:, None] & ~self.views, mask[:, None]), dim=1)
        review_inputs = torch.cat((values[:, None] * review_mask, review_mask.to(values.dtype)), -1)
        critics = torch.arange(EXPERTS + 1, device=values.device).repeat(len(values))
        review_states = self.evidence_filter(review_inputs.flatten(0, 1), before.review_states.flatten(0, 1),
                                            minutes.repeat_interleave(EXPERTS + 1), critics)
        review_states = review_states.reshape(len(values), EXPERTS + 1, WIDTH)
        full_reviews = self.independent(review_states) - F.pad(self.review_offset[:, None], (1, 0))
        independent, counter_values = full_reviews[:, -1], full_reviews[:, :-1]
        rows = torch.arange(len(values), device=values.device).repeat_interleave(EXPERTS)
        experts = torch.arange(EXPERTS, device=values.device).repeat(len(values))
        zero = values.new_zeros(len(rows))
        proposed = self.candidates(values, mask, recall, before, minutes, rows, experts, zero, zero)
        proposals = Proposals(*(p.reshape(len(values), EXPERTS, -1) for p in
                                (proposed.states, proposed.values, proposed.forecasts)))
        first = self.review(proposals, values, mask, independent, counter_values)
        competing = first.available & (first.votes != first.decision[:, None])
        counter_backs_alternative = first.counter_values.argmax(-1) != first.votes
        contested = competing & counter_backs_alternative & review_mask[:, :-1].any(-1)
        disputed = first.available & (first.contradicts | contested | (first.consensus_objection[:, None]
                                     & (first.votes == first.decision[:, None])))
        revised_rows, revised_experts = disputed.nonzero(as_tuple=True)
        if len(revised_rows):
            counter = torch.where(first.consensus_objection[revised_rows, None],
                                  independent[revised_rows], first.counter_values[revised_rows, revised_experts])
            evidence_direction = counter.softmax(-1)[:, 1] * 2 - 1
            if self.task == "heat":
                cold = mask[revised_rows, 0] & (values[revised_rows, 0] < -.5)
                evidence_direction = torch.where(cold, 1., evidence_direction)
            revised = self.candidates(values, mask, recall, before, minutes, revised_rows,
                                      revised_experts, torch.ones_like(evidence_direction), evidence_direction)
            proposals = Proposals(*(old.index_put((revised_rows, revised_experts), new) for old, new in
                                    zip((proposals.states, proposals.values, proposals.forecasts),
                                        (revised.states, revised.values, revised.forecasts), strict=True)))
        final = self.review(proposals, values, mask, independent, counter_values)
        accepted = final.quorum & ~final.consensus_objection
        eligible = final.available & (~final.contradicts | (final.validity.sigmoid() >= .8))
        agrees = eligible & (final.votes == final.decision[:, None])
        commits = final.available if force_commit else agrees & accepted[:, None]
        weights = final.validity.sigmoid() * commits
        draft = (proposals.states * weights[:, :, None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(.01)
        publish = commits.any(-1)
        next_memory = self.attention(before.memory, draft, minutes)
        state = torch.where(commits[:, :, None], proposals.states, before.states)
        reviewed = torch.where(publish[:, None, None] & review_mask.any(-1, keepdim=True),
                               review_states, before.review_states)
        board = Board(state, reviewed, torch.where(publish[:, None, None, None], next_memory, before.memory),
                      torch.where(publish, final.decision, before.last_decision), before.commits + publish,
                      torch.where(publish[:, None], evidence, before.evidence),
                      torch.where(publish[:, None], final.votes, before.votes))
        forecasts = (proposals.forecasts * final.available[:, :, None]).sum(1)
        forecasts = forecasts / final.available.sum(1, keepdim=True).clamp_min(1)
        return Decision(board, first, final, disputed, accepted, independent, evidence, values + forecasts)


def sensor_loss(result, observed):
    valid = observed.isfinite()
    safe = torch.where(valid, observed, result.forecast.detach())
    return (F.smooth_l1_loss(result.forecast, safe, reduction="none") * valid).sum() / valid.sum().clamp_min(1)


def classification_loss(result, labels, weights):
    loss = (F.cross_entropy(result.final.aggregate, labels, weight=weights)
            + .25 * F.cross_entropy(result.independent, labels, weight=weights))
    for round_ in (result.first, result.final):
        expected = labels[:, None].expand_as(round_.votes)
        relevant = round_.available
        correct = (round_.votes == expected).to(round_.validity.dtype)
        forecast = F.cross_entropy(round_.proposals.values.transpose(1, 2), expected, reduction="none")
        critic = F.binary_cross_entropy_with_logits(round_.validity, correct, reduction="none")
        counter = F.cross_entropy(round_.counter_values.transpose(1, 2), expected, reduction="none")
        scaled = relevant * weights[labels, None]
        loss += ((.25 * forecast + .15 * critic + .1 * counter) * scaled).sum() / scaled.sum().clamp_min(1)
    return loss


def action_loss(result, action, target):
    own = result.final.proposals.values
    options = action[:, None, None].expand(-1, EXPERTS, 1)
    logged = own.gather(-1, options).squeeze(-1)
    combined = result.final.aggregate.gather(1, action[:, None]).squeeze(1)
    critic_return = result.independent.gather(1, action[:, None]).squeeze(1)
    # Review the estimate against the measured bootstrapped return for the logged action.
    reliable = ((logged.detach() - target[:, None]).abs() < .4).to(logged.dtype)
    critic = F.binary_cross_entropy_with_logits(result.final.validity, reliable)
    dissent_return = result.final.counter_values.gather(-1, options).squeeze(-1)
    conservative = own.logsumexp(-1) - logged
    return (F.smooth_l1_loss(combined, target) + .4 * F.smooth_l1_loss(logged, target[:, None].expand_as(logged))
            + .25 * F.smooth_l1_loss(critic_return, target)
            + .1 * F.smooth_l1_loss(dissent_return, target[:, None].expand_as(dissent_return))
            + .1 * conservative.mean() + .1 * critic)
