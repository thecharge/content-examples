"""Parallel liquid experts with a transactional decision board."""

from dataclasses import dataclass, fields
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "liquid-workflow-v3"))
from blackboard import DeltaAttention, HEADS, LiquidODE, WIDTH

VIEWS = {
    "occupancy": ((0, 1, 5, 6), (2, 3, 7, 8), (4, 9), (5, 6, 7, 8, 9), (0, 2, 3, 4)),
    "heat": ((0, 1), (0, 4), (1, 2, 3), (0, 2, 4), (2, 3, 4)),
}


@dataclass
class State:
    evidence: torch.Tensor
    judgment: torch.Tensor
    memory: torch.Tensor
    answer: torch.Tensor

    def detached(self):
        return State(*(getattr(self, item.name).detach() for item in fields(self)))


@dataclass
class Round:
    evidence_state: torch.Tensor
    judgment_state: torch.Tensor
    witness: torch.Tensor
    logits: torch.Tensor
    action_value: torch.Tensor
    confidence: torch.Tensor
    drive: torch.Tensor
    forecast: torch.Tensor
    process: torch.Tensor
    proposal: torch.Tensor
    available: torch.Tensor
    speaking: torch.Tensor
    witness_rejected: torch.Tensor
    vote_rejected: torch.Tensor
    supporters: torch.Tensor
    answer: torch.Tensor
    answer_confidence: torch.Tensor
    quorum: torch.Tensor
    active: torch.Tensor


@dataclass
class Decision:
    state: State
    rounds: list[Round]
    answer: torch.Tensor
    confidence: torch.Tensor
    committed: torch.Tensor
    forecast: torch.Tensor
    action_value: torch.Tensor


class Board(nn.Module):
    def __init__(self, task, views=None, max_rounds=4):
        super().__init__()
        views = VIEWS[task] if views is None else views
        self.width = 10 if task == "occupancy" else 5
        self.sensor_count = 5 if task == "occupancy" else 2
        self.count = len(views)
        self.quorum = self.count // 2 + 1
        if self.count < 5 or max_rounds < 1:
            raise ValueError("Use at least five experts and at least one round")
        if any(not view or any(i < 0 or i >= self.width for i in view) for view in views):
            raise ValueError("Every expert needs a valid sensor view")
        self.max_rounds = max_rounds
        self.task = task
        self.register_buffer("views", torch.tensor([[i in view for i in range(self.width)] for view in views]))
        self.register_buffer("fallback", torch.tensor(0))
        self.pack_width = 2 * self.width
        self.evidence = LiquidODE(self.pack_width + WIDTH + 3, self.count)
        self.judgment = LiquidODE(self.pack_width + 2 * WIDTH + 3, self.count)
        self.witness = nn.Linear(WIDTH, self.sensor_count)
        self.query = nn.Linear(self.pack_width, WIDTH)
        self.attention = DeltaAttention()
        self.instant = nn.Parameter(torch.zeros(self.count, self.pack_width + WIDTH, 2))
        self.values = nn.Linear(WIDTH, 2)
        self.drive = nn.Linear(WIDTH, 1)
        self.forecast = nn.Linear(WIDTH, self.sensor_count)
        self.counter = nn.Sequential(nn.Linear(self.pack_width, 16), nn.Tanh(), nn.Linear(16, 1))
        self.process = nn.Sequential(nn.Linear(self.pack_width + 4, 16), nn.Tanh(), nn.Linear(16, 1))

    def start(self, x):
        batch = len(x)
        return State(x.new_zeros(batch, self.count, WIDTH),
                     x.new_zeros(batch, self.count, WIDTH),
                     x.new_zeros(batch, HEADS, WIDTH // HEADS, WIDTH // HEADS),
                     self.fallback.expand(batch))

    def round(self, values, mask, minutes, before, memory, feedback, counter, mode, active):
        view = mask[:, None] & self.views
        other = mask[:, None] & ~self.views
        own = torch.cat((values[:, None] * view, view.to(values.dtype)), -1)
        independent = torch.cat((values[:, None] * other, other.to(values.dtype)), -1)
        expert = torch.arange(self.count, device=values.device).repeat(len(values))
        dt = minutes.repeat_interleave(self.count)
        recall = memory[:, None].expand(-1, self.count, -1)
        context = torch.cat((own, recall, feedback), -1)
        sensed = self.evidence(context.flatten(0, 1), before.evidence.flatten(0, 1), dt, expert)
        sensed = sensed.reshape(len(values), self.count, WIDTH)
        witness = self.witness(sensed)
        witness_mask = other[..., :self.sensor_count]
        residual = torch.where(witness_mask, (witness - values[:, None, :self.sensor_count]).abs(), 0.)
        witness_error = residual.sum(-1) / witness_mask.sum(-1).clamp_min(1)
        witness_rejected = witness_mask.any(-1) & witness_error.gt(1.0)
        retry = torch.cat((context, sensed), -1)
        state = self.judgment(retry.flatten(0, 1), before.judgment.flatten(0, 1), dt, expert)
        state = state.reshape_as(sensed)
        logits = torch.einsum("bew,ewv->bev", torch.cat((own, state), -1), self.instant)
        action_value = self.values(state)
        # In the heater task a claim is a choice over expected action returns.
        proposal_scores = action_value if self.task == "heat" else logits
        probabilities = proposal_scores.softmax(-1)
        confidence, proposal = probabilities.max(-1)
        drive = .25 * self.drive(state).squeeze(-1).sigmoid()
        available = view.any(-1)
        speaking = available & (2 * confidence - 1 + drive >= .20)
        learned_counter = self.counter(independent).squeeze(-1)
        counter_prob = learned_counter.sigmoid() if counter is None else counter
        vote_margin = proposal_scores[..., 1] - proposal_scores[..., 0]
        process_input = torch.cat((independent, torch.stack(
            (vote_margin, confidence, witness_error, feedback[..., 2]), -1)), -1)
        process = self.process(process_input).squeeze(-1)
        vote_rejected = (other.any(-1) & proposal.ne(counter_prob.ge(.5)) &
                         ((counter_prob - .5).abs() >= .25) & process.sigmoid().lt(.75))
        if mode == "off":
            witness_rejected = torch.zeros_like(witness_rejected)
            vote_rejected = torch.zeros_like(vote_rejected)

        eligible = speaking & ~witness_rejected & ~vote_rejected
        support = eligible.unsqueeze(-1) & F.one_hot(proposal, 2).bool()
        count = support.sum(1)
        confidence_sum = (support * (confidence - .5).unsqueeze(-1)).sum(1)
        # Support count wins. Confidence only breaks an equal-size split.
        choice = (count + confidence_sum / self.count).argmax(-1)
        tied = count[:, 0].eq(count[:, 1]) & confidence_sum[:, 0].eq(confidence_sum[:, 1])
        choice = torch.where(tied, before.answer, choice)
        supporters = support.gather(2, choice[:, None, None].expand(-1, self.count, 1)).squeeze(-1)
        board_confidence = .5 + (supporters * (confidence - .5)).sum(-1) / self.count
        quorum = supporters.sum(-1).ge(self.quorum)
        forecast = values[:, None, :self.sensor_count] + self.forecast(state)
        return Round(sensed, state, witness, proposal_scores, action_value, confidence, drive,
                     forecast, process, proposal, available, speaking, witness_rejected,
                     vote_rejected, supporters, choice, board_confidence, quorum, active), counter_prob

    def forward(self, x, minutes, before=None, *, mode="learned", counter=None):
        if mode not in ("off", "learned", "tree") or (mode == "tree") != (counter is not None):
            raise ValueError("Tree review requires independent tree probabilities")
        mask = x.isfinite()
        values = torch.where(mask, x, 0.)
        before = self.start(x) if before is None else before
        memory = self.attention.read(before.memory, self.query(torch.cat((values, mask.to(x.dtype)), -1)))
        feedback = x.new_zeros(len(x), self.count, 3)
        active = torch.ones(len(x), dtype=torch.bool, device=x.device)
        rounds = []
        evidence, judgment, board_memory = before.evidence, before.judgment, before.memory
        answer, confidence = before.answer, x.new_full((len(x),), .5)
        forecast = values[:, :self.sensor_count]
        action_value = x.new_zeros(len(x), 2)
        for index in range(self.max_rounds):
            result, counter_prob = self.round(values, mask, minutes, before, memory, feedback, counter, mode, active)
            rounds.append(result)
            available = result.available
            support = result.supporters
            accepted = active & result.quorum
            answer = torch.where(active, result.answer, answer)
            confidence = torch.where(active, result.answer_confidence, confidence)
            summed = (result.forecast * available.unsqueeze(-1)).sum(1)
            combined = summed / available.sum(1, keepdim=True).clamp_min(1)
            forecast = torch.where((active & available.any(1))[:, None], combined, forecast)
            mean_value = (result.action_value * available.unsqueeze(-1)).sum(1)
            mean_value = mean_value / available.sum(1, keepdim=True).clamp_min(1)
            action_value = torch.where(active[:, None], mean_value, action_value)
            promote = accepted[:, None] & support
            evidence = torch.where(promote.unsqueeze(-1), result.evidence_state, evidence)
            judgment = torch.where(promote.unsqueeze(-1), result.judgment_state, judgment)
            draft = (result.judgment_state * support.unsqueeze(-1)).sum(1)
            draft = draft / support.sum(1, keepdim=True).clamp_min(1)
            updated_memory = self.attention(before.memory, draft, minutes)
            board_memory = torch.where(accepted[:, None, None, None], updated_memory, board_memory)
            active = active & ~accepted
            if not active.any():
                break
            challenge = result.witness_rejected | result.vote_rejected
            check_direction = torch.where(challenge, 2 * counter_prob - 1, 0.)
            group_direction = (2 * result.answer.to(x.dtype) - 1) * (2 * result.answer_confidence - 1)
            feedback = torch.stack((check_direction, group_direction[:, None].expand_as(check_direction),
                                    torch.full_like(check_direction, (index + 1) / self.max_rounds)), -1)
        state = State(evidence, judgment, board_memory, torch.where(active, before.answer, answer))
        return Decision(state, rounds, answer, confidence, ~active, forecast, action_value)


def observation_loss(model, result, x, later):
    observed = later[:, :model.sensor_count]
    valid = observed.isfinite()
    error = F.smooth_l1_loss(result.forecast, torch.where(valid, observed, result.forecast.detach()),
                            reduction="none")
    loss = (error * valid).sum() / valid.sum().clamp_min(1)
    for item in result.rounds:
        other = x[:, None, :model.sensor_count].isfinite() & ~model.views[:, :model.sensor_count]
        other = other & item.available.unsqueeze(-1) & item.active[:, None, None]
        target = torch.where(other, x[:, None, :model.sensor_count], item.witness.detach())
        loss = loss + .15 * F.smooth_l1_loss(item.witness, target, reduction="sum") / other.sum().clamp_min(1)
    return loss


def attempt_reward(item, correct, index):
    rejected = item.witness_rejected | item.vote_rejected
    clean_round = item.quorum & ~(rejected & item.speaking).any(-1)
    verified = torch.where(correct, .90 + .15 * clean_round[:, None], -1.10 * 1.25**index)
    calibration = (item.confidence.detach() - correct.float()).square()
    return verified - calibration + .10 * item.process.detach().sigmoid()


def claim_loss(model, result, inputs, truth, weights):
    """Train claims, the process judge, and expected payoff for speaking."""
    losses = []
    for index, item in enumerate(result.rounds):
        active = item.active[:, None] & item.available
        target = truth[:, None].expand_as(item.proposal)
        correct = item.proposal.eq(target)
        evidence = inputs.isfinite()[:, None] & ~model.views
        other = torch.cat((torch.where(evidence, inputs[:, None], 0.), evidence.to(inputs.dtype)), -1)
        counter = model.counter(other).squeeze(-1)
        vote_loss = F.cross_entropy(item.logits.transpose(-1, -2), target, reduction="none")
        process = F.binary_cross_entropy_with_logits(item.process, correct.to(inputs.dtype), reduction="none")
        check = F.binary_cross_entropy_with_logits(counter, target.to(inputs.dtype), reduction="none")
        # Train the attempt policy with verified payoff and a small process score.
        payoff = attempt_reward(item, correct, index)
        attempt = (2 * item.confidence.detach() - 1 + item.drive - .20).mul(12).sigmoid()
        loss = vote_loss + .20 * process + .15 * check * evidence.any(-1) - .10 * attempt * payoff
        scaled = active * weights[truth, None]
        losses.append((loss * scaled).sum() / scaled.sum().clamp_min(1))
    return torch.stack(losses).mean()


def reward_loss(result, action, reward, next_values, continuation):
    """Fit discounted measured reward for the recorded action, with conservative Q learning."""
    have_reward = reward.isfinite()
    expected = reward.nan_to_num() + .95 * next_values.detach().amax(-1) * continuation
    logged = result.action_value.gather(1, action[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(logged, expected, reduction="none")
    conservative = result.action_value.logsumexp(-1) - logged
    total = (loss + .10 * conservative) * have_reward
    auxiliaries = []
    for index, item in enumerate(result.rounds):
        active = item.active[:, None] & item.available & have_reward[:, None]
        logged_vote = item.action_value.gather(
            2, action[:, None, None].expand_as(item.action_value[..., :1])).squeeze(-1)
        error = (logged_vote.detach() - expected[:, None]).abs()
        reliable = error.le(.5).to(logged_vote.dtype)
        process = F.binary_cross_entropy_with_logits(item.process, reliable, reduction="none")
        # There is a measured outcome only for the logged action.
        judged = active & item.proposal.eq(action[:, None])
        payoff = attempt_reward(item, reliable.bool(), index)
        attempt = (2 * item.confidence.detach() - 1 + item.drive - .20).mul(12).sigmoid()
        mse = F.smooth_l1_loss(logged_vote, expected[:, None].expand_as(logged_vote), reduction="none")
        auxiliaries.append(((.30 * mse + .10 * process - .10 * attempt * payoff * judged) *
                            active).sum() / active.sum().clamp_min(1))
    return total.sum() / have_reward.sum().clamp_min(1) + torch.stack(auxiliaries).mean()
