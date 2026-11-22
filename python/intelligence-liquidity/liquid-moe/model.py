"""Causal occupancy models and the optional one-revision route.

Inputs are train-normalized readings and per-step elapsed minutes. Dormant
experts retain their last state and timestamp. On reactivation their liquid
update uses the full elapsed gap and the current drive. Review branches only
at the endpoint, with both proposals starting from the same prefix state.
Expert counts measure per-example cell evaluations, including challengers.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 16
EXPERTS = 3
OCCUPIED_MISS = 2.0
EMPTY_FALSE_ALARM = 1.0
REVISION_COST = 0.06
ABSTAIN_REWARD = -0.20


class LiquidCell(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.drive = nn.Linear(inputs + width, 2 * width)

    def forward(self, inputs, state, minutes):
        target, rate = self.drive(torch.cat((inputs, state), dim=-1)).chunk(2, -1)
        fraction = -torch.expm1(-(F.softplus(rate) + 1e-4) * minutes[:, None] / 5)
        return state + fraction * (target.tanh() - state)


class DeltaBoard(nn.Module):
    def __init__(self, width, size=8):
        super().__init__()
        self.size = size
        self.key = nn.Linear(width, size)
        self.query = nn.Linear(width, size)
        self.value = nn.Linear(width, size)
        self.gate = nn.Linear(width, 1)
        self.decay = nn.Parameter(torch.zeros(()))
        self.output = nn.Linear(size, width, bias=False)

    def forward(self, features, board, minutes):
        key = F.normalize(self.key(features), dim=-1)
        query = F.normalize(self.query(features), dim=-1)
        old = board * torch.exp(-(F.softplus(self.decay) + 1e-4) *
                                minutes[:, None, None] / 5)
        predicted = torch.bmm(key[:, None], old).squeeze(1)
        error = self.value(features).tanh() - predicted
        gate = self.gate(features).sigmoid() * -torch.expm1(-minutes[:, None] / 5)
        updated = old + key[:, :, None] * (gate * error)[:, None, :]
        read = torch.bmm(query[:, None], updated).squeeze(1)
        return updated, self.output(read)


class Checker(nn.Module):
    """An independent liquid encoder for one expert's decisions."""

    def __init__(self):
        super().__init__()
        self.cell = LiquidCell(5, WIDTH)
        self.correct = nn.Linear(WIDTH + 2, 1)
        self.gain = nn.Sequential(nn.Linear(2 * (WIDTH + 2), WIDTH),
                                  nn.Tanh(), nn.Linear(WIDTH, 1))

    def encode(self, features, clock):
        state = features.new_zeros(features.shape[0], WIDTH)
        for step in range(features.shape[1]):
            state = self.cell(features[:, step], state, clock[:, step])
        return state


@dataclass
class Proposal:
    logits: torch.Tensor
    expert: torch.Tensor
    states: torch.Tensor
    timestamps: torch.Tensor
    board: torch.Tensor
    valid: torch.Tensor

    def choose(self, alternate, commit):
        def select(initial, tentative):
            mask = commit.reshape((-1,) + (1,) * (initial.ndim - 1))
            return torch.where(mask, tentative, initial)

        return Proposal(
            select(self.logits, alternate.logits),
            select(self.expert, alternate.expert),
            select(self.states, alternate.states),
            select(self.timestamps, alternate.timestamps),
            select(self.board, alternate.board),
            select(self.valid, alternate.valid),
        )


def causal_features(x, clock):
    return torch.cat((x, x - x[:, :1], torch.log1p(clock.clamp_min(0))[:, :, None]), -1)


class OccupancyModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.routed = name in ("moe", "moe_delta")
        if name == "current":
            self.readout = nn.Linear(2, 1)
            return
        self.core = nn.GRUCell(5, WIDTH) if name == "gru" else LiquidCell(5, WIDTH)
        self.readout = nn.Linear(WIDTH + 5, 1)
        if self.routed:
            self.router = nn.Linear(WIDTH + 5, EXPERTS)
            self.experts = nn.ModuleList(LiquidCell(WIDTH + 5, WIDTH)
                                         for _ in range(EXPERTS))
        self.memory = DeltaBoard(WIDTH) if name == "moe_delta" else None
        if self.memory is not None:
            self.checkers = nn.ModuleList(Checker() for _ in range(EXPERTS))

    def _logit(self, state, sample):
        return self.readout(torch.cat((state, sample), -1)).squeeze(-1)

    def _propose(self, shared, sample, drive, now, minutes, states,
                 timestamps, board, choices, weights):
        residual = torch.zeros_like(shared)
        for expert_id, cell in enumerate(self.experts):
            rows = torch.where(choices == expert_id)[0]
            if rows.numel():
                elapsed = now[rows] - timestamps[rows, expert_id]
                update = cell(drive[rows], states[rows, expert_id], elapsed)
                residual = residual.index_copy(0, rows, update)

        mask = F.one_hot(choices, EXPERTS).bool()
        new_states = torch.where(mask[:, :, None], residual[:, None], states)
        new_timestamps = torch.where(mask, now[:, None], timestamps)
        # Hard top-1 forward execution with a selected-softmax surrogate gradient.
        gate = weights.gather(1, choices[:, None])
        combined = shared + residual * (1 + (gate - gate.detach()))
        if self.memory is not None:
            board, read = self.memory(combined, board, minutes)
            combined = combined + read
        logits = self._logit(combined, sample)
        valid = (torch.isfinite(logits) & torch.isfinite(new_states).all((1, 2)) &
                 torch.isfinite(board).all((1, 2)) & torch.isfinite(new_timestamps).all(1) &
                 (new_timestamps <= now[:, None]).all(1))
        return Proposal(logits, choices, new_states, new_timestamps, board, valid)

    def _run(self, x, clock, *, branch=False):
        features = causal_features(x, clock)
        batch, steps, _ = features.shape
        shared = x.new_zeros(batch, WIDTH)
        states = x.new_zeros(batch, EXPERTS, WIDTH)
        timestamps = x.new_zeros(batch, EXPERTS)
        size = self.memory.size if self.memory is not None else 0
        board = x.new_zeros(batch, size, size)
        now = x.new_zeros(batch)
        routes = []
        for step in range(steps):
            sample, minutes = features[:, step], clock[:, step]
            if self.name == "gru":
                shared = self.core(sample, shared)
            else:
                shared = self.core(sample, shared, minutes)
            if self.routed:
                now = now + minutes
                drive = torch.cat((shared, sample), -1)
                scores = self.router(drive)
                weights = scores.softmax(-1)
                primary = scores.argmax(-1)
                proposal = self._propose(shared, sample, drive, now, minutes,
                                         states, timestamps, board, primary, weights)
                routes.append(primary)
                if branch and step == steps - 1:
                    # Capture the pre-proposal transaction and execute only requested rows.
                    context = (shared, sample, drive, now, minutes, states,
                               timestamps, board, weights)
                    alternative = scores.masked_fill(
                        F.one_hot(primary, EXPERTS).bool(), -torch.inf).argmax(-1)
                states, timestamps, board = proposal.states, proposal.timestamps, proposal.board

        if self.routed:
            route = torch.stack(routes, dim=1)
            counts = F.one_hot(route, EXPERTS).sum((0, 1))
            result = dict(logits=proposal.logits, route=route,
                          expert_counts=counts, expert_calls=counts.sum())
            if branch:
                return result, features, proposal, context, alternative
            return result
        return dict(logits=self._logit(shared, sample),
                    expert_counts=x.new_zeros(EXPERTS, dtype=torch.long),
                    expert_calls=x.new_zeros((), dtype=torch.long))

    def forward(self, x, clock):
        if self.name == "current":
            return dict(logits=self.readout(x[:, -1]).squeeze(-1),
                        expert_counts=x.new_zeros(EXPERTS, dtype=torch.long),
                        expert_calls=x.new_zeros((), dtype=torch.long))
        return self._run(x, clock)

    def _check(self, features, clock, choices, logits, cutoff):
        # Stop checker gradients at the proposal. Only the relevant checker runs.
        logits = logits.detach()
        decisions = logits.sigmoid() >= cutoff
        evidence = features.new_zeros(features.shape[0], WIDTH + 2)
        correctness = torch.zeros_like(logits)
        for expert_id, checker in enumerate(self.checkers):
            rows = torch.where(choices == expert_id)[0]
            if rows.numel():
                state = checker.encode(features[rows], clock[rows])
                candidate = torch.cat((state, logits[rows, None].tanh(),
                                       decisions[rows, None].to(state.dtype)), -1)
                evidence = evidence.index_copy(0, rows, candidate)
                correctness = correctness.index_copy(0, rows, checker.correct(candidate).squeeze(-1))
        return correctness, evidence

    def _gain(self, choices, initial, alternate):
        gain = initial.new_zeros(initial.shape[0])
        for expert_id, checker in enumerate(self.checkers):
            rows = torch.where(choices == expert_id)[0]
            if rows.numel():
                pair = torch.cat((initial[rows], alternate[rows]), -1)
                gain = gain.index_copy(0, rows, checker.gain(pair).squeeze(-1))
        return gain


def build(name):
    if name not in ("current", "gru", "liquid", "moe", "moe_delta"):
        raise ValueError(f"Unknown arm: {name}")
    return OccupancyModel(name)


def _challenger(model, context, choices, rows):
    shared, sample, drive, now, minutes, states, timestamps, board, weights = (
        tensor[rows] for tensor in context
    )
    return model._propose(shared, sample, drive, now, minutes, states,
                          timestamps, board, choices[rows], weights)


def _expected_reward(correctness, decision):
    error_cost = torch.where(decision, EMPTY_FALSE_ALARM, OCCUPIED_MISS)
    return correctness - (1 - correctness) * error_cost


def _observed_reward(correct, decision):
    return _expected_reward(correct.float(), decision)


def training_loss(model, x, clock, y, pos_weight):
    if model.name == "moe_delta":
        result, features, initial, context, choices = model._run(x, clock, branch=True)
    else:
        result = model(x, clock)
    logits = result["logits"]
    y = y.reshape_as(logits).to(logits)
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    task = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
    if model.name != "moe_delta":
        return task

    with torch.no_grad():
        rows = torch.arange(x.shape[0], device=x.device)
        alternate = _challenger(model, context, choices, rows)
        # Train on a fixed operating point and sampled probability thresholds
        # so correctness and revision value can generalize to calibration.
        cutoff = torch.rand_like(logits) * 0.8 + 0.1
        cutoff[::2] = 0.5
        decision = logits.sigmoid() >= cutoff
        other_decision = alternate.logits.sigmoid() >= cutoff
        correct = decision == y.bool()
        other_correct = other_decision == y.bool()
        value = (_observed_reward(other_correct, other_decision) -
                 _observed_reward(correct, decision))
    initial_check, initial_evidence = model._check(
        features, clock, initial.expert, logits, cutoff)
    alternate_check, alternate_evidence = model._check(
        features, clock, choices, alternate.logits, cutoff)
    gain = model._gain(choices, initial_evidence, alternate_evidence)
    correctness_loss = (
        F.binary_cross_entropy_with_logits(initial_check, correct.to(logits)) +
        F.binary_cross_entropy_with_logits(alternate_check, other_correct.to(logits))
    ) / 2
    return task + correctness_loss + F.smooth_l1_loss(gain, value.to(gain))


@torch.no_grad()
def review(model, x, clock, cutoff):
    """Review endpoint proposals; cutoff is the ordinary probability threshold.

    accepted marks covered predictions. revision_accepted is a separate flag.
    logits always carry the committed decision for full-coverage scoring.
    action codes are 0 keep, 1 commit, 2 rollback, 3 abstain.
    """
    if model.name != "moe_delta":
        raise ValueError("Review requires a fitted moe_delta model")
    if not math.isfinite(float(cutoff)) or not 0 <= float(cutoff) <= 1:
        raise ValueError("cutoff must be a finite probability in [0, 1]")
    result, _, initial, context, choices = model._run(x, clock, branch=True)
    time_valid = torch.isfinite(clock).all(1) & (clock >= 0).all(1)
    input_valid = time_valid & torch.isfinite(x).all((1, 2))
    initial_valid = initial.valid & input_valid
    # Keep invalid rows out of both learned checks and challenger execution.
    valid_rows = torch.where(initial_valid)[0]
    features = causal_features(x[valid_rows], clock[valid_rows])
    check, evidence = model._check(features, clock[valid_rows], initial.expert[valid_rows],
                                   initial.logits[valid_rows], cutoff)
    initial_check = torch.full_like(initial.logits, torch.nan)
    initial_check[valid_rows] = check.sigmoid()
    initial_reward = _expected_reward(initial_check, initial.logits.sigmoid() >= cutoff)
    # Spend the one revision only if the initial estimate leaves room to
    # recover the attempt cost. The challenger must then clear that cost.
    attempted = initial_valid & torch.isfinite(initial_reward) & (
        initial_reward < 1 - REVISION_COST)
    rows = torch.where(attempted)[0]
    revision_accepted = torch.zeros_like(attempted)
    challenger_valid = torch.zeros_like(attempted)
    challenger_check = torch.full_like(initial_check, torch.nan)
    challenger_logits = torch.full_like(initial.logits, torch.nan)
    process_gain = torch.full_like(initial_check, torch.nan)
    revision_gain = torch.full_like(initial_check, torch.nan)
    committed_logits = initial.logits.clone()
    committed_expert = initial.expert.clone()
    committed_states = initial.states.clone()
    committed_timestamps = initial.timestamps.clone()
    committed_board = initial.board.clone()
    committed_reward = initial_reward.clone()
    if rows.numel():
        alternate = _challenger(model, context, choices, rows)
        finite_logits = torch.where(alternate.valid, alternate.logits, 0)
        alternative_check, alternate_evidence = model._check(
            causal_features(x[rows], clock[rows]), clock[rows], alternate.expert,
            finite_logits, cutoff)
        challenger_valid[rows] = alternate.valid
        challenger_check[rows] = alternative_check.sigmoid()
        challenger_logits[rows] = alternate.logits
        alternate_reward = _expected_reward(challenger_check[rows],
                                            finite_logits.sigmoid() >= cutoff)
        learned_gain = model._gain(alternate.expert, evidence[attempted[valid_rows]],
                                   alternate_evidence)
        process_gain[rows] = learned_gain
        estimated_gain = (learned_gain + alternate_reward - initial_reward[rows]) / 2
        revision_gain[rows] = estimated_gain
        commit = alternate.valid & torch.isfinite(estimated_gain) & (estimated_gain > REVISION_COST)
        revision_accepted[rows] = commit
        accepted_proposal = Proposal(initial.logits[rows], initial.expert[rows],
                                     initial.states[rows], initial.timestamps[rows],
                                     initial.board[rows], initial.valid[rows])
        final = accepted_proposal.choose(alternate, commit)
        committed_logits[rows], committed_expert[rows] = final.logits, final.expert
        committed_states[rows], committed_timestamps[rows] = final.states, final.timestamps
        committed_board[rows] = final.board
        committed_reward[rows] = torch.where(commit, alternate_reward, initial_reward[rows])

    accepted = initial_valid & torch.isfinite(committed_reward) & (committed_reward >= ABSTAIN_REWARD)
    revision_rejected = attempted & ~revision_accepted
    action = revision_accepted.long() + 2 * revision_rejected.long()
    action = torch.where(accepted, action, 3)
    counts = result["expert_counts"] + F.one_hot(choices[rows], EXPERTS).sum(0)
    # A malformed primary abstains and contributes a deterministic empty decision
    # when the caller scores every window.
    logits = torch.where(initial_valid, committed_logits, -torch.inf)
    return dict(
        logits=logits, ordinary_logits=initial.logits, accepted=accepted,
        attempted=attempted, revision_accepted=revision_accepted,
        revision_rejected=revision_rejected, revision_count=attempted.long(),
        budget_remaining=1 - attempted.long(), coverage=accepted.float().mean(),
        route=result["route"], expert_counts=counts, expert_calls=counts.sum(),
        initial_expert=initial.expert, challenger_expert=torch.where(attempted, choices, -1),
        returned_expert=torch.where(initial_valid, committed_expert, -1),
        initial_logits=initial.logits, challenger_logits=challenger_logits,
        initial_check=initial_check, challenger_check=challenger_check,
        initial_valid=initial_valid, challenger_valid=challenger_valid, time_valid=time_valid,
        process_gain=process_gain, revision_gain=revision_gain, action=action,
        decision=torch.where(accepted, (logits.sigmoid() >= cutoff).long(), -1),
        committed_board=committed_board, committed_states=committed_states,
        committed_timestamps=committed_timestamps,
    )
