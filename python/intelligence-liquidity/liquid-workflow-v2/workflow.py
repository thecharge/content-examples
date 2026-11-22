"""A liquid expert workflow with transactional state and causal feedback."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

WIDTH = 8
EXPERTS = 3
DEPTH = 3
HEADS = 2


class LiquidODE(nn.Module):
    """A conductance LTC. Trainable settings live inside the derivative."""

    def __init__(self, inputs, count=1, substeps=6):
        super().__init__()
        self.substeps = substeps
        edges = inputs + WIDTH
        self.strength = nn.Parameter(torch.full((count, edges, WIDTH), -2.0))
        self.slope = nn.Parameter(torch.randn(count, edges, WIDTH) * .6)
        self.center = nn.Parameter(torch.zeros(count, edges, WIDTH))
        self.target = nn.Parameter(torch.randn(count, edges, WIDTH))
        self.leak = nn.Parameter(torch.full((count, WIDTH), -1.0))
        self.rest = nn.Parameter(torch.zeros(count, WIDTH))

    def forward(self, x, state, minutes, expert):
        strength = F.softplus(self.strength[expert])
        slope, center = self.slope[expert], self.center[expert]
        target = self.target[expert].tanh()
        leak = F.softplus(self.leak[expert]) + .01
        rest = self.rest[expert].tanh()
        inputs = x.shape[-1]
        sensor = strength[:, :inputs] * ((x.unsqueeze(-1) - center[:, :inputs]) * slope[:, :inputs]).sigmoid()
        sensor_drive = (sensor * target[:, :inputs]).sum(1) + leak * rest
        sensor_decay = sensor.sum(1) + leak
        dt = minutes[:, None] / (5 * self.substeps)
        for _ in range(self.substeps):
            gates = ((state.unsqueeze(-1) - center[:, inputs:]) * slope[:, inputs:]).sigmoid()
            recurrent = strength[:, inputs:] * gates
            drive = sensor_drive + (recurrent * target[:, inputs:]).sum(1)
            decay = sensor_decay + recurrent.sum(1)
            state = (state + dt * drive) / (1 + dt * decay)
        return state


class DeltaAttention(nn.Module):
    """Diagonal forgetting and key-addressed delta writes in two heads."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(WIDTH, WIDTH)
        self.write = nn.Linear(WIDTH, 3 * WIDTH + HEADS)

    def heads(self, value):
        return value.reshape(-1, HEADS, WIDTH // HEADS)

    def read(self, board, query):
        query = F.normalize(self.heads(self.query(query)), dim=-1)
        return torch.einsum("bhi,bhij->bhj", query, board).flatten(1)

    @staticmethod
    def update(board, key, target, forgetting, rate):
        past = board * forgetting.unsqueeze(-1)
        gap = target - torch.einsum("bhi,bhij->bhj", key, past)
        return past + key.unsqueeze(-1) * (gap * rate.unsqueeze(-1)).unsqueeze(-2)

    def forward(self, board, draft, minutes):
        key, target, forget, rate = self.write(draft).split((WIDTH, WIDTH, WIDTH, HEADS), -1)
        alpha = (-F.softplus(self.heads(forget)) * minutes[:, None, None] / 15).exp()
        return self.update(board, F.normalize(self.heads(key), dim=-1),
                           self.heads(target).tanh(), alpha, rate.sigmoid())


@dataclass(frozen=True)
class Ledger:
    experts: tuple
    reviewers: tuple
    board: torch.Tensor

    def detach(self):
        return Ledger(tuple(t.detach() for t in self.experts),
                      tuple(t.detach() for t in self.reviewers), self.board.detach())


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Linear(1 + 2 * WIDTH, EXPERTS)
        self.expert = LiquidODE(1 + 2 * WIDTH, EXPERTS)
        self.reviewer = LiquidODE(4 + 2 * WIDTH)
        self.draft = nn.Linear(2 * WIDTH + 1, WIDTH)
        self.shadow = nn.Linear(WIDTH, 1)
        self.process = nn.Linear(WIDTH, 1)
        self.delta = DeltaAttention()


class Blackboard(nn.Module):
    def __init__(self, *, use_delta=True, fixed_expert=False, use_checks=True):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(DEPTH))
        self.input = nn.Linear(1, WIDTH)
        self.occupancy = nn.Linear(WIDTH, 1)
        self.next_sensor = nn.Linear(WIDTH, 1)
        self.action_value = nn.Linear(WIDTH, 2)
        self.use_delta, self.fixed_expert, self.use_checks = use_delta, fixed_expert, use_checks

    def start(self, x):
        return Ledger(tuple(x.new_zeros(len(x), EXPERTS, WIDTH) for _ in self.layers),
                      tuple(x.new_zeros(len(x), WIDTH) for _ in self.layers),
                      x.new_zeros(len(x), HEADS, WIDTH // HEADS, WIDTH // HEADS))

    def step(self, primary, witness, minutes, ledger=None, *, tolerance=.55, training=False, replay=False):
        # The witness never enters a specialist, router, or attention projection.
        original = self.start(primary) if ledger is None else ledger
        batch = torch.arange(len(primary), device=primary.device)
        attempts = []
        active = torch.ones_like(primary, dtype=torch.bool)
        truth_available = witness.isfinite()
        evidence = witness.nan_to_num(nan=0., posinf=0., neginf=0.)
        initial = self.input(primary[:, None]).tanh()

        def path(retry_at=None, retry_rows=None, rank=0):
            draft, board = initial, original.board
            states, reviewers, errors, scores, claims, finite = [], [], [], [], [], []
            for index, layer in enumerate(self.layers):
                memory = layer.delta.read(board, draft) if self.use_delta else torch.zeros_like(draft)
                inputs = torch.cat((primary[:, None], draft, memory), dim=-1)
                logits = layer.route(inputs)
                order = logits.argsort(dim=-1, descending=True)
                if self.fixed_expert:
                    order = torch.zeros_like(order)
                expert = order[:, 0]
                if training and not replay and not self.fixed_expert:
                    expert = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
                if retry_at == index:
                    expert = torch.where(retry_rows, order[:, rank], expert)
                state = layer.expert(inputs, original.experts[index][batch, expert], minutes, expert)
                if training:
                    score = logits.softmax(-1)[batch, expert, None]
                    state = state * (1 + score - score.detach())
                candidate = layer.draft(torch.cat((draft, state, primary[:, None]), -1)).tanh()
                shadow = layer.shadow(candidate).squeeze(-1)
                residual = (shadow - evidence).abs()
                review_input = torch.cat((draft, candidate, torch.stack(
                    (shadow, evidence, residual, truth_available.to(primary.dtype)), -1)), -1)
                review = layer.reviewer(review_input, original.reviewers[index], minutes, torch.zeros_like(expert))
                process = layer.process(review).squeeze(-1)
                valid = candidate.isfinite().all(-1) & review.isfinite().all(-1)
                board = layer.delta(board, candidate, minutes) if self.use_delta else board
                valid = valid & board.isfinite().flatten(1).all(-1)
                next_experts = original.experts[index].scatter(
                    1, expert[:, None, None].expand(-1, 1, WIDTH), state[:, None, :])
                states.append(next_experts)
                reviewers.append(review)
                errors.append(residual)
                scores.append(process)
                claims.append(shadow)
                finite.append(valid)
                draft = candidate
                attempts.append(dict(layer=index, retry_from=retry_at, retry_rank=rank, expert=expert.detach(),
                                     claim=shadow.detach(), error=residual.detach(), process=process.detach()))
            error = torch.stack(errors, -1)
            process = torch.stack(scores, -1)
            passes = (error <= tolerance) & torch.stack(finite, -1) & truth_available[:, None]
            reward = passes.to(primary.dtype) + .25 * process.sigmoid()
            return draft, Ledger(tuple(states), tuple(reviewers), board), passes, reward, torch.stack(claims, -1), process

        proposed, candidate, passes, rewards, claims, process = path()
        allowed = passes.all(-1)
        if self.use_checks and not training and (~allowed & truth_available).any():
            # One rewind per row. Try the two remaining specialists; the verifier has veto power.
            retry_at = ((~passes).to(torch.int64).argmax(-1) - 1).clamp_min(0)
            for index in range(len(self.layers) - 1):
                retry_rows = ~allowed & truth_available & (retry_at == index)
                if not retry_rows.any():
                    continue
                best_reward = rewards.sum(-1)
                for rank in (1, 2) if not self.fixed_expert else (1,):
                    other_draft, other, other_passes, other_reward, other_claims, other_process = path(
                        index, retry_rows, rank)
                    total = other_reward.sum(-1)
                    choose = retry_rows & other_passes.all(-1) & (total > best_reward)
                    proposed = torch.where(choose[:, None], other_draft, proposed)
                    candidate = select(choose, other, candidate)
                    claims = torch.where(choose[:, None], other_claims, claims)
                    process = torch.where(choose[:, None], other_process, process)
                    best_reward = torch.where(choose, total, best_reward)
                    allowed = allowed | choose
        accepted = allowed if self.use_checks and not training else active
        return Result(select(accepted, candidate, original), self.occupancy(proposed).squeeze(-1),
                      self.next_sensor(proposed).squeeze(-1), self.action_value(proposed),
                      accepted, claims, process, attempts)


def select(mask, yes, no):
    return Ledger(tuple(torch.where(mask[:, None, None], a, b) for a, b in zip(yes.experts, no.experts, strict=True)),
                  tuple(torch.where(mask[:, None], a, b) for a, b in zip(yes.reviewers, no.reviewers, strict=True)),
                  torch.where(mask[:, None, None, None], yes.board, no.board))


@dataclass(frozen=True)
class Result:
    ledger: Ledger
    logit: torch.Tensor
    forecast: torch.Tensor
    action_value: torch.Tensor
    accepted: torch.Tensor
    claims: torch.Tensor
    process: torch.Tensor
    attempts: list


def loss(result, witness, label, next_sensor, tolerance=.55):
    verified = ((result.claims.detach() - witness[:, None]).abs() <= tolerance).float()
    return (F.binary_cross_entropy_with_logits(result.logit, label)
            + .3 * F.smooth_l1_loss(result.claims, witness[:, None].expand_as(result.claims))
            + .12 * F.binary_cross_entropy_with_logits(result.process, verified)
            + .25 * F.smooth_l1_loss(result.forecast, next_sensor))


@dataclass(frozen=True)
class Observation:
    time: int
    primary: torch.Tensor
    witness: torch.Tensor
    minutes: torch.Tensor
    before: Ledger | None


class Online:
    """Forecast, then learn on the later reading after its own prediction is emitted."""

    def __init__(self, model, lr=.001, max_gap=450):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.pending = None
        self.max_gap = max_gap

    def feedback(self, current, *, observed_action=None, observed_reward=None,
                 next_action_return=None):
        if (observed_action is None) != (observed_reward is None):
            raise ValueError("An action and its measured reward must arrive together")
        previous = self.pending
        if previous is not None and current.time <= previous.time:
            raise ValueError("Observations must arrive in time order")
        self.pending = current
        if previous is None or current.time - previous.time > self.max_gap:
            return None
        self.optimizer.zero_grad()
        replay = self.model.step(previous.primary, previous.witness, previous.minutes,
                                 previous.before, training=True, replay=True)
        error = F.smooth_l1_loss(replay.forecast, current.primary)
        if observed_reward is not None:
            target = replay.forecast.new_tensor(observed_reward)
            if next_action_return is not None:
                target = target + .95 * replay.forecast.new_tensor(next_action_return).detach()
            q = replay.action_value[:, observed_action]
            error = error + F.smooth_l1_loss(q, target.expand_as(q))
        if not error.isfinite():
            return None
        error.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25, error_if_nonfinite=True)
        self.optimizer.step()
        return float(error.detach())
