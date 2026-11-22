"""Liquid experts with a shared delta record, bounded retries, and online learning."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


WIDTH = 8
DEPTH = 3
EXPERTS = 3


class Liquid(nn.Module):
    """Positive conductances pull state toward bounded destinations."""

    def __init__(self, inputs: int, width: int, count: int = 1):
        super().__init__()
        self.sensor = nn.Parameter(torch.randn(count, inputs, width) * .3)
        self.recurrent = nn.Parameter(torch.randn(count, width, width) * .3)
        self.destination = nn.Parameter(torch.randn(count, inputs, width))
        self.leak = nn.Parameter(torch.zeros(count, width))
        self.rest = nn.Parameter(torch.zeros(count, width))

    def forward(self, x, old, elapsed, expert):
        sensor = x.unsqueeze(-1)
        conductance = F.softplus(self.sensor[expert]) * torch.sigmoid(sensor)
        recurrent = F.softplus(self.recurrent[expert]) * old.unsqueeze(-1).sigmoid()
        leak = F.softplus(self.leak[expert]) + .01
        drive = (conductance * self.destination[expert].tanh()).sum(1)
        drive += (recurrent * old.unsqueeze(-1)).sum(1) + leak * self.rest[expert].tanh()
        total = conductance.sum(1) + recurrent.sum(1) + leak
        dt = elapsed[:, None] / 5
        return (old + dt * drive) / (1 + dt * total)


class Delta(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_value = nn.Linear(WIDTH, 3 * WIDTH + 1)
        self.decay = nn.Linear(WIDTH, WIDTH)

    def read(self, board, query):
        return torch.einsum("bi,bij->bj", F.normalize(query, dim=-1), board)

    def forward(self, board, value, elapsed):
        key, query, target, write = self.key_value(value).split((WIDTH, WIDTH, WIDTH, 1), -1)
        key = F.normalize(key, dim=-1)
        prior = board * torch.exp(-F.softplus(self.decay(value)) * elapsed[:, None] / 15).unsqueeze(-1)
        gap = target.tanh() - self.read(prior, key)
        board = prior + key.unsqueeze(-1) * (gap * write.sigmoid()).unsqueeze(1)
        return board, self.read(board, query)


@dataclass
class Ledger:
    experts: tuple
    checks: tuple
    board: torch.Tensor
    last: torch.Tensor
    forecast: torch.Tensor

    def detach(self):
        return Ledger(tuple(s.detach() for s in self.experts), tuple(s.detach() for s in self.checks),
                      self.board.detach(), self.last.detach(), self.forecast.detach())


@dataclass
class Claim:
    expert: torch.Tensor
    state: torch.Tensor
    draft: torch.Tensor
    review: torch.Tensor
    process: torch.Tensor
    current_error: torch.Tensor
    forecast: torch.Tensor
    logit: torch.Tensor
    q: torch.Tensor
    reconstructed: torch.Tensor


@dataclass
class Result:
    memory: Ledger
    logit: torch.Tensor
    q: torch.Tensor
    accepted: torch.Tensor
    attempts: list


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Linear(6 + 2 * WIDTH, EXPERTS)
        self.experts = Liquid(4 + 2 * WIDTH, WIDTH, EXPERTS)
        self.checker = Liquid(2 + 2 * WIDTH, WIDTH)
        self.mix = nn.Linear(2 * WIDTH, WIDTH)
        self.process = nn.Linear(WIDTH, 1)
        self.sensor = nn.Linear(WIDTH, 4)
        self.occupied = nn.Linear(WIDTH, 1)
        self.value = nn.Linear(WIDTH, 2)

    def propose(self, signal, incoming, memory, old, check, elapsed, evidence, choice, routed):
        rows = torch.arange(len(choice), device=choice.device)
        groups = torch.stack((signal[:, :2], signal[:, 2:]), dim=1)
        mask = torch.stack((choice != 1, choice != 0), dim=1).unsqueeze(-1)
        values = torch.cat(((groups * mask).flatten(1), incoming, memory), dim=-1)
        expert_state = self.experts(values, old[rows, choice], elapsed, choice)
        # Route exactly one expert. Give its learned selection score a training gradient.
        if routed is not None:
            score = routed[rows, choice].unsqueeze(-1)
            expert_state = expert_state * (1 + score - score.detach())
        draft = self.mix(torch.cat((incoming, expert_state), -1)).tanh()
        reviewer_input = torch.cat((incoming, draft, evidence), -1)
        review = self.checker(reviewer_input, check, elapsed, torch.zeros_like(choice))
        process = self.process(review).squeeze(-1).tanh()
        reconstructed, deviation = self.sensor(draft).split(2, -1)
        current_error = (reconstructed - signal[:, :2]).abs().mean(-1)
        return Claim(choice, expert_state, draft, review, process, current_error,
                     signal[:, :2] + deviation, self.occupied(draft).squeeze(-1),
                     self.value(draft), reconstructed)


class Workflow(nn.Module):
    def __init__(self, *, route=True, delta=True, checks=True):
        super().__init__()
        self.stages = nn.ModuleList(Stage() for _ in range(DEPTH))
        self.first = nn.Linear(4, WIDTH)
        self.memory = Delta()
        self.route, self.delta, self.checks = route, delta, checks

    def start(self, x):
        return Ledger(tuple(x.new_zeros(len(x), EXPERTS, WIDTH) for _ in self.stages),
                      tuple(x.new_zeros(len(x), WIDTH) for _ in self.stages),
                      x.new_zeros(len(x), WIDTH, WIDTH), x, x)

    def step(self, x, elapsed, ledger=None, *, train_alternatives=False,
             tolerance=.8, force_objection=False, trace=False, backtrack=True):
        prior = self.start(x) if ledger is None else ledger
        signal = torch.cat((x, (x - prior.last).clamp(-4, 4)), -1)
        incoming, board = self.first(signal).tanh(), prior.board
        evidence = (x - prior.forecast).clamp(-4, 4)
        states, checks, attempts = [], [], []
        accepted = torch.ones(len(x), dtype=torch.bool, device=x.device)
        index = 0
        returned = None
        history = []
        resume = False
        while index < len(self.stages):
            stage = self.stages[index]
            history.append((incoming, board, accepted))
            memory = self.memory.read(board, incoming) if self.delta else torch.zeros_like(incoming)
            logits = stage.route(torch.cat((signal, incoming, memory, evidence), -1))
            routed = logits.softmax(-1) if self.route else None
            order = logits.argsort(dim=-1, descending=True) if self.route else torch.zeros_like(logits, dtype=torch.long)
            primary = 1 if resume else 0
            expert = order[:, primary]
            first = stage.propose(signal, incoming, memory, prior.experts[index],
                                  prior.checks[index], elapsed, evidence, expert, routed)
            legal = first.current_error.isfinite() & (first.current_error <= tolerance)
            # The next checker can object to the previous group's work. Do not
            # advance the clock, memory or expert state during the retry.
            if (backtrack and self.route and self.checks and len(x) == 1
                    and not train_alternatives and index > 0 and returned is None
                    and (force_objection or not bool(legal[0]) or bool(first.process[0] < -.15))):
                returned = index
                if trace:
                    attempts.append(dict(objection_from=index + 1, revisit=index,
                                         measured_error=first.current_error.detach().tolist()))
                index -= 1
                incoming, board, accepted = history[index]
                del history[index:]
                del states[index:]
                del checks[index:]
                resume = True
                continue
            resume = False
            retry = (~legal | (first.process < 0) | force_objection) & self.checks
            retry_rows = torch.arange(len(x), device=x.device) if train_alternatives else retry.nonzero().flatten()
            chosen = first
            retry_expert = None
            take = torch.zeros_like(legal)
            if len(retry_rows) and self.route and self.checks:
                r = retry_rows
                # Reject the claim, rerun that expert stage from its original state.
                # Write to the shared board only after the outcome of the review.
                alternative = stage.propose(signal[r], incoming[r], memory[r], prior.experts[index][r],
                                            prior.checks[index][r], elapsed[r], evidence[r], order[r, primary + 1],
                                            routed[r])
                retry_expert = alternative.expert
                allowed = alternative.current_error <= tolerance
                prefer = alternative.process - alternative.current_error > first.process[r] - first.current_error[r]
                selected = (allowed & (~legal[r] | prefer) & retry[r]).nonzero().flatten()
                rows = r[selected]
                take[rows] = True
                chosen = Claim(*(a.index_copy(0, rows, b[selected])
                                 for a, b in zip(first.__dict__.values(), alternative.__dict__.values(), strict=True)))
                if train_alternatives:
                    attempts.append((first, alternative, order, routed))
            verified = chosen.current_error.isfinite() & (chosen.current_error <= tolerance)
            accepted = accepted & (verified | (not self.checks))
            old = prior.experts[index]
            update = F.one_hot(chosen.expert, EXPERTS).bool().unsqueeze(-1)
            states.append(torch.where(update, chosen.state[:, None, :], old))
            checks.append(chosen.review)
            incoming = chosen.draft
            if self.delta:
                board, recalled = self.memory(board, incoming, elapsed)
                incoming = (incoming + recalled).tanh()
            if trace:
                attempts.append(dict(stage=index + 1, first=first.expert.tolist(),
                                     rejected=retry.tolist(), retry=None if retry_expert is None else retry_expert.tolist(),
                                     used_retry=take.tolist(), verified=verified.tolist(),
                                     measured_error=chosen.current_error.detach().tolist()))
            index += 1
        memory = Ledger(tuple(states), tuple(checks), board, x, chosen.forecast)
        return Result(memory, chosen.logit, chosen.q, accepted, attempts)


def process_target(claim, x, next_x, y=None):
    error = (claim.forecast.detach() - next_x).abs().mean(-1)
    reward = -error - claim.current_error.detach()
    if y is not None:
        reward -= F.binary_cross_entropy_with_logits(claim.logit.detach(), y, reduction="none")
    return torch.exp(reward) * 2 - 1


def train_loss(result, x, next_x, y):
    loss = F.binary_cross_entropy_with_logits(result.logit, y)
    loss = loss + .4 * F.smooth_l1_loss(result.memory.forecast, next_x)
    for first, alt, order, routed in result.attempts:
        for claim in (first, alt):
            loss = loss + .08 * F.smooth_l1_loss(claim.reconstructed, x)
            loss = loss + .08 * F.smooth_l1_loss(claim.forecast, next_x)
            loss = loss + .10 * F.mse_loss(claim.process, process_target(claim, x, next_x, y))
            loss = loss + .08 * F.binary_cross_entropy_with_logits(claim.logit, y)
        best = torch.where(process_target(first, x, next_x, y) >= process_target(alt, x, next_x, y),
                           order[:, 0], order[:, 1])
        loss = loss + .06 * F.nll_loss(routed.clamp_min(1e-8).log(), best)
    return loss


def learn_from_observation(model, optimizer, previous, next_x):
    """The previous answer exists before this function sees the next reading."""
    x, elapsed, prior = previous
    result = model.step(x, elapsed, prior, train_alternatives=True)
    loss = F.smooth_l1_loss(result.memory.forecast, next_x)
    for first, alternative, _, _ in result.attempts:
        for claim in (first, alternative):
            loss = loss + .12 * F.smooth_l1_loss(claim.forecast, next_x)
            loss = loss + .08 * F.mse_loss(claim.process, process_target(claim, x, next_x))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
    optimizer.step()
    return float(loss.detach())
