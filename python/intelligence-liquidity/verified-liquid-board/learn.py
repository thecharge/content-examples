"""Commit expert updates only when separate delayed outcomes validate them."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from model import State, measured_reward


@dataclass(frozen=True)
class Example:
    timestamp: int
    inputs: torch.Tensor
    before: State
    truth: torch.Tensor
    reward: torch.Tensor


class Feedback:
    def __init__(self, model, *, minimum=32, threshold=.001, rate=.008):
        self.model = model
        self.minimum = minimum
        self.threshold = threshold
        self.rate = rate
        self.history = deque(maxlen=144)
        self.latest = None
        self.next_check = minimum
        self.receipts = []

    def observe(self, timestamp, now, row, before, issued, actual):
        if timestamp >= now or (self.latest is not None and timestamp <= self.latest):
            raise ValueError("Only previously issued, newly revealed outcomes can grade updates")
        if actual.numel() != 1 or int(actual) not in (0, 1):
            raise ValueError("An external binary occupancy answer is required")
        self.latest = timestamp
        reward = measured_reward(issued.rounds, actual).detach()
        self.history.append(Example(timestamp, row.detach(), before.detached(), actual, reward))
        self.next_check -= 1
        if self.next_check > 0:
            return reward
        self.next_check = self.minimum
        examples = list(self.history)
        cut = max(len(examples) // 2, len(examples) - self.minimum)
        fit, check = examples[:cut], examples[cut:]
        if not all(torch.cat([sample.truth for sample in group]).bincount(minlength=2).min() >= 2
                   for group in (fit, check)):
            self.receipts.append(dict(revealed_at=now, status="frozen_insufficient_classes", accepted=[]))
            return reward
        receipt = self.try_update(fit, check)
        self.receipts.append(dict(revealed_at=now, **receipt))
        return reward

    def try_update(self, fit, check):
        candidate = deepcopy(self.model)
        params = tuple(candidate.expert_parameters())
        # Penalized decisions receive more fitting weight. The gate uses separate answers.
        loss, _ = measurements(candidate, fit, train=True)
        gradients = torch.autograd.grad(loss.mean(), params, allow_unused=True)
        with torch.no_grad():
            for param, gradient in zip(params, gradients, strict=True):
                if gradient is not None:
                    param.add_(gradient.clamp(-1., 1.), alpha=-self.rate)
            original_loss, original_correct = measurements(self.model, check)
            candidate_loss, candidate_correct = measurements(candidate, check)
            gain = original_loss - candidate_loss
            accept = gain.isfinite() & gain.ge(self.threshold) & candidate_correct.ge(original_correct)
            for old, new in zip(self.model.expert_parameters(), candidate.expert_parameters(), strict=True):
                accept_expert = accept.reshape(-1, *([1] * (old.ndim - 1)))
                old.copy_(torch.where(accept_expert, new, old))
        return dict(status="accepted" if accept.any() else "frozen_no_gain",
                    fit_first=fit[0].timestamp, fit_last=fit[-1].timestamp,
                    check_first=check[0].timestamp, check_last=check[-1].timestamp,
                    fit_rows=len(fit), check_rows=len(check),
                    gate_log_loss_gain=gain.tolist(),
                    accepted=accept.nonzero().flatten().add(1).tolist())


def measurements(model, examples, train=False):
    inputs = torch.cat([item.inputs for item in examples])
    before = State(tuple(torch.cat(group) for group in zip(*(item.before.cells for item in examples), strict=True)),
                   torch.cat([item.before.memory for item in examples]),
                   torch.cat([item.before.last for item in examples]),
                   torch.cat([item.before.elapsed for item in examples]))
    actual = torch.cat([item.truth for item in examples])
    # Score each expert on a fresh first-round proposal. Membership or votes do
    # not grade the forecast. The model is never given the label as input.
    result = model.round(inputs, inputs.new_full((len(inputs),), 5.), before,
                         inputs.new_zeros(len(inputs), model.experts, 3),
                         torch.ones(len(inputs), dtype=torch.bool, device=inputs.device), False, assess=True)
    raw = result.margins[..., -1]
    target = actual[:, None].expand_as(raw)
    costs = F.binary_cross_entropy_with_logits(raw, target.float(), reduction="none")
    if train:
        penalty = torch.cat([item.reward for item in examples]).clamp(max=0).abs()
        costs = costs * (1 + .25 * penalty)
        # Train the speak decision against verified outcomes too.
        drive_target = raw.detach().ge(0).eq(target).float()
        costs = (costs + .10 * F.binary_cross_entropy_with_logits(result.drive, drive_target, reduction="none")
                 + .12 * F.binary_cross_entropy_with_logits(result.interest, drive_target, reduction="none"))
    correct = raw.ge(0).eq(target).float()
    return tuple(torch.stack([metric[actual == k].mean(0) for k in (0, 1)]).mean(0)
                 for metric in (costs, correct))
