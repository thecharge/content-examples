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


def board_predictions(model, examples, checks=True):
    state, previous = examples[0].before, examples[0].timestamp - 300
    answers, probabilities, committed = [], [], []
    with torch.no_grad():
        for item in examples:
            if item.timestamp - previous != 300:
                state = model.start(item.inputs)
            result = model(item.inputs, item.inputs.new_tensor([5.]), state, checks=checks)
            answers.append(result.answer)
            probabilities.append(result.probability)
            committed.append(result.committed)
            state, previous = result.state, item.timestamp
    return tuple(torch.cat(group) for group in (answers, probabilities, committed))


def by_class(values, labels):
    return torch.stack([values[labels == value].float().mean(0) for value in (0, 1)]).mean(0)


def paired_margin(correct_gain, actual, *, block=24, samples=400):
    # Resample continuous chunks to retain some of the dependence between neighbors.
    generator = torch.Generator().manual_seed(531)
    size = len(actual)
    if size < block:
        return None
    origins = torch.randint(size, (samples, (size + block - 1) // block), generator=generator)
    indices = (origins.unsqueeze(-1) + torch.arange(block)).flatten(1)[:, :size] % size
    labels = actual[indices]
    valid = (labels.eq(0).sum(1) >= 16) & (labels.eq(1).sum(1) >= 16)
    if int(valid.sum()) < samples // 2:
        return None
    delta = correct_gain[indices][valid]
    labels = labels[valid]
    gains = torch.stack([(delta * labels.eq(c)).sum(1) / labels.eq(c).sum(1) for c in (0, 1)]).mean(0)
    return float(torch.quantile(gains, .05))


def deployment_gain(original, candidate, check, checks=True):
    expected = torch.cat([item.truth for item in check])
    old_answer, old_probability, old_quorum = board_predictions(original, check, checks)
    new_answer, new_probability, new_quorum = board_predictions(candidate, check, checks)
    old_right = old_answer.eq(expected).float()
    new_right = new_answer.eq(expected).float()
    gain = new_right - old_right
    return dict(accuracy_gain=float(by_class(gain, expected)),
                accuracy_lower=paired_margin(gain, expected),
                brier_gain=float(by_class((old_probability - expected).square() -
                                         (new_probability - expected).square(), expected)),
                quorum_gain=float(new_quorum.float().mean() - old_quorum.float().mean()),
                class_gain=[float(gain[expected.eq(c)].mean()) for c in (0, 1)])


class Feedback:
    def __init__(self, model, *, minimum=288, minority=16, rate=.025, checks=True):
        self.model = model
        self.minimum = minimum
        self.minority = minority
        self.rate = rate
        self.checks = checks
        self.history = (deque(maxlen=224), deque(maxlen=224))
        self.check = []
        self.latest = None
        self.receipts = []

    def observe(self, timestamp, now, row, before, issued, actual):
        if timestamp >= now or (self.latest is not None and timestamp <= self.latest):
            raise ValueError("Only previously issued, newly revealed outcomes can grade updates")
        if actual.numel() != 1 or int(actual) not in (0, 1):
            raise ValueError("An external binary occupancy answer is required")
        self.latest = timestamp
        reward = measured_reward(issued.rounds, actual).detach()
        self.check.append(Example(timestamp, row.detach(), before.detached(), actual.detach(), reward))
        if len(self.check) < self.minimum:
            return reward
        fit = sorted((*self.history[0], *self.history[1]), key=lambda item: item.timestamp)
        classes = torch.cat([sample.truth for sample in self.check]).bincount(minlength=2)
        if min(map(len, self.history)) < self.minority or classes.min() < self.minority:
            receipt = dict(status="frozen_insufficient_classes", accepted=[])
        else:
            receipt = self.try_update(fit, self.check)
        self.receipts.append(dict(revealed_at=now, check_first=self.check[0].timestamp,
                                  check_last=self.check[-1].timestamp, check_classes=classes.tolist(),
                                  **receipt))
        for item in self.check:
            self.history[int(item.truth)].append(item)
        self.check.clear()
        return reward

    def try_update(self, fit, check):
        if fit[-1].timestamp >= check[0].timestamp:
            raise ValueError("Candidate fitting must precede its independent check")
        candidate = deepcopy(self.model)
        params = tuple(candidate.expert_parameters())
        initial, _ = measurements(candidate, fit, train=True)
        gradients = torch.autograd.grad(initial.mean(), params, allow_unused=True)
        with torch.no_grad():
            for param, gradient in zip(params, gradients, strict=True):
                if gradient is not None:
                    param.add_(gradient.clamp(-1., 1.), alpha=-self.rate)
            after, _ = measurements(candidate, fit, train=True)
            # Pick one proposal using the fitting block only. One proposal gets
            # one look at the later block. A rejected proposal cannot reuse it.
            chosen = int((initial - after).argmax())
            for old, new in zip(self.model.expert_parameters(), candidate.expert_parameters(), strict=True):
                keep = new[chosen].clone()
                new.copy_(old)
                new[chosen].copy_(keep)
            gain = deployment_gain(self.model, candidate, check, self.checks)
            accept = (gain["accuracy_gain"] >= .01 and gain["brier_gain"] >= .002
                      and gain["accuracy_lower"] is not None and gain["accuracy_lower"] > 0
                      and min(gain["class_gain"]) >= -.02 and gain["quorum_gain"] >= -.02)
            if accept:
                self.model.load_state_dict(candidate.state_dict())
        return dict(status="accepted" if accept else "frozen_no_board_gain",
                    fit_first=fit[0].timestamp, fit_last=fit[-1].timestamp,
                    fit_rows=len(fit), check_rows=len(check), candidate_expert=chosen + 1,
                    board_gain=gain, accepted=[chosen + 1] if accept else [])


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
