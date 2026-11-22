"""Causal online updates for sensor forecasts and logged reward."""

from dataclasses import dataclass

import torch

from blackboard import DISCOUNT, action_loss, sensor_loss


@dataclass(frozen=True)
class Event:
    time: int
    evidence: torch.Tensor
    minutes: torch.Tensor
    before: object


class Online:
    def __init__(self, model, max_gap, rate=.002):
        self.model = model
        self.max_gap = max_gap
        self.optimizer = torch.optim.SGD(model.parameters(), lr=rate)
        self.previous = None

    def feedback(self, current, decision, *, action=None, reward=None):
        if (action is None) != (reward is None):
            raise ValueError("Feedback requires both the previous executed action and its outcome")
        previous = self.previous
        if previous is not None and current.time <= previous.time:
            raise ValueError("Feedback must be in time order")
        self.previous = current
        if previous is None or current.time - previous.time > self.max_gap:
            return None
        self.optimizer.zero_grad()
        result = self.model(previous.evidence, previous.minutes, previous.before)
        self_supervised = sensor_loss(result, current.evidence)
        value_error = self_supervised.new_zeros(())
        if action is not None:
            target = reward + DISCOUNT * decision.final.aggregate.detach().amax(-1).clamp_max(0.)
            value_error = action_loss(result, action, target)
        loss = self_supervised + value_error
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite online loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25, error_if_nonfinite=True)
        self.optimizer.step()
        return {"sensor_loss": self_supervised.detach().item(),
                "reward_loss": value_error.detach().item() if action is not None else None}
