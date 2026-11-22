"""Clock-aware liquid stages and transactional upstream replay."""
import torch
from torch import nn
from torch.nn import functional as F


def liquid_step(h, a, b, c, dt, exponential):
    if exponential:
        return h + (b / a - h) * (-torch.expm1(-dt * a / c))
    return (c * h + dt * b) / (c + dt * a)


class Liquid(nn.Module):
    def __init__(self, features, width, exponential=True, substeps=3):
        super().__init__()
        self.sensory = nn.Linear(features, width)
        self.recurrent = nn.Linear(width, width, bias=False)
        self.capacitance = nn.Parameter(torch.zeros(width))
        self.leak = nn.Parameter(torch.zeros(width))
        self.rest = nn.Parameter(torch.zeros(width))
        self.reversal = nn.Parameter(torch.ones(width))
        self.exponential = exponential
        self.substeps = substeps

    def forward(self, x, minutes):
        c = F.softplus(self.capacitance) + 1e-4
        leak = F.softplus(self.leak) + 1e-4
        h = x.new_zeros(len(x), len(c))
        states = []
        for drive, gap in zip(self.sensory(x).unbind(1), minutes.unbind(1)):
            dt = gap[:, None] / self.substeps
            for _ in range(self.substeps):
                conductance = F.softplus(drive + self.recurrent(h))
                h = liquid_step(h, leak + conductance,
                                leak * self.rest + conductance * self.reversal,
                                c, dt, self.exponential)
            states.append(h)
        return torch.stack(states, 1)


class Delta(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.project = nn.Linear(features, 5 * width)
        self.rate = nn.Parameter(torch.full((width,), -2.0))
        self.strength = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, minutes):
        width = len(self.rate)
        state = x.new_zeros(len(x), width, width)
        outputs = []
        rate = F.softplus(self.rate) + 1e-6
        for token, gap in zip(self.project(x).unbind(1), minutes.unbind(1)):
            query, key, value, write, output = token.chunk(5, -1)
            query, key = F.normalize(query, dim=-1), F.normalize(key, dim=-1)
            state = state * torch.exp(-gap[:, None, None] * rate[None, :, None])
            residual = value - (key[:, None] @ state).squeeze(1)
            state = state + key[:, :, None] * (write.sigmoid() * residual)[:, None]
            outputs.append((query[:, None] @ state).squeeze(1) * output.sigmoid())
        return self.strength * torch.stack(outputs, 1)


class Adaptive(nn.Module):
    def __init__(self, features=3, width=12, delta=True, correction=True):
        super().__init__()
        self.width = width
        self.delta = Delta(features, width) if delta else None
        self.correction = correction
        self.stages = nn.ModuleList(Liquid(features + 2 * width, width) for _ in range(3))
        self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(3))
        self.critics = nn.ModuleList(Liquid(2 * width + 1, width) for _ in range(3))
        self.scores = nn.ModuleList(nn.Linear(width, 1) for _ in range(3))
        self.revisions = nn.ModuleList(nn.Linear(width, width) for _ in range(2))

    def forward(self, x, minutes, revise=None, budget=2):
        hard = torch.isfinite(x).all((1, 2)) & (torch.isfinite(minutes) & (minutes > 0)).all(1)
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        minutes = torch.where(torch.isfinite(minutes) & (minutes > 0), minutes, 1.)
        zero = x.new_zeros(*x.shape[:2], self.width)
        base = torch.cat((x, zero if self.delta is None else self.delta(x, minutes)), -1)
        states, proposals, evidence, scores, trace = [], [], [], [], []
        remaining = torch.full_like(hard, min(2, max(0, budget)), dtype=torch.long)
        abstained = torch.zeros_like(hard)

        def verify(i, state, proposal, upstream):
            value = self.critics[i](torch.cat((state, proposal, upstream), -1).detach(), minutes)
            return value, self.scores[i](value[:, -1]).squeeze(-1)

        def select(mask, new, previous):
            return torch.where(mask.reshape((-1,) + (1,) * (new.ndim - 1)), new, previous)

        for i in range(3):
            previous = states[-1] if i else zero
            upstream = evidence[-1] if i else zero
            state = self.stages[i](torch.cat((base, previous), -1), minutes)
            proposal = self.heads[i](state)
            report, score = verify(i, state, proposal, upstream)
            trace.append(dict(stage=i, proposal=proposal, score=score))
            if i and self.correction and revise is not None:
                eligible = hard & ~abstained & (remaining > 0) & torch.isfinite(report).all((1, 2))
                requested = revise(i, proposal[:, -1, 0], score, scores[-1], remaining)
                abstained = abstained | (requested == 2)
                attempted = eligible & (requested == 1)
                trace[-1].update(requested=requested, eligible=eligible)
                if attempted.any():
                    remaining = remaining - attempted.long()
                    revised = previous + self.revisions[i - 1](report.detach()).tanh()
                    prior_prediction = self.heads[i - 1](revised)
                    prior_report, prior_score = verify(
                        i - 1, revised, prior_prediction, evidence[i - 2] if i > 1 else zero)
                    replay = self.stages[i](torch.cat((base, revised), -1), minutes)
                    replay_prediction = self.heads[i](replay)
                    replay_report, replay_score = verify(i, replay, replay_prediction, prior_report)
                    retained = attempted.clone()
                    for value in (revised, prior_prediction, prior_report, prior_score,
                                  replay, replay_prediction, replay_report, replay_score):
                        retained &= torch.isfinite(value.reshape(len(x), -1)).all(1)
                    prior_revision = (revised, prior_prediction, prior_report, prior_score)
                    for history, value in zip((states, proposals, evidence, scores), prior_revision):
                        history[-1] = select(retained, value, history[-1])
                    state = select(retained, replay, state)
                    proposal = select(retained, replay_prediction, proposal)
                    report = select(retained, replay_report, report)
                    score = select(retained, replay_score, score)
                    trace[-1].update(attempted=attempted, retained=retained,
                                     revised_prior=prior_prediction, revised_evidence=prior_report,
                                     revised_score=prior_score, replay=replay_prediction,
                                     replay_score=replay_score, remaining=remaining)
            states.append(state)
            proposals.append(proposal)
            evidence.append(report)
            scores.append(score)
        accepted = hard & ~abstained
        for proposal, report, score in zip(proposals, evidence, scores):
            accepted &= (torch.isfinite(proposal).all((1, 2)) &
                         torch.isfinite(report).all((1, 2)) & torch.isfinite(score))
        return [p[:, -1, 0] for p in proposals], scores, accepted, trace
