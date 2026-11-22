"""Serial conductance stages with causal delta memory and local critics."""
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attention-benchmark"))
from models import Conductance, UNITS


class Delta(nn.Module):
    """Reference recurrence inspired by https://arxiv.org/abs/2510.26692."""
    def __init__(self, features):
        super().__init__()
        self.project = nn.Linear(features, 5 * UNITS + 1)

    def forward(self, x):
        state = x.new_zeros(len(x), UNITS, UNITS)
        outputs = []
        for token in self.project(x).unbind(1):
            query, key, value, forget, output, write = token.split([UNITS] * 5 + [1], -1)
            query = F.normalize(query, dim=-1)
            key = F.normalize(key, dim=-1)
            decayed = forget.sigmoid().unsqueeze(-1) * state
            residual = value - (key.unsqueeze(1) @ decayed).squeeze(1)
            state = decayed + key.unsqueeze(-1) * (write.sigmoid() * residual).unsqueeze(1)
            outputs.append((query.unsqueeze(1) @ state).squeeze(1) * output.sigmoid())
        return torch.stack(outputs, 1)


class Adaptive(nn.Module):
    def __init__(self, task, variant):
        super().__init__()
        self.task = task
        self.variant = variant
        features = 3 if task == "occupancy" else 4
        targets = 1 if task == "occupancy" else 3
        self.delta = Delta(features)
        self.stages = nn.ModuleList(Conductance(features + 2 * UNITS) for _ in range(3))
        self.heads = nn.ModuleList(nn.Linear(UNITS, targets) for _ in range(3))
        self.critics = nn.ModuleList(Conductance(3 * UNITS + targets) for _ in range(3))
        self.scores = nn.ModuleList(nn.Linear(UNITS, 1) for _ in range(3))
        self.corrections = nn.ModuleList(nn.Linear(UNITS, UNITS) for _ in range(2))

    def forward(self, x, gaps, action=None, force=False, corrupt=False):
        hard = torch.isfinite(x).all(dim=(1, 2)) & (torch.isfinite(gaps) & (gaps > 0)).all(1)
        if action is not None:
            hard = hard & ((action == 0) | (action == 1))
            action = F.one_hot(action.clamp(0, 1), 2).to(x).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat((x, action), -1)
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        gaps = torch.where(torch.isfinite(gaps) & (gaps > 0), gaps, 1.)
        memory = self.delta(x)
        if self.variant == "no_delta":
            memory = torch.zeros_like(memory)
        context = x.new_zeros(*x.shape[:2], UNITS)
        evidence = torch.zeros_like(context)
        base = torch.cat((x, memory), -1)
        trace = []
        predictions, scores = [], []
        accepted = hard.clone()
        for i in range(3):
            prior = context
            context = self.stages[i](torch.cat((base, prior), -1), gaps)
            proposal = self.heads[i](context)
            upstream = evidence
            def verify(hidden, prediction):
                critic_input = torch.cat((hidden, prediction, memory, upstream), -1).detach()
                if corrupt:
                    critic_input = critic_input + 2 * torch.sign(critic_input + 1e-6)
                verdict = self.critics[i](critic_input, gaps.detach())
                return verdict, self.scores[i](verdict[:, -1]).squeeze(-1)

            evidence, score = verify(context, proposal)
            local_hard = hard & torch.isfinite(proposal).all(dim=(1, 2))
            initial_pass = local_hard & ((score >= 0) | (self.variant == "hard_only"))
            retry = local_hard & ~initial_pass
            if force and i == 1:
                retry = local_hard
            if i == 0 or self.variant in ("no_feedback", "delta_only"):
                retry = torch.zeros_like(retry)
            if self.variant == "hard_only" and not force:
                retry = torch.zeros_like(retry)
            change = torch.zeros_like(score)
            replay_change = torch.zeros_like(score)
            initial_score = score
            if retry.any():
                correction = self.corrections[i - 1](evidence.detach()).tanh()
                revised = prior + correction
                change = (self.heads[i - 1](revised) - self.heads[i - 1](prior)).abs().amax(dim=(1, 2)) * retry
                before = proposal
                context = torch.where(retry[:, None, None],
                                      self.stages[i](torch.cat((base, revised), -1), gaps), context)
                proposal = self.heads[i](context)
                replay_change = (proposal - before).abs().amax(dim=(1, 2))
                evidence, score = verify(context, proposal)
                local_hard = local_hard & torch.isfinite(proposal).all(dim=(1, 2))
            passed = local_hard & ((score >= 0) | (self.variant == "hard_only"))
            accepted = accepted & passed
            predictions.append(proposal[:, -1])
            scores.append(score)
            trace.append(dict(initial_pass=initial_pass, revise=retry, pass_after=passed,
                              hard=local_hard, abstain=~passed, initial_score=initial_score,
                              score=score, upstream_change=change, replay_change=replay_change))
            if self.variant != "delta_only":
                context = context + 0.1 * evidence.detach()
            else:
                evidence = torch.zeros_like(evidence)
        return predictions, scores, accepted, trace
