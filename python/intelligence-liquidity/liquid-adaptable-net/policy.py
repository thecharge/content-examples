"""Per-expert contextual action values and learned feature-group selectors."""

import torch
from torch import nn
from torch.nn import functional as F

from reference import WIDTH

ACTIONS = (
    "learn",
    "spawn",
    "deepen",
    "retire",
    "share",
    "split",
    "grow_heads",
    "prune_heads",
    "add_layer",
    "remove_layer",
    "branch_attention",
    "prune_attention",
    "prune_neuron",
    "fork_network",
    "retire_network",
    "split_liquid",
    "prune_liquid",
)


class Policy(nn.Module):
    def __init__(self, features, replay_capacity=64):
        super().__init__()
        self.action = nn.Sequential(
            nn.Linear(WIDTH + 8, 24), nn.SiLU(), nn.Linear(24, len(ACTIONS))
        )
        self.group = nn.Linear(WIDTH, features)
        self.register_buffer("replay_context", torch.zeros(replay_capacity, WIDTH + 8))
        self.register_buffer(
            "replay_action", torch.zeros(replay_capacity, dtype=torch.long)
        )
        self.register_buffer("replay_reward", torch.zeros(replay_capacity))
        self.register_buffer("replay_count", torch.tensor(0))
        self.register_buffer("replay_cursor", torch.tensor(0))
        with torch.no_grad():
            self.action[-1].weight.zero_()
            self.action[-1].bias.zero_()
            self.action[-1].bias[0] = 0.01
            self.group.weight.zero_()
            self.group.bias.zero_()

    def extend(self, old_names, new_names):
        old = self.group
        group = nn.Linear(WIDTH, len(new_names)).to(old.weight)
        with torch.no_grad():
            group.weight.zero_()
            group.bias.zero_()
            for previous, name in enumerate(old_names):
                current = new_names.index(name)
                group.weight[current].copy_(old.weight[previous])
                group.bias[current].copy_(old.bias[previous])
        self.group = group

    def choose(self, context, allowed, generator, exploration):
        with torch.no_grad():
            utilities = self.action(context).flatten()
            mask = torch.tensor(allowed, dtype=torch.bool)
            if len(mask) < len(ACTIONS):
                mask = F.pad(mask, (0, len(ACTIONS) - len(mask)), value=False)
            utilities = utilities.masked_fill(~mask, -torch.inf)
            if float(torch.rand((), generator=generator)) < exploration:
                # Exploration belongs to the learned contextual bandit, not a rotation.
                index = int(torch.multinomial(mask.float(), 1, generator=generator))
            else:
                index = int(utilities.argmax())
        return index, float(utilities[index])

    def learn_action(self, context, action, measured_return, rate=0.02, updates=8):
        with torch.no_grad():
            position = int(self.replay_cursor)
            self.replay_context[position].copy_(context)
            self.replay_action[position] = action
            self.replay_reward[position] = measured_return
            self.replay_cursor.fill_((position + 1) % len(self.replay_reward))
            self.replay_count.fill_(
                min(len(self.replay_reward), int(self.replay_count) + 1)
            )
        count = int(self.replay_count)
        optimizer = torch.optim.Adam(self.action.parameters(), lr=rate)
        for _ in range(updates):
            optimizer.zero_grad()
            value = (
                self.action(self.replay_context[:count])
                .gather(1, self.replay_action[:count, None])
                .flatten()
            )
            loss = F.smooth_l1_loss(value, self.replay_reward[:count])
            loss.backward()
            optimizer.step()

    def learn_group(self, hidden, relevance, rate=0.02, updates=12):
        optimizer = torch.optim.Adam(self.group.parameters(), lr=rate)
        for _ in range(updates):
            optimizer.zero_grad()
            loss = F.mse_loss(self.group(hidden).sigmoid().flatten(), relevance)
            loss.backward()
            optimizer.step()

    def select_group(self, hidden, names, fraction=0.5):
        scores = self.group(hidden).detach().sigmoid().flatten()
        eligible = [i for i, name in enumerate(names) if not name.startswith("parent:")]
        count = max(1, int(torch.ceil(torch.tensor(len(eligible) * fraction))))
        chosen = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:count]
        return [names[i] for i in chosen]
