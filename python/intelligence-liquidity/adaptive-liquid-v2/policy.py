"""Two-decision offline fitted Q for the occupancy replay graph."""
import torch

ACTIONS = ("keep", "revise", "abstain")
RETRY = -0.06
ABSTAIN = -0.20
PROCESS_WEIGHT = 0.10


def outcome(logits, labels, cutoff):
    predicted = logits >= cutoff
    return torch.where(predicted == labels, 1., torch.where(labels, -2., -1.))


def state_features(proposal, score, previous, remaining):
    return torch.stack((proposal, score, previous, remaining.to(proposal)), 1)


def basis(x):
    x = x.clamp(-10, 10)
    return torch.cat((torch.ones_like(x[:, :1]), x, x.square()), 1)


def regress(x, targets):
    x = basis(x).double()
    penalty = 0.01 * torch.eye(x.shape[1], dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + penalty, x.T @ targets.double()).float()


def values(x, weights):
    return basis(x) @ weights


def replay(model, x, clock, route):
    states = {}
    def choose(stage, proposal, score, previous, remaining):
        states[stage] = state_features(proposal, score, previous, remaining)
        return torch.full_like(remaining, route[stage - 1])
    predictions, scores, accepted, trace = model(x, clock, choose)
    return states, predictions, scores, accepted, trace


def fit_policy(model, batches, cutoff):
    examples = [[], []]
    with torch.no_grad():
        for x, clock, y in batches:
            for first in (0, 1):
                keep = replay(model, x, clock, (first, 0))
                revise = replay(model, x, clock, (first, 1))
                keep_payoff = torch.where(keep[3], outcome(keep[1][-1], y, cutoff), ABSTAIN)
                revise_payoff = torch.where(revise[3], outcome(revise[1][-1], y, cutoff), ABSTAIN)
                attempted = revise[4][2].get("attempted", torch.zeros_like(y))
                quality = ((revise[1][-2] >= cutoff) == y).float()
                unchanged = ((keep[1][-2] >= cutoff) == y).float()
                examples[first].append((keep[0][1], keep[0][2], keep_payoff,
                                        revise_payoff + RETRY * attempted, quality - unchanged,
                                        attempted))
    data = [tuple(map(torch.cat, zip(*group))) for group in examples]
    second = torch.cat([group[1] for group in data])
    target_keep = torch.cat([group[2] for group in data])
    target_revise = torch.cat([group[3] for group in data])
    process = regress(second, torch.cat([group[4] for group in data]))
    q2 = regress(second, torch.stack((target_keep, target_revise,
                                     torch.full_like(target_keep, ABSTAIN)), 1))
    first_state = data[0][0]
    policies = {}
    for weight in (0., PROCESS_WEIGHT):
        continuation = []
        for _, next_state, *rest in data:
            q = values(next_state, q2)
            q[:, 1] += weight * values(next_state, process)
            continuation.append(q.max(1).values)
        early_quality = ((data[1][2] == 1).float() - (data[0][2] == 1).float())
        p1 = regress(first_state, early_quality)
        q1 = regress(first_state, torch.stack((
            continuation[0], RETRY + continuation[1],
            torch.full_like(continuation[0], ABSTAIN)), 1))
        policies[str(weight)] = dict(q1=q1, q2=q2, p1=p1, p2=process, weight=weight)
    return policies


def callback(policy, log):
    def choose(stage, proposal, score, previous, remaining):
        x = state_features(proposal, score, previous, remaining)
        q = values(x, policy["q" + str(stage)])
        quality = values(x, policy["p" + str(stage)])
        q[:, 1] += policy["weight"] * quality
        q[:, 1].masked_fill_(remaining <= 0, -torch.inf)
        action = q.argmax(1)
        log.append((x, q, quality, action))
        return action
    return choose
