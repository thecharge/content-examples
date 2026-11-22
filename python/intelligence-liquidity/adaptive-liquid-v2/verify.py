"""Numerical, causal and transactional checks, before any fresh scoring."""
import json
from math import exp

import torch
from torch.nn import functional as F

from adaptive import Adaptive, Delta, Liquid, liquid_step
from experiment import UCI, office, occupancy_windows, pack, normalizer
from policy import callback, fit_policy, replay


def solver():
    h, a, b, c, dt = map(torch.tensor, (0.3, 2., 0.8, 0.7, 0.6))
    expected = 0.4 + (0.3 - 0.4) * exp(-0.6 * 2 / 0.7)
    assert abs(float(liquid_step(h, a, b, c, dt, True)) - expected) < 1e-7
    assert abs(float(liquid_step(h, a, b, c, dt, False)) - (0.7 * 0.3 + 0.6 * 0.8) /
               (0.7 + 0.6 * 2)) < 1e-7
    cell = Liquid(3, 4).double()
    with torch.no_grad():
        cell.recurrent.weight.fill_(0.8)
        x = torch.randn(4, 3, 3, dtype=torch.double)
        clock = torch.tensor([[0.4, 1.8, 0.7]] * len(x), dtype=x.dtype)
        cell.substeps = 2048
        reference = cell(x, clock)
        errors = {}
        for flag in (False, True):
            cell.exponential = flag
            errors[str(flag)] = []
            for steps in (2, 4, 8, 16):
                cell.substeps = steps
                errors[str(flag)].append(float((cell(x, clock) - reference).abs().max()))
            assert all(a > b for a, b in zip(errors[str(flag)], errors[str(flag)][1:]))
    return dict(fixed_expected=expected, refinement_max_error=errors)


def memory(x, dt):
    model = Delta(3, 4)
    actual = model(x, dt)
    assert torch.allclose(actual[:, :4], model(x[:, :4], dt[:, :4]), atol=1e-8)
    elapsed = dt.clone()
    elapsed[:, 4] *= 5
    changed = model(x, elapsed)
    assert torch.equal(actual[:, :4], changed[:, :4])
    assert not torch.allclose(actual[:, 4], changed[:, 4])
    state = x.new_zeros(len(x), 4, 4)
    outputs = []
    for t in range(x.shape[1]):
        q, k, v, w, o = model.project(x[:, t]).chunk(5, 1)
        q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
        decay = torch.exp(-(F.softplus(model.rate) + 1e-6) * dt[:, t:t + 1])
        state = torch.einsum("bi,bij->bij", decay, state)
        residual = v - torch.einsum("bi,bij->bj", k, state)
        state = state + torch.einsum("bi,bj->bij", k, w.sigmoid() * residual)
        outputs.append(torch.einsum("bi,bij->bj", q, state) * o.sigmoid())
    assert torch.allclose(actual, model.strength * torch.stack(outputs, 1), atol=1e-8)
    return float((actual - changed).abs().max())


def transaction(model, x, dt):
    reports = [[], [], []]
    inputs = [[], [], []]
    def capture(i):
        def hook(module, args, result):
            inputs[i].append(args[0].clone())
            reports[i].append(result.clone())
        return hook
    handles = [m.register_forward_hook(capture(i)) for i, m in enumerate(model.critics)]
    predictions, scores, accepted, trace = model(
        x, dt, lambda stage, p, s, prev, left: torch.full_like(left, int(stage == 1)))
    for h in handles:
        h.remove()
    assert [len(v) for v in reports] == [2, 2, 1]
    assert accepted.all() and trace[1]["retained"].all()
    assert torch.equal(predictions[0], trace[1]["revised_prior"][:, -1, 0])
    assert torch.equal(scores[0], trace[1]["revised_score"])
    assert torch.equal(inputs[1][1][:, :, -model.width:], reports[0][1])
    assert torch.equal(inputs[2][0][:, :, -model.width:], reports[1][1])
    old = trace[0]["proposal"][:, -1, 0]
    upstream = float((predictions[0] - old).abs().max())
    assert upstream > 1e-5
    assert not torch.equal(reports[0][0], reports[0][1])
    assert not torch.equal(predictions[1], trace[1]["proposal"][:, -1, 0])
    invalid = dt.clone()
    invalid[0, 2] = 0
    veto = replay(model, x, invalid, (1, 1))
    assert not veto[3][0] and not veto[4][1]["attempted"][0]
    exhausted = model(x, dt, lambda stage, p, s, prev, left: torch.ones_like(left), budget=1)
    assert exhausted[2].all() and exhausted[3][1]["attempted"].all()
    assert "attempted" not in exhausted[3][2]
    seen = []
    def poison_recheck(module, args, result):
        seen.append(1)
        return result * torch.nan if len(seen) == 2 else result
    handle = model.critics[0].register_forward_hook(poison_recheck)
    rejected = replay(model, x, dt, (1, 0))
    handle.remove()
    assert len(seen) == 2 and rejected[4][1]["attempted"].all()
    assert not rejected[4][1]["retained"].any()
    assert torch.equal(rejected[1][0], old)
    return dict(calls=[len(v) for v in reports], max_prior_change=upstream,
                max_rechecked_evidence_change=float((reports[0][0] - reports[0][1]).abs().max()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(41)
    raw = office(UCI, "datatraining.txt")[:500]
    x, dt, y = pack(occupancy_windows([raw], 5, 1), normalizer([r[2] for r in raw]))
    model = Adaptive(width=8)
    with torch.no_grad():
        receipt = dict(solver=solver(), delta_gap_change=memory(x[:64], dt[:64]),
                       transaction=transaction(model, x[:64], dt[:64]))
        policies = fit_policy(model, [(x, dt, y)], 0.)
        log = []
        predictions = model(x[:16], dt[:16], callback(policies["0.1"], log))
        assert len(log) == 2 and torch.isfinite(policies["0.1"]["q1"]).all()
        receipt["policy_sample"] = [a.bincount(minlength=3).tolist() for _, _, _, a in log]
    print(json.dumps(receipt, indent=2))
