"""Learn from logged heater actions and compare both actions in a fitted simulator."""

import copy
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from model import Workflow, process_target
from run import dump, io_data, save, sha


def tensors(rows):
    x, next_x, action, reward = io_data.heat_tensors(rows)
    measured = torch.tensor([(row[4], row[5]) for row in rows])
    return x, next_x, action, reward, measured


def control_loss(model, target, rows, prior=None):
    x, next_x, action, reward, _ = rows
    elapsed = x.new_full((len(x),), 15.)
    attempt = model.step(x, elapsed, prior, train_alternatives=True)
    with torch.no_grad():
        next_value = target.step(next_x, elapsed, attempt.memory.detach()).q.max(-1).values
    estimate = attempt.q.gather(1, action[:, None]).squeeze(1)
    actual_error = (attempt.memory.forecast.detach()-next_x).abs().mean(-1)
    hybrid = reward - .1 * actual_error.clamp_max(1)
    loss = F.smooth_l1_loss(estimate, hybrid + .6 * next_value.clamp(-5,0))
    loss = loss + .12 * (attempt.q.logsumexp(-1)-estimate).mean()
    loss = loss + .4 * F.smooth_l1_loss(attempt.memory.forecast, next_x)
    for first, alternative, order, router in attempt.attempts:
        predictors = []
        for claim in (first, alternative):
            choice = claim.q.gather(1,action[:, None]).squeeze(1)
            observed = process_target(claim,x,next_x)
            # Grade the process using observed forecast error and the action's
            # measured reward. Penalize optimistic scores on the logged action.
            measured = torch.exp((reward-(choice.detach()-reward).abs()).clamp_max(0))*2-1
            quality = .65*observed + .35*measured
            predictors.append(quality)
            loss = loss + .08 * F.mse_loss(quality, claim.process)
            loss = loss + .08 * F.smooth_l1_loss(claim.reconstructed, x)
            loss = loss + .08 * F.smooth_l1_loss(claim.forecast, next_x)
            loss = loss + .07 * F.smooth_l1_loss(choice, reward)
        best = torch.where(predictors[0]>=predictors[1],order[:,0],order[:,1])
        loss = loss + .05*F.nll_loss(router.clamp_min(1e-8).log(),best)
    return loss, attempt


def thermal_features(x, action):
    """Indoor response to outside temperature and logged heater state."""
    indoor, outdoors = x[:, 0] * 4 + 22, x[:, 1] * 10 + 5
    return torch.stack((torch.ones_like(indoor), outdoors-indoor, action.float()), -1)


def simulate(x, action, calibration):
    rates, energy = calibration
    delta = thermal_features(x, action) @ rates
    start = x[:,0]*4+22
    endpoint = start + delta
    # Average distance outside the target band over the 15-minute step.
    points = start[:,None] + delta[:,None]*torch.arange(1,16)/15
    discomfort = torch.maximum((20-points).clamp_min(0), (points-24).clamp_min(0)).mean(-1)*.25
    consumption = energy[action]
    return endpoint, consumption, discomfort


def fit(registry, measured):
    x, next_x, action, _, real = registry
    rate = torch.linalg.lstsq(thermal_features(x, action), (next_x[:,0]-x[:,0])*4).solution
    energy = torch.stack([real[action==i,0].mean() for i in (0,1)])
    # A heater must supply heat, and indoor temperature must respond to outside temperature.
    if not bool((rate[1:]>=0).all()):
        raise ValueError(f"Measured fit does not support a positive thermal response: {rate.tolist()}")
    actual_x, actual_next, actual_action, _, actual = measured
    endpoint, power, discomfort = simulate(actual_x,actual_action,(rate,energy))
    stats = dict(temperature_mae_c=float((endpoint-(actual_next[:,0]*4+22)).abs().mean()),
                 energy_mae_kwh=float((power-actual[:,0]).abs().mean()),
                 discomfort_mae_degree_hours=float((discomfort-actual[:,1]).abs().mean()),
                 observed_rows=len(actual), coefficients=rate.tolist(), energy_kwh=energy.tolist())
    return (rate,energy),stats


def fit_model(rows, out):
    model = Workflow()
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    index = round(.8 * len(rows[0]))
    training = tuple(t[:index] for t in rows)
    valid = tuple(t[index:] for t in rows)
    best, curve, selected = float("inf"), [], 0
    for epoch in range(1, 16):
        values = torch.randperm(index, generator=torch.Generator().manual_seed(41+epoch))
        total, train_rows = 0., 0
        for i in values.split(128):
            optimizer.zero_grad()
            loss, _ = control_loss(model,target,tuple(t[i] for t in training))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total+=float(loss.detach())*len(i)
            train_rows+=len(i)
        target.load_state_dict(model.state_dict())
        with torch.no_grad():
            checks = []
            for i in torch.arange(len(valid[0])).split(128):
                x, _, action, outcome, _ = tuple(t[i] for t in valid)
                p = model.step(x,x.new_full((len(x),),15.)).q.gather(1,action[:,None]).squeeze(1)
                checks += (p-outcome).abs().tolist()
        score = sum(checks)/len(checks)
        curve.append(dict(epoch=epoch,training_loss=total/train_rows,validation_reward_error=score))
        if score<best:
            best,selected=score,epoch
            save(out/"heater.pt",model.state_dict())
        save(out/"heater-resume.pt",dict(model=model.state_dict(),optimizer=optimizer.state_dict(),curve=curve))
        if epoch-selected>=4:
            break
    return dict(curve=curve,selected=selected,train_rows=index,validation_rows=len(valid[0]),sha256=sha(out/"heater.pt"))


def replay(log, train, test, out, calibration):
    model = Workflow()
    start = torch.load(out/"heater.pt",weights_only=True,map_location="cpu")
    model.load_state_dict(start)
    target = copy.deepcopy(model).eval()
    optimizer = torch.optim.SGD(model.parameters(),lr=.001)
    ledger, previous_end = None, None
    trace = []
    train_states = [torch.tensor([r[1] for r in train if r[3]==action]) for action in (0,1)]
    normalized = (torch.tensor([r[1] for r in log])-torch.tensor((22.,5.)))/torch.tensor((4.,10.))
    scaled = torch.tensor((.7,2.))
    for i,row in enumerate(log):
        if previous_end!=row[0]:
            ledger=None
        one = tuple(t[i:i+1] for t in test)
        actual_x = torch.tensor(row[1])
        counts = [int((((states-actual_x).abs()/scaled)<=1).all(-1).sum()) for states in train_states]
        with torch.no_grad():
            result = model.step(normalized[i:i+1],normalized.new_tensor([15.]),ledger,trace=True)
            predicted = result.q[0].tolist()
            chosen = int(result.q[0].argmax()) if min(counts)>=8 and bool(result.accepted[0]) else None
            a0, a1 = torch.tensor([0,1]), normalized[i:i+1].repeat(2,1)
            temp, energy, discomfort = simulate(a1,a0,calibration)
            estimates = [dict(action=j,temperature_c=float(temp[j]),energy_kwh=float(energy[j]),
                              discomfort_degree_hours=float(discomfort[j]),
                              reward=float(-energy[j]-2*discomfort[j])) for j in (0,1)]
            trace.append(dict(minute=row[0],chosen=chosen,logged_action=row[3],considered=counts,
                              recorded_energy_kwh=row[4],recorded_discomfort_degree_hours=row[5],
                              action_value=predicted,simulated_pair=estimates,
                              attempts=result.attempts if i<5 else []))
        # All fields above are recorded before the outcome updates the model.
        optimizer.zero_grad()
        loss, attempt = control_loss(model,target,one,ledger)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        optimizer.step()
        if i%32==31:
            target.load_state_dict(model.state_dict())
        ledger=result.memory.detach()
        previous_end=row[0]+15
    change = sum(float((p.detach()-start[n]).abs().sum()) for n,p in model.named_parameters()
                 if ".experts." in n)
    return trace,change


def main(data,out):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    archive = data/"six-buildings.zip"
    train, test = io_data.heat(archive)
    train_t, test_t = tensors(train),tensors(test)
    thermal, real_errors = fit(train_t,test_t)
    training = fit_model(train_t,out)
    trace,change = replay(test,train,test_t,out,thermal)
    active = [row for row in trace if row["chosen"] is not None]
    averages = {}
    if active:
        for policy in ("model","always_on","always_off"):
            values = [r["simulated_pair"][r["chosen"] if policy=="model" else int(policy=="always_on")] for r in active]
            averages[policy] = {k:sum(v[k] for v in values)/len(values)
                                for k in ("energy_kwh","discomfort_degree_hours","reward")}
    dump(out/"heating.json",dict(dataset_sha256=sha(archive),fit=training,
        recorded_rows=len(trace),simulator_checked_on_logged_actions=real_errors,
        simulator_kind="Fitted one-step thermal model; action pairs are simulated, not measured.",
        simulated_same_start=averages,expert_weight_change=change, eligible=len(active),
        picked_on=sum(r["chosen"]==1 for r in active),picked_off=sum(r["chosen"]==0 for r in active),
        trace=trace))


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:]))
