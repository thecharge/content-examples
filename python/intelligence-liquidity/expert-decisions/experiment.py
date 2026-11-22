"""Five selective liquid experts and an optional independent evidence review."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "liquid-workflow-v3"))
from blackboard import LiquidODE, WIDTH, OCCUPANCY
from data import occupancy, sha
from train import auc, balanced, boundaries, windows

NAMES = ("air", "thermal", "light", "change", "mixed")
VIEWS = ((0, 1, 5, 6), (2, 3, 7, 8), (4, 9), (5, 6, 7, 8, 9), (0, 2, 3, 4))
COST = .10
CURIOSITY = .15


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("views", torch.tensor([[j in view for j in range(10)] for view in VIEWS]))
        self.liquid = LiquidODE(20, len(VIEWS))
        self.read = nn.Parameter(torch.zeros(len(VIEWS), 20 + WIDTH))
        self.bias = nn.Parameter(torch.zeros(len(VIEWS)))

    def forward(self, x, state=None):
        present = x.isfinite()[:, None] & self.views
        values = torch.where(present, x[:, None], 0.)
        sensor = torch.cat((values, present.float()), -1)
        ids = torch.arange(len(VIEWS), device=x.device).repeat(len(x))
        if state is None:
            state = x.new_zeros(len(ids), WIDTH)
        out = self.liquid(sensor.flatten(0, 1), state, x.new_full((len(ids),), 5.), ids)
        # Age the hidden state on missing evidence without emitting an unsupported vote.
        active = present.any(-1).flatten()
        state = torch.where(active[:, None], out, state * .9)
        scores = (torch.cat((sensor, state.reshape(len(x), len(VIEWS), WIDTH)), -1) * self.read).sum(-1)
        return scores + self.bias, state


@torch.no_grad()
def infer(model, stream):
    times, x, _ = stream
    output, state = [], None
    for i, row in enumerate(x):
        if i == 0 or times[i]-times[i-1] != 300:
            state = None
        score, state = model(row[None], state)
        output.append(score[0])
    return torch.stack(output)


def fit(train, validation, epochs):
    torch.manual_seed(51)
    model = Experts()
    opt = torch.optim.AdamW(model.parameters(), lr=.008)
    labels = torch.cat([s[2] for s in train])
    freq = labels.bincount(minlength=2).float()
    weights = freq.mean()/freq
    groups = windows(train,300)
    best, curve, checkpoint = -float("inf"), [], None
    for epoch in range(1, epochs+1):
        for x,y in groups:
            for ids in torch.randperm(len(x)).split(128):
                opt.zero_grad()
                state, losses = None, []
                for i in range(x.shape[1]-1):
                    signals, state = model(x[ids,i], state)
                    if i < 3:
                        continue
                    truth = y[ids,i,None].expand_as(signals)
                    present = (x[ids,i].isfinite()[:,None] & model.views).any(-1)
                    cost = F.binary_cross_entropy_with_logits(signals,truth.float(),reduction="none")
                    losses.append((cost*present*weights[truth]).sum()/present.sum().clamp_min(1))
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
        result = infer(model, validation)
        score = sum(auc(validation[2], result[:,i]) for i in range(len(VIEWS)))/len(VIEWS)
        curve.append(score)
        if score > best:
            best, selected = score, epoch
            checkpoint = {k:v.detach().clone() for k,v in model.state_dict().items()}
        print("epoch",epoch,"validation mean expert AUC",round(score,4),flush=True)
        if epoch-selected>=4:
            break
    model.load_state_dict(checkpoint)
    return model, selected, curve


def calibrate(raw, y):
    # One slope and intercept per expert. Calibration only sees development labels.
    slope, bias = nn.Parameter(raw.new_ones(raw.shape[1])), nn.Parameter(raw.new_zeros(raw.shape[1]))
    opt = torch.optim.LBFGS([slope,bias], max_iter=60, line_search_fn="strong_wolfe")
    truth = y[:,None].expand_as(raw).float()
    def objective():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(raw*slope+bias,truth)+.005*(slope-1).square().mean()
        loss.backward()
        return loss
    opt.step(objective)
    return slope.detach(), bias.detach()


def choose(p, coverage, stamps):
    vote = p >= .5
    confidence = torch.where(vote,p,1-p)
    utility = 2*confidence-1-COST
    # One deterministic experiment slot per row. The bonus changes willingness,
    # never confidence or the outcome reward. An empty view always abstains.
    curious = ((stamps//300) % len(VIEWS))[:,None] == torch.arange(len(VIEWS))
    drive = utility + CURIOSITY*curious*coverage
    act = coverage.gt(0) & drive.ge(0)
    return vote, confidence, act, utility, curious & act & utility.lt(0)


def combine(p, active):
    count = active.sum(-1)
    probability = torch.where(active,p,0.).sum(-1)/count.clamp_min(1)
    return torch.where(count>0,probability,.5), count>0


def verify(p, act, counter, counter_available, threshold):
    vote = p>=.5
    objection = act & counter_available & (vote != (counter>=.5)) & (torch.maximum(counter,1-counter)>=threshold)
    return act & ~objection, objection


def fit_counter(train, dev):
    x=torch.cat([s[1] for s in train]); y=torch.cat([s[2] for s in train])
    models, raw = [], []
    for view in VIEWS:
        columns = [i for i in range(10) if i not in view]
        fit=xgb.DMatrix(x[:,columns].numpy(),label=y.numpy())
        valid=xgb.DMatrix(dev[1][:,columns].numpy(),label=dev[2].numpy())
        model=xgb.train(dict(objective="binary:logistic",eval_metric="logloss",max_depth=2,
                             eta=.05,nthread=1,seed=51),fit,150,evals=[(valid,"H4")],
                        early_stopping_rounds=12,verbose_eval=False)
        models.append((columns,model))
        raw.append(torch.from_numpy(model.predict(valid,output_margin=True,iteration_range=(0,model.best_iteration+1))))
    return models, torch.stack(raw,-1)


def counter_scores(models,x):
    scores=[]
    for fields,model in models:
        matrix=xgb.DMatrix(x[:,fields].numpy())
        scores.append(torch.from_numpy(model.predict(matrix,output_margin=True,iteration_range=(0,model.best_iteration+1))))
    return torch.stack(scores,-1)


def metrics(p, called, y):
    decision=p>=.5; good=decision.eq(y.bool()) & called
    return dict(rows=len(y), called=int(called.sum()), right=int(good.sum()), wrong=int((called&~good).sum()),
                withheld=int((~called).sum()), balanced_accuracy=balanced(y,good), auc=auc(y,p),
                detected=int((good&y.bool()).sum()), occupied=int(y.sum()))


def evaluate(model, stream, p, counter, threshold):
    t,x,y = stream
    coverage=(x.isfinite()[:,None] & model.views).sum(-1)/model.views.sum(-1)
    vote,confidence,act,utility,curious=choose(p,coverage,t)
    without,called=combine(p,act)
    counter_available=(x.isfinite()[:,None] & ~model.views).any(-1)
    kept,objection=verify(p,act,counter,counter_available,threshold)
    reviewed,called_review=combine(p,kept)
    truth=vote.eq(y[:,None].bool())
    reward=torch.where(act,2*truth.float()-1-COST,0.)
    calibration=1-(confidence-truth.float()).square()
    confidence_reward=torch.where(act,.2*(calibration-.75),0.)
    points=[]
    for i,(lo,hi) in enumerate(zip((.5,.6,.7,.8,.9),(.6,.7,.8,.9,1.01))):
        rows=act & (confidence>=lo) & (confidence<hi)
        points.append(dict(bin=i,count=int(rows.sum()),confidence=confidence[rows].mean().item() if rows.any() else None,
                           observed_accuracy=truth[rows].float().mean().item() if rows.any() else None))
    coords=((curious.any(-1)) | objection.any(-1) | (~act).any(-1)).nonzero().flatten()[:7]
    trace=[]
    for i in coords.tolist():
        actors=[]
        for j,name in enumerate(NAMES):
            values={OCCUPANCY[k]:float(x[i,k]) for k in VIEWS[j] if x[i,k].isfinite()}
            reason=("missing evidence" if not values else "exploration allowance" if curious[i,j]
                    else "expected payoff covered the cost" if act[i,j] else "expected payoff below the cost")
            actors.append(dict(expert=name, evidence=values, proposed=int(vote[i,j]) if act[i,j] else None,
                               confidence=float(confidence[i,j]) if act[i,j] else None, reason=reason,
                               expected_payoff=float(utility[i,j]), reward=round(float(reward[i,j]),5),
                               confidence_reward=round(float(confidence_reward[i,j]),5),
                               reviewed_out=bool(objection[i,j]),
                               counter_probability_occupied=float(counter[i,j]) if counter_available[i,j] else None))
        trace.append(dict(row=i, recorded_occupancy=int(y[i]), experts=actors))
    return dict(no_falsifier=metrics(without,called,y), independent_review=metrics(reviewed,called_review,y),
                proposals=int(act.sum()), abstentions=int((~act).sum()), low_confidence_trials=int(curious.sum()),
                low_confidence_correct=int((curious&truth).sum()), total_reward=round(float(reward.sum()),3),
                total_confidence_reward=round(float(confidence_reward.sum()),3),
                average_calibration_score=float(calibration[act].mean()) if act.any() else None,
                vetoed_correct=int((objection&truth).sum()),vetoed_wrong=int((objection&~truth).sum()),
                calibration=points,trace=trace)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    sources={}; streams=[]
    for home in range(1,7):
        stream, hashes=occupancy(args.data,home,14 if home<=3 else 7)
        streams.append(stream); sources.update(hashes)
    train,dev,tests=streams[:3],streams[3],streams[4:]
    # Separate model-selection and calibration portions of H4 in chronological order.
    mid=len(dev[0])//2
    select=tuple(v[:mid] for v in dev); calibrate_on=tuple(v[mid:] for v in dev)
    model,best,curve=fit(train,select,args.epochs)
    cp=calibrate(infer(model,calibrate_on),calibrate_on[2])
    refs,_=fit_counter(train,select)
    cr=calibrate(counter_scores(refs,calibrate_on[1]),calibrate_on[2])
    model_path=args.output/"experts.pt"
    torch.save(dict(weights=model.state_dict(),expert_calibration=cp,counter_calibration=cr),model_path)
    for j,(_,ref) in enumerate(refs):
        ref.save_model(args.output/f"review-{j}.ubj")
    threshold=.75
    reports={}
    for name, stream in zip(("H5","H6"),tests,strict=True):
        p=(infer(model,stream)*cp[0]+cp[1]).sigmoid()
        review=(counter_scores(refs,stream[1])*cr[0]+cr[1]).sigmoid()
        reports[name]=evaluate(model,stream,p,review,threshold)
        print(name,{k:v for k,v in reports[name].items() if k not in ("trace","calibration")},flush=True)
    artifacts={p.name:sha(p) for p in [model_path,*sorted(args.output.glob("review-*.ubj"))]}
    record=dict(rows={f"H{i}":len(value[0]) for i,value in enumerate(streams,1)},data=sources,
                selected_epoch=best,curve=curve,threshold=threshold,act_cost=COST,curiosity_bonus=CURIOSITY,
                code_sha256=sha(Path(__file__)),checkpoints=artifacts,
                dependencies={str(p.relative_to(HERE.parent)):sha(p) for p in
                              [HERE.parent/"liquid-workflow-v3"/name for name in ("blackboard.py","data.py","train.py")]
                              + [HERE.parent/"liquid-workflow-v2"/"workflow.py"]}, homes=reports)
    (args.output/"results.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--epochs",type=int,default=15)
    torch.set_num_threads(1)
    run(parser.parse_args())
