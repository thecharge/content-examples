"""Recheck fixed V2 delta on already explored Aalto under matched training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

def run(inputs):
    root, data, seed, use_delta = inputs
    torch.set_num_threads(1)
    base=Path(root)/"python/intelligence-liquidity"
    sys.path[:0]=[str(base/"adaptive-liquid-v2"),str(base/"attention-and-reward")]
    import experiment as exp
    from adaptive import Adaptive, Delta
    exp.INPUT=Path(data)
    exp.UCI=exp.INPUT/"occupancy-fresh.zip"
    raw=exp.office(exp.UCI,"datatraining.txt")
    blocks=exp.fitting_blocks(raw,100)
    stats=exp.normalizer([row[2] for block in blocks for row in block])
    fit=exp.pack(exp.occupancy_windows(blocks,5,1),stats)
    validation=exp.pack(exp.occupancy_windows([raw[6500:]],5,1),stats)
    torch.manual_seed(seed)
    model=Adaptive(width=8,delta=False)
    if use_delta:
        model.delta=Delta(3,8)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    positives=int(fit[2].sum())
    weight=torch.tensor((len(fit[2])-positives)/positives)
    generator=torch.Generator().manual_seed(seed)
    route_gen=torch.Generator().manual_seed(seed+1000)
    def scheduled(stage,proposal,score,prior,remaining):
        return torch.randint(0,2,(len(proposal),),generator=route_gen)
    curve=[]
    best=None
    for step in range(1,241):
        batch=torch.randint(len(fit[2]),(64,),generator=generator)
        x,clock,y=(part[batch] for part in fit)
        pred,scores,_,_=model(x,clock,scheduled)
        loss=torch.stack([F.binary_cross_entropy_with_logits(p,y.float(),pos_weight=weight) for p in pred]).mean()
        loss=loss+.2*torch.stack([F.binary_cross_entropy_with_logits(s,((p.detach()>=0)==y).float()) for s,p in zip(scores,pred)]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%24==0:
            dev=exp.predict(model,validation)[0]
            dev_loss=float(F.binary_cross_entropy_with_logits(dev,validation[2].float(),pos_weight=weight))
            curve.append([step,dev_loss])
            if best is None or dev_loss<best[0]:
                best=dev_loss,step,{k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best[2])
    dev=exp.predict(model,validation)[0]
    cutoff=exp.calibration(dev,validation[2])
    windows,_=exp.aalto()
    test=exp.pack(windows,stats)
    logits=exp.predict(model,test)[0]
    balanced=exp.balanced(logits,test[2],cutoff)
    fixed=exp.balanced(logits,test[2],0)
    mask=None
    if use_delta:
        with torch.no_grad():
            model.delta.strength.zero_()
        masked=exp.predict(model,test)[0]
        mask={"same_cutoff":exp.balanced(masked,test[2],cutoff),
              "mean_absolute_logit_change":float((logits-masked).abs().mean())}
    return dict(seed=seed,delta=use_delta,best_step=best[1],validation_loss=best[0],cutoff=cutoff,
                balanced=balanced,fixed_balanced=fixed,curve=curve,masked_after_fit=mask)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    pairs=[(root,args.data,seed,delta) for seed in (41,42,43) for delta in (False,True)]
    with ProcessPoolExecutor(max_workers=min(6,len(pairs))) as pool:
        rows=list(pool.map(run,pairs))
    args.output.write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in row.items() if k!="curve"} for row in rows],indent=2))
