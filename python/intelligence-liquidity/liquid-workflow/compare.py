"""Train a tree benchmark on the same rows, with only causal summaries."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import torch
import xgboost as xgb

from run import dump, external, io_data, metrics, packed, sha


def features(x, times):
    recent = []
    for i, item in enumerate(x):
        if i == 0 or times[i] - times[i-1] > 450:
            recent.clear()
        recent.append(item)
        recent = recent[-4:]
        change = item - recent[-2] if len(recent)>1 else torch.zeros_like(item)
        yield torch.cat((item, change, torch.stack(recent).mean(0), item-recent[0]))


def matrix(x, t):
    return torch.stack(list(features(x, t)))


def fit_one(job):
    train, valid, sites, output, seed = job
    tx, ty = train
    vx, vy = valid
    dtrain, dval = xgb.DMatrix(tx.numpy(), label=ty.numpy()), xgb.DMatrix(vx.numpy(), label=vy.numpy())
    history = {}
    fit = xgb.train(dict(max_depth=4, eta=.12, subsample=.9, nthread=1, seed=seed,
                         objective="binary:logistic", eval_metric="auc"),
                    dtrain, 250, [(dval,"validation")], early_stopping_rounds=25,
                    verbose_eval=False, evals_result=history)
    rounds = fit.best_iteration+1
    path = output/f"xgboost-{seed}.ubj"
    fit.save_model(path)
    cutoff = io_data.source.threshold(torch.from_numpy(fit.predict(dval,iteration_range=(0,rounds))), vy)
    rows = []
    for site, rooms in sites.items():
        values, labels = [], []
        for _, x, y, stamps in rooms:
            data = xgb.DMatrix(matrix(x, stamps).numpy())
            values.append(torch.from_numpy(fit.predict(data, iteration_range=(0,rounds))))
            labels.append(y)
        p, y = torch.cat(values), torch.cat(labels)
        rows.append(dict(site=site, checkpoint=path.name, variant="xgboost",
                         **metrics(p,y,torch.ones_like(y,dtype=torch.bool),cutoff)))
    return dict(checkpoint=path.name, sha256=sha(path), rounds=rounds,
                cutoff=cutoff, validation=history, rows=rows)


def train_matrix(streams, fitting):
    x, y, t = packed(streams)
    count = x.shape[1]-int(fitting)
    xx = torch.cat([matrix(x0,t0)[:count] for x0,t0 in zip(x,t,strict=True)])
    return xx, y[:,:count].flatten()


if __name__ == "__main__":
    torch.set_num_threads(1)
    data, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True,exist_ok=True)
    train, valid, hashes = io_data.fit_sources(data)
    sites = io_data.test_sites(data)
    sites["uci-estimation"] = external(data)
    dtrain, dvalid = train_matrix(train,True), train_matrix(valid,False)
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(fit_one,[(dtrain,dvalid,sites,output,seed) for seed in (41,42,43)]))
    dump(output/"xgboost.json",dict(training_sources=hashes,training_rows=len(dtrain[1]),runs=runs))
