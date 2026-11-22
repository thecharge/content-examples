"""Separate the feature change from the solver on the two known UCI tests."""
from csv import reader
from io import TextIOWrapper
from zipfile import ZipFile
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
names = ("Temperature", "Humidity", "CO2", "Light", "HumidityRatio")


def data(zfile, member):
    with TextIOWrapper(zfile.open(member)) as source:
        records = reader(source)
        header = next(records)
        rows = [dict(zip(header, raw[-len(header):])) for raw in records]
    return (torch.tensor([[float(r[k]) for k in names] for r in rows]),
            torch.tensor([float(r["Occupancy"]) for r in rows]))


with ZipFile(sys.argv[1]) as zfile:
    fit_x, fit_y = data(zfile, "datatraining.txt")
    fit_x, fit_y = fit_x[:6500], fit_y[:6500]
    sites = {n: data(zfile, n) for n in ("datatest.txt", "datatest2.txt")}
out = []
for size in (3,5):
    train = fit_x[:, :size]
    mu, sigma = train.mean(0), train.std(0).clamp_min(1e-6)
    train = (train-mu)/sigma
    torch.manual_seed(41)
    model = nn.Linear(size, 1)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50, line_search_fn="strong_wolfe")
    def loss():
        optimizer.zero_grad()
        value = F.binary_cross_entropy_with_logits(model(train).flatten(), fit_y)
        value.backward()
        return value
    optimizer.step(loss)
    for site, (test, y) in sites.items():
        with torch.inference_mode():
            labels = y.bool()
            picks = model((test[:,:size] - mu) / sigma).flatten() >= 0
            correctly = picks == labels
            out.append(dict(site=site, feature_count=size, ordinary=correctly.float().mean().item(),
                            balanced=(correctly[labels].float().mean().item()+correctly[~labels].float().mean().item())/2))
print(json.dumps(out, indent=2))
