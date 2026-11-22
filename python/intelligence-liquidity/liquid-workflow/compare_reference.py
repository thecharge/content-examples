"""Fit the authors' PyTorch LTC under the two-sensor experiment protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

from ncps.torch import LTCCell
from ncps.wirings import FullyConnected
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from run import dump, external, io_data, metrics, packed, save, sha


class ReferenceLTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.liquid = LTCCell(FullyConnected(8), 2, output_mapping=None, ode_unfolds=6, implicit_param_constraints=True)
        self.output = nn.Linear(8, 1)

    def forward(self, x, time):
        state = x.new_zeros((len(x), 8))
        result = []
        for index in range(x.shape[1]):
            dt = x.new_ones(len(x), 1) if not index else (time[:, index:index+1] - time[:, index-1:index]).float() / 300
            _, updated = self.liquid(x[:, index], state, dt.clamp_min(1 / 300))
            state = torch.where(dt > 0, updated, state)
            result.append(self.output(state).squeeze(-1))
        return torch.stack(result, dim=1)


def sequences(rooms):
    for _, x, y, times in rooms:
        edges = [0, *((times.diff() > 450).nonzero().flatten() + 1).tolist(), len(x)]
        for start, end in zip(edges, edges[1:]):
            yield x[start:end], y[start:end], times[start:end]


@torch.no_grad()
def evaluate(model, rooms, cutoff):
    ps, ys = [], []
    ordered = sorted(sequences(rooms), key=lambda parts: len(parts[0]))
    for start in range(0, len(ordered), 32):
        x, y, time = zip(*ordered[start:start+32], strict=True)
        logits = model(pad_sequence(x, batch_first=True), pad_sequence(time, batch_first=True))
        ps.extend(logit[:len(labels)].sigmoid() for logit, labels in zip(logits, y, strict=True))
        ys.extend(y)
    p, y = torch.cat(ps), torch.cat(ys)
    return metrics(p, y, torch.ones_like(y, dtype=torch.bool), cutoff)


def fit(job):
    seed, train, val, sites, output, epochs = job
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = ReferenceLTC()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    x, y, time = train
    vx, vy, vt = val
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    name = f"reference-ltc-{seed}.pt"
    history, best_auc, best_epoch, cutoff = [], -1., 0, .5
    for epoch in range(1, epochs+1):
        if epoch - best_epoch > 4 and best_epoch:
            break
        ids = order[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))]
        total = 0.
        for idx in ids.split(192):
            optimizer.zero_grad()
            logits = model(x[idx, :-1], time[idx, :-1])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx, :-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        with torch.no_grad():
            probabilities = model(vx, vt).flatten().sigmoid()
        labels = vy.flatten()
        candidate_cutoff = io_data.source.threshold(probabilities, labels)
        score = io_data.source.score(probabilities, labels, candidate_cutoff)
        history.append(dict(epoch=epoch, training_loss=total / len(x), **score))
        if score["auroc"] > best_auc:
            best_auc, best_epoch, cutoff = score["auroc"], epoch, candidate_cutoff
            save(output / name, model.state_dict())
        print(f"reference LTC {seed} epoch {epoch}: {score['auroc']:.4f} validation AUROC", flush=True)
    model.load_state_dict(torch.load(output / name, weights_only=True, map_location="cpu"))
    rows = [dict(site=site, checkpoint=name, variant="reference-ltc", **evaluate(model, rooms, cutoff))
            for site, rooms in sites.items()]
    return dict(seed=seed, checkpoint=name, sha256=sha(output / name), epoch=best_epoch,
                cutoff=cutoff, validation=history, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    training, validation, source_hashes = io_data.fit_sources(args.data)
    held_out = io_data.test_sites(args.data)
    held_out["uci-estimation"] = external(args.data)
    train, val = packed(training), packed(validation)
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(fit, [(seed, train, val, held_out, args.output, args.epochs) for seed in (41,42,43)]))
    dump(args.output / "reference-ltc.json", dict(
        ncps=version("ncps"), torch=str(torch.__version__), source_hashes=source_hashes,
        test_hashes={name: sha(args.data/name) for name in
                     ("graz.zip", "munich.zip", "porto-alegre.csv", "independent-uci.zip")},
        training_rows=len(train[0]) * (train[0].shape[1] - 1), epoch_budget=args.epochs,
        hidden_units=8, solver_steps=6, parameters=sum(p.numel() for p in ReferenceLTC().parameters()),
        code_sha256=sha(Path(__file__)), runs=runs))
