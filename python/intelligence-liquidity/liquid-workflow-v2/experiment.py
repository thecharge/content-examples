"""Fit on the archived source split and evaluate a frozen checkpoint."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
import xgboost as xgb
from ncps.torch import LTC

from workflow import Blackboard, Observation, Online, loss

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "liquid-workflow"))
from run import dump, io_data, metrics, blocks, save, sha
from compare import matrix


class Reference(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.core = LTC(1, 8, batch_first=True) if name == "ltc" else nn.GRU(1, 8, batch_first=True)
        self.read = nn.Linear(8, 1)

    def step(self, reading, minutes, state):
        if isinstance(self.core, LTC):
            if state is None:
                state = reading.new_zeros(len(reading), self.core.state_size)
            hidden, state = self.core.rnn_cell(reading[:, None], state, (minutes / 5)[:, None])
        else:
            sequence, state = self.core(reading[:, None, None], state)
            hidden = sequence[:, 0]
        return self.read(hidden).squeeze(-1), state


def forward(model, batch, training):
    x, y, t = batch
    state, answers, costs = None, [], []
    for i in range(x.shape[1] - 1):
        minutes = x.new_full((len(x),), 5.) if i == 0 else (t[:, i] - t[:, i-1]).float() / 60
        primary, witness = x[:, i].unbind(-1)
        if isinstance(model, Blackboard):
            step = model.step(primary, witness, minutes, state, training=training)
            state = step.ledger
            costs.append(loss(step, witness, y[:, i], x[:, i+1, 0]))
            answers.append(step.logit)
        else:
            logit, state = model.step(primary, minutes, state)
            costs.append(F.binary_cross_entropy_with_logits(logit, y[:, i]))
            answers.append(logit)
    return torch.stack(costs).mean(), torch.stack(answers, 1).flatten().sigmoid(), y[:, :-1].flatten()


FACTORIES = {
    "workflow": Blackboard,
    "fixed-expert": lambda: Blackboard(fixed_expert=True),
    "no-delta": lambda: Blackboard(use_delta=False),
    "ltc": lambda: Reference("ltc"),
    "gru": lambda: Reference("gru"),
}


def train_model(name, train, valid, out, epochs, seed, share=1.):
    torch.manual_seed(seed)
    model = FACTORIES[name]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.006)
    sample = torch.randperm(len(train[0]), generator=torch.Generator().manual_seed(seed))
    sample = sample[:round(len(sample) * share)]
    curve, best_loss, best_epoch, cutoff = [], float("inf"), 0, .5
    checkpoint = out / f"{name}.pt"
    resume = out / f"{name}-resume.pt"
    fingerprint = {n: sha(BASE/n) for n in ("workflow.py", "experiment.py")}
    if resume.exists():
        recorded = torch.load(resume, weights_only=True)
        if (recorded["seed"], recorded["share"], recorded["code"]) != (seed, share, fingerprint):
            raise ValueError(f"Training parameters changed for {name}")
        model.load_state_dict(recorded["model"])
        optimizer.load_state_dict(recorded["optimizer"])
        curve = recorded["curve"]
        best_loss, best_epoch, cutoff = recorded["best_loss"], recorded["best_epoch"], recorded["cutoff"]
    for epoch in range(len(curve) + 1, epochs + 1):
        if epoch - best_epoch > 4:
            break
        torch.manual_seed(seed + epoch)
        shuffled = sample[torch.randperm(len(sample), generator=torch.Generator().manual_seed(seed + epoch))]
        for ix in shuffled.split(128):
            optimizer.zero_grad()
            cost, _, _ = forward(model, tuple(v[ix] for v in train), True)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            _, p, y = forward(model, valid, False)
        validation = float(F.binary_cross_entropy(p, y))
        curve.append(validation)
        if validation < best_loss - 1e-5:
            best_loss, best_epoch = validation, epoch
            cutoff = float(io_data.source.threshold(p, y))
            save(checkpoint, model.state_dict())
        save(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), curve=curve,
                          best_loss=best_loss, best_epoch=best_epoch, cutoff=cutoff, seed=seed,
                          share=share, code=fingerprint))
        print(name, epoch, round(validation, 4), flush=True)
    return dict(best_epoch=best_epoch, cutoff=cutoff, validation_binary_loss=curve,
                best_sha256=sha(checkpoint), resume_sha256=sha(resume), training_blocks=len(sample),
                seed=seed, fraction=share, early_stop=len(curve) - best_epoch >= 4,
                parameters=sum(p.numel() for p in model.parameters()))


def tree_input(blocks):
    x, y, times = blocks
    parts = [matrix(a[:, :1], t)[:-1] for a, t in zip(x, times, strict=True)]
    return xgb.DMatrix(torch.cat(parts).numpy(), label=y[:, :-1].flatten().numpy())


def fit_job(payload):
    name, train, valid, out, epochs, share = payload
    torch.set_num_threads(1)
    model = name if share == 1 else "workflow"
    return name, train_model(model, train, valid, out, epochs, 41, share)


def fit(args):
    train, valid, hashes = io_data.fit_sources(args.data)
    train, valid = [tuple(torch.stack(c) for c in zip(*blocks(split, 6), strict=True))
                    for split in (train, valid)]
    jobs = [(name, train, valid, args.out, args.epochs, 1.) for name in FACTORIES]
    for fraction in (.25, .50):
        split = args.out / str(round(fraction * 100))
        split.mkdir(exist_ok=True)
        jobs.append((f"workflow-{round(fraction*100)}", train, valid, split, args.epochs, fraction))
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as workers:
        records = dict(workers.map(fit_job, jobs))
    tx, vx = tree_input(train), tree_input(valid)
    tree = xgb.train(dict(max_depth=4, eta=.12, nthread=1, seed=41, objective="binary:logistic"),
                     tx, 250, [(vx, "validation")], early_stopping_rounds=20, verbose_eval=False)
    rounds = tree.best_iteration + 1
    path = args.out / "xgboost.ubj"
    tree.save_model(path)
    cutoff = io_data.source.threshold(torch.from_numpy(tree.predict(vx, iteration_range=(0, rounds))),
                                      torch.from_numpy(vx.get_label()))
    records["xgboost"] = dict(cutoff=float(cutoff), rounds=rounds, best_sha256=sha(path))
    dump(args.out / "fit.json", dict(models=records, fit_sources=hashes,
         code={n: sha(BASE/n) for n in ("workflow.py", "experiment.py")},
         training_blocks=len(train[0]), validation_blocks=len(valid[0]), torch=torch.__version__))


def replay(model, room, mode, *, maximum=None):
    name, x, y, stamps = room
    if maximum:
        x, y, stamps = x[:maximum], y[:maximum], stamps[:maximum]
    ledger, timestamp, trace, forecasts, errors = None, None, [], None, []
    learner = Online(model) if mode == "live" else None
    p, calls, updates, elapsed_total = [], [], 0, 0.
    for row, (reading, time) in enumerate(zip(x, stamps.tolist(), strict=True)):
        dt = time - timestamp if timestamp is not None else 300
        minutes = x.new_tensor([dt / 60 if 0 < dt <= 450 else 5.])
        primary, witness = reading.unsqueeze(0).unbind(-1)
        if dt > 450:
            ledger, forecasts = None, None
        begin = perf_counter()
        with torch.no_grad():
            if isinstance(model, Blackboard):
                step = model.step(primary, witness, minutes, ledger)
                p.append(step.logit.sigmoid()[0])
                calls.append(step.accepted[0])
                if (not step.accepted.item() or len(step.attempts) > 3) and len(trace) < 3:
                    trace.append(dict(row=row, time=time, witness=float(witness.item()),
                        accepted=bool(step.accepted.item()), records=[{
                            **{key: rec[key] for key in ("layer", "retry_from", "retry_rank")},
                            **{key: rec[key].item() for key in ("claim", "error", "process", "expert")}
                        } for rec in step.attempts]))
                if forecasts is not None:
                    errors.append(abs(forecasts - primary.item()))
                forecasts = step.forecast.item()
            else:
                logit, ledger = model.step(primary, minutes, ledger)
                p.append(logit.sigmoid()[0])
                calls.append(x.new_ones((), dtype=torch.bool))
        if isinstance(model, Blackboard):
            if learner:
                updates += learner.feedback(Observation(time, primary, witness, minutes, ledger)) is not None
            ledger = step.ledger.detach()
        elapsed_total += perf_counter() - begin
        timestamp = time
    return torch.stack(p), y, torch.stack(calls), dict(room=name, trace=trace,
        updates=updates, milliseconds_per_row=1000 * elapsed_total / len(x),
        co2_forecast_mae_ppm=200 * sum(errors) / len(errors) if errors else None)


def score(args):
    from transfer import h6
    report = json.loads((args.out / "fit.json").read_text())
    if any(sha(BASE/name) != digest for name, digest in report["code"].items()):
        raise ValueError("Source changed since fit")
    rooms, hashes = h6(args.external)
    rows = []
    variants = ("workflow", "live", "no-check", "fixed-expert", "no-delta", "ltc", "gru",
                "xgboost", "workflow-25", "workflow-50")
    for variant in variants:
        key = "workflow" if variant in ("live", "no-check") else variant
        record = report["models"][key]
        params = args.out / f"{key}.pt"
        if key in ("workflow-25", "workflow-50"):
            params = args.out / key.split("-")[-1] / "workflow.pt"
        model = (Blackboard(use_checks=variant != "no-check") if key.startswith("workflow")
                 else FACTORIES[key]() if key != "xgboost" else None)
        if model is None:
            params = args.out / "xgboost.ubj"
        if sha(params) != record["best_sha256"]:
            raise ValueError(f"Checkpoint changed for {variant}")
        outputs = []
        for room in rooms:
            if model is None:
                tree = xgb.Booster(model_file=params)
                _, x, y, times = room
                p = torch.from_numpy(tree.predict(xgb.DMatrix(matrix(x[:, :1], times).numpy()),
                                                  iteration_range=(0, record["rounds"])))
                pred = p, y, torch.ones_like(y, dtype=torch.bool), {}
            else:
                model.load_state_dict(torch.load(params, weights_only=True))
                pred = replay(model, room, variant)
            outputs.append(pred)
        p, y, accepted = (torch.cat(values) for values in zip(*(result[:3] for result in outputs), strict=True))
        called = int(accepted.sum())
        rows.append(dict(variant=variant, correct=int(((p >= record["cutoff"]) == y.bool()).logical_and(accepted).sum()),
                         coverage=called/len(y), rows=len(y), **metrics(p, y, accepted, record["cutoff"]),
                         accepted_accuracy=(float(((p[accepted] >= record["cutoff"]) == y[accepted].bool()).float().mean())
                                            if called else None),
                         details=[result[3] for result in outputs]))
        print(variant, rows[-1]["correct"], "/", len(y), flush=True)
    dump(args.out / "h6.json", dict(fit_sha256=sha(args.out/"fit.json"), files=hashes, results=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score", "control"))
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "control":
        from control import run
        run(args)
    else:
        {"fit": fit, "score": score}[args.phase](args)


if __name__ == "__main__":
    main()
