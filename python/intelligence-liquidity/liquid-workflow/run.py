"""Fit without the test buildings. Score released snapshots in time order."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import torch
from torch.nn import functional as F

from model import Workflow, learn_from_observation, train_loss

BASE = Path(__file__).resolve().parent
OLD = BASE.parent / "liquid-blackboard"


def data_reader():
    # Use the audited archive parsers without importing the earlier model here.
    original = sys.modules.pop("model")
    sys.path.insert(0, str(OLD))
    try:
        spec = importlib.util.spec_from_file_location("archived_reader", OLD / "experiment.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
    finally:
        sys.path.pop(0)
        sys.modules["model"] = original
    return reader


io_data = data_reader()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def save(path, data):
    pending = path.with_suffix(".tmp")
    torch.save(data, pending)
    pending.replace(path)


def blocks(streams, length=9):
    for name, x, y, stamps in streams:
        edges = [0, *((stamps.diff() > 450).nonzero().flatten() + 1).tolist(), len(x)]
        for i, j in zip(edges, edges[1:]):
            for start in range(i, j-length+1, length):
                yield x[start:start+length], y[start:start+length], stamps[start:start+length]


def train_batch(model, x, y, times):
    prior = None
    total = 0
    for i in range(x.shape[1] - 1):
        elapsed = x.new_full((len(x),), 5.) if i == 0 else (times[:, i] - times[:, i-1]).float()/60
        result = model.step(x[:, i], elapsed, prior, train_alternatives=True)
        total = total + train_loss(result, x[:, i], x[:, i+1], y[:, i])
        prior = result.memory
    return total / (x.shape[1]-1)


@torch.no_grad()
def validation(model, x, y, times):
    prior = None
    guesses = []
    for i in range(x.shape[1]):
        elapsed = x.new_full((len(x),), 5.) if i == 0 else (times[:, i] - times[:, i-1]).float()/60
        result = model.step(x[:, i], elapsed, prior, backtrack=False)
        prior = result.memory
        guesses.append(result.logit.sigmoid())
    return torch.stack(guesses, 1).flatten(), y.flatten()


def fit_one(payload):
    train, val, out, seed, share, epochs, resume = payload
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = Workflow()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    x, y, time = train
    n = round(len(x) * share)
    take = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:n]
    tag = f"{seed}-{round(share*100)}"
    snapshot = out / f"{tag}-resume.pt"
    best_path = out / f"{tag}.pt"
    curve = []
    best = -1.
    best_epoch = 0
    cutoff = .5
    if resume and snapshot.exists():
        state = torch.load(snapshot, map_location="cpu", weights_only=True)
        if state["seed"] != seed or state["training_blocks"] != n:
            raise ValueError("Snapshot does not match training selection")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        curve, best, best_epoch, cutoff = (state[k] for k in ("curve", "best", "best_epoch", "cutoff"))
    for epoch in range(len(curve)+1, epochs+1):
        if curve and len(curve) - best_epoch >= 4:
            break
        ids = take[torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))]
        total = 0.
        for part in ids.split(192):
            optimizer.zero_grad()
            loss = train_batch(model, x[part], y[part], time[part])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())*len(part)
        p, label = validation(model, *val)
        local_cutoff = io_data.source.threshold(p, label)
        measure = io_data.source.score(p, label, local_cutoff)
        auc = measure["auroc"]
        curve.append(dict(epoch=epoch, training_loss=total/n, **measure))
        if auc > best:
            best, best_epoch, cutoff = auc, epoch, local_cutoff
            save(best_path, model.state_dict())
        save(snapshot, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            curve=curve, best=best, best_epoch=best_epoch, cutoff=cutoff,
                            seed=seed, training_blocks=n))
        print(tag, epoch, "validation AUROC", round(auc, 4), flush=True)
    return best_path.name, dict(seed=seed, share=share, training_blocks=n, epoch=best_epoch,
                                cutoff=cutoff, state_sha256=sha(best_path), resume_sha256=sha(snapshot),
                                stop="validation plateau" if len(curve)-best_epoch>=4 else "epoch budget", curve=curve)


def packed(streams):
    return tuple(torch.stack(c) for c in zip(*blocks(streams), strict=True))


def fit(options):
    train, val, source_hashes = io_data.fit_sources(options.data)
    table, comparison = packed(train), packed(val)
    jobs = [(table, comparison, options.out, seed, share, options.epochs, options.resume)
            for seed in options.seeds for share in options.shares]
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        snapshots = dict(pool.map(fit_one, jobs))
    dump(options.out/"fit.json", dict(training_sources=source_hashes, code={n: sha(BASE/n) for n in ("model.py","run.py")},
                                      training_blocks=len(table[0]), validation_blocks=len(comparison[0]),
                                      checkpoints=snapshots))


def external(data):
    path = data / "independent-uci.zip"
    expected = "a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd"
    if sha(path) != expected:
        raise ValueError("Room Occupancy Estimation archive changed")
    selected = {}
    with zipfile.ZipFile(path) as archive, io.TextIOWrapper(archive.open("Occupancy_Estimation.csv")) as file:
        for row in csv.DictReader(file):
            stamp = int(datetime.strptime(row["Date"]+" "+row["Time"], "%Y/%m/%d %H:%M:%S")
                        .replace(tzinfo=timezone.utc).timestamp())
            selected.setdefault(stamp//300, (stamp, [float(row["S5_CO2"]),float(row["S1_Temp"])],
                                            float(int(row["Room_Occupancy_Count"])>0)))
    t, x, y = map(torch.tensor, zip(*(selected[k] for k in sorted(selected)), strict=True))
    normalized = (x-torch.tensor((500,22)))/torch.tensor((200,3))
    return [("Room Occupancy Estimation", normalized, y, t)]


def metrics(p, y, accepted, cutoff):
    called = p >= cutoff
    correct = (called == y.bool()) & accepted
    occupied, empty = y.bool(), ~y.bool()
    return dict(accuracy=float(correct.float().mean()),
                equal_weight=float((correct[occupied].float().mean() + correct[empty].float().mean())/2),
                occupied=int(occupied.sum()), empty=int(empty.sum()),
                missed=int((occupied & ~called & accepted).sum()),
                false_alarm=int((empty & called & accepted).sum()),
                withheld_occupied=int((occupied & ~accepted).sum()),
                withheld_empty=int((empty & ~accepted).sum()),
                undecided=int((~accepted).sum()))


def stream(model, room, variant):
    _, x, _, stamps = room
    ledger, previous = None, None
    optimizer = torch.optim.SGD(model.parameters(), lr=.001) if variant == "live" else None
    count = Counter()
    example = None
    answers, passes = [], []
    for i, reading in enumerate(x):
        seconds = int(stamps[i]-stamps[i-1]) if i else 300
        if seconds > 450:
            ledger, previous = None, None
        current = reading.unsqueeze(0)
        duration = x.new_tensor([min(seconds, 450)/60])
        with torch.no_grad():
            result = model.step(current, duration, ledger, trace=True)
            answers.append(result.logit.sigmoid()[0])
            passes.append(result.accepted[0])
            events = [e for e in result.attempts if "revisit" in e]
            count["sent_back"] += len(events)
            if events and example is None:
                example = dict(row=i, time=int(stamps[i]), attempts=result.attempts, accepted=bool(result.accepted[0]))
            count["alternate_picked"] += sum(e.get("used_retry", [False])[0] for e in result.attempts)
        if optimizer and previous is not None:
            learn_from_observation(model, optimizer, previous, current)
            count["learning_updates"] += 1
        previous = current, duration, ledger
        ledger = result.memory.detach()
    return torch.stack(answers), torch.stack(passes), dict(count), example


def score_model(payload):
    out, entry, rec, site, rooms, variant = payload
    torch.set_num_threads(1)
    model = Workflow(route=variant != "fixed-expert", delta=variant != "no-delta", checks=variant != "no-check")
    state = torch.load(out/entry, weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    ps, ys, checks, evidence = [], [], [], []
    for room in rooms:
        model.load_state_dict(state)
        p, pass_check, events, example = stream(model, room, variant)
        ps.append(p)
        ys.append(room[2])
        checks.append(pass_check)
        change = sum(float((parameter.detach()-state[name]).abs().sum()) for name, parameter in model.named_parameters()
                     if ".experts." in name)
        evidence.append(dict(room=room[0], events=events, trace=example, expert_weight_change=change))
    row = metrics(torch.cat(ps), torch.cat(ys), torch.cat(checks), rec["cutoff"])
    return dict(checkpoint=entry, site=site, variant=variant, **row, evidence=evidence)


def score(options):
    manifest = json.loads((options.out/"fit.json").read_text())
    if any(sha(BASE/name) != value for name,value in manifest["code"].items()):
        raise ValueError("Fit code has changed")
    for name, record in manifest["checkpoints"].items():
        if sha(options.out/name) != record["state_sha256"]:
            raise ValueError(f"Checkpoint changed: {name}")
    sites = io_data.test_sites(options.data)
    sites["uci-estimation"] = external(options.data)
    if options.sites:
        sites = {key: sites[key] for key in options.sites}
    jobs = [(options.out, name, rec, site, rooms, variant)
            for name,rec in manifest["checkpoints"].items()
            for site, rooms in sites.items()
            for variant in (options.variants if rec["share"]==1 else ("frozen",))]
    with ProcessPoolExecutor(max_workers=options.workers) as pool:
        rows = list(pool.map(score_model, jobs))
    dump(options.out/"scores.json", dict(fit_sha256=sha(options.out/"fit.json"),
         test_hashes={n:sha(options.data/n) for n in ("graz.zip","munich.zip","porto-alegre.csv","independent-uci.zip")},
         rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit","score"))
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41,42,43])
    parser.add_argument("--shares", type=float, nargs="+", default=[.25,.5,1.])
    parser.add_argument("--sites", nargs="+")
    parser.add_argument("--variants", nargs="+", default=["frozen","live","fixed-expert","no-delta","no-check"])
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    torch.set_num_threads(1)
    options.out.mkdir(parents=True, exist_ok=True)
    (fit if options.phase=="fit" else score)(options)


if __name__ == "__main__":
    main()
