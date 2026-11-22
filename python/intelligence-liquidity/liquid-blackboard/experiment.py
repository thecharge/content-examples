"""Train in chronological blocks and test by predicting before learning."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import torch
from torch.nn import functional as F

from model import Blackboard, logged_q_loss, sensor_loss


ROOT = Path(__file__).resolve().parent.parent
MODES = ("plain", "moe", "delta", "full")
SEEDS = (41, 42, 43)


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = load_module("fidelity_sources", ROOT / "liquid-fidelity" / "experiment.py")
entry_module = sys.modules.get("experiment")
try:
    sys.modules["experiment"] = source
    evaluation = load_module("fidelity_eval", ROOT / "liquid-fidelity" / "evaluate.py")
finally:
    if entry_module is None:
        del sys.modules["experiment"]
    else:
        sys.modules["experiment"] = entry_module
control = load_module("logged_control", ROOT / "building-control" / "replay.py")


def build(mode):
    return Blackboard(mode != "plain", mode in ("delta", "full"), mode == "full")


def continuous(windows):
    """Keep observations at their unique endpoints and use five-minute cadence."""
    result = []
    for name, (x, _, y, times) in windows.items():
        indices = [i for i, t in enumerate(times) if i == 0 or t // 300 > times[i-1] // 300]
        idx = torch.tensor(indices)
        if not len(indices):
            raise ValueError(f"No five-minute endpoints: {name}")
        seconds = torch.tensor([times[i] for i in indices])
        values = (x[idx, -1] - torch.tensor((500, 22))) / torch.tensor((200, 3))
        result.append((name, values, y[idx], seconds))
    return result


def fit_sources(data):
    raw = {key: source.source(data, key) for key in ("uci", "aalborg", "calibrate", "select")}
    train = {k: source.windows(x) for k, x in source.training(raw).items()}
    val = {k: source.windows(source.aalto(raw[k])) for k in ("calibrate", "select")}
    return continuous(train), continuous(val), {name: source.sha(b) for name, b in raw.items()}


def batches(streams, length=12):
    parts = []
    for _, x, y, time in streams:
        ends = (time.diff() > 450).nonzero().flatten().tolist()
        starts = [0] + [end + 1 for end in ends]
        for start, end in zip(starts, ends + [len(time) - 1], strict=True):
            for i in range(start, end + 2 - length, length):
                parts.append((x[i:i+length], y[i:i+length]))
    return tuple(torch.stack(col) for col in zip(*parts, strict=True))


def occupancy_loss(model, xs, ys, *, train):
    state = None
    logits = []
    loss = xs.new_zeros(())
    for i in range(xs.shape[1] - 1):
        step = model.step(xs[:, i], xs.new_full((len(xs),), 5.), state)
        loss = loss + F.binary_cross_entropy_with_logits(step.occupied, ys[:, i])
        if train:
            loss = loss + .3 * sensor_loss(xs[:, i], xs[:, i+1], step)
        logits.append(step.occupied)
        state = step.memory
    return loss / len(logits), torch.stack(logits, 1)


def occupancy_worker(args):
    tx, ty, vx, vy, out, epochs, seed, mode, share = args
    torch.set_num_threads(1)
    if share < 1:
        ix = torch.arange(len(tx))[((torch.arange(len(tx)) * 0.61803398875) % 1) < share]
        tx, ty = tx[ix], ty[ix]
    torch.manual_seed(seed)
    model = build(mode)
    optim = torch.optim.AdamW(model.parameters(), lr=.004)
    best = None
    curve = []
    for epoch in range(1, epochs + 1):
        ids = torch.randperm(len(tx), generator=torch.Generator().manual_seed(seed + epoch))
        for ix in ids.split(96):
            optim.zero_grad()
            loss, _ = occupancy_loss(model, tx[ix], ty[ix], train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()
        with torch.no_grad():
            _, score = occupancy_loss(model, vx, vy, train=False)
            result = source.score(score.flatten().sigmoid(), vy[:, :-1].flatten(), .5)
        curve.append({"epoch": epoch, **result})
        key = (result["auroc"], -result["brier"])
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    source.threshold(score.flatten().sigmoid(), vy[:, :-1].flatten()))
        if epoch - best[1] >= 8:
            break
    _, epoch, weights, cutoff = best
    tag = "" if share == 1 else f"-budget{int(share*100)}"
    path = out / f"occupancy-{mode}-{seed}{tag}.pt"
    torch.save(weights, path)
    print(path.name, "kept epoch", epoch, "of", len(curve), flush=True)
    return path.name, {"mode": mode, "seed": seed, "epoch": epoch, "training_blocks": len(tx), "share": share, "cutoff": cutoff,
                           "sha256": source.sha(path.read_bytes()), "curve": curve}



def occupancy_fit(train, val, out, epochs, workers):
    tx, ty = batches(train)
    vx, vy = batches(val)
    jobs = [(tx, ty, vx, vy, out, epochs, seed, mode, 1.)
            for seed in SEEDS for mode in MODES]
    jobs.extend((tx, ty, vx, vy, out, epochs, SEEDS[0], "full", share)
                for share in (.25, .5))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(occupancy_worker, jobs))


def porto(path):
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != "895cc9cfd1043b83b20b4004d9d7ccf1":
        raise ValueError("Publisher checksum mismatch for Porto Alegre")
    readings = []
    previous = None
    for row in source.rows(raw):
        time = source.timestamp(row["createdAt"])
        co2, temp, label = map(source.number, (row["co2"], row["temperature_1"], row["ground_truth"]))
        if not all(map(math.isfinite, (co2, temp, label))) or label < 0:
            continue
        bucket = time // 300
        if previous is not None and bucket < previous:
            raise ValueError("Porto Alegre rows must be chronological")
        if bucket == previous:
            continue
        previous = bucket
        readings.append((time, (co2, temp), float(label > 0)))
    times, x, y = map(torch.tensor, zip(*readings, strict=True))
    return [("CONSEG office", (x - torch.tensor((500, 22))) / torch.tensor((200, 3)), y, times)]


def test_sites(data):
    graz = evaluation.graz(source.source(data, "test"))
    raw = (data / "munich.zip").read_bytes()
    check_hash(raw, "22c02aa064bc4c67086717671a48cade26c0e04576ba8adbac5145b49f072d35")
    return {"graz": continuous(graz), "munich": continuous(evaluation.munich(raw)),
            "porto": porto(data / "porto-alegre.csv")}


def check_hash(data, expected):
    if source.sha(data) != expected:
        raise ValueError("SHA-256 mismatch")


def adapter_for(model, enabled):
    if not enabled:
        return None
    for name, value in model.named_parameters():
        value.requires_grad_(name in ("adapter_scale", "adapter_offset"))
    return torch.optim.SGD((model.adapter_scale, model.adapter_offset), lr=.015)


def update_from_arrival(model, previous, x, optimizer):
    previous_x, minutes, prior = previous
    optimizer.zero_grad()
    estimate = model.step(previous_x, minutes, prior)
    loss = sensor_loss(previous_x, x, estimate)
    loss.backward()
    torch.nn.utils.clip_grad_norm_((model.adapter_scale, model.adapter_offset), .5)
    optimizer.step()
    with torch.no_grad():
        model.adapter_scale.clamp_(.75, 1.25)
        model.adapter_offset.clamp_(-.5, .5)
    return float(loss.detach())


def stream(model, x, t, carry=True, learn=False, feedback_budget=None):
    """Emit the current decision before the current arrival can update weights."""
    optimizer = adapter_for(model, learn)
    previous, state, last = None, None, None
    readings, forecasts, changes, routes, corrections, gains = [], [], [], [], [], []
    for reading, stamp in zip(x, t.tolist(), strict=True):
        reading = reading[None]
        connected = last is not None and 0 < stamp - last <= 450
        minutes = x.new_tensor([(stamp - last) / 60 if connected else 5.])
        prior = state if connected and carry else None
        with torch.no_grad():
            step = model.step(reading, minutes, prior)
            readings.append(step.occupied.item())
            routes.append(step.routes.squeeze(0))
            corrections.append(step.corrections.squeeze(0))
            gains.append(step.process.squeeze(0))
            if previous is not None and connected:
                forecasts.append(float((reading - state.previous_forecast).abs().mean()))
        if (previous is not None and connected and optimizer is not None
                and (feedback_budget is None or len(changes) < feedback_budget)):
            changes.append(update_from_arrival(model, previous, reading, optimizer))
        previous, state, last = (reading, minutes, prior), step.memory, stamp
    return torch.tensor(readings).sigmoid(), {
        "arrival_updates": len(changes),
        "forecast_mae": sum(forecasts) / len(forecasts) if forecasts else None,
        "adapter_scale": model.adapter_scale.detach().tolist(),
        "adapter_offset": model.adapter_offset.detach().tolist(),
        "expert_weight": torch.stack(routes).mean(0).tolist(),
        "mean_revision_gate": torch.stack(corrections).mean(0).tolist(),
        "predicted_process_gain": torch.stack(gains).mean(0).tolist(),
    }


def restore(out, item, name):
    raw = (out / name).read_bytes()
    check_hash(raw, item["sha256"])
    model = build(item["mode"])
    model.load_state_dict(torch.load(io.BytesIO(raw), weights_only=True))
    return model


def periods(p, y, cutoff):
    result = {}
    for first, stop in ((0, 288), (288, 864), (864, 2016), (2016, len(y))):
        actual = y[first:stop]
        if len(actual) and 0 < actual.sum() < len(actual):
            result[f"{first}:{stop}"] = source.score(p[first:stop], actual, cutoff)
    return result


def score_site(out, manifest, site, rooms):
    scores = []
    for name, item in manifest.items():
        variants = [("state", True, False, None)]
        if item["mode"] in ("plain", "full"):
            variants.append(("learn-all", True, True, None))
        if item["mode"] in ("plain", "full") and item["share"] == 1:
            variants.append(("cold", False, False, None))
        if item["mode"] == "full" and item["seed"] == SEEDS[0] and item["share"] == 1:
            variants.extend((f"learn-{budget}", True, True, budget) for budget in (288, 864))
        for variant, carry, learn, budget in variants:
            predictions, labels, diagnostics, room_scores = [], [], {}, {}
            for room, x, y, t in rooms:
                # A fixed evaluation horizon bounds the test identically in each arm.
                x, y, t = x[:2016], y[:2016], t[:2016]
                p, details = stream(restore(out, item, name), x, t, carry, learn, budget)
                predictions.append(p)
                labels.append(y)
                diagnostics[room] = details
                room_scores[room] = {
                    "whole": source.score(p, y, item["cutoff"]),
                    "periods": periods(p, y, item["cutoff"]),
                }
            scores.append({"site": site, "model": name, "variant": variant,
                           "metrics": source.score(torch.cat(predictions), torch.cat(labels), item["cutoff"]),
                           "diagnostics": diagnostics, "rooms": room_scores})
            print(site, name, variant, round(scores[-1]["metrics"]["balanced_accuracy"], 4), flush=True)
    return scores



def score_worker(args):
    torch.set_num_threads(1)
    return score_site(*args)


def heat(path):
    raw = path.read_bytes()
    check_hash(raw, control.ARCHIVE_HASH)
    with zipfile.ZipFile(io.BytesIO(raw)) as zip_file:
        text = io.TextIOWrapper(zip_file.open(control.CSV_NAME))
        rows = {source.timestamp(r["Time"]) // 60: r for r in csv.DictReader(text)}
    transitions = []
    start = min(rows) + (-min(rows) % 15)
    for minute in range(start, max(rows) - 14, 15):
        try:
            series = [rows[m] for m in range(minute, minute + 16)]
            prev = rows[minute - 15]
            features = lambda row: (float(row["RTD31"]), float(row["Tout"]))
            x, y = features(series[0]), features(series[-1])
            commands = {(float(row["CpSp3"]), float(row["EvRad"])) for row in series[:-1]}
            if len(commands) != 1 or next(iter(commands)) not in control.ACTION:
                continue
            action = int(control.ACTION[next(iter(commands))] == "on")
            temps = [float(row["RTD31"]) for row in series[1:]]
            power = [float(row["PhRad_actual_power"]) for row in series[1:]]
            if not all(map(math.isfinite, (*x, *y, *temps, *power))) or min(power) < 0:
                continue
            energy = sum(power) / 60_000
            discomfort = sum(max(20 - t, t - 24, 0) for t in temps) / 60
            transitions.append((minute, x, y, action, energy, discomfort, control.state(series[0], prev)))
        except (KeyError, ValueError):
            continue
    split = int(control.HOLDOUT_START.timestamp()) // 60
    return ([r for r in transitions if r[0] + 15 < split],
            [r for r in transitions if r[0] >= split])


def heat_tensors(records):
    _, x, y, actions, energy, discomfort, _ = zip(*records, strict=True)
    center, scale = torch.tensor((22., 5.)), torch.tensor((4., 10.))
    return ((torch.tensor(x) - center) / scale, (torch.tensor(y) - center) / scale,
            torch.tensor(actions), -torch.tensor(energy) - 2 * torch.tensor(discomfort))


def value_loss(model, current, nxt, action, outcome, continues, prior, target=None, target_prior=None):
    elapsed = current.new_full((len(current),), 15.)
    step = model.step(current, elapsed, prior)
    target_model = target if target is not None else model
    with torch.no_grad():
        source_state = (target_model.step(current, elapsed, target_prior).memory
                        if target is not None else step.memory)
        follow = target_model.step(nxt, elapsed, source_state)
        if step.process.numel():
            errors = (step.candidate_forecasts - (nxt - current)[:, None, None, :]).abs().mean(-1)
            process = (errors[:, :, 0] - errors[:, :, 1]).mean(-1)
        else:
            process = torch.zeros_like(outcome)
    loss = logged_q_loss(step.q, follow.q, action, outcome, continues, process)
    return loss + .3 * sensor_loss(current, nxt, step), step


def fit_heat(data, out, epochs):
    train, _ = heat(data / "six-buildings.zip")
    x, nxt, actions, reward = heat_tensors(train)
    continuation = torch.tensor([train[i+1][0] == r[0] + 15
                                 if i+1 < len(train) else False for i,r in enumerate(train)]).float()
    entries = {}
    val = len(x) * 4 // 5
    for seed in SEEDS:
        for mode in ("plain", "full"):
            torch.manual_seed(seed)
            model, target = build(mode), build(mode)
            target.load_state_dict(model.state_dict())
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            best = None
            curve = []
            for epoch in range(1, epochs + 1):
                perm = torch.randperm(val - 1, generator=torch.Generator().manual_seed(seed+epoch))
                for ix in perm.split(96):
                    optimizer.zero_grad()
                    loss, _ = value_loss(model, x[ix], nxt[ix], actions[ix], reward[ix],
                                         continuation[ix], None, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                if epoch % 2 == 0:
                    target.load_state_dict(model.state_dict())
                with torch.no_grad():
                    v, _ = value_loss(model, x[val:], nxt[val:], actions[val:], reward[val:],
                                      continuation[val:], None, target)
                    val_loss = float(v)
                curve.append(val_loss)
                if best is None or val_loss < best[0]:
                    best = (val_loss, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
                if epoch - best[1] >= 8:
                    break
            name = f"heating-{mode}-{seed}.pt"
            path = out / name
            torch.save(best[2], path)
            entries[name] = {"mode": mode, "seed": seed, "epoch": best[1],
                             "sha256": source.sha(path.read_bytes()), "curve": curve}
            print(name, "kept epoch", best[1], "of", len(curve), flush=True)
    return entries


def score_heat(data, out, manifest):
    train, hold = heat(data / "six-buildings.zip")
    counts = Counter((r[-1], r[3]) for r in train[:len(train)*4//5])
    x, nxt, actions, reward = heat_tensors(hold)
    results = []
    for name, item in manifest.items():
        for learn in (False, True):
            model = restore(out, item, name)
            target = restore(out, item, name)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            prior, target_prior, last = None, None, None
            matches = []
            trace = []
            for i, row in enumerate(hold):
                if last is None or row[0] - last > 15:
                    prior, target_prior = None, None
                sample, elapsed = x[i:i+1], x.new_tensor([15.])
                with torch.no_grad():
                    step = model.step(sample, elapsed, prior)
                    target_step = target.step(sample, elapsed, target_prior)
                    if not all(counts[row[-1], a] >= 8 for a in (0, 1)):
                        choice = None
                    else:
                        choice = int(step.q.argmax(-1).item())
                if choice == row[3]:
                    matches.append(i)
                trace.append({"minute_utc": row[0], "logged_action": row[3], "recommendation": choice,
                              "outcome_reward": float(reward[i]), "thermal_kwh": row[4],
                              "comfort_degree_hours": row[5], "q": step.q.flatten().tolist()})
                if learn:
                    optimizer.zero_grad()
                    has_next = i + 1 < len(hold) and hold[i+1][0] == row[0] + 15
                    loss, _ = value_loss(model, sample, nxt[i:i+1], actions[i:i+1], reward[i:i+1],
                                         sample.new_tensor([float(has_next)]), prior, target, target_prior)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                    optimizer.step()
                    if i % 32 == 31:
                        target.load_state_dict(model.state_dict())
                prior, target_prior, last = step.memory.detach(), target_step.memory.detach(), row[0]
            results.append({"model": name, "updates_after_outcome": len(hold) if learn else 0,
                            "transitions": len(hold), "covered": sum(t["recommendation"] is not None for t in trace),
                            "logged_matches": len(matches), "matched_logged_reward": float(reward[matches].sum()),
                            "matching_thermal_kwh": sum(hold[i][4] for i in matches),
                            "matching_degree_hours": sum(hold[i][5] for i in matches), "trace": trace})
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("fit", "score"))
    cli.add_argument("data", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--epochs", type=int, default=24)
    cli.add_argument("--freeze")
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    torch.set_num_threads(1)
    out = args.output
    own = Path(__file__).resolve().parent
    files = {str(path.relative_to(ROOT)): source.sha(path.read_bytes()) for path in (
        own/"experiment.py", own/"model.py", ROOT/"liquid-fidelity"/"experiment.py",
        ROOT/"liquid-fidelity"/"evaluate.py", ROOT/"building-control"/"replay.py")}
    if args.command == "fit":
        out.mkdir()
        train, val, input_hashes = fit_sources(args.data)
        manifest = {"code": files, "torch": str(torch.__version__), "epochs": args.epochs,
                    "source_sha256": input_hashes, "occupancy": occupancy_fit(train, val, out, args.epochs, args.workers),
                    "heating": fit_heat(args.data, out, args.epochs)}
        content = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
        (out / "frozen.json").write_bytes(content)
        print("Freeze SHA-256", source.sha(content), flush=True)
    else:
        raw = (out / "frozen.json").read_bytes()
        check_hash(raw, args.freeze)
        manifest = json.loads(raw)
        if manifest["code"] != files:
            raise ValueError("Code differs from training seal")
        sites = test_sites(args.data)
        jobs = [(out, {key: item}, site, rooms)
                for site, rooms in sites.items() for key, item in manifest["occupancy"].items()]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            sites = list(executor.map(score_worker, jobs))
        results = {"freeze": args.freeze, "occupancy": [r for scores in sites for r in scores],
                   "heating": score_heat(args.data, out, manifest["heating"])}
        (out / "scores.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
