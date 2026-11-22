"""Measure each model reading one record at a time on the same CPU."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import xgboost as xgb

from experiment import FACTORIES, dump, sha
from compare import features
from transfer import h6
from workflow import Observation, Online


def measure(evidence, source, output):
    model_receipt = json.loads((evidence / "fit.json").read_text())["models"]
    (room,), archives = h6(source)
    _, readings, _, timestamps = room
    readings, timestamps = readings[:300], timestamps[:300]
    torch.set_num_threads(1)
    timings = {}
    for variant in ("workflow", "live", "ltc", "gru", "xgboost"):
        key = "workflow" if variant == "live" else variant
        model = FACTORIES[key]() if key != "xgboost" else None
        path = evidence / (f"{key}.pt" if model else "xgboost.ubj")
        rec = model_receipt[key]
        if sha(path) != rec["best_sha256"]:
            raise ValueError(f"Changed weights for {key}")
        state, idle_minutes = None, 0.
        if model:
            model.load_state_dict(torch.load(path, weights_only=True))
        else:
            tree = xgb.Booster(model_file=path)
            causal = iter(features(readings[:, :1], timestamps))
        learner = Online(model) if variant == "live" else None
        values = []
        for i, (reading, stamp) in enumerate(zip(readings, timestamps.tolist(), strict=True)):
            seconds = stamp - int(timestamps[i-1]) if i else 300
            if seconds > 450:
                state, idle_minutes = None, 0.
            minutes = readings.new_tensor([seconds/60 if seconds <= 450 else 5.])
            primary, witness = reading.reshape(1, 2).unbind(-1)
            start = perf_counter_ns()
            with torch.no_grad():
                if key == "workflow":
                    result = model.step(primary, witness, minutes, state, idle_minutes=idle_minutes)
                elif model:
                    _, state = model.step(primary, minutes, state)
                else:
                    tree.inplace_predict(next(causal).reshape(1, -1).numpy(),
                                         iteration_range=(0, rec["rounds"]))
            if learner:
                learner.feedback(Observation(stamp, primary, witness, minutes, state, idle_minutes))
            if key == "workflow":
                idle_minutes = 0. if result.accepted.item() else idle_minutes + minutes.item()
                state = result.ledger.detach()
            if i >= 40:
                values.append((perf_counter_ns() - start)/1_000_000)
        timings[variant] = dict(p50=median(values), p95=sorted(values)[(95*len(values)+99)//100-1],
                                samples=values)
    dump(output, dict(unit="milliseconds per row", warmup_rows=40, timed_rows=len(readings)-40,
                      fit_sha256=sha(evidence/"fit.json"), archives=archives, repetitions=1, timings=timings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("h6", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.evidence, args.h6, args.output)
