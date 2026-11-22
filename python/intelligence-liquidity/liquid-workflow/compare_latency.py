"""Compare single-reading CPU latency for the saved liquid and XGBoost models."""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import torch
import xgboost as xgb

from compare import features
from run import dump, io_data, sha
from serve import Stream, percentile


def measure(data, evidence):
    torch.set_num_threads(1)
    room, readings, _, times = io_data.test_sites(data)["munich"][0]
    readings = readings[:400]
    times = times[:400]
    offsets, scales = torch.tensor([500., 22.]), torch.tensor([200., 3.])
    raw = readings * scales + offsets
    records = [dict(timestamp=int(t), co2_ppm=float(x[0]), temperature_c=float(x[1]))
               for x, t in zip(raw, times, strict=True)]
    liquid_path, tree_path = evidence / "41-100.pt", evidence / "xgboost-41.ubj"

    fit = json.loads((evidence / "fit.json").read_text())["checkpoints"][liquid_path.name]
    tree_fit = json.loads((evidence / "xgboost.json").read_text())["runs"][0]
    if sha(liquid_path) != fit["state_sha256"] or sha(tree_path) != tree_fit["sha256"]:
        raise ValueError("Checkpoint hashes do not match the fit receipts")
    results = {name: [] for name in ("tree", "liquid_frozen", "liquid_live")}
    for order in (tuple(results), tuple(reversed(results)), tuple(results)):
        for mode in order:
            if mode == "tree":
                tree = xgb.Booster(params={"nthread": 1})
                tree.load_model(tree_path)
                tree.set_param({"nthread": 1})
                inputs = ((torch.tensor([r["co2_ppm"], r["temperature_c"]]) - offsets) / scales
                          for r in records)
                rows = features(inputs, times)
                for i in range(len(records)):
                    tick = perf_counter_ns()
                    row = next(rows).unsqueeze(0).numpy()
                    probability = tree.inplace_predict(row, iteration_range=(0, tree_fit["rounds"]))
                    _ = float(probability[0]) >= tree_fit["cutoff"]
                    if i >= 20:
                        results[mode].append((perf_counter_ns() - tick) / 1e6)
            else:
                stream = Stream(liquid_path, fit["cutoff"], live=mode == "liquid_live")
                for i, record in enumerate(records):
                    tick = perf_counter_ns()
                    stream.observe(record)
                    if i >= 20:
                        results[mode].append((perf_counter_ns() - tick) / 1e6)
    cpu = platform.processor()
    proc = Path("/proc/cpuinfo")
    if proc.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in proc.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return dict(
        room=room, cpu=cpu, threads=1, torch=str(torch.__version__), xgboost=xgb.__version__,
        checkpoint_sha256={liquid_path.name: sha(liquid_path), tree_path.name: sha(tree_path)},
        rows_per_pass=len(records), discarded_per_pass=20, passes=3,
        milliseconds={name: dict(p50=percentile(samples, .50), p95=percentile(samples, .95),
                                  count=len(samples)) for name, samples in results.items()},
        scope="In-process, one reading at a time. Includes normalization, features/state, checks and "
              "live feedback where applicable. Excludes loading, input/output, persistence and energy.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dump(args.output, measure(args.data, args.evidence))
