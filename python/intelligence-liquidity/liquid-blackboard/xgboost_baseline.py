"""Fit a reproducible tree baseline with the same inputs, blocks and holdout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

from experiment import SEEDS, batches, fit_sources, source, test_sites


def features(readings, timestamps):
    """Describe the current sample and the preceding history, never a future row."""
    history = []
    rows = []
    previous = None
    for reading, stamp in zip(readings, timestamps, strict=True):
        if previous is not None and stamp - previous > 450:
            history.clear()
        history.append(reading)
        history = history[-11:]
        recent = np.stack(history)
        delta = reading - recent[-2] if len(recent) > 1 else np.zeros(2)
        rows.append(np.concatenate([reading, delta, recent.mean(0), recent.min(0),
                                    recent.max(0), reading - recent[0]]))
        previous = stamp
    return np.stack(rows)


def fitting_matrix(streams):
    x, y = batches(streams)
    clock = np.arange(11) * 300
    return xgb.DMatrix(np.concatenate([features(block[:-1].numpy(), clock) for block in x]),
                       label=y[:, :-1].flatten().numpy())


def fit(data, out):
    train, validation, inputs = fit_sources(data)
    pool, check = fitting_matrix(train), fitting_matrix(validation)
    entries = {}
    for seed in SEEDS:
        configuration = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                             max_depth=4, eta=.05, min_child_weight=10, subsample=.8,
                             colsample_bytree=.8, seed=seed, nthread=2)
        curve = {}
        model = xgb.train(configuration, pool, num_boost_round=400, evals=[(check, "aalto")],
                          early_stopping_rounds=25, evals_result=curve, verbose_eval=False)
        best = model.best_iteration + 1
        val = torch.from_numpy(model.predict(check, iteration_range=(0, best)))
        cutoff = source.threshold(val, torch.from_numpy(check.get_label()))
        file = out / f"xgboost-{seed}.ubj"
        model.save_model(file)
        entries[file.name] = dict(sha256=source.sha(file.read_bytes()), cutoff=cutoff,
                                  best_round=best, config=configuration, validation=curve)
    manifest = dict(version=xgb.__version__, source_sha256=inputs, train_rows=pool.num_row(),
                    validation_rows=check.num_row(), checkpoints=entries)
    raw = (json.dumps(manifest, indent=2) + "\n").encode()
    (out / "xgboost.json").write_bytes(raw)
    print("Baseline manifest SHA-256", source.sha(raw), flush=True)


def score(data, out, digest):
    raw = (out / "xgboost.json").read_bytes()
    if source.sha(raw) != digest:
        raise ValueError("Baseline manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    sites = test_sites(data)
    results = []
    for name, item in manifest["checkpoints"].items():
        file = out / name
        if source.sha(file.read_bytes()) != item["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {name}")
        model = xgb.Booster(params={"nthread": 2})
        model.load_model(file)
        for site, rooms in sites.items():
            predictions, observed = [], []
            for _, x, y, t in rooms:
                causal = features(x[:2016].numpy(), t[:2016].numpy())
                matrix = xgb.DMatrix(causal)
                p = model.predict(matrix, iteration_range=(0, item["best_round"]))
                predictions.append(torch.from_numpy(p))
                observed.append(y[:2016])
            result = source.score(torch.cat(predictions), torch.cat(observed), item["cutoff"])
            results.append(dict(site=site, model=name, metrics=result))
    (out / "xgboost-scores.json").write_text(json.dumps({"freeze": digest, "rows": results}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "score"))
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--freeze")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "fit":
        args.output.mkdir()
        fit(args.data, args.output)
    else:
        score(args.data, args.output, args.freeze)


if __name__ == "__main__":
    main()
