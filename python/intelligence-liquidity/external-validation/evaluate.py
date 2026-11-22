"""Evaluate the frozen occupancy models on a separate room without fitting."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from datetime import datetime
from math import isfinite
from pathlib import Path
from zipfile import ZipFile

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_SHA256 = "97d9da53a69e98566f4e3caf7b7c1a50ae5bf19be5975da0c277d27260f83d43"
ARCHIVE_SHA256 = "a0db79ee96ea297c22458ac77a28fba20d41246e75d24b2655ca3ab624cc08fd"


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def checked(path, expected):
    payload = path.read_bytes()
    if sha256(payload) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    return payload


def read_rows(payload, protocol, features, mean):
    with ZipFile(io.BytesIO(payload)) as archive:
        contents = archive.read(protocol["archive_csv"])
    parsed = []
    for record in csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))):
        instant = datetime.strptime(f"{record['Date']} {record['Time']}", "%Y/%m/%d %H:%M:%S")
        values = [
            mean[index] if column is None else float(record[column])
            for index, feature in enumerate(features)
            for column in [protocol["input_map"][feature]]
        ]
        count = int(record["Room_Occupancy_Count"])
        if count not in range(4) or not all(isfinite(value) for value in values):
            raise ValueError(f"Invalid measurement or count at {instant}")
        parsed.append((instant.isoformat(sep=" "), values, int(count > 0)))
    times = [datetime.fromisoformat(row[0]) for row in parsed]
    if len(parsed) != 10129 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Publisher row count or chronological order differs")
    return parsed, times, sha256(contents)


def windows(prior, rows, times, protocol, checkpoint):
    indices = list(range(len(rows)))
    width = protocol["window"]
    cap = protocol["max_gap_minutes"] * 60
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    targets = {
        i for i in indices[width - 1:]
        if max(gaps[i - width + 1:i]) <= cap
    }
    mean, scale = map(torch.tensor, (checkpoint["mean"], checkpoint["scale"]))
    x, g, y, ids = prior.windows(rows, indices, targets, mean, scale)
    if set(ids) != targets or len(set(y.tolist())) != 2:
        raise ValueError("Evaluation targets differ or a class is missing")
    return x, g, y


def evaluate(archive_path):
    protocol = json.loads(checked(HERE / "protocol.json", PROTOCOL_SHA256))
    candidate = json.loads(checked(ROOT / "lifecycle/candidate.json", protocol["checkpoint_sha256"]))
    baselines = json.loads(checked(ROOT / "custom-ltc/checkpoint.json", protocol["baseline_checkpoint_sha256"]))
    for field in ("features", "mean", "scale", "window", "schema", "units", "substeps"):
        if candidate[field] != baselines[field]:
            raise ValueError(f"Training metadata differs for {field}")
    source = ROOT / "custom-ltc/model.py"
    spec = importlib.util.spec_from_file_location("frozen_models", source)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    if prior.WINDOW != protocol["window"] or protocol["threshold"] != 0.5:
        raise ValueError("Model shape or metric threshold differs")

    data = checked(archive_path, ARCHIVE_SHA256)
    rows, times, csv_hash = read_rows(data, protocol, candidate["features"], candidate["mean"])
    x, g, y = windows(prior, rows, times, protocol, candidate)
    seed = str(protocol["seed"])
    metrics = {}
    for label, kind, entry in (
        ("candidate", "ltc", candidate["runs"][seed]["ltc"]),
        ("previous_ltc", "ltc", baselines["runs"][seed]["ltc"]),
        ("gru", "gru", baselines["runs"][seed]["gru"]),
        ("latest_reading", "current", baselines["runs"][seed]["current"]),
    ):
        model = prior.load_model(kind, entry)
        logits = prior.predict(model, x, g)
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite prediction from {label}")
        metrics[label] = prior.metrics(logits, y)

    positive = int(y.sum())
    negative = len(y) - positive
    return {
        "archive_sha256": sha256(data),
        "csv_sha256": csv_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "model_source_sha256": sha256(source.read_bytes()),
        "runtime": str(torch.__version__),
        "source_rows": len(rows),
        "first": times[0].isoformat(),
        "last": times[-1].isoformat(),
        "windows": len(y),
        "occupied_windows": positive,
        "empty_windows": negative,
        "excluded_first_rows": protocol["window"] - 1,
        "excluded_gap_targets": len(rows) - len(y) - protocol["window"] + 1,
        "metrics": metrics,
        "always_empty": {
            "tp": 0, "tn": negative, "fp": 0, "fn": positive,
            "accuracy": negative / len(y), "balanced_accuracy": 0.5,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expect", type=Path)
    args = parser.parse_args()
    try:
        result = json.dumps(evaluate(args.archive), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.expect and args.expect.read_text() != result:
            raise ValueError("Receipt differs")
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(result, end="")


if __name__ == "__main__":
    main()
