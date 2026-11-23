"""Audit occupancy labels at five-minute boundaries and fit a tree baseline."""

import argparse
from collections import Counter, defaultdict
import csv
from io import TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import torch
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "debug-training")]
from experiment import occupancy, sha
from data import ENV_FIELDS, epoch
from train_debug import fit_cutoff, measure


def audit_labels(path, days, stream):
    # Group the selected raw labels independently before comparing with the
    # production loader. Sort within each bucket by timestamp.
    buckets = defaultdict(list)
    with ZipFile(path) as archive:
        files = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        selected = 0
        for file in files:
            with TextIOWrapper(archive.open(file), encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            if len(rows) < 8640:
                continue
            for row in rows:
                if row["occupied"] in ("0", "1"):
                    stamp = epoch(row["timestamp"])
                    buckets[stamp // 300].append((stamp, int(row["occupied"])))
            selected += 1
            if selected == days:
                break

    stops, inputs, labels = stream
    mixed = Counter()
    transition_labels = []
    label_age = []
    for stop, label in zip(stops.tolist(), labels.tolist(), strict=True):
        rows = sorted(buckets[stop // 300 - 1])
        latest, actual = rows[-1]
        if actual != label or any(value != actual for t, value in rows if t == latest):
            raise ValueError("Loader label differs from the latest unambiguous raw label")
        label_age.append(stop - latest)
        changed = len({value for _, value in rows}) > 1
        transition_labels.append(changed)
        if changed:
            mixed["changed_within_window"] += 1
            if rows[0][1] != actual:
                mixed["arrivals" if actual else "departures"] += 1
            else:
                mixed["returned_to_start_label"] += 1

    present = inputs.isfinite()
    return dict(
        scored_windows=len(labels),
        vacant=int((labels == 0).sum()),
        occupied=int((labels == 1).sum()),
        labeled_windows_without_environment=len(buckets) - len(labels),
        max_seconds_from_last_label_to_decision=max(label_age),
        decision_labels_matching_raw_end=len(labels),
        changed_between_contiguous_window_ends=int(((stops.diff() == 300) &
                                                    (labels.diff() != 0)).sum()),
        changed_within_window=mixed["changed_within_window"],
        arrivals=mixed["arrivals"],
        departures=mixed["departures"],
        returned_to_start_label=mixed["returned_to_start_label"],
        available_levels=dict(zip(ENV_FIELDS, present[:, :5].sum(0).tolist(), strict=True)),
    ), torch.tensor(transition_labels)


def predict(model, inputs):
    return torch.from_numpy(model.predict(xgb.DMatrix(inputs.numpy(), nthread=1)))


def replay(model, stream, cutoff, transitions):
    _, inputs, truth = stream
    scores = predict(model, inputs)
    answered = inputs.isfinite().any(-1)
    right = (scores >= cutoff).eq(truth.bool()) & answered
    return dict(
        selected_cutoff=measure(scores, answered, truth, cutoff),
        fixed_half=measure(scores, answered, truth, .5),
        transition_rows=int(transitions.sum()),
        transition_correct=int(right[transitions].sum()),
    )


def run(data, output):
    streams, audits, transitions, archives = [], {}, [], {}
    for home in range(1, 7):
        name = f"H{home}"
        days = 14 if home <= 3 else 7
        stream, hashes = occupancy(data, home, days)
        audit, changed = audit_labels(data / f"{name}_GROUNDTRUTH.zip", days, stream)
        streams.append(stream)
        audits[name] = audit
        transitions.append(changed)
        archives.update(hashes)

    inputs = torch.cat([stream[1] for stream in streams[:3]])
    labels = torch.cat([stream[2] for stream in streams[:3]])
    counts = labels.bincount(minlength=2)
    weights = len(labels) / (2 * counts[labels]).float()
    rounds = 100
    settings = dict(max_depth=3, eta=.05, objective="binary:logistic",
                    tree_method="hist", nthread=1, seed=51)
    batch = xgb.DMatrix(inputs.numpy(), label=labels.numpy(), weight=weights.numpy(), nthread=1)
    model = xgb.train(settings, batch, num_boost_round=rounds)

    selection = streams[3]
    mid = len(selection[0]) // 2
    first = tuple(column[:mid] for column in selection)
    second = tuple(column[mid:] for column in selection)
    probabilities = predict(model, first[1])
    cutoff = fit_cutoff(probabilities, first[1].isfinite().any(-1), first[2])
    cases = {
        "H4 first half": (first, transitions[3][:mid]),
        "H4 second half": (second, transitions[3][mid:]),
        "H5": (streams[4], transitions[4]),
        "H6": (streams[5], transitions[5]),
    }
    results = {name: replay(model, stream, cutoff, changed)
               for name, (stream, changed) in cases.items()}

    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "trees.json"
    model.save_model(model_path)
    source_paths = (Path(__file__), HERE.parent / "experiment.py", HERE.parent.parent / "liquid-workflow-v3" / "data.py",
                    HERE.parent.parent / "liquid-workflow-v3" / "train.py",
                    HERE.parent / "debug-training" / "train_debug.py")
    sources = {str(path.resolve().relative_to(HERE.parents[3])): sha(path) for path in source_paths}
    record = dict(archives=archives, sources=sources, model_sha256=sha(model_path),
                  settings=dict(rounds=rounds, **settings), fit_rows=len(labels), fit_class_counts=counts.tolist(),
                  cutoff=cutoff, alignment=audits, results=results)
    (output / "results.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.archives, args.output)
