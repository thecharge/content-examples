"""Replay saved heating snapshots, infer offline, or list exact occupancy windows."""
import argparse
import gzip
import base64
import json
import math
from pathlib import Path

import torch

from benchmark import recommendation, reload, score_heat, heating_prediction
from data import WINDOW, fitting_blocks, heat_rows, heat_windows, office, occupancy_windows, support_bucket


def snapshot(path, name):
    entry, = (row for row in json.loads(Path(path).read_text())["runs"] if row["model"] == name)
    counts = {tuple(key): count for key, count in entry["support_table"]}
    means = {tuple(key): torch.tensor(avg) for key, _, avg in entry.get("lookup", [])}
    fallback = {int(key): torch.tensor(avg) for key, avg in entry.get("action_means", {}).items()}
    model = None if name == "mean" else reload("heat", name, entry["weights"])
    return entry, model, counts, means, fallback


def replay(path, name, source):
    entry, model, counts, means, fallback = snapshot(path, name)
    rows, _ = heat_windows(heat_rows(source, True))
    output = score_heat(model, rows, entry["x_stats"], entry["y_stats"], counts, means, fallback)
    records = output.pop("records")
    assert output == entry["test"], "Replayed heating metrics or traces differ from the receipt"
    return records


@torch.no_grad()
def infer(path, name, request):
    entry, model, counts, means, fallback = snapshot(path, name)
    invalid = dict(action=None, rank=None, reason="invalid_input")
    try:
        series = request["history"]
        if len(series) != WINDOW:
            return invalid
        if any(len(reading) != 3 for reading in series):
            return invalid
        times, indoor, outdoor = zip(*series)
        prior_time, prior_indoor = request["prior"]
        gaps = [1., *(b-a for a, b in zip(times, times[1:]))]
        if (not all(map(math.isfinite, [*times, *indoor, *outdoor, prior_time, prior_indoor]))
                or any(gap != 1 for gap in gaps) or times[-1] - prior_time != 15
                or request["now"] != times[-1]):
            return invalid
        row = dict(x=list(zip(indoor, outdoor)), gaps=gaps, action=0, y=[0., 0., 0.],
                   bucket=support_bucket(indoor[-1], outdoor[-1], prior_indoor))
        outs = [heating_prediction(model, [row], a, entry["x_stats"], entry["y_stats"], means, fallback)
                for a in (0, 1)]
        predictions = torch.cat([out[0] for out in outs], dim=0)
        attention = None if outs[0][1] is None else outs[0][1][0].tolist()
        return recommendation(row, predictions, counts, attention)
    except (KeyError, TypeError, ValueError, OverflowError):
        return invalid


def ranges(ids):
    compressed = []
    for value in ids:
        if compressed and value == compressed[-1][1] + 1:
            compressed[-1][1] = value
        else:
            compressed.append([value, value])
    return compressed


def membership(path):
    rows = office(path, "datatraining.txt")
    result = {}
    for key, blocks in [*((str(b), fitting_blocks(rows, b)) for b in (25, 50, 100)),
                        ("validation", [rows[6500:]])]:
        windows = occupancy_windows(blocks, 5, 1)
        result[key] = dict(row_ranges=[[b[0][0], b[-1][0]] for b in blocks],
                           endpoint_ranges=ranges(row["id"] for row in windows),
                           auxiliary_endpoint_ranges=ranges(row["id"] for row in windows if row["next_id"] is not None))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    subs.add_parser("membership").add_argument("archive")
    for mode in ("replay", "infer"):
        command = subs.add_parser(mode)
        command.add_argument("snapshot")
        command.add_argument("model", choices=("mean", "liquid", "attention"))
        command.add_argument("input")
        if mode == "replay":
            command.add_argument("--compressed", action="store_true")
    args = parser.parse_args()
    if args.mode == "membership":
        output = membership(args.archive)
    elif args.mode == "replay":
        output = replay(args.snapshot, args.model, args.input)
        if args.compressed:
            raw = json.dumps(output, separators=(",", ":"), allow_nan=False).encode()
            print(base64.b64encode(gzip.compress(raw, mtime=0)).decode())
            return
    else:
        output = infer(args.snapshot, args.model, json.loads(Path(args.input).read_text()))
    print(json.dumps(output, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
