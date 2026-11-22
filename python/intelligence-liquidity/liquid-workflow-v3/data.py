"""Read public occupancy archives and real logged heater actions."""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from io import TextIOWrapper
import math
from pathlib import Path
from statistics import median
from zipfile import ZipFile

import torch

import sys

ARTICLES = ((14368457, 14368460), (14368538, 14368541), (14368580, 14368574),
            (14368619, 14368613), (14368670, 14368658), (14368703, 14368706))
ENV_FIELDS = ("co2eq_ppm", "tvoc_ppb", "temp_c", "rh_percent", "light_lux")
INTERVAL = {"occupancy": 300, "heat": 900}


def sha(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def epoch(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def normalize(raw):
    co2, voc, temp, humidity, light = raw
    return [max(co2 - 400, 0) / 500 if co2 is not None else None,
            math.log1p(max(voc, 0)) / 5 if voc is not None else None,
            (temp - 24) / 5 if temp is not None else None,
            (humidity - 40) / 25 if humidity is not None else None,
            math.log1p(max(light, 0)) / 6 if light is not None else None]


def occupancy(folder, home, days):
    environmental, truth = [folder / f"H{home}_{kind}.zip" for kind in ("ENVIRONMENTAL", "GROUNDTRUTH")]
    wanted, labels = [], {}
    with ZipFile(truth) as archive:
        for file in sorted(n for n in archive.namelist() if n.endswith(".csv")):
            with TextIOWrapper(archive.open(file), encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
                if len(rows) < 8640:
                    continue
                date = Path(file).name[:10]
                wanted.append(date)
                for row in rows:
                    if row["occupied"] in ("0", "1"):
                        stamp = epoch(row["timestamp"])
                        labels[stamp // 300] = int(row["occupied"])
                if len(wanted) == days:
                    break
    if not wanted:
        raise ValueError(f"H{home} has no complete labeled days")
    # Keep the most recent observation for each field and hub inside the current
    # five-minute bucket. The decision is made at the end of that bucket.
    hubs = defaultdict(dict)
    with ZipFile(environmental) as archive:
        for file in sorted(n for n in archive.namelist() if n.endswith(".csv")
                           and Path(n).name[:10] in wanted):
            channel = Path(file).parent.name
            with TextIOWrapper(archive.open(file), encoding="utf-8-sig") as source:
                for row in csv.DictReader(source):
                    stamp = epoch(row[""])
                    bucket = stamp // 300
                    if bucket not in labels:
                        continue
                    for idx, field in enumerate(ENV_FIELDS):
                        value = number(row[field])
                        key = channel, idx
                        if value is not None and stamp >= hubs[bucket].get(key, (-1, None))[0]:
                            hubs[bucket][key] = stamp, value
    timestamps, inputs, outcomes = [], [], []
    previous, last = None, None
    for bucket, readings in sorted(hubs.items()):
        groups = [[v for (_, idx), (_, v) in readings.items() if idx == i] for i in range(len(ENV_FIELDS))]
        window = normalize([(median(g) if i in (2, 3) else max(g)) if g else None
                            for i, g in enumerate(groups)])
        current = torch.tensor([float("nan") if f is None else f for f in window])
        delta = current - previous if last == bucket - 1 else torch.full_like(current, float("nan"))
        timestamps.append((bucket + 1) * 300)
        inputs.append(torch.cat((current, delta)))
        outcomes.append(labels[bucket])
        previous, last = current, bucket
    return (torch.tensor(timestamps), torch.stack(inputs), torch.tensor(outcomes)), {
        p.name: sha(p) for p in (environmental, truth)}


def heater_rows(records):
    seconds, inputs, actions, rewards = [], [], [], []
    prev = None
    for row in records:
        minute, sensors, _, action, energy, discomfort, _ = row
        lag = (2 * prev[3] - 1, prev[4], prev[5]) if prev is not None and minute - prev[0] == 15 else (float("nan"),) * 3
        inputs.append(((sensors[0] - 22) / 4, (sensors[1] - 5) / 10, *lag))
        seconds.append(minute * 60)
        actions.append(action)
        rewards.append(-energy - 2 * discomfort)
        prev = row
    return torch.tensor(seconds), torch.tensor(inputs), torch.tensor(actions), torch.tensor(rewards)


def heater(path):
    # Use the released archive parser and its fixed chronological split.
    previous = Path(__file__).resolve().parents[1] / "liquid-workflow"
    sys.path.insert(0, str(previous))
    try:
        from run import io_data
    finally:
        sys.path.pop(0)
    original, test = io_data.heat(path)
    split = len(original) * 4 // 5
    return tuple(heater_rows(records) for records in (original[:split], original[split:], test))
