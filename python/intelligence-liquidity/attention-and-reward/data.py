"""Time-safe source mappings. Row numbers in receipts are zero-based."""
import csv
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timezone
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import torch

WINDOW = 8
OCC_FEATURES = ("Temperature", "Humidity", "CO2")
SOURCES = {
    "uci": ("https://doi.org/10.24432/C5X01N", "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"),
    "danish": ("https://zenodo.org/records/10761326", "c08081438816078bdce5d607383dc3e610e2584c479de3d0a4d9b5ded1215ee9"),
    "heat": ("https://data.mendeley.com/datasets/xztfbtsgys/3", "393c5c5d106d1597e0a059ea43b799d7d2e58baf74a6388beb256c46f2400fe8"),
}
HOLDOUT = int(datetime(2019, 3, 7, tzinfo=timezone.utc).timestamp()) // 60
COMMANDS = {(0., 0.): 0, (100., 100.): 1}


def verify(path, source):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCES[source][1]:
        raise ValueError(f"Unexpected SHA-256 for {source}")


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return math.nan
    return result


def minute(value):
    date = datetime.fromisoformat(value)
    return (date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date).timestamp() / 60


def office(path, member):
    verify(path, "uci")
    with ZipFile(path) as archive, TextIOWrapper(archive.open(member)) as stream:
        reader = csv.reader(stream)
        next(reader)
        return [(i, minute(r[1]), tuple(map(number, r[2:4] + r[5:6])), int(r[-1]))
                for i, r in enumerate(reader)]


def apartments(path):
    verify(path, "danish")
    groups = defaultdict(list)
    fields = ("indoor_operative_temperature", "indoor_relative_humidity",
              "indoor_co2_concentration")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            time = minute(datetime.strptime(row["datetime"], "%d-%m-%y %H:%M").isoformat())
            values = tuple(number(row[field]) for field in fields)
            label = number(row["occupancy_ground_truth"])
            groups[(row["apartment_number"], row["room_number"])].append((i, time, values, label))
    return {key: sorted(rows, key=lambda r: r[1]) for key, rows in groups.items()}


def fitting_blocks(rows, budget):
    pool = rows[:6500]
    count = len(pool) * budget // 100
    if budget == 100:
        return [pool]
    blocks = []
    for index in range(10):
        left, right = index * 650, (index + 1) * 650
        size = (index + 1) * count // 10 - index * count // 10
        start = (left + right - size) // 2
        blocks.append(pool[start:start + size])
    return blocks


def valid_history(values, gaps, max_gap):
    return (all(all(map(math.isfinite, reading)) for reading in values)
            and all(0 < gap <= max_gap for gap in gaps))


def occupancy_windows(blocks, max_gap, first_gap):
    result = []
    for block in blocks:
        for index in range(WINDOW - 1, len(block)):
            history = block[index - WINDOW + 1:index + 1]
            row_id, time, _, label = history[-1]
            values = [r[2] for r in history]
            gaps = [b[1] - a[1] for a, b in zip(history, history[1:])]
            if label not in (0, 1) or not valid_history(values, gaps, max_gap):
                continue
            future = None
            if index + 1 < len(block):
                following = block[index + 1]
                if 0 < following[1] - time <= max_gap and all(map(math.isfinite, following[2])):
                    future = following
            result.append(dict(id=row_id, time=time, x=values, gaps=[first_gap, *gaps],
                               y=label, next_id=None if future is None else future[0],
                               next_x=None if future is None else future[2]))
    return result


def heat_rows(path, held_out):
    verify(path, "heat")
    rows = {}
    with ZipFile(path) as archive, TextIOWrapper(archive.open("ds2-detailed-PT1M.csv")) as stream:
        for row in csv.DictReader(stream):
            time = int(minute(row["Time"]))
            if (time >= HOLDOUT) != held_out:
                continue
            if time in rows:
                raise ValueError("Duplicate heating timestamp")
            rows[time] = row
    return rows


def heat_windows(rows):
    result = []
    skipped = defaultdict(int)
    for time in range(min(rows) + (-min(rows) % 15), max(rows) + 1, 15):
        try:
            history = [(number(rows[t]["RTD31"]), number(rows[t]["Tout"]))
                       for t in range(time - WINDOW + 1, time + 1)]
            prior = number(rows[time - 15]["RTD31"])
            commands = {(number(rows[t]["CpSp3"]), number(rows[t]["EvRad"]))
                        for t in range(time, time + 15)}
            future = [rows[t] for t in range(time + 1, time + 16)]
        except KeyError:
            skipped["missing_minute_or_split_boundary"] += 1
            continue
        action = COMMANDS.get(next(iter(commands))) if len(commands) == 1 else None
        if action is None:
            skipped["other_or_changing_command"] += 1
            continue
        temperatures = [number(row["RTD31"]) for row in future]
        powers = [number(row["PhRad_actual_power"]) for row in future]
        if (not valid_history(history, [1] * (WINDOW - 1), 1)
                or not all(map(math.isfinite, [prior, *temperatures, *powers]))
                or min(powers) < 0):
            skipped["invalid_reading"] += 1
            continue
        indoor, outdoor = history[-1]
        bucket = support_bucket(indoor, outdoor, prior)
        result.append(dict(id=time, x=history, gaps=[1.] * WINDOW, action=action, bucket=bucket,
                           y=[temperatures[-1], sum(powers) / 60000,
                              sum(max(20-t, t-24, 0) for t in temperatures) / 60]))
    return result, dict(skipped)


def support_bucket(indoor, outdoor, prior):
    return (math.floor(indoor), math.floor(outdoor / 4), indoor - prior < -0.2)


def normalizer(values):
    raw = torch.tensor(values, dtype=torch.float32)
    return raw.mean(0).tolist(), raw.std(0, correction=0).clamp_min(1e-6).tolist()
