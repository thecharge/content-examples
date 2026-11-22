"""Fit a two-action heating policy from published lab logs, entirely offline."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import TextIOWrapper
import json
import math
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_HASH = "393c5c5d106d1597e0a059ea43b799d7d2e58baf74a6388beb256c46f2400fe8"
CSV_NAME = "ds2-detailed-PT1M.csv"
HOLDOUT_START = datetime(2019, 3, 7, tzinfo=timezone.utc)
STEP_MINUTES = 15
MIN_OBSERVATIONS = 8
DISCOUNT = 0.95
COMFORT_WEIGHT = 2
ACTION = {(0.0, 0.0): "off", (100.0, 100.0): "on"}


def state(row, previous):
    indoor = float(row["RTD31"])
    outdoor = float(row["Tout"])
    trend = indoor - float(previous["RTD31"])
    if not all(map(math.isfinite, (indoor, outdoor, trend))):
        raise ValueError("Missing temperature")
    return (math.floor(indoor), math.floor(outdoor / 4), trend < -0.2)


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Missing reading")
    return number


def load(archive):
    if sha256(archive.read_bytes()).hexdigest() != ARCHIVE_HASH:
        raise ValueError("Dataset archive hash does not match Mendeley version 3")
    with ZipFile(archive) as data, data.open(CSV_NAME) as raw:
        csv_sha256 = sha256(data.read(CSV_NAME)).hexdigest()
        rows = list(csv.DictReader(TextIOWrapper(raw, encoding="utf-8")))
    rows_by_minute = {
        int(datetime.fromisoformat(row["Time"]).timestamp()) // 60: row for row in rows
    }
    if len(rows_by_minute) != len(rows):
        raise ValueError("Duplicate timestamps")
    cutoff = int(HOLDOUT_START.timestamp()) // 60
    train, holdout = [], []
    skipped = Counter()
    start = min(rows_by_minute) + STEP_MINUTES
    start += (-start) % STEP_MINUTES
    for minute in range(start, max(rows_by_minute) - STEP_MINUTES + 1, STEP_MINUTES):
        end = minute + STEP_MINUTES
        if minute - STEP_MINUTES < cutoff <= end:
            skipped["split_boundary"] += 1
            continue
        try:
            window = [rows_by_minute[index] for index in range(minute, end + 1)]
            prior = rows_by_minute[minute - STEP_MINUTES]
        except KeyError:
            skipped["missing_minute"] += 1
            continue
        try:
            commands = {(finite(row["CpSp3"]), finite(row["EvRad"])) for row in window[:-1]}
            action = ACTION.get(next(iter(commands))) if len(commands) == 1 else None
            if action is None:
                skipped["other_or_changing_command"] += 1
                continue
            before, after = state(window[0], prior), state(window[-1], window[0])
            # The readings are minute samples. Approximate exposure and thermal kWh.
            temperature = [finite(row["RTD31"]) for row in window[1:]]
            power = [finite(row["PhRad_actual_power"]) for row in window[1:]]
            if min(power) < 0:
                raise ValueError("Negative thermal power")
        except ValueError:
            skipped["invalid_reading"] += 1
            continue
        energy = sum(power) / 60_000
        discomfort = sum(max(20 - temp, temp - 24, 0) for temp in temperature) / 60
        transition = (before, action, after, energy, discomfort)
        (train if end < cutoff else holdout).append(transition)
    return train, holdout, skipped, csv_sha256


def fit(train):
    observations = defaultdict(list)
    counts = defaultdict(Counter)
    for before, action, after, energy, discomfort in train:
        observations[before, action].append((after, -energy - COMFORT_WEIGHT * discomfort))
        counts[before][action] += 1

    # Abstain unless both commands have repeat observations in that training state.
    options = {
        before: ["off", "on"] for before, actions in counts.items()
        if min(actions["off"], actions["on"]) >= MIN_OBSERVATIONS
    }
    values = dict.fromkeys(options, 0.0)
    for _ in range(200):
        scores = {}
        for before, actions in options.items():
            scores[before] = {}
            for action in actions:
                trials = observations[before, action]
                scores[before][action] = sum(
                    reward + DISCOUNT * values.get(after, 0) for after, reward in trials
                ) / len(trials)
        values = {before: max(actions.values()) for before, actions in scores.items()}
    policy = {before: max(actions, key=actions.__getitem__) for before, actions in scores.items()}
    behavior = {before: counts[before].most_common(1)[0][0] for before in options}
    return policy, behavior


def measure(policy, holdout):
    commands = Counter(policy.get(before, "abstain") for before, *_ in holdout)
    matched = [row for row in holdout if policy.get(row[0]) == row[1]]
    return {
        "recommendations": commands,
        "exact_logged_action_matches": len(matched),
        "matched_logged_thermal_kwh": round(sum(row[3] for row in matched), 6),
        "matched_logged_degree_hours": round(sum(row[4] for row in matched), 6),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("--expect", type=Path)
    args = cli.parse_args()
    train, holdout, skipped, csv_hash = load(args.archive)
    if not train or not holdout:
        raise ValueError("Missing training or held-out transitions")
    policy, behavior = fit(train)
    receipt = {
        "archive_sha256": ARCHIVE_HASH,
        "csv_sha256": csv_hash,
        "holdout_starts_utc": HOLDOUT_START.isoformat(),
        "training_transitions": len(train),
        "held_out_transitions": len(holdout),
        "training_actions": Counter(row[1] for row in train),
        "held_out_logged_actions": Counter(row[1] for row in holdout),
        "nonzero_comfort_penalty": {
            "training": sum(row[4] > 0 for row in train),
            "held_out": sum(row[4] > 0 for row in holdout),
        },
        "held_out_behavior_thermal_kwh": round(sum(row[3] for row in holdout), 6),
        "held_out_behavior_degree_hours": round(sum(row[4] for row in holdout), 6),
        "skipped": skipped,
        "dual_action_training_states": len(policy),
        "policy_table": [
            {"state": before, "learned": policy[before], "training_mode": behavior[before]}
            for before in sorted(policy)
        ],
        "reward_policy": measure(policy, holdout),
        "behavior_baseline": measure(behavior, holdout),
    }
    if args.expect and json.loads(json.dumps(receipt)) != json.loads(args.expect.read_text()):
        raise ValueError("Results differ from the recorded receipt")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
