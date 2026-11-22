"""Evaluate frozen decisions against independently labeled rooms in Graz or Munich."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import torch

from experiment import (
    SEEDS, alternate, number, predictions, probability, reading, restore, revise,
    rows, score, sha, source, timestamp, windows,
)


def graz(archive):
    result = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in ("Room_A", "Room_B"):
            folder = f"data/intermi/{name}"
            counts = {}
            for row in rows(bundle.read(f"{folder}/df_occ.csv")):
                time = timestamp(row["Date"])
                label = number(row["Image-based-occupancy"])
                if time in counts and counts[time] != label:
                    raise ValueError(f"Conflicting Graz labels in {name}")
                counts[time] = label
            records = [
                reading(row["Date"], row["co2"], row["temperature"], counts.get(timestamp(row["Date"])))
                for row in rows(bundle.read(f"{folder}/df_features.csv"))
            ]
            result[name] = windows(records)
    return result


def munich(archive):
    """Read office days in publisher order, keeping gaps between days."""
    result = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in ("A", "B"):
            filename = next(f for f in bundle.namelist() if f.endswith(f"/office_{name}.csv"))
            records = []
            for row in rows(bundle.read(filename)):
                day = int(row["DAY"])
                hour, minute, second = map(int, row["TIME"].split(":"))
                if minute % 5 or second:
                    continue
                elapsed = 3600 * hour + 60 * minute
                # DAY indexes recorded days, which are often weeks apart.
                # Put a gap between day indices to prevent a window spanning them.
                time = day * 172800 + elapsed
                count = number(row["OCCUPANCY"])
                y = float(count > 0) if count >= 0 and count < float("inf") else float("nan")
                records.append((time, number(row["CO2"]), number(row["Tin"]), y))
            result["Office_" + name] = windows(records)
    return result


def stack_stats(results, key):
    values = [item[key] for item in results]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def stats(prob, ds, cutoff, final=None):
    y = ds[2]
    decisions = None if final is None else final >= cutoff
    return score(prob, y, cutoff, decisions)


def evaluate(data, out, expected, site):
    manifest, models = restore(out, expected)
    if site == "graz":
        archive = source(data, "test")
        datasets = graz(archive)
    else:
        archive = (data / "munich.zip").read_bytes()
        if sha(archive) != "22c02aa064bc4c67086717671a48cade26c0e04576ba8adbac5145b49f072d35":
            raise ValueError("Munich archive checksum mismatch")
        datasets = munich(archive)
    report = {
        "frozen_sha256": expected, "site": site,
        "test_sha256": sha(archive), "evaluator_sha256": sha(Path(__file__).read_bytes()),
        "run_scores": [], "summary": {},
    }
    for model_id, model in models.items():
        entry = manifest["models"][model_id]
        cutoff = entry["cutoff"]
        arm, seed = entry["arm"], entry["seed"]
        base_parts, review_parts, all_routes, attempts = [], [], [], []
        changes, corrections, errors = 0, 0, 0
        rooms, rooms_review = {}, {}
        for name, ds in datasets.items():
            logit, route, experts = predictions(model, ds)
            base = probability(logit, entry["calibration"])
            base_parts.append(base)
            rooms[name] = stats(base, ds, cutoff)
            if route is not None:
                all_routes.append(route)
            if arm == "delta":
                policy = entry["policy"]
                final = base
                if policy:
                    verifier = probability(predictions(models[f"logistic-{seed}"], ds)[0],
                                           manifest["models"][f"logistic-{seed}"]["calibration"])
                    proposal = probability(alternate(experts), entry["alternate_calibration"])
                    final, tried, changed = revise(
                        base, proposal, verifier, cutoff, policy["margin"], policy["gain_limit"])
                    attempts.append(tried)
                    corrected = int((changed & ((final >= cutoff) == ds[2].bool())).sum())
                    corrections += corrected
                    changes += int(changed.sum())
                    errors += int(changed.sum()) - corrected
                review_parts.append(final)
                rooms_review[name] = stats(final, ds, cutoff, final)
        labels = torch.cat([ds[2] for ds in datasets.values()])
        primary = torch.cat(base_parts)
        record = {"arm": arm, "seed": seed, "cutoff": cutoff, "rooms": rooms,
                  "pooled": score(primary, labels, cutoff)}
        if all_routes:
            route = torch.cat(all_routes)
            record["expert_weight_share"] = route.mean((0, 1)).tolist()
            record["expert_top_share"] = (
                torch.bincount(route.argmax(-1).flatten(), minlength=3) / route.shape[:2].numel()
            ).tolist()
        report["run_scores"].append(record)
        if arm == "delta":
            final = torch.cat(review_parts)
            report["run_scores"].append({
                "arm": "delta_review", "seed": seed, "cutoff": cutoff, "rooms": rooms_review,
                "pooled": score(final, labels, cutoff), "attempts": sum(int(a.sum()) for a in attempts),
                "changes": changes, "corrected": corrections, "broken": errors,
                "guard_enabled": entry["policy"] is not None,
            })
    for arm in dict.fromkeys(r["arm"] for r in report["run_scores"]):
        group = [r for r in report["run_scores"] if r["arm"] == arm]
        if sorted(r["seed"] for r in group) != list(SEEDS):
            raise ValueError(f"Missing or duplicate seed for {arm}")
        pooled = [r["pooled"] for r in group]
        stats_keys = ("accuracy", "balanced_accuracy", "auroc", "brier", "calibration_error",
                      "occupied_recall", "empty_recall")
        report["summary"][arm] = {
            **{key: stack_stats(pooled, key) for key in stats_keys},
            "per_room_balanced_accuracy": {
                room: stack_stats([r["rooms"][room] for r in group], "balanced_accuracy")
                for room in datasets
            },
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("frozen_sha256")
    parser.add_argument("--site", choices=("graz", "munich"), default="munich")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(evaluate(args.data, args.out, args.frozen_sha256, args.site), indent=2, allow_nan=False))
