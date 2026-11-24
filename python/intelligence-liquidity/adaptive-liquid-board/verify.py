"""Recompute scores and gate decisions, check hashes and restore saved graphs."""

import argparse
import json
from pathlib import Path
import tempfile

import torch

from compare import hashes, run, summarize
from control import Guards
from reference import REFERENCE, by_class, paired_margin, sha
from runtime import Runtime, metrics


def check_trial(receipt, complexity):
    if "check_first" not in receipt:
        return
    if not receipt["fit_revealed_through"] < receipt["check_first"]:
        raise ValueError("Future-label leakage")
    rows = receipt["check_predictions"]
    if any(r["revealed_at"] <= r["stamp"] for r in rows):
        raise ValueError("Noncausal check answer")
    if receipt["status"] == "insufficient_check_classes":
        if receipt["accepted"]:
            raise ValueError("Accepted unsupported change")
        return
    active, candidate = metrics(rows, "active"), metrics(rows, "candidate")
    if active != receipt["active"] or candidate != receipt["candidate"]:
        raise ValueError("Trial metrics changed")
    truth = torch.tensor([r["truth"] for r in rows])
    delta = torch.tensor(
        [
            float(r["candidate_answer"] == r["truth"])
            - float(r["active_answer"] == r["truth"])
            for r in rows
        ]
    )
    balanced = float(by_class(delta, truth))
    classes = [float(delta[truth == c].mean()) for c in (0, 1)]
    lower = paired_margin(delta, truth)
    quorum = sum(
        int(r["candidate_quorum"]) - int(r["active_quorum"]) for r in rows
    ) / len(rows)
    brier = active["brier"] - candidate["brier"]
    saving = (receipt["before_parameters"] - receipt["candidate_parameters"]) / receipt[
        "before_parameters"
    ]
    penalized = brier + complexity * saving
    from runtime import Settings

    settings = Settings(**receipt.get("acceptance", {}))
    if receipt["action"] in ("share", "retire", "prune_heads"):
        accepted = saving > 0 and brier >= 0 and min(classes) >= 0 and quorum >= 0
    else:
        accepted = (
            balanced >= settings.minimum_accuracy_gain
            and brier >= settings.minimum_brier_gain
            and lower is not None
            and lower > 0
            and min(classes) >= -settings.maximum_class_loss
            and quorum >= -settings.maximum_quorum_loss
            and penalized > 0
        )
    for key, value in dict(
        balanced_gain=balanced,
        class_gain=classes,
        paired_lower=lower,
        quorum_gain=quorum,
        brier_gain=brier,
        parameter_saving=saving,
        penalized_gain=penalized,
        accepted=accepted,
    ).items():
        if receipt[key] != value:
            raise ValueError("Trial gate differs: " + key)
    preservation = receipt.get("preservation")
    if preservation and not preservation["passed"]:
        raise ValueError("Unpreserved growth was tested")
    if not receipt["budget"]["allowed"]:
        raise ValueError("Over-budget trial was tested")


def verify(evidence, archives=None):
    record = json.loads((evidence / "results.json").read_text())
    if record["code_sha256"] != hashes():
        raise ValueError("Source fingerprint changed")
    for name, expected in record["artifact_sha256"].items():
        if sha(evidence / name) != expected:
            raise ValueError("Artifact changed: " + name)
    for name, expected in record["archive_sha256"].items():
        if archives is not None and sha(archives / name) != expected:
            raise ValueError("Archive changed: " + name)
    base = REFERENCE / f'evidence/seed-{record["seed"]}/board.pt'
    if sha(base) != record["initial_checkpoint_sha256"]:
        raise ValueError("Starting checkpoint changed")
    original = REFERENCE / f'evidence/online/context-seed-{record["seed"]}.json'
    for home, item in record["homes"].items():
        if sha(original) != item["original_contextual_record_sha256"]:
            raise ValueError("Original evidence changed")
        if (
            json.loads(original.read_text())["homes"][home]
            != item["original_contextual_record"]
        ):
            raise ValueError("Original comparison changed")
        rows = json.loads((evidence / f"{home}-predictions.json").read_text())
        if len(rows) != item["total_rows"]:
            raise ValueError("Missing predictions")
        for range_name, subset in (
            ("first_week", rows[: item["first_week_rows"]]),
            ("after_day_14", rows[item["later_start_index"] :]),
        ):
            for variant in ("frozen", "contextual", "adaptive", "weight_only", "tree"):
                if summarize(subset, variant) != item["ranges"][range_name][variant]:
                    raise ValueError("Score mismatch")
            expected = torch.tensor([r["truth"] for r in subset])
            gains = torch.tensor(
                [
                    float(r["adaptive"]["answer"] == r["truth"])
                    - float(r["frozen"]["answer"] == r["truth"])
                    for r in subset
                ]
            )
            lower, upper = paired_margin(gains, expected), paired_margin(
                -gains, expected
            )
            interval = None if lower is None or upper is None else [lower, -upper]
            if interval != item["ranges"][range_name]["adaptive_paired_gain_90"]:
                raise ValueError("Paired interval differs")
        for receipt in item["adaptive_receipts"] + item["weight_only_receipts"]:
            check_trial(receipt, record["settings"]["complexity_rate"])
        restored = Runtime.load(evidence / f"{home}-adaptive.pt")
        if restored.graph.layout() != item["final_layout"]:
            raise ValueError("Checkpoint topology differs")
    print(
        f'Verified seed {record["seed"]}: hashes, predictions, scores, gates, chronology and restored topology',
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--archives", type=Path)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    verify(args.evidence, args.archives)
    if args.rerun:
        if args.archives is None:
            parser.error("--rerun needs --archives")
        with tempfile.TemporaryDirectory() as directory:
            record = json.loads((args.evidence / "results.json").read_text())
            fresh = Path(directory)
            from runtime import Settings

            run(
                args.archives,
                fresh,
                record["seed"],
                Guards(**record["guards"]),
                Settings(**record["settings"]),
                record["limits"],
            )
            for name in ("H5-predictions.json", "H6-predictions.json"):
                if sha(fresh / name) != sha(args.evidence / name):
                    raise ValueError("Replay differs: " + name)
            print("Exact prediction replay matched")
