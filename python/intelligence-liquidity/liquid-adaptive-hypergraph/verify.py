"""Recompute scores and gate decisions, check hashes and restore saved graphs."""

import argparse
from pathlib import Path

import torch

from reference import by_class, paired_margin
from runtime import metrics


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
    old_work = receipt.get("before_profile", {}).get("work_units", 1)
    new_work = receipt.get("candidate_profile", {}).get("work_units", old_work)
    work_saving = (old_work - new_work) / old_work
    from runtime import Settings

    compute = receipt.get("compute_rate", Settings().compute_rate)
    penalized = brier + complexity * saving + compute * work_saving
    from runtime import Settings

    settings = Settings(**receipt.get("acceptance", {}))
    if receipt["action"] in (
        "share",
        "retire",
        "prune_heads",
        "remove_layer",
        "prune_attention",
        "prune_neuron",
        "retire_network",
        "prune_liquid",
        "merge_edge",
        "retire_edge",
    ):
        accepted = (
            max(saving, work_saving) > 0
            and brier >= 0
            and min(classes) >= 0
            and quorum >= 0
        )
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
    from evaluate import verify_evaluation

    return verify_evaluation(evidence, archives)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    verify(args.evidence, args.archives)
