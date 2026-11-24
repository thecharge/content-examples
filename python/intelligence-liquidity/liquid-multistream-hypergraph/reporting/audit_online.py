"""Recompute shared room vetoes and causal availability from saved online receipts."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multistream import MultiRuntime
from runtime import Settings, metrics
from verify import check_trial


def audit(folder):
    result = json.loads((folder / "results.json").read_text())
    runtime = MultiRuntime.load(folder / "final.pt")
    settings = Settings(**result["settings"])
    graded = accepted = 0
    for receipt in result["receipts"]:
        if "check_predictions" not in receipt:
            assert not receipt.get("accepted"), "Unsupported accepted receipt"
            continue
        rows = receipt["check_predictions"]
        assert receipt["fit_revealed_through"] < min(r["stamp"] for r in rows)
        assert all(r["revealed_at"] > r["stamp"] for r in rows)
        rooms = receipt["per_room"]
        assert rooms, "Missing room protections"
        reports = {}
        for room in rooms:
            local = [r for r in rows if r["room"] == room]
            counts = Counter(r["truth"] for r in local)
            supported = min(counts.get(c, 0) for c in (0, 1)) >= settings.minority
            active = metrics(local, "active") if supported else None
            candidate = metrics(local, "candidate") if supported else None
            quorum = sum(
                int(r["candidate_quorum"]) - int(r["active_quorum"]) for r in local
            ) / max(1, len(local))
            passed = bool(
                supported
                and candidate["brier"] <= active["brier"]
                and all(b >= a for a, b in zip(active["recall"], candidate["recall"]))
                and quorum >= 0
            )
            reports[room] = dict(
                supported=supported,
                passed=passed,
                active=active,
                candidate=candidate,
                quorum_gain=quorum,
            )
        assert reports == rooms, "Stored room gate does not reproduce"
        room_safe = all(r["passed"] for r in reports.values())
        if "balanced_gain" in receipt:
            copy = deepcopy(receipt)
            shrink = receipt["action"] in (
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
            )
            if shrink:
                aggregate = (
                    max(receipt["parameter_saving"], receipt["work_saving"]) > 0
                    and receipt["brier_gain"] >= 0
                    and min(receipt["class_gain"]) >= 0
                    and receipt["quorum_gain"] >= 0
                )
            else:
                lower = receipt["paired_lower"]
                aggregate = (
                    receipt["balanced_gain"] >= settings.minimum_accuracy_gain
                    and receipt["brier_gain"] >= settings.minimum_brier_gain
                    and lower is not None
                    and lower > 0
                    and min(receipt["class_gain"]) >= -settings.maximum_class_loss
                    and receipt["quorum_gain"] >= -settings.maximum_quorum_loss
                    and receipt["penalized_gain"] > 0
                )
            copy["accepted"] = aggregate
            check_trial(copy, settings.complexity_rate)
            assert receipt["accepted"] == (aggregate and room_safe)
        else:
            assert not receipt["accepted"]
        graded += 1
        accepted += int(receipt["accepted"])
    calibration = Counter()
    for room in runtime.rooms.values():
        learner = room["learner"]
        if learner is None:
            continue
        for receipt in learner.receipts:
            calibration[receipt["status"]] += 1
            if "fit_revealed_through" in receipt:
                assert (
                    receipt["fit_revealed_through"] < receipt["check_prediction_first"]
                )
            if receipt.get("accepted"):
                gain = receipt["board_gain"]
                assert gain["accuracy_gain"] >= 0.01 and gain["brier_gain"] >= 0.002
                assert gain["accuracy_lower"] > 0 and min(gain["class_gain"]) >= 0
                assert gain["quorum_gain"] >= 0
    assert len(runtime.streams) == 36 and not runtime.pending
    return dict(
        shared_graded_trials=graded,
        shared_accepted_trials=accepted,
        causal_room_calibration=dict(calibration),
        streams=36,
        pending_labels=0,
        independently_recomputed_room_gates=True,
    )


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    record = audit(folder)
    (folder / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
