"""Independently reload tree artifacts and join their future rows to liquid evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/xgboost-seed-73"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    data = torch.load(ROOT / "evidence/seed-73/dataset.pt", weights_only=False)
    results = read(OUT / "results.json")
    assert results["script_sha256"] == sha(ROOT / "reporting/xgboost_compare.py")
    assert results["protocol_sha256"] == sha(ROOT / "reporting/xgboost-protocol.json")
    assert results["dataset_sha256"] == sha(ROOT / "evidence/seed-73/dataset.pt")
    comparison = {}
    for variant, report in results["variants"].items():
        assert sha(OUT / f"{variant}.ubj") == report["model_sha256"]
        assert sha(OUT / f"{variant}-predictions.json") == report["predictions_sha256"]
        rows = read(OUT / f"{variant}-training-rows.json")
        assert (
            sha(OUT / f"{variant}-training-rows.json") == report["training_rows_sha256"]
        )
        assert len(rows) == len(set(map(tuple, rows))) == report["training_rows"]
        allowed = {(s["room"], int(t)) for s in data for t in s["times"][s["fit"]]}
        assert set(map(tuple, rows)) <= allowed
        if variant == "all-prefix":
            assert set(map(tuple, rows)) == allowed
        else:
            # Independently reconstruct the complete 36-window sequences.
            expected = set()
            for s in data:
                ts = s["times"][s["fit"]].tolist()
                segments = [[]]
                for t in ts:
                    if segments[-1] and t - segments[-1][-1] != 300:
                        segments.append([])
                    segments[-1].append(t)
                for segment in segments:
                    expected.update(
                        (s["room"], t) for t in segment[: len(segment) // 36 * 36]
                    )
            assert set(map(tuple, rows)) == expected
        model = xgb.Booster()
        model.load_model(OUT / f"{variant}.ubj")
        model.set_param({"nthread": 1})
        predictions = read(OUT / f"{variant}-predictions.json")
        for partition in ("validation", "future"):
            for s in data:
                rows0 = predictions[partition][s["room"]]
                m = s[partition]
                assert [r["stamp"] for r in rows0] == s["times"][m].tolist()
                assert [r["truth"] for r in rows0] == s["y"][m].tolist()
                p = model.predict(xgb.DMatrix(s["x"][m].numpy(), nthread=1))
                assert np.array_equal(p, np.array([r["probability"] for r in rows0]))
                correct = [
                    sum(r["answer"] == c for r in rows0 if r["truth"] == c)
                    for c in (0, 1)
                ]
                counts = [sum(r["truth"] == c for r in rows0) for c in (0, 1)]
                balanced = np.mean([a / b for a, b in zip(correct, counts) if b])
                assert (
                    balanced
                    == report["partitions"][partition]["room_scores"][s["room"]][
                        "balanced_accuracy"
                    ]
                )
        comparison[variant] = {
            s["room"]: report["partitions"]["future"]["room_scores"][s["room"]][
                "balanced_accuracy"
            ]
            for s in data
        }
    for name in ("scratch", "online", "pretrained_frozen", "pretrained-retrained"):
        folder = ROOT / "evidence/acceptance-seed-73" / name
        recorded = read(folder / "predictions.json")
        report = read(folder / "results.json")
        # Liquid prediction receipts use a room mapping, just like the tree receipts.
        for s in data:
            rows = recorded[s["room"]]
            assert [(r["stamp"], r["truth"]) for r in rows] == list(
                zip(s["times"][s["future"]].tolist(), s["y"][s["future"]].tolist())
            )
        comparison[name] = {
            room: score["balanced_accuracy"]
            for room, score in report["room_scores"].items()
        }
    result = dict(
        passed=True,
        exact_future_timestamp_truth_joins=True,
        fitting_rows_unique_and_fit_only=True,
        complete_block_rows_independently_reconstructed=True,
        saved_tree_probabilities_exact=True,
        scores=comparison,
        macro_balanced_accuracy={
            k: sum(v.values()) / len(v) for k, v in comparison.items()
        },
    )
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
