"""Run a fixed CPU tree baseline on the existing six-room temporal split."""

import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(y, p):
    a = p >= 0.5
    classes = [
        dict(rows=int((y == c).sum()), right=int(((y == c) & (a == c)).sum()))
        for c in (0, 1)
    ]
    recalls = [c["right"] / c["rows"] for c in classes if c["rows"]]
    return dict(
        rows=len(y),
        balanced_accuracy=float(np.mean(recalls)),
        accuracy=float(np.mean(a == y)),
        brier=float(np.mean((p - y) ** 2)),
        classes=classes,
        both_classes_present=all(c["rows"] for c in classes),
    )


def run():
    from cold_start import blocks

    torch.set_num_threads(1)
    protocol_path = ROOT / "reporting/xgboost-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    dataset_path = ROOT / "evidence/seed-73/dataset.pt"
    streams = torch.load(dataset_path, weights_only=False)
    output = ROOT / "evidence/xgboost-seed-73"
    output.mkdir(parents=True, exist_ok=True)
    matched = {
        (room, int(t)) for _, _, stamps, room in blocks(streams, 36) for t in stamps
    }
    record = dict(
        protocol_sha256=digest(protocol_path),
        dataset_sha256=digest(dataset_path),
        dataset_metadata_sha256=digest(ROOT / "evidence/seed-73/dataset.json"),
        script_sha256=digest(Path(__file__)),
        versions=dict(
            python=platform.python_version(),
            xgboost=xgb.__version__,
            numpy=np.__version__,
            torch=torch.__version__,
        ),
        variants={},
    )
    for variant in protocol["variants"]:
        masks = [
            s["fit"]
            if variant == "all-prefix"
            else torch.tensor([(s["room"], int(t)) in matched for t in s["times"]])
            for s in streams
        ]
        assert all(
            not bool((mask & ~s["fit"]).any()) for s, mask in zip(streams, masks)
        )
        x = torch.cat([s["x"][m] for s, m in zip(streams, masks)]).numpy()
        y = torch.cat([s["y"][m] for s, m in zip(streams, masks)]).numpy()
        assert x.shape[1] == 17
        counts = np.bincount(y, minlength=2).astype(np.float32)
        weights = counts.mean() / counts[y]
        training_rows = [
            [s["room"], int(t)] for s, m in zip(streams, masks) for t in s["times"][m]
        ]
        dump(output / f"{variant}-training-rows.json", training_rows)
        start = time.perf_counter()
        train = xgb.DMatrix(x, label=y, weight=weights, nthread=1)
        model = xgb.train(
            protocol["params"], train, num_boost_round=protocol["num_boost_round"]
        )
        fit_seconds = time.perf_counter() - start
        model_path = output / f"{variant}.ubj"
        model.save_model(model_path)
        loaded = xgb.Booster()
        loaded.load_model(model_path)
        loaded.set_param({"nthread": 1})
        result = dict(
            training_rows=len(y),
            training_classes=counts.astype(int).tolist(),
            per_room_training_rows={
                s["room"]: int(m.sum()) for s, m in zip(streams, masks)
            },
            training_rows_sha256=digest(output / f"{variant}-training-rows.json"),
            fit_seconds=fit_seconds,
            model_bytes=model_path.stat().st_size,
            model_sha256=digest(model_path),
            partitions={},
        )
        predictions = {}
        for partition in ("validation", "future"):
            scores, rows = {}, {}
            start = time.perf_counter()
            for s in streams:
                m = s[partition]
                data = xgb.DMatrix(s["x"][m].numpy(), nthread=1)
                p = loaded.predict(data)
                assert np.array_equal(p, model.predict(data)), (
                    "Reload changed predictions"
                )
                replicas = loaded.predict(
                    xgb.DMatrix(np.repeat(s["x"][m].numpy(), 6, axis=0), nthread=1)
                ).reshape(-1, 6)
                assert np.array_equal(replicas, np.repeat(p[:, None], 6, axis=1))
                y_test = s["y"][m].numpy()
                scores[s["room"]] = metrics(y_test, p)
                rows[s["room"]] = [
                    dict(
                        stamp=int(t),
                        truth=int(y0),
                        probability=float(p0),
                        answer=int(p0 >= 0.5),
                    )
                    for t, y0, p0 in zip(s["times"][m], y_test, p)
                ]
            result["partitions"][partition] = dict(
                room_scores=scores,
                macro_balanced_accuracy=float(
                    np.mean([s["balanced_accuracy"] for s in scores.values()])
                ),
                scoring_and_reload_replica_check_seconds=time.perf_counter() - start,
            )
            predictions[partition] = rows
        dump(output / f"{variant}-predictions.json", predictions)
        result["predictions_sha256"] = digest(output / f"{variant}-predictions.json")
        record["variants"][variant] = result
    record["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["verification"] = dict(
        fit_only=True, saved_model_reload_exact=True, six_replicas_exact=True
    )
    dump(output / "results.json", record)
    print(
        json.dumps(
            {k: v["partitions"]["future"] for k, v in record["variants"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
