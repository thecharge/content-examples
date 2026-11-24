"""Train only before day 14, then freeze weights/topology for unseen inference.

Run each model folder in a separate process to keep its original imports isolated.
Scoring labels never enter the model during the held-out interval.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def fingerprint(graph):
    result = hashlib.sha256()
    import torch

    for name, tensor in sorted(graph.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(
            json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def run(folder, archives, output, seed):
    sys.path.insert(0, str(folder.resolve()))
    import torch
    from graph import Graph
    from reference import (
        Board,
        FIELDS,
        REFERENCE,
        add_history,
        data,
        score,
        sensor_data,
        sha,
    )
    from runtime import Runtime
    from workflow import load_configuration

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.perf_counter()
    settings, guards, limits = load_configuration(folder / "configuration.json")
    sources = {p.name: sha(p) for p in sorted(folder.glob("*.py"))}
    _, archive_hashes = data(archives)
    reference = json.loads(
        (REFERENCE / f"evidence/seed-{seed}/results.json").read_text()
    )
    if archive_hashes != reference["archive_sha256"]:
        raise ValueError("Original archive fingerprints differ")
    output.mkdir(parents=True, exist_ok=True)
    initial = REFERENCE / f"evidence/seed-{seed}/board.pt"
    base = Board(features=17).eval()
    base.load_state_dict(torch.load(initial, weights_only=True))
    homes = {}
    for home in (5, 6):
        (times, features, truth), _ = sensor_data.occupancy(archives, home, 21)
        warm, _ = sensor_data.occupancy(archives, home, 14)
        features = add_history(times, features)
        cutoff = len(warm[0])
        if not torch.equal(times[:cutoff], warm[0]):
            raise ValueError("Chronology differs")
        runtime = Runtime(
            Graph(base, **limits),
            settings=settings,
            guards=guards,
            seed=seed,
            stage_directory=output / f"H{home}-stages",
        )
        previous = None
        training = []
        for index in range(cutoff):
            stamp = float(times[index])
            if training:
                runtime.pending[training[-1]["prediction"]["token"]]["label_id"] = (
                    f"H{home}:{index - 1}:label"
                )
                runtime.observe(
                    training[-1]["prediction"]["token"], int(truth[index - 1]), stamp
                )
            if previous is None or stamp - previous != 300:
                runtime.reset_state()
            values = {
                n: float(v) if __import__("math").isfinite(float(v)) else None
                for n, v in zip(FIELDS, features[index])
            }
            prediction = runtime.predict(stamp + 0.001, values)
            runtime.pending[prediction["token"]].update(
                observation_id=f"H{home}:{index}",
                context=f"H{home}:" + str(int(features[index, -2] >= 0)),
            )
            training.append(
                dict(stamp=stamp, truth=int(truth[index]), prediction=prediction)
            )
            previous = stamp
            if (index + 1) % 1000 == 0:
                print(
                    folder.name,
                    seed,
                    f"H{home}",
                    "training",
                    index + 1,
                    cutoff,
                    flush=True,
                )
        # Reveal only the final training label, at the next reading's arrival.
        # No label belonging to index >= cutoff is ever passed to this runtime.
        runtime.pending[training[-1]["prediction"]["token"]]["label_id"] = (
            f"H{home}:{cutoff - 1}:label"
        )
        runtime.observe(
            training[-1]["prediction"]["token"],
            int(truth[cutoff - 1]),
            float(times[cutoff]),
        )
        uncompleted_trial = runtime.trial
        monitored = runtime.fallback is not None
        # A shadow is not an adopted model. A still-monitored candidate uses its fallback.
        if monitored:
            runtime.graph, runtime.state = runtime.fallback, runtime.fallback_state
        runtime.shadow, runtime.trial, runtime.fallback = None, None, None
        runtime.contextual = None
        runtime.structural = False
        runtime.stage_ready = False
        runtime.settings = replace(
            runtime.settings, full_weight_learning=False, contextual=False
        )
        for parameter in runtime.graph.parameters():
            parameter.requires_grad_(False)
        layout = json.loads(json.dumps(runtime.graph.layout()))
        weights = fingerprint(runtime.graph)
        runtime.save(output / f"H{home}-trained-frozen.pt")
        issued = []
        for index in range(cutoff, len(times)):
            stamp = float(times[index])
            if stamp - previous != 300:
                runtime.reset_state()
            values = {
                n: float(v) if __import__("math").isfinite(float(v)) else None
                for n, v in zip(FIELDS, features[index])
            }
            prediction = runtime.predict(stamp + 0.001, values)
            runtime.discard(prediction["token"])
            issued.append(
                dict(stamp=stamp, truth=int(truth[index]), prediction=prediction)
            )
            previous = stamp
        if layout != runtime.graph.layout() or weights != fingerprint(runtime.graph):
            raise AssertionError("Inference changed model weights or architecture")
        expected = torch.tensor([r["truth"] for r in issued])
        answer = torch.tensor([r["prediction"]["answer"] for r in issued])
        probability = torch.tensor([r["prediction"]["probability"] for r in issued])
        metrics = score(expected, answer, probability, with_interval=True)
        predictions = output / f"H{home}-unseen-predictions.json"
        predictions.write_text(
            json.dumps(issued, separators=(",", ":"), allow_nan=False) + "\n"
        )
        (output / f"H{home}-training-predictions.json").write_text(
            json.dumps(training, separators=(",", ":"), allow_nan=False) + "\n"
        )
        runtime.save(output / f"H{home}-after-inference.pt")
        homes[f"H{home}"] = dict(
            training_rows=cutoff,
            test_rows=len(issued),
            training_last_stamp=float(times[cutoff - 1]),
            test_first_stamp=float(times[cutoff]),
            metrics=metrics,
            training_receipts=list(runtime.receipts),
            dropped_incomplete_shadow=uncompleted_trial is not None,
            used_monitored_fallback=monitored,
            layout=layout,
            profile=runtime.graph.profile(),
            weights_before=weights,
            weights_after=fingerprint(runtime.graph),
            labels_supplied_during_test=0,
            topology_frozen=True,
        )
        print(folder.name, seed, f"H{home}", "unseen", metrics, flush=True)
    if sources != {p.name: sha(p) for p in sorted(folder.glob("*.py"))}:
        raise ValueError("Runtime source changed during evaluation")
    record = dict(
        kind="strict-unseen-inference-v1",
        seed=seed,
        model=folder.name,
        protocol="Adapt only on first 14 selected days of each room; freeze weights and architecture; infer after day 14 without labels or rewards. Recurrent state may update.",
        script_sha256=sha(Path(__file__)),
        runtime_source_sha256=sources,
        configuration_sha256=sha(folder / "configuration.json"),
        archive_sha256=archive_hashes,
        initial_checkpoint_sha256=sha(initial),
        settings=asdict(settings),
        guards=asdict(guards),
        limits=limits,
        homes=homes,
        environment=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            threads=1,
            device="cpu",
        ),
        elapsed_seconds=time.perf_counter() - started,
    )
    record["artifact_sha256"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "results.json"
    }
    (output / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def verify(folder, output, archives):
    sys.path.insert(0, str(folder.resolve()))
    import torch
    from reference import REFERENCE, score, sha
    from runtime import Runtime

    record = json.loads((output / "results.json").read_text())
    if record["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Evaluation script changed")
    for name, expected in record["runtime_source_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("Model source changed: " + name)
    if sha(folder / "configuration.json") != record["configuration_sha256"]:
        raise ValueError("Configuration changed")
    if (
        sha(REFERENCE / f"evidence/seed-{record['seed']}/board.pt")
        != record["initial_checkpoint_sha256"]
    ):
        raise ValueError("Initial checkpoint changed")
    for name, expected in record["archive_sha256"].items():
        if sha(archives / name) != expected:
            raise ValueError("Archive changed")
    for name, expected in record["artifact_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError("Artifact changed: " + name)
    for home, item in record["homes"].items():
        rows = json.loads((output / f"{home}-unseen-predictions.json").read_text())
        train = json.loads((output / f"{home}-training-predictions.json").read_text())
        if len(rows) != item["test_rows"] or len(train) != item["training_rows"]:
            raise ValueError("Row count differs")
        if not max(r["stamp"] for r in train) < min(r["stamp"] for r in rows):
            raise ValueError("Holdout overlaps training")
        expected = torch.tensor([r["truth"] for r in rows])
        answer = torch.tensor([r["prediction"]["answer"] for r in rows])
        probability = torch.tensor([r["prediction"]["probability"] for r in rows])
        if score(expected, answer, probability, with_interval=True) != item["metrics"]:
            raise ValueError("Score differs")
        before = Runtime.load(output / f"{home}-trained-frozen.pt")
        after = Runtime.load(output / f"{home}-after-inference.pt")
        if (
            not fingerprint(before.graph)
            == item["weights_before"]
            == item["weights_after"]
            == fingerprint(after.graph)
        ):
            raise ValueError("Inference changed weights")
        if (
            before.graph.layout() != after.graph.layout()
            or before.graph.layout() != item["layout"]
        ):
            raise ValueError("Inference changed topology")
        if item["labels_supplied_during_test"] != 0:
            raise ValueError("Feedback entered holdout")
    print("Verified strict unseen inference:", folder.name, record["seed"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(73, 83, 97), default=73)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.folder, args.output, args.archives)
    else:
        run(args.folder, args.archives, args.output, args.seed)
