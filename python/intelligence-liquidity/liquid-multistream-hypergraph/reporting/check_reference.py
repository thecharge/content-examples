"""Serve each original trained H5/H6 checkpoint through 36 isolated stream contexts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from batching import fingerprint
from cold_start import dump
from compare import clean
from multistream import MultiRuntime
from reference import score, sha
from runtime import Runtime, Settings


def run(args):
    torch.set_num_threads(1)
    streams = torch.load(args.dataset, weights_only=True)
    baseline = (
        Path(__file__).resolve().parents[2]
        / "liquid-adaptive-hypergraph"
        / f"evidence/strict/new-seed-{args.seed}"
    )
    record = json.loads((baseline / "results.json").read_text())
    reports = {}
    for room in ("H5", "H6"):
        source = baseline / f"{room}-trained-frozen.pt"
        if sha(source) != record["artifact_sha256"][source.name]:
            raise AssertionError("Checkpoint changed")
        trained = Runtime.load(source)
        model = trained.graph
        initial = fingerprint(model)
        runtime = MultiRuntime(
            model, settings=Settings(contextual=False, full_weight_learning=False)
        )
        for i in range(36):
            runtime.open(str(i), f"context-{i // 6}")
            slot = runtime.streams[str(i)]
            slot["state"] = deepcopy(trained.state)
            slot["last_prediction"] = trained.last_prediction
        old = json.loads((baseline / f"{room}-unseen-predictions.json").read_text())
        stream = next(s for s in streams if s["room"] == room)
        indices = stream["future"].nonzero().flatten().tolist()
        if len(indices) != len(old):
            raise AssertionError("Prior future row count differs")
        maximum = 0.0
        answers = []
        probabilities = []
        for index, reference in zip(indices, old, strict=True):
            stamp = int(stream["times"][index])
            if (
                stamp != reference["stamp"]
                or int(stream["y"][index]) != reference["truth"]
            ):
                raise AssertionError("Prior future chronology differs")
            outputs = []
            for left in range(0, 36, 8):
                requests = [
                    dict(
                        stream_id=str(i),
                        room=f"context-{i // 6}",
                        stamp=stamp,
                        values=clean(stream["x"][index]),
                        evidence_id=f"{room}:{stamp}",
                    )
                    for i in range(left, min(left + 8, 36))
                ]
                batch = runtime.predict_batch(requests)
                outputs.extend(batch)
                for output in batch:
                    runtime.discard(output["token"])
            for output in outputs:
                if (
                    output["answer"] != reference["prediction"]["answer"]
                    or output["quorum"] != reference["prediction"]["quorum"]
                ):
                    raise AssertionError("Original and 36-stream decisions differ")
                maximum = max(
                    maximum,
                    abs(output["probability"] - reference["prediction"]["probability"]),
                )
            answers.append(outputs[0]["answer"])
            probabilities.append(outputs[0]["probability"])
        if maximum > 1e-6 or fingerprint(model) != initial:
            raise AssertionError("Frozen reference fidelity failed")
        reports[room] = dict(
            logical_streams=36,
            rows_per_stream=len(indices),
            logical_predictions=36 * len(indices),
            original_checkpoint_sha256=sha(source),
            exact_answers_and_quorum=True,
            maximum_probability_difference=maximum,
            metrics=score(
                stream["y"][stream["future"]],
                torch.tensor(answers),
                torch.tensor(probabilities),
            ),
        )
        print(
            json.dumps(
                dict(room=room, passed=True, maximum_probability_difference=maximum)
            ),
            flush=True,
        )
    dump(
        args.output,
        dict(
            seed=args.seed,
            homes=reports,
            scope="Each pre-existing per-room trained model is separately served through 36 copies of its private historical context. This tests execution fidelity, not shared six-room training or independent data.",
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--seed", type=int, default=73)
    run(cli.parse_args())
