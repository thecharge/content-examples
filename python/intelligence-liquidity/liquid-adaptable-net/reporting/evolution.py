"""Capture and render observed architecture evolution without changing model code."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = {
    60: "add_layer",
    62: "split_liquid",
    63: "split",
    65: "branch_attention",
    68: "deepen",
    88: "retire",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def weights_digest(graph):
    result = hashlib.sha256()
    for name, tensor in sorted(graph.state_dict().items()):
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(str(tensor.dtype).encode())
        result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def graph_record(graph):
    return canonical(
        dict(
            layout=deepcopy(graph.layout()),
            profile=graph.profile(),
            weights_sha256=weights_digest(graph),
            gates={
                name: dict(
                    layers=core.layer_gain.detach().tolist(),
                    child_routes=core.child_gain.detach().tanh().tolist(),
                    attention={
                        path + ":" + key: float(node.gains[key].detach().tanh())
                        for path, node in core.delta.nodes()
                        for key in node.gains
                    },
                )
                for name, core in graph.cores.items()
            },
        )
    )


def capture(seed, output):
    # Install the same process limits before importing PyTorch. Capture is a
    # read-only observer around the unchanged training function, not a new model.
    from worker import impose

    impose(json.loads((ROOT / "worker-configuration.json").read_text()))
    import torch
    import runtime as implementation
    from control import Guards
    from graph import Graph
    from reference import Board, FIELDS, REFERENCE
    from runtime import Runtime, Settings

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    board = Board(features=17)
    checkpoint = REFERENCE / "evidence/seed-73/board.pt"
    board.load_state_dict(torch.load(checkpoint, weights_only=True))
    original_train = implementation.train_candidate
    phases, samples, predictions = [], [], []
    sources = {p.name: digest(p) for p in ROOT.glob("*.py")}

    with tempfile.TemporaryDirectory(prefix="liquid-evolution-") as stages:
        model = Runtime(
            Graph(board),
            seed=seed,
            settings=Settings(
                window=48,
                history=96,
                minority=16,
                exploration=1.0,
                fit_rate=0.01,
                fit_passes=8,
            ),
            guards=Guards(minimum_life=48),
            stage_directory=stages,
        )

        def phase(name, candidate=None):
            item = dict(
                phase=name,
                revealed_labels=model.serial,
                active=graph_record(model.graph),
            )
            if candidate is not None:
                item["candidate"] = graph_record(candidate)
            if model.fallback is not None:
                item["predecessor"] = graph_record(model.fallback)
            phases.append(item)

        def observe_training(candidate, rows, settings, **kwargs):
            phase("before_fit", candidate)
            report = original_train(candidate, rows, settings, **kwargs)
            phase("after_fit", candidate)
            return report

        phase("initial")
        implementation.train_candidate = observe_training
        try:
            for index in range(145):
                truth = (index // 3) % 2
                values = dict.fromkeys(FIELDS, 0.0)
                values["new:cue"] = 2.0 * truth - 1
                issued = model.predict(300.0 * (index + 1), values)
                before_receipts, before_monitor = len(model.receipts), len(
                    model.monitor_audit
                )
                model.observe(issued["token"], truth, 300.0 * (index + 1) + 1)
                predictions.append(dict(stamp=300 * (index + 1), truth=truth, **issued))
                if len(model.receipts) > before_receipts:
                    phase("adopted" if model.receipts[-1]["accepted"] else "rejected")
                if len(model.monitor_audit) > before_monitor:
                    phase(
                        "rolled_back"
                        if model.monitor_audit[-1]["rollback"]
                        else "retained"
                    )
                if (index + 1) % 12 == 0 or index == 144:
                    samples.append(
                        dict(
                            revealed_labels=index + 1,
                            active=graph_record(model.graph),
                            candidate=(
                                graph_record(model.shadow)
                                if model.shadow is not None
                                else None
                            ),
                        )
                    )
        finally:
            implementation.train_candidate = original_train
        phase("end")
        if len(model.receipts) != 1 or model.receipts[0]["action"] != SEEDS[seed]:
            raise ValueError("Illustration no longer selects the recorded mechanism")
        if not model.receipts[0]["accepted"] or model.fallback is not None:
            raise ValueError("Illustration did not complete adoption and monitoring")
        if len(model.monitor_audit) != 1 or model.monitor_audit[0]["rollback"]:
            raise ValueError("Illustration did not retain its change")

        previous = ROOT / f"evidence/controlled-{seed}.json"
        if previous.exists():
            old = json.loads(previous.read_text())
            if old["predictions"] != canonical(predictions) or old[
                "receipts"
            ] != canonical(list(model.receipts)):
                raise ValueError("Observation changed the published run")
            old_model = Runtime.load(previous.with_suffix(".pt"))
            if weights_digest(old_model.graph) != weights_digest(model.graph):
                raise ValueError("Observation changed final weights")
            model_path = previous.with_suffix(".pt")
            parity = "Exact predictions, trial receipt and final weight/buffer tensors match the published 145-observation run."
        else:
            model_path = output / f"model-{seed}.pt"
            model.save(model_path)
            parity = "Additional illustrative seed, using the identical controlled stream and settings. Not a house-data result."

        record = dict(
            schema="liquid-evolution-v1",
            seed=seed,
            action=SEEDS[seed],
            source_sha256=sources,
            observer_sha256=digest(__file__),
            original_checkpoint_sha256=digest(checkpoint),
            model=str(model_path.relative_to(ROOT)),
            model_sha256=digest(model_path),
            published_receipt=(
                str(previous.relative_to(ROOT)) if previous.exists() else None
            ),
            parity=parity,
            settings=vars(model.settings),
            guards=vars(model.controller.guards),
            definition="Original inputs zero; truth=(index//3)%2; new:cue=2*truth-1. Prediction precedes its answer by one second. No action forced.",
            phases=phases,
            samples=samples,
            predictions=predictions,
            receipts=list(model.receipts),
            monitor_audit=list(model.monitor_audit),
        )
        if sources != {p.name: digest(p) for p in ROOT.glob("*.py")}:
            raise ValueError("Model source changed during capture")
        path = output / f"trace-{seed}.json"
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print(
            seed,
            SEEDS[seed],
            [(p["phase"], p["revealed_labels"]) for p in phases],
            flush=True,
        )


def validate_trace(record):
    """Reject unsupported chronology, identity drift or unverified model exports."""
    if (
        record["schema"] != "liquid-evolution-v1"
        or record["action"] != SEEDS[record["seed"]]
    ):
        raise ValueError("Unknown trace")
    for name, expected in record["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Changed model source: " + name)
    if digest(__file__) != record["observer_sha256"]:
        raise ValueError("Changed trace observer")
    if digest(ROOT / record["model"]) != record["model_sha256"]:
        raise ValueError("Changed model artifact")
    names = [p["phase"] for p in record["phases"]]
    if names != ["initial", "before_fit", "after_fit", "adopted", "retained", "end"]:
        raise ValueError("Incomplete or reordered lifecycle")
    labels = [p["revealed_labels"] for p in record["phases"]]
    if labels != [0, 48, 48, 96, 144, 145]:
        raise ValueError("Unexpected causal boundary")
    if len(record["predictions"]) != 145 or any(
        p["stamp"] != 300 * (i + 1) or p["truth"] != (i // 3) % 2
        for i, p in enumerate(record["predictions"])
    ):
        raise ValueError("Altered controlled stream")
    from verify import check_trial

    for receipt in record["receipts"]:
        check_trial(receipt, record["settings"]["complexity_rate"])
    audit = record["monitor_audit"]
    if len(audit) != 1 or audit[0]["rollback"] or not audit[0]["supported"]:
        raise ValueError("Unretained change")
    from runtime import Runtime

    model = Runtime.load(ROOT / record["model"])
    final = record["phases"][-1]["active"]
    if graph_record(model.graph) != final:
        raise ValueError("Export differs from final checkpoint")
    if canonical(list(model.monitor_audit)) != audit:
        raise ValueError("Monitoring audit differs")
    from runtime import metrics

    monitor_rows = audit[0]["predictions"]
    if (
        canonical(metrics(monitor_rows, "active")) != audit[0]["active"]
        or canonical(metrics(monitor_rows, "candidate")) != audit[0]["candidate"]
    ):
        raise ValueError("Monitoring metrics differ")
    prior = record.get("published_receipt")
    if prior:
        previous = json.loads((ROOT / prior).read_text())
        if (
            record["predictions"] != previous["predictions"]
            or record["receipts"] != previous["receipts"]
        ):
            raise ValueError("Published parity differs")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--seed", type=int, choices=SEEDS, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "evolution")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        capture(args.seed, args.output)
    else:
        for seed in SEEDS:
            validate_trace(json.loads((args.output / f"trace-{seed}.json").read_text()))
        print(
            "Verified six lifecycle traces, causal boundaries, model exports, monitoring and published parity"
        )
