"""Same HPDmobile archives, checkpoints, inputs and chronological ranges."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time

import torch
import xgboost as xgb

from control import Guards
from graph import Graph
from reference import (
    Board,
    FIELDS,
    REFERENCE,
    REVISION,
    add_history,
    data,
    paired_margin,
    score,
    sensor_data,
    sha,
    source_hashes,
)
from runtime import Runtime, Settings
from contextual import ContextBoard, ContextFeedback

ROOT = Path(__file__).resolve().parent
SOURCES = (
    "reference.py",
    "hierarchy.py",
    "liquid.py",
    "worker.py",
    "worker-configuration.json",
    "test_scaling.py",
    "evaluate.py",
    "policy.py",
    "graph.py",
    "runtime.py",
    "compare.py",
    "verify.py",
    "test_adaptive.py",
    "requirements.txt",
    "control.py",
    "demonstrate.py",
    "guards.json",
    "attention.py",
    "adaptation.py",
    "workflow.py",
    "configuration.json",
    "training-configuration.json",
)


def hashes():
    return {
        **{"adaptive/" + p: sha(ROOT / p) for p in SOURCES},
        **{
            "reference/" + p: sha(REFERENCE / p)
            for p in (*source_hashes(), "online.py", "contextual.py")
        },
    }


def clean(values):
    return {
        name: None if not math.isfinite(float(value)) else float(value)
        for name, value in zip(FIELDS, values)
    }


def summarize(rows, variant):
    expected = torch.tensor([r["truth"] for r in rows])
    answers = torch.tensor([r[variant]["answer"] for r in rows])
    probability = torch.tensor([r[variant]["probability"] for r in rows])
    result = score(expected, answers, probability, with_interval=True)
    if variant != "tree":
        result["quorum"] = sum(r[variant]["quorum"] for r in rows)
    return result


def run(archives, output, seed, guards=None, settings=None, limits=None):
    from workflow import load_configuration

    defaults, default_guards, default_limits = load_configuration()
    guards, settings, limits = (
        guards or default_guards,
        settings or defaults,
        limits or default_limits,
    )
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_sources = hashes()
    streams, archive_hashes = data(archives)
    reference_record = json.loads(
        (REFERENCE / f"evidence/seed-{seed}/results.json").read_text()
    )
    if archive_hashes != reference_record["archive_sha256"]:
        raise ValueError("All 12 archive fingerprints must match the original evidence")
    checkpoint = REFERENCE / f"evidence/seed-{seed}/board.pt"
    base = Board(features=17).eval()
    base.load_state_dict(torch.load(checkpoint, weights_only=True))
    tree = xgb.Booster(params={"nthread": 1})
    tree.load_model(REFERENCE / f"evidence/seed-{seed}/tree.json")
    homes = {}
    for home in (5, 6):
        warm, _ = sensor_data.occupancy(archives, home, 14)
        (times, features, truth), used = sensor_data.occupancy(archives, home, 21)
        if not torch.equal(times[: len(warm[0])], warm[0]) or len(times) <= len(
            warm[0]
        ):
            raise ValueError(
                "Later range must follow exactly the first 14 selected days"
            )
        features = add_history(times, features)
        week = len(streams[home - 1][0])
        later = len(warm[0])
        if not torch.equal(times[:week], streams[home - 1][0]) or not torch.allclose(
            features[:week], streams[home - 1][1], equal_nan=True
        ):
            raise ValueError("First-week input parity failed")
        adaptive = Runtime(
            Graph(base, **limits),
            settings=settings,
            seed=seed,
            guards=guards,
            stage_directory=output / f"H{home}-adaptive-stages",
        )
        control = Runtime(
            Graph(base, **limits),
            settings=settings,
            seed=seed,
            structural=False,
            guards=guards,
            stage_directory=output / f"H{home}-weight-only-stages",
        )
        tree_prob = tree.predict(xgb.DMatrix(features.numpy(), nthread=1))
        original_adaptive = ContextBoard().eval()
        original_adaptive.load_state_dict(base.state_dict(), strict=False)
        original_learner = ContextFeedback(original_adaptive)
        original_state, original_pending = None, None
        issued = []
        previous = None
        native_state = None
        for index, stamp in enumerate(times.tolist()):
            # Reveal the preceding answer at the next call, before making that call.
            # No candidate is fitted or graded with the current window's answer.
            if issued:
                old_stamp, old_x, before, result = original_pending
                original_learner.observe(
                    old_stamp, stamp, old_x, before, result, truth[index - 1 : index]
                )
                for name, runtime in (("adaptive", adaptive), ("weight_only", control)):
                    runtime.observe(
                        issued[-1][name]["token"], int(truth[index - 1]), stamp
                    )
            if previous is None or stamp - previous != 300:
                native_state = None
                original_state = None
                adaptive.reset_state()
                control.reset_state()
            x = features[index : index + 1]
            with torch.no_grad():
                native = base(x, torch.tensor([5.0]), native_state)
            native_state = native.state.detached()
            with torch.no_grad():
                before = (
                    original_adaptive.start(x)
                    if original_state is None
                    else original_state
                )
                original_result = original_adaptive(x, torch.tensor([5.0]), before)
                original_state = original_result.state.detached()
                original_pending = stamp, x, before, original_result
            values = clean(features[index])
            # Runtime event clock puts the prediction after the label reveal.
            predictions = {
                name: runtime.predict(stamp + 0.001, values)
                for name, runtime in (("adaptive", adaptive), ("weight_only", control))
            }
            issued.append(
                dict(
                    stamp=stamp,
                    truth=int(truth[index]),
                    frozen=dict(
                        answer=int(native.answer),
                        probability=float(native.probability),
                        quorum=bool(native.committed),
                    ),
                    tree=dict(
                        answer=int(tree_prob[index] >= 0.5),
                        probability=float(tree_prob[index]),
                    ),
                    contextual=dict(
                        answer=int(original_result.answer),
                        probability=float(original_result.probability),
                        quorum=bool(original_result.committed),
                    ),
                    **predictions,
                )
            )
            previous = stamp
            if (index + 1) % 288 == 0:
                print(
                    f"seed {seed} H{home} {index+1}/{len(times)} active={adaptive.graph.size()} "
                    f'trials={adaptive.trial_number} accepted={sum(r["accepted"] for r in adaptive.receipts)}',
                    flush=True,
                )
        ranges = {}
        original_context = json.loads(
            (REFERENCE / f"evidence/online/context-seed-{seed}.json").read_text()
        )
        for name, rows in (
            ("first_week", issued[:week]),
            ("after_day_14", issued[later:]),
        ):
            ranges[name] = {
                variant: summarize(rows, variant)
                for variant in (
                    "frozen",
                    "contextual",
                    "adaptive",
                    "weight_only",
                    "tree",
                )
            }
            expected = torch.tensor([r["truth"] for r in rows])
            gains = torch.tensor(
                [
                    float(r["adaptive"]["answer"] == r["truth"])
                    - float(r["frozen"]["answer"] == r["truth"])
                    for r in rows
                ]
            )
            lower, upper = paired_margin(gains, expected), paired_margin(
                -gains, expected
            )
            ranges[name]["adaptive_paired_gain_90"] = (
                None if lower is None or upper is None else [lower, -upper]
            )
        # Enforce the published frozen comparisons before accepting new evidence.
        for field in ("accuracy", "balanced_accuracy", "brier"):
            expected = reference_record["replay"][f"H{home}"]["frozen"][field]
            if abs(ranges["first_week"]["frozen"][field] - expected) > 2e-6:
                raise ValueError(f"Original first-week score mismatch: H{home} {field}")
            expected = original_context["homes"][f"H{home}"]["frozen"][field]
            if abs(ranges["after_day_14"]["frozen"][field] - expected) > 2e-6:
                raise ValueError(
                    f"Published later frozen score mismatch: H{home} {field}"
                )
            expected = original_context["homes"][f"H{home}"]["adaptive"][field]
            if abs(ranges["after_day_14"]["contextual"][field] - expected) > 2e-6:
                raise ValueError(
                    f"Published contextual score mismatch: H{home} {field}"
                )
        predictions_path = output / f"H{home}-predictions.json"
        predictions_path.write_text(
            json.dumps(issued, separators=(",", ":"), allow_nan=False) + "\n"
        )
        for name, runtime in (("adaptive", adaptive), ("weight-only", control)):
            runtime.save(output / f"H{home}-{name}.pt")
        homes[f"H{home}"] = dict(
            ranges=ranges,
            first_week_rows=week,
            later_start_index=later,
            total_rows=len(issued),
            evaluation_from=int(times[later]),
            original_contextual_record=original_context["homes"][f"H{home}"],
            original_contextual_record_sha256=sha(
                REFERENCE / f"evidence/online/context-seed-{seed}.json"
            ),
            adaptive_receipts=list(adaptive.receipts),
            weight_only_receipts=list(control.receipts),
            contextual_receipts=list(adaptive.contextual_audit),
            original_contextual_rerun_receipts=original_learner.receipts,
            final_experts=adaptive.graph.size(),
            final_layout=adaptive.graph.layout(),
            incomplete_trial=adaptive.trial,
            final_unrevealed_answers=len(adaptive.pending),
        )
        print(
            f"seed {seed} H{home}: "
            + str(
                {
                    k: {
                        v: round(m[v]["balanced_accuracy"], 4)
                        for v in ("frozen", "contextual", "adaptive", "weight_only")
                    }
                    for k, m in ranges.items()
                }
            ),
            flush=True,
        )
        # Keep completed home evidence even if a later run is interrupted.
        receipt = dict(
            reference_revision=REVISION,
            seed=seed,
            settings=vars(settings),
            limits=limits,
            guards=asdict(guards),
            archive_sha256=archive_hashes,
            initial_checkpoint_sha256=sha(checkpoint),
            code_sha256=hashes(),
            homes=homes,
            environment=dict(
                python=platform.python_version(), torch=torch.__version__, device="cpu"
            ),
            elapsed_seconds=time.monotonic() - started,
        )
        if hashes() != initial_sources:
            raise ValueError("Source changed during the experiment; rerun frozen code")
        receipt["artifact_sha256"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.suffix in (".pt", ".json") and p.name != "results.json"
        }
        (output / "results.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(73, 83, 97), default=73)
    parser.add_argument("--guards", type=Path)
    parser.add_argument(
        "--configuration", type=Path, default=ROOT / "configuration.json"
    )
    args = parser.parse_args()
    from workflow import load_configuration

    settings, guards, limits = load_configuration(args.configuration)
    if args.guards is not None:
        guards = Guards(**json.loads(args.guards.read_text()))
    run(
        args.archives,
        args.output,
        args.seed,
        guards,
        settings,
        limits,
    )
