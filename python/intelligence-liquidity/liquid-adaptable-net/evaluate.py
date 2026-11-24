"""Compare the new CPU graph with fingerprinted, already recorded baselines."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch

from compare import clean, hashes, summarize
from graph import Graph
from reference import Board, FIELDS, REFERENCE, add_history, data, sensor_data, sha
from runtime import Runtime
from workflow import load_configuration

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "adaptive-liquid-board"
VARIANTS = ("frozen", "contextual", "previous", "adaptive", "tree")


def previous_record(seed):
    folder = PREVIOUS / f"evidence/seed-{seed}"
    record = json.loads((folder / "results.json").read_text())
    for name, expected in record["code_sha256"].items():
        prefix, relative = name.split("/", 1)
        owner = PREVIOUS if prefix == "adaptive" else REFERENCE
        if sha(owner / relative) != expected:
            raise ValueError("Previous baseline source changed: " + name)
    for name, expected in record["artifact_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("Previous baseline artifact changed: " + name)
    return folder, record


def evaluate(archives, output, seed, configuration=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    settings, guards, limits = load_configuration(configuration)
    initial_sources = hashes()
    output.mkdir(parents=True, exist_ok=True)
    baseline, old_record = previous_record(seed)
    streams, archive_hashes = data(archives)
    if archive_hashes != old_record["archive_sha256"]:
        raise ValueError("All twelve original archives must match")
    checkpoint = REFERENCE / f"evidence/seed-{seed}/board.pt"
    base = Board(features=17).eval()
    base.load_state_dict(torch.load(checkpoint, weights_only=True))
    homes = {}
    for home in (5, 6):
        home_started = time.perf_counter()
        old_rows = json.loads((baseline / f"H{home}-predictions.json").read_text())
        (times, features, truth), _ = sensor_data.occupancy(archives, home, 21)
        warm, _ = sensor_data.occupancy(archives, home, 14)
        features = add_history(times, features)
        week, later = len(streams[home - 1][0]), len(warm[0])
        if (
            len(old_rows) != len(times)
            or not torch.equal(times[:week], streams[home - 1][0])
            or not torch.allclose(features[:week], streams[home - 1][1], equal_nan=True)
        ):
            raise ValueError("Input or range parity failed")
        if not torch.equal(times[:later], warm[0]):
            raise ValueError("Fourteen-day warmup differs")
        runtime = Runtime(
            Graph(base, **limits),
            settings=settings,
            seed=seed,
            guards=guards,
            stage_directory=output / f"H{home}-stages",
        )
        issued = []
        predict_seconds, feedback_seconds = [], []
        previous = None
        for index, stamp in enumerate(times.tolist()):
            old = old_rows[index]
            if old["stamp"] != stamp or old["truth"] != int(truth[index]):
                raise ValueError("Recorded baseline is not the same stream")
            if issued:
                tick = time.perf_counter()
                runtime.observe(
                    issued[-1]["adaptive"]["token"], int(truth[index - 1]), stamp
                )
                feedback_seconds.append(time.perf_counter() - tick)
            if previous is None or stamp - previous != 300:
                runtime.reset_state()
            tick = time.perf_counter()
            prediction = runtime.predict(stamp + 0.001, clean(features[index]))
            predict_seconds.append(time.perf_counter() - tick)
            issued.append(
                dict(
                    stamp=stamp,
                    truth=int(truth[index]),
                    frozen=old["frozen"],
                    contextual=old["contextual"],
                    previous=old["adaptive"],
                    tree=old["tree"],
                    adaptive=prediction,
                )
            )
            previous = stamp
            if (index + 1) % 288 == 0:
                print(
                    f"seed {seed} H{home} {index + 1}/{len(times)} trials={runtime.trial_number} profile={runtime.graph.profile()}",
                    flush=True,
                )
        predictions = output / f"H{home}-predictions.json"
        predictions.write_text(
            json.dumps(issued, separators=(",", ":"), allow_nan=False) + "\n"
        )
        runtime.save(output / f"H{home}-adaptive.pt")
        ranges = {}
        for name, rows in (
            ("first_week", issued[:week]),
            ("after_day_14", issued[later:]),
        ):
            ranges[name] = {variant: summarize(rows, variant) for variant in VARIANTS}
        homes[f"H{home}"] = dict(
            first_week_rows=week,
            later_start_index=later,
            total_rows=len(issued),
            ranges=ranges,
            adaptive_receipts=list(runtime.receipts),
            contextual_receipts=list(runtime.contextual_audit),
            monitor_audit=list(runtime.monitor_audit),
            final_layout=runtime.graph.layout(),
            final_profile=runtime.graph.profile(),
            incomplete_trial=runtime.trial,
            monitoring=runtime.fallback is not None,
            last_answer_pending=len(runtime.pending),
            elapsed_seconds=time.perf_counter() - home_started,
            predict_seconds=predict_seconds,
            feedback_seconds=feedback_seconds,
        )
        record = dict(
            kind="liquid-adaptable-comparison-v1",
            seed=seed,
            settings=asdict(settings),
            guards=asdict(guards),
            limits=limits,
            archive_sha256=archive_hashes,
            initial_checkpoint_sha256=sha(checkpoint),
            previous_record_sha256=sha(baseline / "results.json"),
            previous_predictions_sha256={
                f"H{h}": sha(baseline / f"H{h}-predictions.json") for h in (5, 6)
            },
            baseline_method="Reuse published predictions after checking all baseline source and artifact hashes; do not rerun baseline training.",
            code_sha256=initial_sources,
            homes=homes,
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                threads=torch.get_num_threads(),
                device="cpu",
            ),
            elapsed_seconds=time.perf_counter() - started,
        )
        if hashes() != initial_sources:
            raise ValueError("Experiment source changed during the run")
        record["artifact_sha256"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.suffix in (".json", ".pt") and p.name != "results.json"
        }
        (output / "results.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        print(f'seed {seed} H{home} completed: {ranges["after_day_14"]}', flush=True)
    return record


def verify_evaluation(folder, archives=None):
    from verify import check_trial

    record = json.loads((folder / "results.json").read_text())
    if record["code_sha256"] != hashes():
        raise ValueError("New sources changed")
    baseline, old = previous_record(record["seed"])
    if sha(baseline / "results.json") != record["previous_record_sha256"]:
        raise ValueError("Baseline receipt changed")
    if record["archive_sha256"] != old["archive_sha256"]:
        raise ValueError("Archive protocol differs")
    if archives is not None:
        for name, expected in record["archive_sha256"].items():
            if sha(archives / name) != expected:
                raise ValueError("Archive changed")
    for name, expected in record["artifact_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError("Artifact changed: " + name)
    checkpoint = REFERENCE / f'evidence/seed-{record["seed"]}/board.pt'
    if sha(checkpoint) != record["initial_checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed")
    for home, item in record["homes"].items():
        rows = json.loads((folder / f"{home}-predictions.json").read_text())
        prior = json.loads((baseline / f"{home}-predictions.json").read_text())
        if (
            sha(baseline / f"{home}-predictions.json")
            != record["previous_predictions_sha256"][home]
        ):
            raise ValueError("Prior predictions changed")
        if len(rows) != item["total_rows"] or len(rows) != len(prior):
            raise ValueError("Missing rows")
        for row, previous in zip(rows, prior):
            for key in ("stamp", "truth", "frozen", "contextual", "tree"):
                if row[key] != previous[key]:
                    raise ValueError("Comparator altered")
            if row["previous"] != previous["adaptive"]:
                raise ValueError("Prior adaptive predictions altered")
        for range_name, subset in (
            ("first_week", rows[: item["first_week_rows"]]),
            ("after_day_14", rows[item["later_start_index"] :]),
        ):
            for variant in VARIANTS:
                if summarize(subset, variant) != item["ranges"][range_name][variant]:
                    raise ValueError("Score differs")
        for receipt in item["adaptive_receipts"]:
            check_trial(receipt, record["settings"]["complexity_rate"])
        model = Runtime.load(folder / f"{home}-adaptive.pt")
        if (
            model.graph.layout() != item["final_layout"]
            or model.graph.profile() != item["final_profile"]
        ):
            raise ValueError("Topology differs")
        if model.monitor_audit and list(model.monitor_audit) != item["monitor_audit"]:
            raise ValueError("Monitoring audit differs")
    print(
        f'Verified new experiment seed {record["seed"]}: baselines, sources, predictions, scores, trials and checkpoints',
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, choices=(73, 83, 97), default=73)
    parser.add_argument("--configuration", type=Path)
    args = parser.parse_args()
    evaluate(args.archives, args.output, args.seed, args.configuration)
