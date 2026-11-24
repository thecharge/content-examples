"""Select policies on validation, then independently grade all frozen future streams."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from batching import fingerprint
from cold_start import dump, quality, replay
from graph import Graph
from multistream import MultiRuntime
from reference import Board, REFERENCE, sha, sensor_data, add_history, score


ROOT = Path(__file__).resolve().parents[1]


def available_brier(report, rows):
    errors = []
    for room, result in report.items():
        if room == "execution":
            continue
        if result["brier"] is not None:
            errors.append(result["brier"])
        else:
            local = rows[room]
            errors.append(
                sum((r["probability"] - r["truth"]) ** 2 for r in local) / len(local)
            )
    return sum(errors) / len(errors)


def choose(model, streams, folder, protocol):
    folder.mkdir(parents=True, exist_ok=True)
    root = model.cores["b0"]
    policies = []
    original = fingerprint(model)
    best = None
    for offset, checker, speaking in protocol["policy_choices"]:
        root.vote_offset.fill_(offset)
        root.checker_logit.fill_(checker)
        root.speaking_confidence.fill_(speaking)
        result, rows = replay(model, streams, "validation", audit=True)
        brier = available_brier(result, rows)
        entry = dict(
            offset=offset,
            checker=checker,
            speaking=speaking,
            balanced_accuracy=quality(result),
            brier=brier,
        )
        policies.append(entry)
        if best is None or (entry["balanced_accuracy"], -brier) > (
            best["balanced_accuracy"],
            -best["brier"],
        ):
            best = entry
    root.vote_offset.fill_(best["offset"])
    root.checker_logit.fill_(best["checker"])
    root.speaking_confidence.fill_(best["speaking"])
    scales = []
    for scale in protocol["probability_scales"]:
        root.probability_scale.fill_(scale)
        result, rows = replay(model, streams, "validation", audit=True)
        scales.append(dict(scale=scale, brier=available_brier(result, rows)))
    chosen = min(scales, key=lambda s: s["brier"])["scale"]
    root.probability_scale.fill_(chosen)
    torch.save(model.pack(), folder / "calibrated.pt")
    dump(
        folder / "selection.json",
        dict(
            protocol_sha256=sha(ROOT / "reporting/protocol.json"),
            partition="validation only",
            before=original,
            after=fingerprint(model),
            policy_choices=policies,
            selected_policy=best,
            scale_choices=scales,
            selected_scale=chosen,
        ),
    )
    return model


def check_source_archive(folder):
    record = json.loads((folder / "executed-sources.json").read_text())
    if sha(folder / "executed-sources.zip") != record["archive_sha256"]:
        raise AssertionError("Executed source archive fingerprint changed")
    return record["archive_sha256"]


def certify(model, streams, folder, *, corrections=None, expected=None):
    folder.mkdir(parents=True, exist_ok=True)
    before = fingerprint(model)
    frozen_corrections = (
        {}
        if corrections is None
        else {room: v.detach().clone() for room, v in corrections.items()}
    )
    solos = {}
    solo_rows = {}
    for stream in streams:
        report, rows = replay(
            model, [stream], "future", audit=True, room_corrections=corrections
        )
        solos.update({k: v for k, v in report.items() if k != "execution"})
        solo_rows.update(rows)
    six, six_rows = replay(
        model, streams, "future", audit=True, room_corrections=corrections
    )
    many, many_rows = replay(
        model, streams, "future", replicas=6, audit=True, room_corrections=corrections
    )
    maximum = 0.0
    for room in solo_rows:
        for a, b, c in zip(
            solo_rows[room], six_rows[room], many_rows[room], strict=True
        ):
            if not (
                a["stamp"] == b["stamp"] == c["stamp"]
                and a["truth"] == b["truth"] == c["truth"]
            ):
                raise AssertionError("Stream scoring rows differ")
            if len({a["answer"], b["answer"], c["answer"]}) != 1:
                raise AssertionError("One/six/thirty-six stream answers differ")
            maximum = max(
                maximum,
                abs(a["probability"] - b["probability"]),
                abs(a["probability"] - c["probability"]),
            )
    if corrections is not None and any(
        not torch.equal(v, corrections[room]) for room, v in frozen_corrections.items()
    ):
        raise AssertionError("Frozen room calibration changed")
    if maximum > 1e-6 or fingerprint(model) != before:
        raise AssertionError("Frozen inference fidelity or tensor preservation failed")
    if expected is not None:
        for room in solos:
            if (
                abs(
                    many[room]["balanced_accuracy"]
                    - expected[room]["balanced_accuracy"]
                )
                > 1e-9
            ):
                raise AssertionError("Archived and current-code accuracy differ")
    report = dict(
        room_scores=solos,
        one_stream_per_room=solos,
        six_streams=six,
        thirty_six_streams=many,
        maximum_probability_difference=maximum,
        frozen_sha256=before,
        equal_answers=True,
        test_feedback_calls=0,
    )
    dump(folder / "results.json", report)
    dump(folder / "predictions.json", many_rows)
    return report


def run(args):
    torch.set_num_threads(1)
    streams = torch.load(args.dataset, weights_only=True)
    protocol = json.loads((ROOT / "reporting/protocol.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    archives = {}
    for i, s in enumerate(streams, 1):
        (t, x, y), files = sensor_data.occupancy(args.archives, i, 21)
        torch.testing.assert_close(s["times"], t, rtol=0, atol=0)
        torch.testing.assert_close(
            s["x"], add_history(t, x), rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(s["y"], y, rtol=0, atol=0)
        archives.update(files)
    # Validate data partitions from the tensors, not a prose claim.
    for s in streams:
        fit, validation, future = [
            s["times"][s[k]] for k in ("fit", "validation", "future")
        ]
        if not (
            int(fit[-1]) < int(validation[0]) and int(validation[-1]) < int(future[0])
        ):
            raise AssertionError("Data chronology overlaps")
        if not torch.equal(
            s["fit"].int() + s["validation"].int() + s["future"].int(),
            torch.ones(len(s["times"]), dtype=torch.int),
        ):
            raise AssertionError("Partitions overlap or omit rows")
    original = Board(features=17)
    checkpoint = REFERENCE / f"evidence/seed-{args.seed}/board.pt"
    original.load_state_dict(torch.load(checkpoint, weights_only=True))
    reports = {}
    # Finish all validation-based selection before opening any future labels here.
    models = {
        "pretrained_frozen": Graph(original).eval(),
        "random_initial": Graph.restore(
            torch.load(args.cold / "random-model.pt", weights_only=True)
        ).eval(),
    }
    for name in ("scratch", "pretrained-retrained"):
        trained = Graph.restore(
            torch.load(args.cold / name / "trained.pt", weights_only=True)
        ).eval()
        models[name] = choose(trained, streams, args.output / name, protocol)
    for name, model in models.items():
        if name == "scratch":
            reports[name] = certify(model, streams, args.output / name)
        else:
            canonical, predictions = replay(model, streams, "future", audit=True)
            reports[name] = dict(
                room_scores={k: v for k, v in canonical.items() if k != "execution"},
                canonical_six_streams=canonical,
                test_feedback_calls=0,
                frozen_sha256=fingerprint(model),
            )
            (args.output / name).mkdir(parents=True, exist_ok=True)
            dump(args.output / name / "results.json", reports[name])
            dump(args.output / name / "predictions.json", predictions)
        print(
            json.dumps(
                dict(
                    variant=name,
                    balanced_accuracy={
                        room: result["balanced_accuracy"]
                        for room, result in reports[name]["room_scores"].items()
                    },
                )
            ),
            flush=True,
        )
    deadline = time.monotonic() + args.wait_online_seconds
    while not (args.online / "results.json").exists():
        if time.monotonic() >= deadline:
            raise FileNotFoundError("Live run is not finished")
        print("Waiting for the independently running live benchmark", flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    online = MultiRuntime.load(args.online / "final.pt", device="cpu")
    correction = {room: s["correction"] for room, s in online.rooms.items()}
    reports["online"] = certify(
        online.graph, streams, args.output / "online", corrections=correction
    )
    previous_folder = (
        ROOT.parent
        / "liquid-adaptive-hypergraph"
        / f"evidence/strict/new-seed-{args.seed}"
    )
    previous_record = json.loads((previous_folder / "results.json").read_text())
    previous = {}
    for room in ("H5", "H6"):
        path = previous_folder / f"{room}-unseen-predictions.json"
        if sha(path) != previous_record["artifact_sha256"][path.name]:
            raise AssertionError("Prior prediction artifact changed")
        old = json.loads(path.read_text())
        stream = next(s for s in streams if s["room"] == room)
        times, truth = stream["times"][stream["future"]], stream["y"][stream["future"]]
        if [int(r["stamp"]) for r in old] != times.tolist() or [
            r["truth"] for r in old
        ] != truth.tolist():
            raise AssertionError("Previous model used different future rows")
        result = score(
            truth,
            torch.tensor([r["prediction"]["answer"] for r in old]),
            torch.tensor([r["prediction"]["probability"] for r in old]),
        )
        if (
            result["balanced_accuracy"]
            != previous_record["homes"][room]["metrics"]["balanced_accuracy"]
        ):
            raise AssertionError("Previous score did not reproduce")
        previous[room] = dict(
            metrics=result,
            predictions_sha256=sha(path),
            checkpoint_sha256=sha(previous_folder / f"{room}-trained-frozen.pt"),
        )
    dump(
        args.output / "previous-hypergraph.json",
        dict(
            results_sha256=sha(previous_folder / "results.json"),
            homes=previous,
            protocol=previous_record["protocol"],
            comparison_caveat="Prior models learned separately per room on 14 days and retained temporal state; new shared-model comparisons start fresh temporal states.",
        ),
    )
    comparison = {}
    for room in protocol["rooms"]:
        scores = {
            name: report["room_scores"][room]["balanced_accuracy"]
            for name, report in reports.items()
        }
        comparison[room] = dict(
            scores=scores,
            scratch_matches_frozen=scores["scratch"] >= scores["pretrained_frozen"],
            scratch_matches_retrained=scores["scratch"]
            >= scores["pretrained-retrained"],
            online_matches_frozen=scores["online"] >= scores["pretrained_frozen"],
            previous_hypergraph=None
            if room not in previous
            else previous[room]["metrics"]["balanced_accuracy"],
            scratch_matches_previous=None
            if room not in previous
            else scores["scratch"] >= previous[room]["metrics"]["balanced_accuracy"],
            online_matches_previous=None
            if room not in previous
            else scores["online"] >= previous[room]["metrics"]["balanced_accuracy"],
        )
    record = dict(
        seed=args.seed,
        protocol_sha256=sha(ROOT / "reporting/protocol.json"),
        dataset_sha256=sha(args.dataset),
        independently_rebuilt_archive_sha256=archives,
        pretrained_sha256=sha(checkpoint),
        cold_sources_sha256=check_source_archive(args.cold),
        online_sources_sha256=check_source_archive(args.online),
        comparison=comparison,
        all_rooms_scratch_match_frozen=all(
            r["scratch_matches_frozen"] for r in comparison.values()
        ),
        all_rooms_online_match_frozen=all(
            r["online_matches_frozen"] for r in comparison.values()
        ),
        one_to_36_fidelity=all(
            reports[name]["equal_answers"] for name in ("scratch", "online")
        ),
        caveats=[
            "One seed; six repeated feeds per recorded room, not 36 independent datasets.",
            "H4 validation has no vacant examples. Its validation accuracy/Brier use the available occupied class; future H4 has both classes.",
            "Scratch and retrained models see all six room prefixes; original pretrained checkpoint saw H1-H3 fitting and H4 selection.",
            "Cold bootstrap and live online learning are distinct protocols.",
            "All policy selection used validation only; test feedback calls are zero.",
            "Hard room gates assess finite observed blocks and cannot guarantee future no-loss accuracy.",
        ],
    )
    dump(args.output / "acceptance.json", record)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", type=Path)
    cli.add_argument("cold", type=Path)
    cli.add_argument("online", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--seed", type=int, default=73)
    cli.add_argument("--archives", type=Path, required=True)
    cli.add_argument("--wait-online-seconds", type=int, default=0)
    run(cli.parse_args())
