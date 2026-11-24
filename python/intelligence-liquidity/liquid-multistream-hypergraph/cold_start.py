"""Reproduce a bounded, chronology-separated cold start and 36-stream comparison.

No future labels enter fitting, early stopping, calibration, or topology control.
Six feeds per room are replicas of one recorded timeline, not new observations.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import time
from zipfile import ZipFile, ZIP_DEFLATED

import torch

from batching import fingerprint
from compare import clean, hashes
from graph import Graph, GraphState
from inspection import inspect
from multistream import MultiRuntime
from reference import (
    Board,
    REFERENCE,
    add_history,
    offline_loss,
    score,
    sensor_data,
    sha,
)
from runtime import Settings


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot_sources(folder):
    root = Path(__file__).resolve().parent
    paths = [
        *root.glob("*.py"),
        *root.glob("*configuration.json"),
        root / "BASELINE.json",
        root / "requirements.txt",
    ]
    from reference import source_hashes

    paths.extend(
        REFERENCE / p for p in (*source_hashes(), "online.py", "contextual.py")
    )
    target = folder / "executed-sources.zip"
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(set(p.resolve() for p in paths)):
            archive.write(path, path.relative_to(root.parent).as_posix())
    record = dict(source_sha256=hashes(), archive_sha256=sha(target))
    dump(folder / "executed-sources.json", record)
    return record


def dataset(source, output):
    streams, archives = [], {}
    for room in range(1, 7):
        (times, x, y), files = sensor_data.occupancy(source, room, 21)
        x = add_history(times, x)
        # First 12 complete selected days fit, following 2 select, later days test.
        days = sorted(set((times // 86400).tolist()))
        # Samples are timestamped at bucket END, so midnight belongs to prior day.
        days = sorted(set(((times - 1) // 86400).tolist()))
        fit = (times - 1) // 86400 < days[12]
        validation = (((times - 1) // 86400) >= days[12]) & (
            ((times - 1) // 86400) < days[14]
        )
        future = ((times - 1) // 86400) >= days[14]
        if not all(int(m.sum()) > 0 for m in (fit, validation, future)):
            raise ValueError(
                f"H{room} lacks independent fit, validation and future ranges"
            )
        streams.append(
            dict(
                room=f"H{room}",
                times=times,
                x=x,
                y=y,
                fit=fit,
                validation=validation,
                future=future,
            )
        )
        archives.update(files)
    torch.save(streams, output / "dataset.pt")
    dump(
        output / "dataset.json",
        dict(
            archives=archives,
            source="HPDmobile original environmental/ground-truth ZIPs",
            split={
                s["room"]: {
                    k: dict(
                        rows=int(s[k].sum()),
                        first=int(s["times"][s[k]][0]),
                        last=int(s["times"][s[k]][-1]),
                        classes=s["y"][s[k]].bincount(minlength=2).tolist(),
                    )
                    for k in ("fit", "validation", "future")
                }
                for s in streams
            },
        ),
    )
    return streams, archives


def blocks(streams, length):
    result = []
    for s in streams:
        t, x, y = (s[k][s["fit"]] for k in ("times", "x", "y"))
        edges = [0, *t.diff().ne(300).nonzero().flatten().add(1).tolist(), len(t)]
        for left, right in zip(edges, edges[1:]):
            for start in range(left, right - length + 1, length):
                result.append(
                    (
                        x[start : start + length],
                        y[start : start + length],
                        t[start : start + length],
                        s["room"],
                    )
                )
    return result


@torch.no_grad()
def replay(
    model, streams, partition, replicas=1, *, audit=False, room_corrections=None
):
    settings = Settings(contextual=False, full_weight_learning=False)
    runtime = MultiRuntime(model, settings=settings)
    results = {s["room"]: [] for s in streams}
    selected = []
    for s in streams:
        ids = s[partition].nonzero().flatten().tolist()
        selected.append((s, ids))
        for copy in range(replicas):
            runtime.open(f"{s['room']}/{copy}", s["room"])
            if room_corrections is not None:
                runtime.rooms[s["room"]]["correction"] = (
                    room_corrections[s["room"]].detach().clone()
                )
                runtime.streams[f"{s['room']}/{copy}"]["correction"] = runtime.rooms[
                    s["room"]
                ]["correction"]
    baseline = fingerprint(model)
    start = time.perf_counter()
    for step in range(max(len(ids) for _, ids in selected)):
        requests = []
        for s, ids in selected:
            if step >= len(ids):
                continue
            i = ids[step]
            # Match the reference's reset at recorded discontinuities.
            if step and int(s["times"][i] - s["times"][ids[step - 1]]) != 300:
                for copy in range(replicas):
                    runtime.reset_stream(f"{s['room']}/{copy}")
            for copy in range(replicas):
                requests.append(
                    dict(
                        stream_id=f"{s['room']}/{copy}",
                        room=s["room"],
                        stamp=int(s["times"][i]),
                        values=clean(s["x"][i]),
                        evidence_id=f"{s['room']}:{int(s['times'][i])}",
                        index=i,
                    )
                )
        issued = []
        for left in range(0, len(requests), runtime.stream_limits.batch_size):
            items = requests[left : left + runtime.stream_limits.batch_size]
            outputs = runtime.predict_batch(items)
            for request, out in zip(items, outputs):
                issued.append((request, out))
                runtime.discard(out["token"])
        for s, ids in selected:
            if step >= len(ids):
                continue
            local = [out for req, out in issued if req["room"] == s["room"]]
            answers = {out["answer"] for out in local}
            spread = max(out["probability"] for out in local) - min(
                out["probability"] for out in local
            )
            if len(answers) != 1 or spread > 1e-6:
                raise AssertionError("Replica predictions diverged")
            chosen = local[0]
            results[s["room"]].append(
                dict(
                    stamp=int(s["times"][ids[step]]),
                    truth=int(s["y"][ids[step]]),
                    probability=chosen["probability"],
                    answer=chosen["answer"],
                    quorum=chosen["quorum"],
                    replica_spread=spread,
                )
            )
    seconds = time.perf_counter() - start
    if fingerprint(model) != baseline:
        raise AssertionError("Frozen inference changed model tensors")
    report = {}
    for room, rows in results.items():
        report[room] = dict(
            score(
                torch.tensor([r["truth"] for r in rows]),
                torch.tensor([r["answer"] for r in rows]),
                torch.tensor([r["probability"] for r in rows]),
            ),
            quorum=sum(r["quorum"] for r in rows),
            maximum_replica_spread=max(r["replica_spread"] for r in rows),
        )
    report["execution"] = dict(
        seconds=seconds,
        logical_streams=len(streams) * replicas,
        replicas_per_room=replicas,
        logical_predictions=sum(len(r) for r in results.values()) * replicas,
        unique_observations=sum(len(r) for r in results.values()),
        frozen_hash=baseline,
        memory_bytes=runtime.resident_bytes(),
        backend="cpu",
    )
    return report, results if audit else None


def quality(report):
    return (
        sum(v["balanced_accuracy"] for k, v in report.items() if k != "execution") / 6
    )


def fit(model, streams, args, folder):
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    chunks = blocks(streams, args.length)
    labels = torch.cat([c[1] for c in chunks])
    counts = labels.bincount(minlength=2).float()
    weight = counts.mean() / counts
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rate, weight_decay=0.005)
    initial, _ = replay(model.eval(), streams, "validation")
    curve = [dict(epoch=0, validation=initial, selected=True)]
    best = quality(initial)
    best_state = deepcopy(model.state_dict())
    selected = 0
    dump(folder / "initial-layout.json", inspect(model))
    torch.save(model.pack(), folder / "initial.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        start = time.perf_counter()
        for indices in torch.randperm(len(chunks)).split(args.batch):
            chosen = [chunks[int(i)] for i in indices]
            x, y, t = [torch.stack([c[k] for c in chosen]) for k in (0, 1, 2)]
            state = GraphState()
            costs = []
            optimizer.zero_grad(set_to_none=True)
            for step in range(args.length):
                output = model(
                    x[:, step], x.new_full((len(x),), 5.0), state, stamp=t[:, step]
                )
                state = output.state
                costs.append(
                    torch.stack(
                        [
                            offline_loss(decision, y[:, step], weight, args.residual)
                            for _, decision in output.trace.values()
                        ]
                    ).mean()
                )
                if (step + 1) % args.truncate == 0 or step + 1 == args.length:
                    loss = torch.stack(costs).mean()
                    loss.backward()
                    costs = []
                    state = state.detached()
                    losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        validation, _ = replay(model.eval(), streams, "validation")
        value = quality(validation)
        if value > best + args.minimum_delta:
            best, selected, best_state = value, epoch, deepcopy(model.state_dict())
        entry = dict(
            epoch=epoch,
            loss=sum(losses) / len(losses),
            validation=validation,
            seconds=time.perf_counter() - start,
            selected=selected == epoch,
            parameters=sum(p.numel() for p in model.parameters()),
        )
        curve.append(entry)
        dump(folder / "curve.json", curve)
        print(
            json.dumps(
                dict(
                    variant=folder.name,
                    epoch=epoch,
                    validation_balanced_accuracy=value,
                    best=best,
                    seconds=entry["seconds"],
                )
            ),
            flush=True,
        )
        if epoch in (1, max(1, args.epochs // 2)):
            torch.save(model.pack(), folder / "intermediate.pt")
            dump(folder / "intermediate-layout.json", inspect(model))
        if epoch - selected >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.pack(), folder / "trained.pt")
    dump(folder / "final-layout.json", inspect(model))
    dump(
        folder / "training.json",
        dict(
            selected_epoch=selected,
            completed_epochs=len(curve) - 1,
            early_stopped=len(curve) - 1 < args.epochs,
            initial_validation=initial,
            unique_training_blocks=len(chunks),
            block_length=args.length,
            repeated_feeds_in_loss=False,
            optimizer="AdamW",
            learning_rate=args.rate,
            validation_target="mean per-room balanced accuracy",
            bootstrap="bounded supervised retraining stages; no topology changes during tensor updates",
            online_structural_learning="MultiRuntime separate prospective stages, not this offline bootstrap",
        ),
    )
    return model


def run(args):
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    executed = snapshot_sources(args.output)
    if args.dataset:
        streams = torch.load(args.dataset, weights_only=True)
        archives = json.loads(args.dataset.with_suffix(".json").read_text())["archives"]
    else:
        streams, archives = dataset(args.source, args.output)
    torch.manual_seed(args.seed)
    model = Graph(Board(features=17)).eval()
    initial_hash = fingerprint(model)
    torch.save(model.pack(), args.output / "random-model.pt")
    original = Board(features=17)
    checkpoint = REFERENCE / f"evidence/seed-{args.seed}/board.pt"
    original.load_state_dict(torch.load(checkpoint, weights_only=True))
    pretrained = Graph(original).eval()
    reports = {}
    # Only validation is inspected during fitting. Future test is opened once below.
    if args.variant in ("both", "scratch"):
        scratch = fit(model, streams, args, args.output / "scratch")
        single, single_rows = replay(scratch, streams, "future", audit=True)
        many, many_rows = replay(scratch, streams, "future", replicas=6, audit=True)
        for room in single_rows:
            for a, b in zip(single_rows[room], many_rows[room], strict=True):
                if (
                    a["answer"] != b["answer"]
                    or abs(a["probability"] - b["probability"]) > 1e-6
                ):
                    raise AssertionError("Single/multiple stream fidelity failed")
        reports.update(scratch_single=single, scratch_36=many)
        dump(args.output / "scratch-future-predictions.json", many_rows)
        dump(args.output / "scratch-reports.json", reports)
    if args.variant in ("both", "pretrained"):
        frozen, _ = replay(pretrained, streams, "future")
        reports["pretrained_frozen"] = frozen
        retrained = fit(pretrained, streams, args, args.output / "pretrained-retrained")
        reports["pretrained_retrained"], _ = replay(retrained, streams, "future")
    if (args.output / "scratch-reports.json").exists():
        reports.update(json.loads((args.output / "scratch-reports.json").read_text()))
    verdict = {}
    if "scratch_36" in reports and "pretrained_frozen" in reports:
        for room in [f"H{i}" for i in range(1, 7)]:
            a = reports["scratch_36"][room]["balanced_accuracy"]
            b = reports["pretrained_frozen"][room]["balanced_accuracy"]
            c = reports["pretrained_retrained"][room]["balanced_accuracy"]
            verdict[room] = dict(
                scratch=a,
                pretrained_frozen=b,
                pretrained_same_prefix_retrained=c,
                matches_frozen=a >= b,
                matches_retrained=a >= c,
            )
    dump(
        args.output / "results.json",
        dict(
            seed=args.seed,
            random_model_sha256=initial_hash,
            pretrained_sha256=sha(checkpoint),
            archive_sha256=archives,
            source_sha256=executed["source_sha256"],
            executed_sources_archive_sha256=executed["archive_sha256"],
            source_changed_during_run=hashes() != executed["source_sha256"],
            configuration={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            reports=reports,
            verdict=verdict,
            environment=dict(
                python=platform.python_version(),
                torch=str(torch.__version__),
                device="cpu",
            ),
            scope="Offline cold-start bootstrap and frozen 36-stream inference. Online mutation/feedback tested separately.",
            limitations=[
                "One recorded timeline per room is replicated six times.",
                "Scratch training includes rooms H5/H6 prefix; original pretrained benchmark did not.",
                "Future suffix is not used by training, validation, calibration or controller.",
            ],
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--dataset", type=Path)
    cli.add_argument("--seed", type=int, default=73)
    cli.add_argument("--epochs", type=int, default=16)
    cli.add_argument("--patience", type=int, default=4)
    cli.add_argument("--minimum-delta", type=float, default=0.0001)
    cli.add_argument("--length", type=int, default=36)
    cli.add_argument("--truncate", type=int, default=12)
    cli.add_argument("--batch", type=int, default=8)
    cli.add_argument("--rate", type=float, default=0.002)
    cli.add_argument("--residual", type=float, default=2.0)
    cli.add_argument(
        "--variant", choices=("both", "scratch", "pretrained"), default="both"
    )
    args = cli.parse_args()
    if (
        min(args.epochs, args.patience, args.length, args.truncate, args.batch) < 1
        or args.batch > 8
    ):
        cli.error("Positive counts and batch <=8 required")
    run(args)
