"""Cold-start 36 streams with delayed labels, live liquid state and bounded stages."""

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from batching import fingerprint
from cold_start import dump, replay, snapshot_sources
from compare import clean, hashes
from graph import Graph
from inspection import inspect
from multistream import MultiRuntime, StreamLimits
from reference import Board, score
from runtime import Settings


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    streams = torch.load(args.dataset, weights_only=True)
    graph = Graph(Board(features=17)).eval()
    runtime = MultiRuntime(
        graph,
        seed=args.seed,
        settings=Settings(maximum_pending=288),
        stream_limits=StreamLimits(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    executed = snapshot_sources(args.output)
    initial = fingerprint(graph)
    runtime.save(args.output / "initial.pt")
    dump(args.output / "initial-layout.json", inspect(graph))
    pending = {}
    rows = {}
    selected = []
    for s in streams:
        indices = s["fit"].nonzero().flatten().tolist()
        if args.steps:
            indices = indices[: args.steps]
        selected.append((s, indices))
        for replica in range(6):
            sid = f"{s['room']}/{replica}"
            runtime.open(sid, s["room"])
            pending[sid] = deque()
            rows[sid] = []
    started = time.perf_counter()
    curves = []
    for step in range(max(len(ids) for _, ids in selected)):
        # Reveal prior labels independently in each stream. Different rooms have
        # different delays; one unavailable room never blocks inference elsewhere.
        for s, ids in selected:
            for replica in reversed(range(6)):
                sid = f"{s['room']}/{replica}"
                while pending[sid] and pending[sid][0][0] <= step:
                    _, token, truth, stamp = pending[sid].popleft()
                    now = max(stamp + 1, runtime.streams[sid]["last_prediction"] + 1)
                    runtime.reward(
                        token,
                        1.0 if runtime.pending[token]["answer"] == truth else -1.0,
                        now,
                    )
                    runtime.observe(token, truth, now)
        requests = []
        truth_by_sid = {}
        for s, ids in selected:
            if step >= len(ids):
                continue
            index = ids[step]
            stamp = int(s["times"][index])
            for replica in range(6):
                sid = f"{s['room']}/{replica}"
                requests.append(
                    dict(
                        stream_id=sid,
                        room=s["room"],
                        stamp=stamp,
                        values=clean(s["x"][index]),
                        evidence_id=f"{s['room']}:{stamp}",
                    )
                )
                truth_by_sid[sid] = int(s["y"][index])
        for left in range(0, len(requests), 8):
            batch = requests[left : left + 8]
            issued = runtime.predict_batch(batch)
            for request, out in zip(batch, issued):
                sid = out["stream_id"]
                truth = truth_by_sid[sid]
                rows[sid].append(dict(stamp=request["stamp"], truth=truth, **out))
                room = int(out["room"][1:])
                pending[sid].append(
                    (step + 1 + (room % 3), out["token"], truth, request["stamp"])
                )
        if (step + 1) % 288 == 0:
            curve = dict(
                step=step + 1,
                version=runtime.version,
                profile=runtime.graph.profile(),
                shared_observations=len(runtime.evidence),
                receipts=list(runtime.receipts),
                per_stream={
                    sid: score(
                        torch.tensor([r["truth"] for r in local[-288:]]),
                        torch.tensor([r["answer"] for r in local[-288:]]),
                        torch.tensor([r["probability"] for r in local[-288:]]),
                    )
                    for sid, local in rows.items()
                },
            )
            curves.append(curve)
            dump(args.output / "curve.json", curves)
            runtime.save(args.output / "latest.pt")
            dump(args.output / "latest-layout.json", inspect(runtime.graph))
            print(
                json.dumps(
                    dict(
                        step=step + 1,
                        version=runtime.version,
                        parameters=runtime.graph.profile()["parameters"],
                        receipts=len(runtime.receipts),
                        seconds=time.perf_counter() - started,
                    )
                ),
                flush=True,
            )
    for queue in pending.values():
        while queue:
            _, token, truth, stamp = queue.popleft()
            slot = runtime.streams[runtime.pending[token]["stream_id"]]
            runtime.observe(
                token,
                truth,
                max(stamp + 1, slot["last_prediction"] + 1, slot["last_reveal"]),
            )
    runtime.save(args.output / "final.pt")
    torch.save(runtime.graph.pack(), args.output / "trained-graph.pt")
    dump(args.output / "final-layout.json", inspect(runtime.graph))
    dump(args.output / "predictions.json", rows)
    # Untouched future inference has no feedback path; no labels reach the model.
    future, _ = replay(
        runtime.graph,
        streams,
        "future",
        replicas=6,
        room_corrections={
            room: shared["correction"] for room, shared in runtime.rooms.items()
        },
    )
    dump(
        args.output / "results.json",
        dict(
            seed=args.seed,
            initial_sha256=initial,
            final_sha256=fingerprint(runtime.graph),
            empty_initialization=True,
            logical_streams=len(runtime.streams),
            rooms=6,
            replicas_per_room=6,
            unique_labeled_rows=sum(len(ids) for _, ids in selected),
            logical_predictions=sum(len(v) for v in rows.values()),
            receipts=list(runtime.receipts),
            monitoring=list(runtime.monitor_audit),
            private_calibration={
                sid: slot["learner"].receipts for sid, slot in runtime.streams.items()
            },
            final_version=runtime.version,
            late_feedback=runtime.late_feedback,
            source_sha256=executed["source_sha256"],
            executed_sources_archive_sha256=executed["archive_sha256"],
            source_changed_during_run=hashes() != executed["source_sha256"],
            settings=asdict(runtime.settings),
            limits=asdict(runtime.stream_limits),
            seconds=time.perf_counter() - started,
            frozen_future=future,
            scope="Online cold start; frozen future keeps trained room calibrations and starts fresh private liquid states.",
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("dataset", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--seed", type=int, default=73)
    cli.add_argument(
        "--steps", type=int, default=0, help="0 uses the complete fitting prefix"
    )
    args = cli.parse_args()
    if args.steps < 0:
        cli.error("steps must be nonnegative")
    run(args)
