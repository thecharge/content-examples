"""Train, fine-tune, inspect and recover CPU models from recorded events."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from graph import Graph
from reference import Board, sha
from runtime import Runtime, Settings, train_candidate
from control import Controller, Guards


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_configuration(path=None):
    source = Path(path) if path else Path(__file__).with_name("configuration.json")
    value = json.loads(source.read_text())
    return Settings(**value["learning"]), Guards(**value["guards"]), value["limits"]


class Journal:
    """An explicit ordered event log. Snapshot only after its event is durable."""

    def __init__(self, runtime, path, cursor=0):
        self.runtime, self.path, self.cursor = runtime, Path(path), cursor
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            last = json.loads(self.path.read_text().splitlines()[-1])
            if last["sequence"] != cursor:
                raise ValueError(
                    "Journal has another future; recover into a new journal"
                )

    def record(self, kind, at, **payload):
        self.cursor += 1
        event = dict(sequence=self.cursor, kind=kind, at=at, **payload)
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def predict(self, stamp, values):
        # JSON uses null for missing evidence, including caller-supplied NaN.
        values = {
            k: None if v is None or math.isnan(float(v)) else float(v)
            for k, v in values.items()
        }
        result = self.runtime.predict(stamp, values)
        self.record("predict", stamp, values=values, token=result["token"])
        return result

    def observe(self, token, truth, revealed_at):
        self.runtime.observe(token, truth, revealed_at)
        self.record("observe", revealed_at, token=token, truth=truth)

    def reward(self, token, value, revealed_at):
        self.runtime.reward(token, value, revealed_at)
        self.record("reward", revealed_at, token=token, value=value)

    def reset_state(self):
        self.runtime.reset_state()
        self.record(
            "reset", max(self.runtime.last_prediction, self.runtime.last_reveal)
        )

    def discard(self, token):
        self.runtime.discard(token)
        self.record(
            "discard",
            max(self.runtime.last_prediction, self.runtime.last_reveal),
            token=token,
        )

    def snapshot(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.runtime.save(path)
        content = self.path.read_bytes() if self.path.exists() else b""
        at = max(self.runtime.last_prediction, self.runtime.last_reveal)
        write_json(
            path.with_suffix(path.suffix + ".snapshot.json"),
            dict(
                schema="liquid-journal-snapshot-v1",
                checkpoint_sha256=sha(path),
                cursor=self.cursor,
                at=None if not math.isfinite(at) else at,
                journal_bytes=len(content),
                journal_prefix_sha256=hashlib.sha256(content).hexdigest(),
            ),
        )


def recover(checkpoint, journal, until, output=None):
    checkpoint, journal = Path(checkpoint), Path(journal)
    metadata = json.loads(
        checkpoint.with_suffix(checkpoint.suffix + ".snapshot.json").read_text()
    )
    if (
        metadata["schema"] != "liquid-journal-snapshot-v1"
        or sha(checkpoint) != metadata["checkpoint_sha256"]
    ):
        raise ValueError("Snapshot fingerprint changed")
    if metadata["at"] is not None and until < metadata["at"]:
        raise ValueError(
            "Recovery time precedes this snapshot; choose an earlier snapshot"
        )
    content = journal.read_bytes()
    prefix = content[: metadata["journal_bytes"]]
    if hashlib.sha256(prefix).hexdigest() != metadata["journal_prefix_sha256"]:
        raise ValueError("Journal history before snapshot changed")
    runtime = Runtime.load(checkpoint)
    cursor = metadata["cursor"]
    previous = metadata["at"]
    for line in content[metadata["journal_bytes"] :].splitlines():
        event = json.loads(line)
        if event["sequence"] != cursor + 1:
            raise ValueError("Journal sequence is incomplete or duplicated")
        if previous is not None and event["at"] < previous:
            raise ValueError("Journal time moved backwards")
        if event["at"] > until:
            break
        kind = event["kind"]
        if kind == "predict":
            issued = runtime.predict(event["at"], event["values"])
            if issued["token"] != event["token"]:
                raise ValueError("Recovered prediction identity differs")
        elif kind == "observe":
            runtime.observe(event["token"], event["truth"], event["at"])
        elif kind == "reward":
            runtime.reward(event["token"], event["value"], event["at"])
        elif kind == "reset":
            runtime.reset_state()
        elif kind == "discard":
            runtime.discard(event["token"])
        else:
            raise ValueError("Unknown journal event")
        cursor, previous = event["sequence"], event["at"]
    if output is not None:
        Journal(runtime, Path(str(output) + ".events.jsonl"), cursor).snapshot(output)
    return runtime, cursor


def train(data, output, mode, checkpoint=None, configuration=None, until=math.inf):
    configuration = configuration or Path(__file__).with_name(
        "training-configuration.json"
    )
    settings, guards, limits = load_configuration(configuration)
    torch.set_num_threads(1)
    torch.manual_seed(73)
    if mode == "finetune":
        if checkpoint is None:
            raise ValueError("Fine-tuning requires a snapshot")
        ancestor = Runtime.load(checkpoint).graph
        for name, value in limits.items():
            setattr(ancestor, name, value)
        ancestor.validate_topology()
        if not Controller(guards).budget(ancestor)["allowed"]:
            raise ValueError("Snapshot exceeds the requested retraining envelope")
        graph = ancestor.fork()
    else:
        if checkpoint is not None:
            raise ValueError("Scratch training does not read a checkpoint")
        ancestor = None
        graph = Graph(Board(features=17), **limits)
    rows = [
        json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row["stamp"] <= until]
    if not rows or any(b["stamp"] <= a["stamp"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Training rows must be nonempty and strictly chronological")
    if {row["truth"] for row in rows} != {0, 1}:
        raise ValueError("Training requires both external classes")
    for row in rows:
        if not math.isfinite(row["stamp"]) or any(
            v is not None
            and (not math.isfinite(float(v)) or abs(float(v)) > guards.max_abs_input)
            for v in row["values"].values()
        ):
            raise ValueError("Invalid training time or input magnitude")
    graph.add_fields(
        sorted({name for row in rows for name in row["values"]} - set(graph.fields))
    )
    budget = Controller(guards).budget(graph)
    if not budget["allowed"]:
        raise ValueError("Offline model exceeds configured training budget")
    # Architecture selection uses only this explicit training fork's earlier
    # labels and subsequent training observations, never a deployment check set.
    selection = Runtime(graph, settings=settings, guards=guards)
    previous = None
    for row in rows:
        if previous is None or row["stamp"] - previous != 300:
            selection.reset_state()
        issued = selection.predict(row["stamp"], row["values"])
        if row.get("reward") is not None:
            selection.reward(issued["token"], row["reward"], row["stamp"] + 1)
        selection.observe(issued["token"], row["truth"], row["stamp"] + 1)
        previous = row["stamp"]
    graph = selection.graph
    unfinished = selection.trial
    # An unfinished proposal is not an adopted topology. Final fitting starts
    # from the last accepted graph with a fresh optimizer and private states.
    report = train_candidate(graph, rows, settings, ancestor=ancestor, guards=guards)
    runtime = Runtime(graph, settings=settings, guards=guards)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    runtime.save(output)
    receipt = dict(
        mode=mode,
        data_sha256=sha(data),
        initial_checkpoint_sha256=None if checkpoint is None else sha(checkpoint),
        through=rows[-1]["stamp"],
        rows=len(rows),
        settings=asdict(settings),
        guards=asdict(guards),
        fit=report,
        final_layout=graph.layout(),
        structural_receipts=list(selection.receipts),
        unfinished_selection_trial=unfinished,
        checkpoint_sha256=sha(output),
    )
    write_json(Path(str(output) + ".training.json"), receipt)
    return runtime, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("mode", choices=("scratch", "finetune"))
    fitting.add_argument("data", type=Path)
    fitting.add_argument("output", type=Path)
    fitting.add_argument("--checkpoint", type=Path)
    fitting.add_argument("--configuration", type=Path)
    fitting.add_argument("--until", type=float, default=math.inf)
    replay = commands.add_parser("recover")
    replay.add_argument("checkpoint", type=Path)
    replay.add_argument("journal", type=Path)
    replay.add_argument("output", type=Path)
    replay.add_argument("--until", type=float, required=True)
    inspection = commands.add_parser("inspect")
    inspection.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        _, receipt = train(
            args.data,
            args.output,
            args.mode,
            args.checkpoint,
            args.configuration,
            args.until,
        )
        print(json.dumps(receipt["fit"], indent=2))
    elif args.command == "recover":
        runtime, cursor = recover(
            args.checkpoint, args.journal, args.until, args.output
        )
        print(
            f"Recovered through event {cursor}; {len(runtime.pending)} pending answers"
        )
    else:
        runtime = Runtime.load(args.checkpoint)
        print(
            json.dumps(
                dict(
                    layout=runtime.graph.layout(),
                    settings=asdict(runtime.settings),
                    guards=asdict(runtime.controller.guards),
                    pending=len(runtime.pending),
                    contextual_updates=(
                        []
                        if runtime.contextual is None
                        else runtime.contextual.receipts
                    ),
                ),
                indent=2,
            )
        )
