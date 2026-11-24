"""Create, feed, inspect or branch the durable liquid hypergraph experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from device import require
from graph import Graph
from inspection import inspect
from ledger import Ledger, StorageLimits
from provenance import Sensor
from reference import Board, REFERENCE, sha
from runtime import Runtime
from session import Session
from workflow import load_configuration


def storage(path):
    config = json.loads(path.read_text())
    return StorageLimits(**config["storage"]), config["maximum_features"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "feed", "inspect", "branch"))
    parser.add_argument("database", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path(__file__).with_name("configuration.json"),
    )
    parser.add_argument(
        "--storage",
        type=Path,
        default=Path(__file__).with_name("storage-configuration.json"),
    )
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--multistream", action="store_true")
    parser.add_argument(
        "--streams",
        type=Path,
        default=Path(__file__).with_name("stream-configuration.json"),
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--revision", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "rocm"), default="cpu")
    args = parser.parse_args()
    if not args.multistream:
        require(args.device)
    elif args.device == "rocm" and (
        not torch.version.hip or not torch.cuda.is_available()
    ):
        raise RuntimeError("A ROCm build and accessible accelerator are required")
    torch.set_num_threads(1)
    limits, maximum = storage(args.storage)
    if args.command == "create" and args.database.exists():
        raise ValueError("Create needs a new database path")
    if args.command != "create" and not args.database.exists():
        raise ValueError("No experiment database exists")
    ledger = Ledger(args.database, limits)
    try:
        if args.command == "create":
            torch.manual_seed(args.seed)
            if args.runtime:
                if args.from_scratch:
                    raise ValueError("Choose runtime import or scratch")
                from session import unpack_runtime

                runtime = unpack_runtime(
                    torch.load(args.runtime, weights_only=True), None
                )
                origin = dict(runtime_sha256=sha(args.runtime), kind="runtime_import")
            else:
                settings, guards, graph_limits = load_configuration(args.configuration)
                base = Board(features=17)
                origin = dict(kind="scratch", seed=args.seed)
                if not args.from_scratch:
                    checkpoint = REFERENCE / f"evidence/seed-{args.seed}/board.pt"
                    base.load_state_dict(torch.load(checkpoint, weights_only=True))
                    origin = dict(
                        kind="verified_board", sha256=sha(checkpoint), seed=args.seed
                    )
                runtime_class, extra = Runtime, {}
                if args.multistream:
                    from multistream import MultiRuntime, StreamLimits

                    runtime_class = MultiRuntime
                    extra = dict(
                        stream_limits=StreamLimits(
                            **json.loads(args.streams.read_text())
                        ),
                        device="cuda" if args.device == "rocm" else "cpu",
                    )
                runtime = runtime_class(
                    Graph(base, **graph_limits),
                    **extra,
                    settings=settings,
                    guards=guards,
                    seed=args.seed,
                )
            Session(ledger, runtime, maximum_features=maximum, origin=origin)
            print(json.dumps(dict(revision=ledger.revision, origin=origin)))
        elif args.command == "feed":
            if args.input is None:
                raise ValueError("Feed requires --input JSONL")
            session = Session(ledger)
            with args.input.open() as stream:
                while True:
                    line = stream.readline(limits.max_event_bytes + 1)
                    if not line:
                        break
                    if len(line.encode()) > limits.max_event_bytes:
                        raise ValueError("Input line exceeds the event ceiling")
                    item = json.loads(line)
                    kind = item.pop("kind")
                    if kind == "observation":
                        item["sensors"] = {
                            k: Sensor(**v) for k, v in item["sensors"].items()
                        }
                        result = session.predict(**item)
                    elif kind == "observation_batch":
                        for observation in item["observations"]:
                            observation["sensors"] = {
                                k: Sensor(**v)
                                for k, v in observation["sensors"].items()
                            }
                        result = session.predict_batch(**item)
                    elif kind == "label":
                        result = session.label(**item)
                    elif kind == "reward":
                        result = session.reward(**item)
                    elif kind == "derived_feature":
                        result = session.derive(**item)
                    elif kind == "correction":
                        result = session.correct(**item)
                    elif kind == "restore":
                        result = session.restore(**item)
                    else:
                        raise ValueError("Unknown event kind: " + kind)
                    print(
                        json.dumps(
                            dict(revision=session.revision, result=result),
                            allow_nan=False,
                        ),
                        flush=True,
                    )
        elif args.command == "inspect":
            session = Session(ledger)
            if args.revision is not None:
                session._install(ledger.recover(args.revision))
            print(
                json.dumps(
                    dict(
                        revision=args.revision or ledger.revision,
                        blocked=session.blocked,
                        model=inspect(session.runtime.graph),
                    ),
                    indent=2,
                    allow_nan=False,
                )
            )
        elif args.command == "branch":
            if args.output is None or args.output.exists():
                raise ValueError("Branch needs a new --output database")
            revision = args.revision or ledger.revision
            row = ledger.db.execute(
                "SELECT snapshot,valid FROM events WHERE revision=?", (revision,)
            ).fetchone()
            if row is None or not row[1]:
                raise ValueError("Branch needs a valid retained revision")
            state = ledger.recover(revision)
            if state["blocked"]:
                raise ValueError("Cannot bless a quarantined branch")
            child = Ledger(args.output, limits)
            try:
                child.commit(
                    "initial",
                    "branch",
                    dict(
                        parent_revision=revision,
                        parent_snapshot=row[0],
                        parent_database_sha256=hashlib.sha256(
                            args.database.read_bytes()
                        ).hexdigest(),
                    ),
                    state,
                    0,
                )
                print(
                    json.dumps(
                        dict(
                            revision=child.revision,
                            parent_revision=revision,
                            parent_snapshot=row[0],
                        )
                    )
                )
            finally:
                child.close()
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
