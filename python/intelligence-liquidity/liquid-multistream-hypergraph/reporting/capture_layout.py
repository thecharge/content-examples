"""Export an actual saved runtime as a named before/during/after graph layout."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from inspection import inspect
from multistream import MultiRuntime


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("checkpoint", type=Path)
    cli.add_argument("output_prefix", type=Path)
    args = cli.parse_args()
    runtime = MultiRuntime.load(args.checkpoint)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "-layout.json").write_text(
        json.dumps(inspect(runtime.graph), indent=2) + "\n"
    )
    torch.save(runtime.graph.pack(), prefix.with_name(prefix.name + "-graph.pt"))
    prefix.with_name(prefix.name + "-state.json").write_text(
        json.dumps(
            dict(
                coordinator_order=runtime.order,
                version=runtime.version,
                streams={
                    sid: slot["last_prediction"]
                    for sid, slot in runtime.streams.items()
                },
                correction_labels=[
                    *[f"value:{field}" for field in runtime.graph.fields],
                    *[f"present:{field}" for field in runtime.graph.fields],
                    "bias",
                ],
                rooms={
                    room: dict(
                        norm=float(value["correction"].norm()),
                        coefficients=value["correction"].tolist(),
                        accepted_updates=sum(
                            bool(receipt.get("accepted"))
                            for receipt in value["learner"].receipts
                        )
                        if value["learner"] is not None
                        else 0,
                    )
                    for room, value in runtime.rooms.items()
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(order=runtime.order, version=runtime.version, captured=True)))


if __name__ == "__main__":
    main()
