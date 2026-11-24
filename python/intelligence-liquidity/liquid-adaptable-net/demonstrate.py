"""A controlled new-input stream; this is not an HPDmobile accuracy result."""

import argparse
import json
from pathlib import Path

import torch

from control import Guards
from graph import Graph
from reference import Board, FIELDS, REFERENCE, sha
from runtime import Runtime, Settings


def demonstrate(output=None, seed=68, windows=97):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    board = Board(features=17)
    checkpoint = REFERENCE / "evidence/seed-73/board.pt"
    board.load_state_dict(torch.load(checkpoint, weights_only=True))
    runtime = Runtime(
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
    )
    predictions = []
    for index in range(windows):
        truth = (index // 3) % 2
        values = {name: 0.0 for name in FIELDS}
        values["new:cue"] = 2.0 * truth - 1
        issued = runtime.predict(300.0 * (index + 1), values)
        # The synthetic source defines the correct class. The network receives
        # it only after it has issued the prediction. No policy action is forced.
        runtime.observe(issued["token"], truth, 300.0 * (index + 1) + 1)
        predictions.append(dict(stamp=300 * (index + 1), truth=truth, **issued))
    receipt = runtime.receipts[0]
    expected = {
        68: "deepen",
        63: "split",
        88: "retire",
        62: "split_liquid",
        60: "add_layer",
        65: "branch_attention",
    }[seed]
    if not receipt["accepted"] or receipt["action"] != expected:
        raise AssertionError("The controlled stream did not retain learned structure")
    record = dict(
        kind="controlled_synthetic_new_input",
        seed=seed,
        settings=vars(runtime.settings),
        guards=vars(runtime.controller.guards),
        initial_checkpoint_sha256=sha(checkpoint),
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        definition="All original inputs are zero. truth=(index//3)%2; new:cue=2*truth-1. "
        "Answers arrive one second after prediction. The first 48 labels fit; "
        "the next 48 independent predictions check the candidate.",
        predictions=predictions,
        receipts=list(runtime.receipts),
        final_layout=runtime.graph.layout(),
    )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model_path = output.with_suffix(".pt")
        runtime.save(model_path)
        record["checkpoint_sha256"] = sha(model_path)
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        f'Controlled stream: {receipt["action"]}, accepted={receipt["accepted"]}, '
        f'balanced gain={receipt["balanced_gain"]:.3f}, experts={runtime.graph.size()}',
        flush=True,
    )
    return runtime, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--seed", type=int, choices=(60, 62, 63, 65, 68, 88), default=68
    )
    parser.add_argument("--windows", type=int, default=193)
    args = parser.parse_args()
    demonstrate(args.output, args.seed, args.windows)
