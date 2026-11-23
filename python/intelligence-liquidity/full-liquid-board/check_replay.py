"""Reload the public checkpoints and verify the published held-out results."""

import argparse
import json
from pathlib import Path

import torch
import xgboost as xgb

from board import Board
from run_board import heater, occupancy, replay, sha


def check(archive, output, variant=None):
    receipt = json.loads((output / "results.json").read_text())
    task = receipt["task"]
    if task == "occupancy":
        streams = {f"H{i}": occupancy(archive, i, 7)[0] for i in (5, 6)}
        for name, digest in receipt["archives"].items():
            if sha(archive / name) != digest:
                raise ValueError(f"Archive changed: {name}")
    else:
        streams = {"held_out_heat": heater(archive)[2]}
        if sha(archive) != receipt["archives"][archive.name]:
            raise ValueError("Heating data changed")
    for name, digest in receipt["source_sha256"].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"Source changed: {name}")
    path = output / "board.pt"
    if sha(path) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    model = Board(task, max_rounds=receipt["max_rounds"])
    model.load_state_dict(torch.load(path, weights_only=True))
    reviewers = []
    for i, digest in enumerate(receipt["reviewer_sha256"]):
        path = output / f"reviewer-{i+1}.json"
        if sha(path) != digest:
            raise ValueError(f"Tree checker {i+1} changed")
        reviewer = xgb.Booster()
        reviewer.load_model(path)
        reviewers.append((~model.views[i], reviewer))
    variants = receipt["results"] if variant is None else {variant: receipt["results"][variant]}
    for name, results in variants.items():
        mode, updates = name.split("_")
        for site, stream in streams.items():
            observed = replay(model, stream, mode=mode, online=updates == "live",
                              reviewers=reviewers, keep_trace=True)
            if observed != results[site]:
                raise ValueError(f"Replay differs for {name} {site}")
            print(f"Verified {name} {site}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--variant", choices=("off_frozen", "learned_frozen", "learned_live",
                                             "tree_frozen", "tree_live"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    check(args.archive, args.evidence, args.variant)
