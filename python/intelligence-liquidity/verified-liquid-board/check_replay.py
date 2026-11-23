"""Reproduce the score table from the saved fit and outcome archives."""

import argparse
import json
from pathlib import Path

import torch
import xgboost as xgb

from benchmark import data, replay, score
from model import Board
from data import sha


def check(source, evidence):
    record = json.loads((evidence / "results.json").read_text())
    streams, hashes = data(source)
    if hashes != record["archive_sha256"]:
        raise ValueError("The data hashes do not match the run")
    for collection, folder in ((record["sha256"], evidence),
                               (record["code_sha256"], Path(__file__).resolve().parent)):
        for name, expected in collection.items():
            if sha(folder / name) != expected:
                raise ValueError(f"Hash mismatch for {name}")
    model = Board(experts=record["experts"], rounds=record["rounds"], depth=record["layers"])
    model.load_state_dict(torch.load(evidence / "board.pt", weights_only=True))
    model.eval()
    tree = xgb.Booster(params={"nthread": 1}, model_file=evidence / "tree.json")
    for i, stream in enumerate(streams[4:], 5):
        home = f"H{i}"
        expected = record["replay"][home]
        features, labels = stream[1:]
        pred = torch.tensor(tree.predict(xgb.DMatrix(features.numpy(), nthread=1))) >= .5
        if score(labels, pred) != expected["xgboost"]:
            raise ValueError(f"Tree result did not reproduce for {home}")
        if score(labels, torch.zeros_like(labels)) != expected["constant"]:
            raise ValueError(f"Constant result did not reproduce for {home}")
        for name, online, use_checks in (("without_checks", False, False), ("frozen", False, True),
                                         ("delayed_feedback", True, True)):
            actual, adapted = replay(model, stream, online=online, checks=use_checks, traces=True)
            if online:
                path = evidence / f"{home}-adapted.pt"
                actual["adapted_sha256"] = sha(path)
                saved = torch.load(path, weights_only=True)
                if not all(torch.equal(value, saved[key]) for key, value in adapted.state_dict().items()):
                    raise ValueError(f"Online weights did not reproduce for {home}")
            if actual != expected[name]:
                raise ValueError(f"Replay did not reproduce for {home} {name}")
            print(f"Reproduced {home} {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check(args.source, args.evidence)
