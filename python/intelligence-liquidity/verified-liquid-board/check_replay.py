"""Reproduce scores from a saved fit, public archives and recorded settings."""

import argparse
import json
import math
from pathlib import Path

import torch
import xgboost as xgb

from benchmark import data, expert_diagnostics, replay, score
from model import Board
from sources import ROOT, sha, source_hashes


def match(actual, expected, key="receipt"):
    """Class counts and decisions must match. Allow float rounding across CPUs."""
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        for field, value in expected.items():
            match(actual[field], value, f"{key}.{field}")
        return
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for index, value in enumerate(expected):
            match(actual[index], value, f"{key}.{index}")
        return
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-6):
            return
    elif type(actual) is type(expected) and not isinstance(expected, (dict, list)) and actual == expected:
        return
    raise ValueError(f"{key} differs from the recorded run")


def check(source, evidence):
    record = json.loads((evidence / "results.json").read_text())
    if source_hashes() != record["code_sha256"]:
        raise ValueError(f"Code under {ROOT} differs from the recorded run")
    for name, expected in record["sha256"].items():
        if sha(evidence / name) != expected:
            raise ValueError(f"Hash mismatch for {name}")
    streams, hashes = data(source, record["sensor_history"])
    if hashes != record["archive_sha256"]:
        raise ValueError("The data hashes do not match the run")
    model = Board(features=record["features"], experts=record["experts"],
                  rounds=record["rounds"], depth=record["layers"])
    model.load_state_dict(torch.load(evidence / "board.pt", weights_only=True))
    model.eval()
    match(expert_diagnostics(model, streams[3]), record["development_experts"], "development experts")
    tree = xgb.Booster(params={"nthread": 1}, model_file=evidence / "tree.json")
    for i, stream in enumerate(streams[4:], 5):
        home = f"H{i}"
        expected = record["replay"][home]
        features, labels = stream[1:]
        probabilities = torch.tensor(tree.predict(xgb.DMatrix(features.numpy(), nthread=1)))
        match(score(labels, probabilities >= .5, probabilities, True), expected["xgboost"], f"{home} tree")
        for bit, name in enumerate(("always_vacant", "always_occupied")):
            result = score(labels, torch.full_like(labels, bit), torch.full_like(labels, bit), True)
            match(result, expected[name], f"{home} {name}")
        for name, online, use_checks in (("without_checks", False, False), ("frozen", False, True),
                                         ("delayed_feedback", True, True)):
            actual, adapted = replay(model, stream, online=online, checks=use_checks, traces=True)
            if online:
                path = evidence / f"{home}-adapted.pt"
                actual["adapted_sha256"] = sha(path)
                saved = torch.load(path, weights_only=True)
                for key, value in adapted.state_dict().items():
                    torch.testing.assert_close(value, saved[key], rtol=2e-5, atol=2e-6, msg=f"{home} {key}")
            match(actual, expected[name], f"{home} {name}")
            print(f"Reproduced {home} {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    check(args.source, args.evidence)
