"""Replay newline-delimited sensor readings with an evaluated checkpoint."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from experiment import sha
from local_reference import FITTING_CENTER_C, LocalTemperature
from workflow import Blackboard, Observation, Online


def replay(lines, model, cutoff, live, local=False):
    state, last = None, None
    reference = LocalTemperature() if local else None
    learner = Online(model) if live else None
    for line in lines:
        reading = json.loads(line)
        timestamp = reading["time"]
        if type(timestamp) is not int or (last is not None and timestamp <= last):
            raise ValueError("time must be an increasing integer in seconds")
        co2 = float(reading["co2_ppm"])
        temperature = reading.get("witness_c")
        temperature = float(temperature) if temperature is not None else None
        if not math.isfinite(co2) or (temperature is not None and not math.isfinite(temperature)):
            raise ValueError("Sensor readings must be finite")
        gap = timestamp - last if last is not None else 300
        if gap > 450:
            state = None
            if reference is not None:
                reference.center = None
        dt = torch.tensor([gap / 60 if gap <= 450 else 5.])
        primary = torch.tensor([(co2-500.) / 200])
        witness = torch.tensor([(temperature-FITTING_CENTER_C) / 3 if temperature is not None else float("nan")])
        if reference is not None:
            witness = torch.tensor([reference.aligned(temperature)])
        center = reference.center if reference is not None else FITTING_CENTER_C
        with torch.no_grad():
            candidate = model.step(primary, witness, dt, state)
            reading_score = candidate.logit.sigmoid().item()
            yield dict(time=timestamp, decision=(
                "occupied" if reading_score >= cutoff else "empty") if candidate.accepted.item() else None,
                score=reading_score, witness_reference_c=center, next_co2_ppm=candidate.forecast.item()*200+500,
                checks=[dict(layer=r["layer"]+1, retry_from=(
                    r["retry_from"]+1 if r["retry_from"] is not None else None), expert=r["expert"].item()+1,
                    expected_witness_c=r["claim"].item()*3+center if center is not None else None,
                    error_c=r["error"].item()*3 if witness.isfinite().item() else None,
                    learned_check_score=r["process"].sigmoid().item()) for r in candidate.attempts])
        if learner is not None:
            learner.feedback(Observation(timestamp, primary, witness, dt, state))
        if reference is not None:
            reference.observe(temperature)
        state, last = candidate.ledger, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--local-reference", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((args.evidence/"fit.json").read_text())["models"]["workflow"]
    path = args.evidence / "workflow.pt"
    if sha(path) != receipt["best_sha256"]:
        raise ValueError("Checkpoint does not match its receipt")
    model = Blackboard()
    model.load_state_dict(torch.load(path, weights_only=True))
    for entry in replay(sys.stdin, model, receipt["cutoff"], args.live, args.local_reference):
        print(json.dumps(entry, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
