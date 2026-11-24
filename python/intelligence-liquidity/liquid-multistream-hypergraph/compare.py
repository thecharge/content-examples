"""Fingerprints, finite inputs and the original benchmark scoring protocol."""

import math
from pathlib import Path
import torch
from reference import FIELDS, REFERENCE, score, sha, source_hashes

ROOT = Path(__file__).resolve().parent


def hashes():
    own = [
        *ROOT.glob("*.py"),
        *ROOT.glob("*configuration.json"),
        ROOT / "requirements.txt",
        ROOT / "BASELINE.json",
    ]
    return {
        **{"adaptive/" + p.name: sha(p) for p in sorted(own)},
        **{
            "reference/" + p: sha(REFERENCE / p)
            for p in (*source_hashes(), "online.py", "contextual.py")
        },
    }


def clean(values):
    return {
        name: None if not math.isfinite(float(value)) else float(value)
        for name, value in zip(FIELDS, values)
    }


def summarize(rows, variant):
    expected = torch.tensor([r["truth"] for r in rows])
    answer = torch.tensor([r[variant]["answer"] for r in rows])
    probability = torch.tensor([r[variant]["probability"] for r in rows])
    result = score(expected, answer, probability, with_interval=True)
    if variant != "tree":
        result["quorum"] = sum(r[variant]["quorum"] for r in rows)
    return result
