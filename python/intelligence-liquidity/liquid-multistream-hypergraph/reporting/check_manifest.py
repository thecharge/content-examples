"""Verify delivered files and unchanged preceding experiment against their hashes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check():
    manifest = json.loads((ROOT / "MANIFEST.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert digest(ROOT / name) == expected, name
    baseline = json.loads((ROOT / "BASELINE.json").read_text())
    for name, expected in baseline["sha256"].items():
        assert digest(ROOT / baseline["directory"] / name) == expected, name
    print(
        json.dumps(
            dict(
                delivered_files=len(manifest["sha256"]),
                preceding_files=len(baseline["sha256"]),
                passed=True,
            )
        )
    )


if __name__ == "__main__":
    check()
