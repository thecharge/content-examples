"""Load shared sources explicitly and record the complete replay inputs."""

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SHARED = ("../liquid-workflow-v2/workflow.py", "../liquid-workflow-v3/data.py")


def load_shared(name, relative):
    spec = spec_from_file_location(name, ROOT / relative)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {relative}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


primitives = load_shared("verified_board_primitives", SHARED[0])
sensor_data = load_shared("verified_board_sensor_data", SHARED[1])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes():
    files = ("sources.py", "model.py", "learn.py", "benchmark.py", "check_replay.py",
             "requirements.txt", *SHARED)
    return {name: sha(ROOT / name) for name in files}
