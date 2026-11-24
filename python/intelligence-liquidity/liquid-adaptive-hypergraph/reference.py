"""Import the unchanged, adjacent verified board and its data protocol."""

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "verified-liquid-board"
sys.path.insert(0, str(REFERENCE))
_model = importlib.import_module("model")
_sources = importlib.import_module("sources")
_benchmark = importlib.import_module("benchmark")
_learn = importlib.import_module("learn")
Board, Decision, ExpertLinear, Round, State = (
    _model.Board,
    _model.Decision,
    _model.ExpertLinear,
    _model.Round,
    _model.State,
)
offline_loss, measured_reward = _model.offline_loss, _model.measured_reward
primitives, sensor_data, sha, source_hashes = (
    _sources.primitives,
    _sources.sensor_data,
    _sources.sha,
    _sources.source_hashes,
)
add_history, data, score = _benchmark.add_history, _benchmark.data, _benchmark.score
by_class, paired_margin = _learn.by_class, _learn.paired_margin

LiquidODE, WIDTH = primitives.LiquidODE, primitives.WIDTH
FIELDS = (
    *sensor_data.ENV_FIELDS,
    *("delta:" + k for k in sensor_data.ENV_FIELDS),
    *("history:" + k for k in sensor_data.ENV_FIELDS),
    "clock:sin",
    "clock:cos",
)
REVISION = "4977126369c64c7a42df2977446e442a5d404121"
