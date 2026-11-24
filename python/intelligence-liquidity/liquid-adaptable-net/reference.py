"""Import the unchanged, adjacent verified board and its data protocol."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "verified-liquid-board"
sys.path.insert(0, str(REFERENCE))
from model import (
    Board,
    Decision,
    ExpertLinear,
    Round,
    State,
    offline_loss,
    measured_reward,
)
from sources import primitives, sensor_data, sha, source_hashes
from benchmark import add_history, data, score
from learn import by_class, paired_margin

LiquidODE, WIDTH = primitives.LiquidODE, primitives.WIDTH
FIELDS = (
    *sensor_data.ENV_FIELDS,
    *("delta:" + k for k in sensor_data.ENV_FIELDS),
    *("history:" + k for k in sensor_data.ENV_FIELDS),
    "clock:sin",
    "clock:cos",
)
REVISION = "4977126369c64c7a42df2977446e442a5d404121"
