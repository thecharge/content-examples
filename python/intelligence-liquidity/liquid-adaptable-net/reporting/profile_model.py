"""Measure fixed-shape inference in an isolated, single-thread CPU process."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("inputs", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.resolve()
allowed = Path(__file__).resolve().parents[2]
if source not in (allowed / "adaptive-liquid-board", allowed / "liquid-adaptable-net"):
    parser.error("Select one of the two adjacent experiment sources")
sys.path.insert(0, str(source))
import torch
from runtime import Runtime
from graph import GraphState

torch.set_num_threads(1)
model = Runtime.load(args.checkpoint).graph.eval()
inputs = json.loads(args.inputs.read_text())
timings, checksums = [], []
for repeat in range(4):
    state, previous = GraphState(), None
    probabilities = []
    for row in inputs:
        if previous is None or row["stamp"] - previous != 300:
            state = GraphState()
        x = model.tensor(row["values"])
        tick = time.perf_counter_ns()
        with torch.no_grad():
            result = model(x, x.new_tensor([5.0]), state, stamp=row["stamp"])
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        if repeat:
            timings.append(elapsed)
        state, previous = result.state.detached(), row["stamp"]
        probabilities.append(float(result.probability))
    checksums.append(hashlib.sha256(json.dumps(probabilities).encode()).hexdigest())
if len(set(checksums)) != 1:
    raise AssertionError("Repeated inference changed predictions")
ordered = sorted(timings)
record = dict(
    source=source.name,
    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
    profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    threads=torch.get_num_threads(),
    warmup_passes=1,
    measured_passes=3,
    observations=len(inputs),
    median_ms=statistics.median(timings),
    p95_ms=ordered[int(0.95 * (len(ordered) - 1))],
    mean_ms=statistics.mean(timings),
    timings_ms=timings,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    parameter_count=sum(p.numel() for p in model.parameters()),
    parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
    buffer_bytes=sum(b.numel() * b.element_size() for b in model.buffers()),
    checkpoint_bytes=args.checkpoint.stat().st_size,
    predictions_sha256=checksums[0],
    method="Graph.forward only; exclude tensor packing, learning, checkpoint loading and state detachment. First 160 chronological H5 rows; reset at gaps; process peak RSS includes imports and loading.",
)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n")
print({k: v for k, v in record.items() if k not in ("timings_ms",)}, flush=True)
