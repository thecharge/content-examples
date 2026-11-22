"""Measure one live replay on one CPU thread, including the feedback step."""
import argparse
import platform
from collections import Counter
from pathlib import Path

import torch

from run import dump, io_data
from serve import Stream, percentile


def main(data, checkpoint, output, cutoff):
    torch.set_num_threads(1)
    room, readings, _, stamps = io_data.test_sites(data)["munich"][0]
    stream = Stream(checkpoint, cutoff, live=True)
    responses = []
    calls = Counter()
    skips = 0
    for row, (reading, stamp) in enumerate(zip(readings[:200], stamps[:200], strict=True)):
        co2, temperature = (reading * torch.tensor((200, 3)) + torch.tensor((500, 22))).tolist()
        response = stream.observe(dict(timestamp=int(stamp), co2_ppm=co2, temperature_c=temperature))
        if row < 10:
            continue
        responses.append(response)
        for attempt in response["trace"]:
            if "stage" in attempt:
                calls[f'{attempt["stage"]}:{attempt["first"][0]+1}'] += 1
                if attempt["retry"] is not None:
                    calls[f'{attempt["stage"]}:{attempt["retry"][0]+1}'] += 1
                skips += 3 - 1 - (attempt["retry"] is not None)
            else:
                calls["abandoned_proposal"] += 1

    def summarize(values):
        return dict(p50=percentile(values, .50), p95=percentile(values, .95), count=len(values))

    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    dump(output, dict(room=room, torch=str(torch.__version__), cpu=cpu, threads=1, warmup=10,
                      prediction_ms=summarize([r["prediction_ms"] for r in responses]),
                      feedback_ms=summarize([r["feedback_ms"] for r in responses if "feedback_ms" in r]),
                      combined_ms=summarize([r["prediction_ms"] + r.get("feedback_ms", 0.) for r in responses]),
                      expert_proposals=dict(calls), skipped_slots_in_recorded_stages=skips))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cutoff", type=float, required=True)
    args = parser.parse_args()
    main(args.data, args.checkpoint, args.output, args.cutoff)
