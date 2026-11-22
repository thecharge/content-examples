"""Turn the repair gate off in an already fitted model and compare its decisions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from experiment import restore, source, stream, test_sites


def compare(job):
    site, rooms, frozen, name, item = job
    torch.set_num_threads(1)
    results = {}
    for room, readings, labels, timestamps in rooms:
        readings, labels, timestamps = readings[:2016], labels[:2016], timestamps[:2016]
        cutoff = item["cutoff"]
        model = restore(frozen, item, name)
        reviewed, diagnostic = stream(model, readings, timestamps)
        model.use_falsifier = False
        unchecked, _ = stream(model, readings, timestamps)
        with_review = (reviewed >= cutoff)
        without_review = (unchecked >= cutoff)
        correct = labels.bool()
        changed = with_review != without_review
        helped = changed & (with_review == correct)
        hurt = changed & (without_review == correct)
        example = {}
        for label, mask in (("helped", helped), ("hurt", hurt)):
            indices = mask.nonzero()
            if len(indices):
                i = int(indices[0, 0])
                example[label] = {"index": i, "label": int(labels[i]),
                                  "without_repair": float(unchecked[i]), "with_repair": float(reviewed[i])}
        results[room] = {
            "cutoff": cutoff, "labels": len(labels),
            "with_repair": source.score(reviewed, labels, cutoff),
            "without_repair": source.score(unchecked, labels, cutoff),
            "helped": int(helped.sum()), "hurt": int(hurt.sum()),
            "mean_repair_weight_by_layer": diagnostic["mean_revision_gate"],
            "examples": example,
        }
    return site, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("frozen", type=Path)
    args = parser.parse_args()
    raw = (args.frozen / "frozen.json").read_bytes()
    name = "occupancy-full-41.pt"
    item = json.loads(raw)["occupancy"][name]
    jobs = [(site, rooms, args.frozen, name, item) for site, rooms in test_sites(args.data).items()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        result = dict(pool.map(compare, jobs))
    print(json.dumps({"freeze": source.sha(raw), "checkpoint": name, "sites": result}, indent=2))


if __name__ == "__main__":
    main()
