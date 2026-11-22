"""Measure separability and expert use on the already scored Aalborg rooms."""
import argparse
import io
import json
import math
from pathlib import Path

import torch

import benchmark
import model


def best_with_labels(logits, occupied):
    """Find the best cutoff with hindsight. This is never a deployment cutoff."""
    order = logits.argsort(descending=True)
    logits, occupied = logits[order], occupied[order]
    end = torch.cat((logits[:-1] != logits[1:], torch.tensor([True])))
    n_positive = occupied.sum()
    n_negative = (~occupied).sum()
    true_positive_rate = occupied.cumsum(0)[end].double() / n_positive
    false_positive_rate = (~occupied).cumsum(0)[end].double() / n_negative
    balanced = (1 + true_positive_rate - false_positive_rate) / 2
    index = int(balanced.argmax())
    return balanced[index].item(), logits[end][index].item()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    frozen = benchmark.verified(args.out / "frozen.json", args.frozen_sha256)
    fit, manifest, weights = benchmark.audit(args.out, args.data)
    if json.loads(frozen) != manifest or not (args.out / "scores.json").is_file():
        raise ValueError("Expected a previously scored, frozen experiment")
    mean, scale = torch.tensor(fit["mean"]), torch.tensor(fit["scale"])
    datasets = [benchmark.normalize(benchmark.windows(rows), mean, scale)
                for rows in benchmark.external_rooms(args.data).values()]
    results = []
    for row in fit["runs"]:
        net = model.build(row["arm"])
        net.load_state_dict(torch.load(io.BytesIO(weights[row["file"]]), weights_only=True))
        net.eval()
        predictions, labels = [], []
        counts = torch.zeros(model.EXPERTS, dtype=torch.long)
        for x, clock, y in datasets:
            for xs, ts, target in zip(x.split(benchmark.BATCH),
                                      clock.split(benchmark.BATCH),
                                      y.split(benchmark.BATCH), strict=True):
                output = net(xs, ts)
                measured = torch.isfinite(target)
                predictions.append(output["logits"][measured])
                labels.append(target[measured] == 1)
                counts += output["expert_counts"]
        logits, target = torch.cat(predictions), torch.cat(labels)
        oracle, cutoff = best_with_labels(logits, target)
        fixed = benchmark.metrics(logits, torch.ones_like(target),
                                  torch.zeros_like(target), target.float(), 0)
        counts = counts.tolist()
        results.append({"arm": row["arm"], "seed": row["seed"],
                        "fixed_zero_balanced": fixed["all_window_balanced_accuracy"],
                        "best_balanced_using_test_labels": oracle,
                        "cutoff_using_test_labels": cutoff if math.isfinite(cutoff) else None,
                        "ordinary_expert_calls_by_index": counts})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
