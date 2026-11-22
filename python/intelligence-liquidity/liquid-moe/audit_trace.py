"""Inspect frozen learned review decisions and their outcomes."""
import argparse
import io
import json
from pathlib import Path

import torch

import benchmark
from model import build, review


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
    if json.loads(frozen) != manifest:
        raise ValueError("Frozen manifest does not match fit")
    rooms = benchmark.external_rooms(args.data)
    mean = torch.tensor(fit["mean"])
    scale = torch.tensor(fit["scale"])
    report = []
    for item in fit["runs"]:
        if item["arm"] != "moe_delta":
            continue
        net = build("moe_delta")
        net.load_state_dict(torch.load(io.BytesIO(weights[item["file"]]), weights_only=True))
        net.eval()
        cutoff = item["cutoff"]
        probability_cutoff = torch.sigmoid(torch.tensor(cutoff)).item()
        totals = dict(windows=0, attempted=0, committed=0, rolled_back=0,
                      abstained=0, changed_label=0, helped=0, hurt=0)
        samples = {}
        for room, raw in rooms.items():
            x, clock, labels = benchmark.normalize(benchmark.windows(raw), mean, scale)
            for start in range(0, len(x), benchmark.BATCH):
                end = start + benchmark.BATCH
                out = review(net, x[start:end], clock[start:end], probability_cutoff)
                attempted = out["attempted"]
                accepted = out["accepted"]
                committed = out["revision_accepted"]
                initial_label = out["ordinary_logits"] >= cutoff
                final_label = out["logits"] >= cutoff
                changed = committed & (initial_label != final_label)
                labeled = torch.isfinite(labels[start:end])
                target = labels[start:end] == 1
                assert (out["revision_count"] <= 1).all()
                assert (out["challenger_expert"][attempted] != out["initial_expert"][attempted]).all()
                assert torch.equal(out["logits"][out["revision_rejected"]],
                                   out["ordinary_logits"][out["revision_rejected"]])
                assert (out["budget_remaining"] == 1 - attempted.long()).all()
                assert torch.equal((out["logits"][accepted] >= cutoff).long(),
                                   out["decision"][accepted])
                for key, mask in {
                    "windows": torch.ones_like(accepted),
                    "attempted": attempted,
                    "committed": committed,
                    "rolled_back": out["revision_rejected"],
                    "abstained": ~accepted,
                    "changed_label": changed,
                    "helped": changed & labeled & (final_label == target),
                    "hurt": changed & labeled & (final_label != target),
                }.items():
                    totals[key] += int(mask.sum())
                for action in range(4):
                    name = ("kept", "committed", "rolled_back", "abstained")[action]
                    if name in samples:
                        continue
                    selected = torch.where(out["action"] == action)[0]
                    if not len(selected):
                        continue
                    index = int(selected[0])
                    row = {"room": room, "window_index": start + index,
                           "actual_occupied": bool(target[index]) if labeled[index] else None}
                    for key in ("initial_expert", "challenger_expert", "returned_expert",
                                "initial_logits", "challenger_logits", "logits",
                                "initial_check", "challenger_check", "process_gain",
                                "revision_gain", "budget_remaining", "decision"):
                        value = out[key][index]
                        row[key] = value.item() if torch.isfinite(value) else None
                    samples[name] = row
        report.append({"seed": item["seed"], "logit_cutoff": cutoff,
                       "totals": totals, "first_example_per_action": samples})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
