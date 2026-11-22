"""Separate the first absolute clamp from learning on the fixed occupancy stream."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from experiment import adapter_for, restore, source, test_sites
from model import sensor_loss


def predict(model, x, stamps, learn):
    origin = (model.adapter_scale.detach().clone(), model.adapter_offset.detach().clone())
    optimizer = adapter_for(model, learn)
    state = prior_input = prior_minutes = prior_state = last = None
    logits = []
    snap_applied = False
    for reading, stamp in zip(x, stamps.tolist(), strict=True):
        reading = reading[None]
        connected = last is not None and 0 < stamp - last <= 450
        minutes = x.new_tensor([(stamp - last) / 60 if connected else 5.])
        previous = state if connected else None
        with torch.no_grad():
            result = model.step(reading, minutes, previous)
            logits.append(result.occupied.item())
        if connected:
            if optimizer is not None:
                optimizer.zero_grad()
                earlier = model.step(prior_input, prior_minutes, prior_state)
                sensor_loss(prior_input, reading, earlier).backward()
                torch.nn.utils.clip_grad_norm_((model.adapter_scale, model.adapter_offset), .5)
                optimizer.step()
                with torch.no_grad():
                    model.adapter_scale.clamp_(origin[0] - .25, origin[0] + .25)
                    model.adapter_offset.clamp_(origin[1] - .5, origin[1] + .5)
            elif not snap_applied:
                with torch.no_grad():
                    model.adapter_scale.clamp_(.75, 1.25)
                    model.adapter_offset.clamp_(-.5, .5)
                snap_applied = True
        prior_input, prior_minutes, prior_state = reading, minutes, previous
        state, last = result.memory, stamp
    return torch.tensor(logits).sigmoid()


def compare(job):
    site, rooms, frozen, name, item = job
    torch.set_num_threads(1)
    return [
        {"site": site, "checkpoint": name, "arm": arm,
         "metrics": source.score(torch.cat([
             predict(restore(frozen, item, name), x[:2016], t[:2016], learn)
             for _, x, _, t in rooms]), torch.cat([y[:2016] for _, _, y, _ in rooms]), item["cutoff"])}
        for arm, learn in (("clamp_only", False), ("centered_learning", True))
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("data", type=Path)
    cli.add_argument("frozen", type=Path)
    args = cli.parse_args()
    raw = (args.frozen / "frozen.json").read_bytes()
    fits = json.loads(raw)["occupancy"]
    jobs = [(site, rooms, args.frozen, name, item)
            for site, rooms in test_sites(args.data).items() for name, item in fits.items()
            if item["mode"] == "full" and item["share"] == 1]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [row for pair in pool.map(compare, jobs) for row in pair]
    print(json.dumps({"freeze": source.sha(raw), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
