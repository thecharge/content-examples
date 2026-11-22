"""Measure revision capacity with a label-aware oracle, for diagnosis only."""
from itertools import product
from statistics import mean

import torch
import experiment
from policy import replay

ROUTES = list(product((0, 1), repeat=2))


@torch.no_grad()
def evaluate(path, windows, checkpoint_digest):
    if experiment.digest(path) != checkpoint_digest:
        raise ValueError(f"Changed checkpoint: {path.name}")
    saved = torch.load(path, weights_only=True)
    model = experiment.build(path.stem.split("-")[0])
    model.load_state_dict(saved["weights"])
    model.eval()
    dataset = experiment.pack(windows, saved["stats"])
    labels = dataset[2]
    cutoff = saved["cutoff"]
    correct, routes_changed = [], []
    for inputs, clock, actual in experiment.batches(dataset):
        candidates = [
            replay(model, inputs, clock, route) for route in ROUTES
        ]
        guess = torch.stack([candidate[1][-1] >= cutoff for candidate in candidates])
        accepted = torch.stack([candidate[3] for candidate in candidates])
        correct.append((guess == actual) & accepted)
        routes_changed.append(torch.any(guess != guess[0], dim=0))
    correct = torch.cat(correct, dim=1)
    has_correct = correct.any(dim=0)
    gained = ~correct[0] & has_correct
    return {
        "direct": float((correct[0][labels].float().mean() + correct[0][~labels].float().mean()) / 2),
        "oracle": float((has_correct[labels].float().mean() + has_correct[~labels].float().mean()) / 2),
        "possible_error_fixes": int(gained.sum()),
        "potential_miss_fixes": int((gained & labels).sum()),
        "potential_false_alarm_fixes": int((gained & ~labels).sum()),
        "decisions_that_could_flip": int(torch.cat(routes_changed).sum()),
    }


def measure(work, checkpoint_hashes):
    windows, membership = experiment.aalto()
    results = {}
    averages = {}
    for model in ("v2", "no_delta"):
        runs = {}
        for path in sorted(work.glob(f"{model}-*.pt")):
            runs[path.stem] = evaluate(path, windows, checkpoint_hashes[path.name])
        if len(runs) != 9:
            raise ValueError(f"Expected nine {model} checkpoints, found {len(runs)}")
        results.update(runs)
        averages[model] = {key: mean(row[key] for row in runs.values()) for key in next(iter(runs.values()))}
    return {"holdout": membership, "runs": results, "mean": averages}
