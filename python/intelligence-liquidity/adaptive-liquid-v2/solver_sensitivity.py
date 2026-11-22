"""Change only the inference integration step on frozen V2 checkpoints."""
from statistics import mean

import torch

from adaptive import Liquid
import experiment


@torch.no_grad()
def measure(work, checkpoint_hashes):
    windows, membership = experiment.aalto()
    files = sorted(work.glob("v2-*.pt"))
    if len(files) != 9:
        raise ValueError(f"Expected nine V2 checkpoints, found {len(files)}.")
    runs = {}
    variants = [(True, 3), (True, 6), (True, 12), (False, 3), (False, 12)]
    for path in files:
        if experiment.digest(path) != checkpoint_hashes[path.name]:
            raise ValueError(f"Changed checkpoint: {path.name}")
        saved = torch.load(path, weights_only=True)
        model = experiment.build("v2")
        model.load_state_dict(saved["weights"])
        model.eval()
        data = experiment.pack(windows, saved["stats"])
        labels, cutoff = data[2], saved["cutoff"]
        cells = [module for module in model.modules() if isinstance(module, Liquid)]
        results = {}
        reference = None
        for exponential, substeps in variants:
            for cell in cells:
                cell.exponential = exponential
                cell.substeps = substeps
            logits, covered, _, _ = experiment.predict(model, data)
            if reference is None:
                reference = logits
            name = "exponential" if exponential else "fused"
            results[f"{name}-{substeps}"] = {
                "balanced": experiment.metrics(logits, labels, covered, cutoff)["effective_balanced"],
                "auroc": experiment.ranking(logits, labels)["auroc"],
                "decision_flips_from_trained_solver": int(((logits >= cutoff) != (reference >= cutoff)).sum()),
                "mean_logit_delta_from_trained_solver": float((logits - reference).abs().mean()),
            }
        runs[path.stem] = results
    averages = {}
    for variant, row in next(iter(runs.values())).items():
        averages[variant] = {key: mean(run[variant][key] for run in runs.values()) for key in row}
    return {"holdout": membership, "runs": runs, "mean": averages}
