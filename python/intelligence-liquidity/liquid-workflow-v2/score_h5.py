"""Score a previously untouched HPDmobile home after choosing the causal reference."""

import argparse
import json
from pathlib import Path

import torch
import xgboost as xgb

from serve import replay as serve
from experiment import FACTORIES, dump, matrix, metrics, replay, sha
from transfer import h5
from workflow import Blackboard


def score(evidence, source):
    fit = json.loads((evidence/"fit.json").read_text())["models"]
    (room,), hashes = h5(source)
    _, x, y, t = room
    model_path = evidence/"workflow.pt"
    if sha(model_path) != fit["workflow"]["best_sha256"]:
        raise ValueError("Changed workflow checkpoint")
    weight = torch.load(model_path, weights_only=True)
    threshold = fit["workflow"]["cutoff"]
    lines = [json.dumps(dict(time=stamp, co2_ppm=float(point[0])*200+500, witness_c=float(point[1])*3+22))
             for point, stamp in zip(x, t.tolist(), strict=True)]
    report = []
    for variant in ("absolute", "local", "local-live", "no-check", "ltc", "gru", "xgboost"):
        errors = []
        if variant in ("absolute", "local", "local-live", "no-check"):
            model = Blackboard(use_checks=variant != "no-check")
            model.load_state_dict(weight)
            outputs = list(serve(lines, model, threshold, variant == "local-live", variant.startswith("local")))
            p = torch.tensor([item["score"] for item in outputs])
            called = torch.tensor([item["decision"] is not None for item in outputs])
            for index, (previous, current) in enumerate(zip(outputs, outputs[1:]), 1):
                gap = current["time"]-previous["time"]
                if gap <= 450:
                    errors.append(abs(previous["next_co2_ppm"] - (float(x[index, 0])*200+500)))
            best = threshold
        else:
            rec = fit[variant]
            path = evidence/(f"{variant}.pt" if variant != "xgboost" else "xgboost.ubj")
            if sha(path) != rec["best_sha256"]:
                raise ValueError(f"Changed weights for {variant}")
            if variant == "xgboost":
                model = xgb.Booster(model_file=path)
                p = torch.from_numpy(model.predict(xgb.DMatrix(matrix(x[:, :1], t).numpy()),
                                                   iteration_range=(0, rec["rounds"])))
                called = torch.ones_like(y, dtype=torch.bool)
            else:
                model = FACTORIES[variant]()
                model.load_state_dict(torch.load(path, weights_only=True))
                p, _, called, _ = replay(model, room, variant)
            best = rec["cutoff"]
        correct = ((p >= best) == y.bool()) & called
        row = dict(variant=variant, correct=int(correct.sum()), rows=len(y), coverage=float(called.float().mean()),
                   co2_forecast_mae_ppm=sum(errors)/len(errors) if errors else None,
                   **metrics(p, y, called, best))
        report.append(row)
        print(variant, row["correct"], "/", row["rows"], "withheld", row["undecided"], flush=True)
    dump(evidence/"h5.json", dict(fit_sha256=sha(evidence/"fit.json"), files=hashes, results=report,
        input_dates=[int(t.min()),int(t.max())], room=room[0],
        time_note="The source records local wall time. The integer representation preserves the sequence and intervals.",
        code={name: sha(Path(__file__).with_name(name)) for name in ("score_h5.py", "local_reference.py", "serve.py", "transfer.py")}))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("evidence", type=Path)
    args.add_argument("h5", type=Path)
    options = args.parse_args()
    torch.set_num_threads(1)
    score(options.evidence, options.h5)
