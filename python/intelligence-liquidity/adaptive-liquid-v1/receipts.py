"""Verify and extract frozen experiment receipts without writing files."""
import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(path):
    payload = json.loads(path.read_text())
    compressed = b"".join(base64.b64decode(part, validate=True)
                          for part in payload["gzip_base64_chunks"])
    assert len(compressed) == payload["gzip_bytes"], path
    assert hashlib.sha256(compressed).hexdigest() == payload["gzip_sha256"], path
    run = json.loads(gzip.decompress(compressed))
    for field in ("task", "model", "seed", "budget", "config", "parameters",
                  "epochs", "updates", "fit_windows", "validation_windows"):
        assert run[field] == payload["summary"][field], (path, field)
    return run


def review():
    freeze = json.loads((ROOT / "freeze.json").read_text())
    for name, value in freeze["adaptive_files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value, name
    for name, value in freeze["files"].items():
        path = ROOT.parent / "attention-and-reward" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value, path

    seeds, budgets, variants = freeze["seeds"], freeze["budgets"], freeze["variants"]
    occ = freeze["occupancy_models"] + variants
    heat = freeze["heating_models"] + variants
    expected = {("occupancy", m, s, b) for m, s, b in product(occ, seeds, budgets)}
    expected |= {("heat", m, s, None) for m, s in product(heat, seeds)}
    seen = set()
    groups, epochs, fit_times, test_times = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    early, match = Counter(), {}
    for path in sorted(ROOT.glob("receipts/*.json")):
        run = read(path)
        task, model, seed, budget = (run[k] for k in ("task", "model", "seed", "budget"))
        key = task, model, seed, budget
        assert key not in seen and key in expected, key
        seen.add(key)
        assert run["config"] == freeze["adaptive_files"], key
        if model != "mean":
            assert run["selected_epoch"] == 1 + min(range(run["epochs"]), key=run["validation_loss"].__getitem__)
            assert run["updates"] == run["epochs"] * ((run["fit_windows"] + freeze["batch"] - 1) // freeze["batch"])
            assert run["reload_exact"]
            early[task] += run["epochs"] < freeze["max_epochs"]
        epoch_key = task, model, budget
        epochs[epoch_key].append([seed, run.get("selected_epoch"), run["epochs"], run["updates"]])
        fit_times[epoch_key].append(run["fit_seconds"])
        if task == "occupancy":
            for site, scored in run["tests"].items():
                all_rows = scored["all"]
                accepted = scored["accepted"]
                joined = scored.get("join")
                membership = all_rows["n"], all_rows["tp"] + all_rows["fn"], joined
                anchor = "occupancy", site
                assert membership == match.setdefault(anchor, membership), key
                assert abs(scored["coverage"] - accepted["n"] / all_rows["n"]) < 1e-6, key
                replay = sum(stage.get("changed_and_replayed", 0) for stage in scored["routes"])
                test_times[(*epoch_key, site)].append(scored["seconds"])
                groups[(*epoch_key, site)].append(
                    [seed, run["parameters"], all_rows["accuracy"], all_rows["balanced_accuracy"],
                     scored["coverage"], accepted["balanced_accuracy"], replay])
        else:
            test = run["test"]
            records = test["records"]
            membership = [(r["id"], r["logged"], r["observed"], r["support"]) for r in records]
            assert membership == match.setdefault("heat", membership), key
            assert all(r["hard"] or r["action"] is None for r in records), key
            supported = [r for r in records if r["hard"]]
            picked = [r for r in supported if r["action"] is not None]
            learned_changed_rank = sum(r["action"] != max(r["rank"], key=r["physical_reward"].__getitem__)
                                       for r in picked)
            veto = sum(any(not v["pass_after"] for v in stage)
                       for r in supported for stage in r["stages"])
            replays = sum(v["revise"] and v["upstream_change"] > 0 and v["replay_change"] > 0
                          for r in records for stage in r["stages"] for v in stage)
            test_times[(*epoch_key, "logged")].append(test["seconds"])
            groups[(*epoch_key, "logged")].append([seed, run["parameters"], *test["mae"], len(picked),
                                                    learned_changed_rank, veto, replays,
                                                    test["picks"].get("0", 0)])
    columns = dict(
        occupancy=["seed", "parameters", "accuracy", "balanced_accuracy", "coverage",
                   "accepted_balanced_accuracy", "stage_replays"],
        heat=["seed", "parameters", "temperature_mae_c", "heat_mae_kwh",
              "discomfort_mae_degree_hours", "physical_reward_mae", "selected_rows",
              "learned_bonus_rank_flips", "critic_vetoed_supported_candidates",
              "stage_replays", "off_picks"],
        training=["seed", "selected_epoch", "stop_epoch", "optimizer_updates"],
    )
    return dict(done=len(seen), expected=len(expected), columns=columns,
                missing=sorted(expected - seen), early_stops=early,
                metrics=[dict(task=t, model=m, budget=b, site=s, seeds=values,
                              mean_test_seconds=mean(test_times[t, m, b, s]))
                         for (t, m, b, s), values in groups.items()],
                training=[dict(task=t, model=m, budget=b, seeds=values,
                               mean_fit_seconds=mean(fit_times[t, m, b]))
                          for (t, m, b), values in epochs.items()])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("extract", "review"))
    cli.add_argument("path", nargs="?", type=Path)
    args = cli.parse_args()
    if args.mode == "extract":
        if args.path is None:
            cli.error("extract requires a receipt path")
        result = read(args.path)
    else:
        result = review()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
