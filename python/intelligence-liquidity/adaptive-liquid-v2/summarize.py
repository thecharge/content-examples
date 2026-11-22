"""Aggregate retained receipts and audit a preregistered run with real rows."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys

import torch

import experiment as e
from policy import ABSTAIN, RETRY, callback, outcome
from verify import transaction


def aggregate(directory):
    rows = defaultdict(list)
    occupancy = []
    for path in sorted(directory.glob("*-aalto.json")):
        name, seed, budget, _ = path.stem.split("-")
        score = json.loads(path.read_text())["summary"]
        fit = json.loads((directory / f"{name}-{seed}-{budget}.json").read_text())
        occupancy.append(dict(fit=fit, score=score))
        for mode in ("base", "0.0", "0.1"):
            if mode not in score:
                continue
            v = score[mode]
            for key in (f"{name}:{mode}", f"{name}:{mode}:{budget}"):
                rows[key].append(dict(
                    fixed=v["original"]["effective_balanced"],
                    calibrated=v["selected"]["effective_balanced"],
                    covered_balanced=v["selected"]["balanced"],
                    accuracy=v["selected"]["effective_accuracy"],
                    false_negatives=v["selected"]["fn"],
                    abstained_positive=v["selected"]["abstain_positive"],
                    coverage=v["selected"]["coverage"],
                    auroc=v["ranking_all_proposals"]["auroc"],
                    average_precision=v["ranking_all_proposals"]["average_precision"],
                    retries=v["retries"], helped=v["retry_help"], hurt=v["retry_hurt"],
                    reward=v["reward_mean"], process_changes=score.get("process_changed_routes", 0),
                    steps=fit["steps"], best_step=fit["best_step"],
                    network_seconds=fit["network_seconds"], policy_seconds=fit["policy_seconds"]))
    assert len(occupancy) == len(e.NAMES) * len(e.SEEDS) * len(e.BUDGETS)
    return dict(aggregates={k: {metric: mean(p[metric] for p in seq if p[metric] is not None)
                                if any(p[metric] is not None for p in seq) else None
                                for metric in seq[0]} for k, seq in rows.items()},
                receipts=occupancy)


@torch.no_grad()
def audit(directory):
    saved = torch.load(directory / "v2-41-100.pt", weights_only=True)
    model = e.build("v2")
    model.load_state_dict(saved["weights"])
    all_windows, _ = e.aalto()
    packed = e.pack(all_windows, saved["stats"])
    forced = transaction(model, packed[0][:64], packed[1][:64])
    receipt = json.loads((directory / "v2-41-100-aalto.json").read_text())
    extracted = {}
    data = e.pack(all_windows, saved["stats"])
    offset = 0
    cutoff = saved["cutoff"]
    for x, gaps, y in e.batches(data):
        logs = {key: [] for key in saved["policies"]}
        results = {key: model(x, gaps, callback(p, logs[key]))
                   for key, p in saved["policies"].items()}
        initial, _, _, _ = model(x, gaps)
        for key, (pred, _, accepted, trace) in results.items():
            observed = receipt["routes"][key]
            assert torch.equal(pred[-1], torch.tensor(observed["logits"][offset:offset + len(x)]))
            assert torch.equal(accepted, torch.tensor(observed["covered"][offset:offset + len(x)]))
        for i in range(len(x)):
            action = [int(row[-1][i]) for row in logs["0.1"]]
            opposite = [int(row[-1][i]) for row in logs["0.0"]]
            pred, _, accepted, trace = results["0.1"]
            attempts = sum(int(t["attempted"][i]) for t in trace if "attempted" in t)
            before = bool((initial[-1][i] >= cutoff) == y[i])
            after = bool((pred[-1][i] >= cutoff) == y[i])
            kinds = []
            if action != opposite:
                kinds.append("process_route_change")
            if attempts:
                kinds.append("revision")
                if accepted[i] and before != after:
                    kinds.append("help" if after else "harm")
            if not accepted[i]:
                kinds.append("abstention")
            for kind in kinds:
                if kind in extracted:
                    continue
                stages = []
                for j, (state, q, quality, selected) in enumerate(logs["0.1"]):
                    t = trace[j + 1]
                    retained = "retained" in t and bool(t["retained"][i])
                    stages.append(dict(stage=j + 2, state=state[i].tolist(),
                                       q=q[i].nan_to_num(neginf=-1e9).tolist(),
                                       process_estimate=float(quality[i]),
                                       action=int(selected[i]), retained=retained,
                                       revised_prior=float(t["revised_prior"][i, -1, 0]) if retained else None,
                                       rechecked_score=float(t["revised_score"][i]) if retained else None))
                classification = float(outcome(pred[-1][i], y[i], cutoff)) if accepted[i] else ABSTAIN
                extracted[kind] = dict(id=all_windows[offset + i]["id"], label=bool(y[i]),
                                       cutoff=cutoff, before=float(initial[-1][i]), after=float(pred[-1][i]),
                                       actions=action, actions_without_process=opposite,
                                       accepted=bool(accepted[i]), stages=stages,
                                       classification_or_abstain=classification, retry_penalty=RETRY * attempts,
                                       total_reward=classification + RETRY * attempts)
        offset += len(x)
    return dict(trained_transaction=forced, traces=extracted)


if __name__ == "__main__":
    torch.set_num_threads(1)
    directory = Path(sys.argv[1])
    report = aggregate(directory)
    report["replay_audit"] = audit(directory)
    (directory / "results.json").write_text(json.dumps(report, indent=2))
    for key, value in report["aggregates"].items():
        if key.count(":") == 1:
            print(key, {k: round(v, 4) for k, v in value.items() if v is not None})
    print(json.dumps(report["replay_audit"]))
