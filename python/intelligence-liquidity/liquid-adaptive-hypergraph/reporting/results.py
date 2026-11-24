"""Summarize the two evaluation protocols without mixing their feedback rules."""

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (73, 83, 97)


def summarize(strict=True):
    online = {
        seed: json.loads((ROOT / f"evidence/seed-{seed}/results.json").read_text())
        for seed in SEEDS
    }
    summary = dict(protocols={}, controlled={}, cost={})
    rows = []
    for home in ("H5", "H6"):
        old = [
            online[s]["homes"][home]["ranges"]["after_day_14"]["previous"][
                "balanced_accuracy"
            ]
            for s in SEEDS
        ]
        new = [
            online[s]["homes"][home]["ranges"]["after_day_14"]["adaptive"][
                "balanced_accuracy"
            ]
            for s in SEEDS
        ]
        frozen = [
            online[s]["homes"][home]["ranges"]["after_day_14"]["frozen"][
                "balanced_accuracy"
            ]
            for s in SEEDS
        ]
        rows.append(
            dict(
                home=home,
                previous=old,
                new=new,
                original_frozen=frozen,
                previous_mean=statistics.mean(old),
                new_mean=statistics.mean(new),
                original_frozen_mean=statistics.mean(frozen),
            )
        )
    summary["protocols"]["online"] = rows
    if strict:
        rows = []
        for home in ("H5", "H6"):
            old, new = [], []
            for seed in SEEDS:
                a = json.loads(
                    (
                        ROOT / f"evidence/strict/previous-seed-{seed}/results.json"
                    ).read_text()
                )
                b = json.loads(
                    (ROOT / f"evidence/strict/new-seed-{seed}/results.json").read_text()
                )
                if (
                    a["homes"][home]["training_rows"]
                    != b["homes"][home]["training_rows"]
                ):
                    raise ValueError("Strict training cutoffs differ")
                for record in (a, b):
                    item = record["homes"][home]
                    if (
                        item["weights_before"] != item["weights_after"]
                        or item["labels_supplied_during_test"] != 0
                    ):
                        raise ValueError("Strict protocol was not frozen")
                old.append(a["homes"][home]["metrics"]["balanced_accuracy"])
                new.append(b["homes"][home]["metrics"]["balanced_accuracy"])
            rows.append(
                dict(
                    home=home,
                    previous=old,
                    new=new,
                    previous_mean=statistics.mean(old),
                    new_mean=statistics.mean(new),
                )
            )
        summary["protocols"]["strict_frozen"] = rows
    receipts = [
        r
        for seed in SEEDS
        for h in online[seed]["homes"].values()
        for r in h["adaptive_receipts"]
    ]
    summary["online_decisions"] = dict(
        completed=len(receipts),
        accepted=sum(bool(r["accepted"]) for r in receipts),
        by_action={
            name: sum(r.get("action") == name for r in receipts)
            for name in sorted(set(r.get("action", "unsupported") for r in receipts))
        },
        final_profiles={
            f"seed-{seed}/{home}": item["final_profile"]
            for seed in SEEDS
            for home, item in online[seed]["homes"].items()
        },
    )
    record = json.loads((ROOT / "evidence/evolution/evolution.json").read_text())
    receipt = record["receipts"][0]
    summary["controlled"] = dict(
        action=receipt["action"],
        accepted=receipt["accepted"],
        balanced_gain=receipt["balanced_gain"],
        brier_gain=receipt["brier_gain"],
        monitor_rollback=record["monitor"][0]["rollback"],
        ledger_revision=record["ledger_revision"],
        ledger_bytes=record["ledger_bytes"],
        intervention=record["intervention"],
        profiles=[
            dict(stage=s["label"], **s["model"]["profile"])
            for s in record["snapshots"][:4]
        ],
    )
    old_record = json.loads(
        (ROOT.parent / "liquid-adaptable-net/evidence/seed-73/results.json").read_text()
    )
    before = old_record["homes"]["H5"]["final_profile"]
    after = online[73]["homes"]["H5"]["final_profile"]
    summary["cost"] = dict(
        previous_parameters=before["parameters"],
        new_parameters=after["parameters"],
        parameter_increase_fraction=after["parameters"] / before["parameters"] - 1,
        previous_work=before["work_units"],
        new_work=after["work_units"],
        note="Operation proxy is not latency. Runs shared the host; SQLite durability excluded from neural benchmark.",
    )
    return summary


def write(summary):
    output = ROOT / "reporting"
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = list(summary["protocols"])
    fig, axes = plt.subplots(
        1, len(protocols), figsize=(6 * len(protocols), 4), squeeze=False
    )
    for axis, name in zip(axes[0], protocols):
        rows = summary["protocols"][name]
        x = list(range(len(rows)))
        axis.bar(
            [v - 0.18 for v in x],
            [r["previous_mean"] * 100 for r in rows],
            width=0.36,
            label="previous",
        )
        axis.bar(
            [v + 0.18 for v in x],
            [r["new_mean"] * 100 for r in rows],
            width=0.36,
            label="hypergraph",
        )
        axis.set_xticks(x, [r["home"] for r in rows])
        axis.set_ylim(0, 100)
        axis.set_ylabel("Mean balanced accuracy (%)")
        axis.set_title(
            "Online delayed-label adaptation"
            if name == "online"
            else "Frozen inference after completed training"
        )
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy.svg")
    plt.close(fig)
    # All timing arrays remain in raw records; do not present shared-host timings as a speedup.
    lines = [
        "# Measured results",
        "",
        "The new experiment adds overlapping liquid memberships and durable revisions.",
        "The following protocols answer different questions. Their scores are kept separate.",
        "",
        "## Online adaptation on the original streams",
        "",
        "Every prediction precedes its label. Later labels can update the model before subsequent predictions.",
        "The same twelve HPDmobile archives, original seed checkpoints, features and chronology were checked.",
        "The preceding model's predictions were reused after source/artifact fingerprint verification.",
        "",
        "| Room | Previous, seeds 73 / 83 / 97 | Hypergraph, seeds 73 / 83 / 97 | Mean previous / new |",
        "| --- | --- | --- | --- |",
    ]

    def values(items):
        return " / ".join(f"{v * 100:.2f}%" for v in items)

    for row in summary["protocols"]["online"]:
        lines.append(
            f"| {row['home']} | {values(row['previous'])} | {values(row['new'])} | {row['previous_mean'] * 100:.2f}% / {row['new_mean'] * 100:.2f}% |"
        )
    decision = summary["online_decisions"]
    lines += [
        "",
        f"Completed structural/weight proposals: {decision['completed']}. Accepted: {decision['accepted']}.",
        "All six final models retained five experts, one board, two attention heads and zero computational hyperedges.",
        "Seed 73 in H6 accepted a neuron contraction: 34,754 parameters and 70,492 work units at the end.",
        "Its later balanced accuracy rose from 62.54% to 62.78%. The other five comparisons matched the preceding model.",
        "This dataset run demonstrates one accepted contraction, but no useful learned hyperedge growth.",
        "The controlled run below establishes the new mechanism separately.",
        "",
    ]
    if "strict_frozen" in summary["protocols"]:
        lines += [
            "## Completed training, then unseen inference",
            "",
            "Both models were trained separately on the first 14 selected days of each room.",
            "The later chronological block was held out. No label or reward from that block entered the model.",
            "Learned tensors and topology were frozen and verified unchanged after inference.",
            "Only recurrent/delta state could evolve from sensor readings. A still-monitored candidate used its retained fallback.",
            "",
            "| Room | Previous, seeds 73 / 83 / 97 | Hypergraph, seeds 73 / 83 / 97 | Mean previous / new |",
            "| --- | --- | --- | --- |",
        ]
        for row in summary["protocols"]["strict_frozen"]:
            lines.append(
                f"| {row['home']} | {values(row['previous'])} | {values(row['new'])} | {row['previous_mean'] * 100:.2f}% / {row['new_mean'] * 100:.2f}% |"
            )
        lines += [
            "",
            "These are unseen future observations in rooms with an earlier adaptation prefix.",
            "They are not entirely unseen rooms. Raw training/test timestamps and both frozen snapshots are retained under `evidence/strict`.",
            "",
        ]
    cost = summary["cost"]
    lines += [
        "## Cost",
        "",
        f"The initial model and five final models use {cost['new_parameters']:,} parameters versus {cost['previous_parameters']:,} previously: {cost['parameter_increase_fraction'] * 100:.2f}% more.",
        f"The operation proxy remains {cost['new_work']:,} units for those unchanged serving topologies; the accepted contraction reduces it to 70,492.",
        "The additional parameters belong to the larger action policy. This is added flexibility with a parameter cost, not a measured speedup.",
        "Neural benchmark timings exclude SQLite persistence and were measured while independent seed runs shared the host.",
        "Do not use them as an isolated old/new latency comparison.",
        "",
        "## Controlled hyperedge lifecycle",
        "",
        "The controlled cue directly determines the class. The action-value bias explicitly requests a connection.",
        "The first 48 labels fit; the following 48 predictions check; another 48 monitor.",
        f"The connection was accepted with balanced-accuracy gain {summary['controlled']['balanced_gain']:.3f} and Brier gain {summary['controlled']['brier_gain']:.6f}.",
        f"Monitoring rollback: {summary['controlled']['monitor_rollback']}. The ledger retains {summary['controlled']['ledger_revision']} committed revisions.",
        "The serving model changed from zero to one hyperedge. Its fitted schema also included the new cue.",
        "This is a disclosed mechanism test. It does not show that the hyperedge was necessary or selected autonomously.",
        "The final three pictures explicitly construct overlapping neuron memberships, edge splitting and edge merging.",
        "They are excluded from counts of learned mutations.",
        "",
        "## Inspect the evidence",
        "",
        "- [Accuracy chart](reporting/accuracy.svg) and [machine-readable summary](reporting/summary.json).",
        "- [Controlled evolution](evidence/evolution/index.html), [prediction trend](evidence/evolution/trend.svg), and full stage JSON files.",
        "- Online records: [seed 73](evidence/seed-73/results.json), [seed 83](evidence/seed-83/results.json), [seed 97](evidence/seed-97/results.json).",
        "- Strict inference records and snapshots: `evidence/strict/{new,previous}-seed-{73,83,97}`.",
        "- [Verification instructions](TESTING.md) and [full vision mapping](VISION.md).",
        "",
        "## Limits of these results",
        "",
        "H5 contributes only 288 later observations, including 12 vacant observations; H6 contributes 2,016 with 1,030 vacant and 986 occupied.",
        "The three seeds reuse the same two rooms. They are not three independent deployments.",
        "No result guarantees future accuracy, useful growth on every stream, or AMD GPU performance.",
        "No GitHub Actions run was used.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-only", action="store_true")
    args = parser.parse_args()
    write(summarize(not args.online_only))
