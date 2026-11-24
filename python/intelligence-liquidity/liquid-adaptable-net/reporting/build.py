"""Verify evidence and build the comparison report and standalone trend figures."""

import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import verify_evaluation
from reference import sha
from runtime import Runtime
from verify import check_trial

SEEDS = (73, 83, 97)
records = {}
for seed in SEEDS:
    folder = ROOT / f"evidence/seed-{seed}"
    verify_evaluation(folder)
    records[seed] = json.loads((folder / "results.json").read_text())
controlled = []
for seed in (60, 62, 65, 88):
    path = ROOT / f"evidence/controlled-{seed}.json"
    record = json.loads(path.read_text())
    for name, expected in record["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert sha(path.with_suffix(".pt")) == record["checkpoint_sha256"]
    model = Runtime.load(path.with_suffix(".pt"))
    assert json.loads(json.dumps(model.graph.layout())) == record["final_layout"]
    assert model.fallback is None, "Controlled monitoring unfinished"
    assert model.monitor_audit and all(
        x["supported"] and not x["rollback"] for x in model.monitor_audit
    ), model.monitor_audit
    for receipt in record["receipts"]:
        check_trial(receipt, record["settings"]["complexity_rate"])
    controlled.append(record)

from evolution import SEEDS as EVOLUTION_SEEDS, validate_trace

for seed in EVOLUTION_SEEDS:
    validate_trace(json.loads((ROOT / f"evolution/trace-{seed}.json").read_text()))

viewer_check = json.loads((ROOT / "evolution/inspector-check.json").read_text())
for field, path in (
    ("viewer_sha256", "evolution/inspector.html"),
    ("checker_sha256", "reporting/check_inspector.py"),
    ("screenshot_sha256", "evolution/inspector-preview.png"),
):
    if sha(ROOT / path) != viewer_check[field]:
        raise ValueError("Inspector check artifact changed: " + path)
if viewer_check["status"] != "passed" or viewer_check["checks"] != 116:
    raise ValueError("Inspector checks did not pass")

lines = [
    "# Measured results",
    "",
    "This is a capability extension, not a demonstrated accuracy or efficiency improvement over the preceding adaptive model. The default controller declined structural changes on both house streams in all three seeds. It continued learning weights and liquid state.",
    "",
    "[Inspect before/during/after topology, lifecycle figures and the offline interactive viewer](EVOLUTION.md). The accuracy trend below is committed as both PNG and SVG.",
    "",
    "## Same-data accuracy",
    "",
    "Balanced accuracy after the original 14-day warmup, in percent. Every value is recomputed from committed predictions. Seeds are 73, 83 and 97; these are repeats, not independent houses.",
    "",
    "| House / seed | Frozen original | Original contextual | Previous adaptive | New adaptable |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for home in ("H5", "H6"):
    for seed in SEEDS:
        scores = records[seed]["homes"][home]["ranges"]["after_day_14"]
        lines.append(
            "| "
            + home
            + " / "
            + str(seed)
            + " | "
            + " | ".join(
                f"{100*scores[v]['balanced_accuracy']:.2f}"
                for v in ("frozen", "contextual", "previous", "adaptive")
            )
            + " |"
        )
    means = {
        v: statistics.mean(
            records[s]["homes"][home]["ranges"]["after_day_14"][v]["balanced_accuracy"]
            for s in SEEDS
        )
        for v in ("frozen", "contextual", "previous", "adaptive")
    }
    lines.append(
        "| "
        + home
        + " mean | "
        + " | ".join(f"{100*x:.2f}" for x in means.values())
        + " |"
    )
lines += [
    "",
    "H5 has only 288 later observations, including 12 vacant examples. H6 has 2,016 later observations. H5 is therefore weak evidence for generalization. The saved receipts also include the original first-week scores, per-class counts, Brier scores and paired trial evidence.",
    "",
    "![Rolling balanced accuracy](trend.png)",
    "",
    "Each curve uses the preceding 576 observations (up to two days), requires at least 16 examples of each class, and is sampled every 48 observations. The line is the seed mean; the band is the seed range. Missing class support produces gaps. Elapsed days include recording gaps. This is descriptive, not a statistical confidence band.",
    "",
    "The previous H6 seed-97 run accepted a `learn` update with class-recall gains of 0 and 5.43 points but a quorum change of -1.04 points on its checking block. This experiment defaults to zero permitted quorum loss, so it cannot accept that update under the new gate. Its later balanced accuracy is 62.64% versus 63.08% previously. The tighter acceptance condition has a measurable tradeoff; it does not prove improved future accuracy.",
    "",
    "## What the controller actually did",
    "",
    "| House / seed | Completed proposals | Adopted proposals | Monitor events | Final experts / boards / attention nodes / heads | Parameters | Work proxy |",
    "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
]
for home in ("H5", "H6"):
    for seed in SEEDS:
        item = records[seed]["homes"][home]
        p = item["final_profile"]
        lines.append(
            f"| {home} / {seed} | {len(item['adaptive_receipts'])} | {sum(bool(r['accepted']) for r in item['adaptive_receipts'])} | {len(item['monitor_audit'])} | {p['experts']} / {p['boards']} / {p['attention_nodes']} / {p['heads']} | {p['parameters']} | {p['work_units']} |"
        )
lines += [
    "",
    "An unfinished proposal is retained in the checkpoint and excluded from completed/adopted counts. The last prediction has no supplied later answer. No house evidence justifies claiming that extra topology helps this dataset.",
    "",
    "## Controlled mechanism evidence",
    "",
    "These hand-selected mechanism seeds use a perfectly informative new input, not the house dataset. Each candidate is trained on 48 labels, checked on the next 48 predictions and monitored on a further 48. All four completed monitoring and remained adopted. Candidate balanced accuracy rises from 50% to 100% on the checking block. This does not establish that added structure is necessary; ordinary weight learning is also involved.",
    "",
    "| Seed | Selected action | Balanced checking gain | Final topology |",
    "| --- | --- | ---: | --- |",
]
for r in controlled:
    c = r["final_layout"]["cores"]["b0"]
    receipt = r["receipts"][0]
    lines.append(
        f"| {r['seed']} | {receipt['action']} | {100*receipt['balanced_gain']:.1f} points | {c['experts']} experts, {c['depth']} liquid stages, recurrent widths {', '.join(str(len(x['units'])) for x in c['liquid'])} |"
    )
lines += [
    "",
    "Saved models and detailed raw paired predictions are in `evidence/controlled-*.pt` and `.json`. Unit tests separately exercise contraction, shared-network forks, head bounds, routing, rollback and allocation rejection.",
    "",
    "## CPU costs",
    "",
    "The new starting model contains 34,749 parameters versus 32,664 previously: 6.38% more. Additional policy actions, liquid readouts and gates have a real cost even when no topology change is accepted. The deterministic work score is a control proxy, not measured FLOPs or wall time.",
    "",
    "Isolated forward timing uses the same 160 H5 observations for each saved model, one warmup and three measured passes, one PyTorch CPU thread. Models are measured in separate worker processes after the comparison jobs finish. Timing excludes input packing, learning, checkpoint loading and state detachment. RSS includes imports and loading. Timings describe this host and run, not a portable speed guarantee.",
    "",
    "| House / seed | Previous median ms | New median ms | Previous p95 ms | New p95 ms |",
    "| --- | ---: | ---: | ---: | ---: |",
]
for home in ("H5", "H6"):
    for seed in SEEDS:
        paths = [
            ROOT / f"reporting/profile-{kind}-{seed}-{home}.json"
            for kind in ("old", "new")
        ]
        if all(p.exists() for p in paths):
            old, new = [json.loads(p.read_text()) for p in paths]
            for kind, profile in (("old", old), ("new", new)):
                source = (
                    ROOT.parent / "adaptive-liquid-board" if kind == "old" else ROOT
                )
                assert profile["checkpoint_sha256"] == sha(
                    source / f"evidence/seed-{seed}/{home}-adaptive.pt"
                )
                assert profile["inputs_sha256"] == sha(ROOT / "reporting/inputs.json")
                assert profile["profiler_sha256"] == sha(
                    ROOT / "reporting/profile_model.py"
                )
                assert profile["threads"] == 1 and profile["measured_passes"] == 3
            lines.append(
                f"| {home} / {seed} | {old['median_ms']:.3f} | {new['median_ms']:.3f} | {old['p95_ms']:.3f} | {new['p95_ms']:.3f} |"
            )
        else:
            raise ValueError("Run all isolated profiles first")
lines += [
    "",
    "Whole-stream prediction and feedback timings are saved separately in each comparison receipt. Those concurrent experiment timings include controller work and are not used as a clean old/new latency comparison.",
    "",
    "## Verification and limits",
    "",
    "The new 57-test suite, previous 36-test suite and original 20-test suite pass locally. Verification checks source and artifact hashes, all twelve original archive hashes, identical timestamps/truths, recomputed scores and trial decisions, and full checkpoint layout restoration. Controlled checkpoints are restored and their monitoring audits checked. See [TESTING.md](TESTING.md).",
    "",
    "The worker bounds address space (4 GiB), CPU time (7,200 seconds), wall time (7,500 seconds) and individual file size (256 MiB). Model allocation estimates and OS process ceilings serve different purposes. Allocation refusal and cancellation cleanup are tested. No GPU or GitHub Actions was used.",
    "",
    "Finite prospective checks and a later rollback window cannot prove zero future degradation under arbitrary drift. Labels are required for accuracy gates; external reward is additional contextual-bandit credit. This experiment does not implement a general actor-critic environment trainer.",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False, layout="constrained")
for ax, home in zip(axes, ("H5", "H6")):
    allrows = {
        seed: json.loads(
            (ROOT / f"evidence/seed-{seed}/{home}-predictions.json").read_text()
        )
        for seed in SEEDS
    }
    for variant, label, color in (
        ("frozen", "Frozen", "#777777"),
        ("contextual", "Original contextual", "#d99400"),
        ("previous", "Previous adaptive", "#3175bc"),
        ("adaptive", "New adaptable", "#15965d"),
    ):
        xs = []
        means = []
        lows = []
        highs = []
        for end in range(576, len(allrows[73]) + 1, 48):
            values = []
            for seed, rows in allrows.items():
                sample = rows[end - 576 : end]
                groups = [[r for r in sample if r["truth"] == c] for c in (0, 1)]
                if min(map(len, groups)) < 16:
                    break
                values.append(
                    100
                    * statistics.mean(
                        statistics.mean(
                            int(r[variant]["answer"] == r["truth"]) for r in group
                        )
                        for group in groups
                    )
                )
            xs.append((allrows[73][end - 1]["stamp"] - allrows[73][0]["stamp"]) / 86400)
            means.append(statistics.mean(values) if len(values) == 3 else float("nan"))
            lows.append(min(values) if len(values) == 3 else float("nan"))
            highs.append(max(values) if len(values) == 3 else float("nan"))
        ax.plot(xs, means, label=label, color=color, lw=1.6)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.12)
    ax.set(
        title=home,
        ylabel="Balanced accuracy (%)",
        xlabel="Elapsed days",
        ylim=(20, 101),
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
fig.savefig(ROOT / "trend.png", dpi=150)
fig.savefig(ROOT / "trend.svg")
p = ROOT / "trend.svg"
p.write_text("\n".join(s.rstrip() for s in p.read_text().splitlines()) + "\n")
manifest = {
    str(p.relative_to(ROOT)): sha(p)
    for p in sorted(ROOT.rglob("*"))
    if p.is_file()
    and (
        p.suffix in (".py", ".json", ".md", ".png", ".svg", ".pt", ".html")
        and "__pycache__" not in p.parts
        and p.name != "manifest.json"
        and p.name != "TASK.md"
    )
}
(ROOT / "reporting/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Verified and built report, figures and manifest")
