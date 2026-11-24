"""Derive the review report and rolling curves from checked prediction records."""

from collections import Counter
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/liquid-board-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compare import hashes
from reference import sha
from verify import verify

SEEDS = (73, 83, 97)
VARIANTS = ("frozen", "contextual", "adaptive", "weight_only", "tree")


def rolling(rows, variant, capacity=576):
    """Trailing observations only; do not report a one-class rolling score."""
    truth = np.array([r["truth"] for r in rows])
    correct = np.array([r[variant]["answer"] == r["truth"] for r in rows])
    result = []
    for end in range(1, len(rows) + 1):
        start = max(0, end - capacity)
        labels, right = truth[start:end], correct[start:end]
        result.append(
            np.mean([right[labels == c].mean() for c in (0, 1)])
            if all(np.sum(labels == c) >= 16 for c in (0, 1))
            else np.nan
        )
    return np.array(result)


def plot(records):
    colors = dict(
        frozen="#64748b",
        contextual="#9333ea",
        adaptive="#059669",
        weight_only="#2563eb",
    )
    labels = dict(
        frozen="Frozen",
        contextual="Original adaptive",
        adaptive="New adaptive",
        weight_only="Context + weight learning",
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.6), constrained_layout=True)
    raw = {}
    for ax, home in zip(axes, ("H5", "H6")):
        streams = [
            json.loads(
                (ROOT / f"evidence/seed-{seed}/{home}-predictions.json").read_text()
            )
            for seed in SEEDS
        ]
        raw[home] = {}
        for variant in colors:
            values = np.stack([rolling(rows, variant) for rows in streams])
            valid = np.isfinite(values).all(0)
            center = np.full(values.shape[1], np.nan)
            low, high = center.copy(), center.copy()
            center[valid] = values[:, valid].mean(0) * 100
            low[valid], high[valid] = (
                values[:, valid].min(0) * 100,
                values[:, valid].max(0) * 100,
            )
            x = np.arange(1, len(center) + 1)
            ax.fill_between(x, low, high, color=colors[variant], alpha=0.07)
            ax.plot(
                x,
                center,
                color=colors[variant],
                label=labels[variant],
                linewidth=1.7 if variant == "adaptive" else 1.2,
                linestyle="--" if variant in ("contextual", "weight_only") else "-",
            )
            raw[home][variant] = [
                None if not np.isfinite(v) else float(v) for v in center
            ]
        item = records[73]["homes"][home]
        ax.axvline(item["first_week_rows"], color="#94a3b8", linestyle=":", linewidth=1)
        ax.axvline(
            item["later_start_index"], color="#334155", linestyle=":", linewidth=1
        )
        ax.set(
            title=f"{home}: trailing 576 observations, mean and range across three seeds",
            ylabel="Balanced accuracy (%)",
            ylim=(20, 102),
            xlabel="Chronological observation index",
        )
        ax.grid(alpha=0.15)
        ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.suptitle(
        "Continuing contextual learning alongside guarded structural adaptation",
        fontsize=14,
    )
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / f"reporting/learning-trends.{suffix}", dpi=150)
    svg = ROOT / "reporting/learning-trends.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    (ROOT / "reporting/rolling-scores.json").write_text(
        json.dumps(
            dict(window=576, minimum_per_class=16, seeds=SEEDS, mean_percent=raw),
            separators=(",", ":"),
        )
        + "\n"
    )


def main():
    records = {}
    for seed in SEEDS:
        directory = ROOT / f"evidence/seed-{seed}"
        verify(directory)
        records[seed] = json.loads((directory / "results.json").read_text())
        if (
            set(records[seed]["homes"]) != {"H5", "H6"}
            or records[seed]["code_sha256"] != hashes()
        ):
            raise ValueError("Incomplete or stale house evidence")
    plot(records)
    lines = [
        "# Measured learning and structural outcomes",
        "",
        "These are CPU runs of the supplied HPDmobile protocol. All 12 archive hashes, original checkpoints and frozen reference scores are checked. The original contextual learner is actually rerun, and its later scores must match the published receipts. H1–H3 supply the original 12,087 fitting rows; H4 supplies 2,016 selection rows. Those initial fits are reused.",
        "",
        "The new model carries the original contextual learner forward. Its structural proposals are an additional learning path. The weight-only control also retains contextual learning, while disabling every structural action. No settings or checkpoint were selected using the new test scores.",
        "",
        "## Learning over time",
        "",
        "![Trailing balanced accuracy across the stream](reporting/learning-trends.svg)",
        "",
        "Curves use only the trailing 576 observations, with at least 16 examples of each class. Lines show the mean of seeds 73, 83 and 97; shading shows their range. Dotted lines mark the end of the original first week and the start of scoring after day 14. The index counts observed windows, not uninterrupted calendar time. Gaps reset recurrent states. Overlapping curves indicate matching behavior, not missing series. The separate range below contains only 12 vacant H5 examples, so its final score cannot be read directly from the class-filtered rolling curve.",
        "",
        "## Exact comparison ranges",
        "",
        "| Range | Home | Rows | Vacant | Occupied |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for period, label in (
        ("first_week", "First week"),
        ("after_day_14", "After day 14"),
    ):
        for home in ("H5", "H6"):
            m = records[73]["homes"][home]["ranges"][period]["frozen"]
            lines.append(
                f'| {label} | {home} | {m["rows"]} | {m["classes"][0]["rows"]} | {m["classes"][1]["rows"]} |'
            )
    lines += [
        "",
        "Each runtime begins on day one. Previous answers arrive before the next prediction. The later range therefore follows the same 14-day online warmup as the article. The final current answer remains unrevealed.",
        "",
    ]
    for period, label in (
        ("first_week", "First-week balanced accuracy"),
        ("after_day_14", "Balanced accuracy after day 14"),
    ):
        lines += [
            "## " + label,
            "",
            "Values are percentages. Balanced accuracy gives vacant and occupied recall equal weight. Raw accuracy, class counts, Brier error, quorum and paired intervals remain in each receipt.",
            "",
            "| Seed | Home | Frozen | Original adaptive rerun | New adaptive | Context + weights only | Saved XGBoost |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for seed in SEEDS:
            for home, item in records[seed]["homes"].items():
                values = [
                    item["ranges"][period][variant]["balanced_accuracy"] * 100
                    for variant in VARIANTS
                ]
                lines.append(
                    f"| {seed} | {home} | "
                    + " | ".join(f"{v:.2f}" for v in values)
                    + " |"
                )
        lines.append("")
    trials = [
        r
        for record in records.values()
        for home in record["homes"].values()
        for r in home["adaptive_receipts"]
        if "check_first" in r
    ]
    retained = [r for r in trials if r["accepted"]]
    lines += [
        "## Architecture decisions",
        "",
        f"The six streams completed {len(trials)} prospective adaptive trials and retained {len(retained)}. Contextual updates run separately and are recorded under `contextual_receipts`. A rejected structural proposal therefore does not mean the running model stopped learning.",
        "",
        "| Action | Completed | Retained |",
        "| --- | ---: | ---: |",
    ]
    for action in (
        "learn",
        "spawn",
        "deepen",
        "split",
        "grow_heads",
        "prune_heads",
        "retire",
        "share",
    ):
        subset = [r for r in trials if r["action"] == action]
        lines.append(
            f'| {action} | {len(subset)} | {sum(r["accepted"] for r in subset)} |'
        )
    lines += [
        "",
        "Each growth trial first checks function preservation and the candidate's resource bounds. Its earlier fit and validation data precede the subsequent check predictions. Both paired graphs continue receiving the same accepted contextual correction. A quality gate measures a finite block; it does not guarantee that every later window improves monotonically.",
        "",
        "| Seed | Home | Accepted contextual updates | Retained trial actions | Final experts | Final root heads |",
        "| ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for seed in SEEDS:
        for home, item in records[seed]["homes"].items():
            contextual = sum(bool(r["accepted"]) for r in item["contextual_receipts"])
            actions = (
                ", ".join(
                    r["action"] for r in item["adaptive_receipts"] if r["accepted"]
                )
                or "None"
            )
            heads = len(item["final_layout"]["cores"]["b0"]["attention"]["identities"])
            lines.append(
                f'| {seed} | {home} | {contextual} | {actions} | {item["final_experts"]} | {heads} |'
            )
    lines += [
        "",
        "Do not interpret ordinary contextual gains as evidence that extra experts or heads were necessary. The tables show separately which changes were retained. Inspect each prospective receipt for its own paired scores and class-specific effects.",
        "",
        "## Controlled adaptation",
        "",
        "These artificial streams are separate from occupancy accuracy. All original inputs are zero; a new input supplies `2 * truth - 1`, with `truth = (index // 3) % 2`. Labels arrive after prediction. The first 48 labels fit a candidate and the next 48 grade it. No action or candidate prediction is forced.",
        "",
        "| Seed | Retained first action | Active / candidate balanced accuracy | Final experts |",
        "| ---: | --- | ---: | ---: |",
    ]
    for seed in SEEDS:
        record = json.loads((ROOT / f"evidence/new-input-seed-{seed}.json").read_text())
        receipt = record["receipts"][0]
        experts = sum(c["experts"] for c in record["final_layout"]["cores"].values())
        lines.append(
            f'| {seed} | {receipt["action"]} | {receipt["active"]["balanced_accuracy"]*100:.1f}% / {receipt["candidate"]["balanced_accuracy"]*100:.1f}% | {experts} |'
        )
    lines += [
        "",
        "Separate behavioral tests check learned proposals for attention growth and pruning, the minimum head count, preservation across recurrent sequences, replayed reinforcement credit, native contextual parity, early stopping, snapshot fine-tuning and point-in-time restoration. A constructed perfect cue is not evidence of universal occupancy accuracy.",
        "",
        "## Inspect and reproduce",
        "",
        "- [Seed 73 receipt](evidence/seed-73/results.json)",
        "- [Seed 83 receipt](evidence/seed-83/results.json)",
        "- [Seed 97 receipt](evidence/seed-97/results.json)",
        "- [Architecture and Mermaid diagrams](ARCHITECTURE.md)",
        "- [Model inspection, training and recovery commands](OPERATIONS.md)",
        "",
        "Each seed directory contains raw H5/H6 predictions, full active and weight-only CPU checkpoints, and recoverable mutation boundaries. `verify.py` recomputes scores, intervals, structural gates and chronology; checks source/archive/checkpoint/artifact fingerprints; and restores the final topology. `--rerun` performs the complete computation again and compares prediction hashes.",
        "",
        "Local verification passed 36 adaptive behavioral tests and 20 unchanged reference tests. House records are independently checked by the verifier before this report is generated. No GitHub Actions were used.",
        "",
        "Regenerate this report with `python reporting/generate.py` after installing `reporting/requirements.txt`. Plot/report fingerprints and the generator's own fingerprint are saved in `reporting/manifest.json`.",
        "",
        "These are two previously studied homes and three seeds. Later H5 has only 12 vacant examples. Finite check blocks, shared research history and one configured controller limit the conclusions. The reported outcomes establish behavior on these streams; they do not guarantee monotonic accuracy, unlimited adaptation or better scores from every larger graph.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    artifacts = [
        ROOT / "RESULTS.md",
        *[
            ROOT / f"reporting/{name}"
            for name in (
                "learning-trends.svg",
                "learning-trends.png",
                "rolling-scores.json",
            )
        ],
    ]
    manifest = dict(
        generator_sha256=sha(Path(__file__)),
        matplotlib=matplotlib.__version__,
        receipt_sha256={
            str(seed): sha(ROOT / f"evidence/seed-{seed}/results.json")
            for seed in SEEDS
        },
        artifacts={str(p.relative_to(ROOT)): sha(p) for p in artifacts},
    )
    (ROOT / "reporting/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Report: {len(trials)} completed trials, {len(retained)} retained")


if __name__ == "__main__":
    main()
