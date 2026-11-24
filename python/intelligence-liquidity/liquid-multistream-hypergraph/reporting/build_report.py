"""Build inspectable plots from measured histories and actual graph layouts."""

import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np


def read(path):
    return json.loads(path.read_text())


def layout(source, target, title):
    data = read(source)
    cores = data["layout"]["cores"]
    figure, axes = plt.subplots(
        len(cores), 1, figsize=(12, 3.6 * len(cores)), squeeze=False
    )
    for ax, (board, core) in zip(axes[:, 0], cores.items()):
        experts = core["ids"]
        layers = core["layer_ids"]
        ax.set_xlim(-0.8, max(5, len(experts)) - 0.2)
        ax.set_ylim(-1.6, len(layers) + 0.9)
        for i, expert in enumerate(experts):
            ax.text(i, len(layers) + 0.25, expert, ha="center", weight="bold")
            for j, (layer, neurons) in enumerate(zip(layers, core["liquid"])):
                y = len(layers) - 1 - j
                ax.add_patch(
                    FancyBboxPatch(
                        (i - 0.42, y - 0.3),
                        0.84,
                        0.6,
                        boxstyle="round,pad=.04",
                        facecolor="#e4f3fb",
                        edgecolor="#22577a",
                    )
                )
                units = neurons["units"]
                for k in range(len(units)):
                    ax.add_patch(
                        Circle(
                            (i - 0.28 + (k % 4) * 0.18, y + 0.12 - (k // 4) * 0.2),
                            0.045,
                            color="#22577a",
                        )
                    )
                if i == 0:
                    ax.text(-0.58, y, layer, ha="right", va="center")
                if j:
                    ax.annotate(
                        "",
                        (i, y + 0.35),
                        (i, y + 0.65),
                        arrowprops=dict(arrowstyle="->", color="#22577a"),
                    )
        heads = len(core["attention"]["identities"])
        ax.text(
            0,
            -0.9,
            f"Delta attention: {heads} root heads | nested branches: {len(core['attention']['branches'])}",
            fontsize=10,
        )
        ax.text(
            0,
            -1.25,
            f"Board {board}: shared parameters; every stream has private liquid / delta state",
            fontsize=9,
            color="#444",
        )
        ax.set_title(
            f"{board}: {len(experts)} experts, {len(layers)} liquid layers, {sum(len(n['units']) for n in core['liquid'])} recurrent units per expert"
        )
        ax.axis("off")
    p = data["profile"]
    figure.suptitle(
        f"{title}\n{p['parameters']:,} parameters | {p['hyperedges']} hyperedges | {p['work_units']:,} estimated work units",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(target, dpi=140)
    figure.savefig(target.with_suffix(".png"), dpi=140)
    plt.close(figure)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for name, label in (
        ("scratch", "Random initialization"),
        ("pretrained-retrained", "Pretrained initialization"),
    ):
        curve = read(args.cold / name / "curve.json")
        for i, ax in enumerate(axes.flat, 1):
            ax.plot(
                [r["epoch"] for r in curve],
                [100 * r["validation"][f"H{i}"]["balanced_accuracy"] for r in curve],
                marker=".",
                label=label,
            )
            ax.set_title(
                f"H{i}: validation only" + (" (occupied class only)" if i == 4 else "")
            )
            ax.set_xlabel("Training epoch")
            ax.set_ylabel("Balanced accuracy (%)")
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Learning curves: model selection sees validation, never future labels"
    )
    fig.tight_layout()
    fig.savefig(args.output / "learning.png", dpi=150)
    fig.savefig(args.output / "learning.svg")
    plt.close(fig)

    final_state = args.online / "learning-state.json"
    if final_state.exists():
        learned = read(final_state)["rooms"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(list(learned), [v["accepted_updates"] for v in learned.values()])
        axes[0].set_title("Accepted room-calibration updates")
        axes[0].set_ylabel("Updates")
        axes[1].bar(list(learned), [v["norm"] for v in learned.values()])
        axes[1].set_title("Final correction-vector norm")
        axes[1].set_ylabel("Euclidean norm")
        fig.suptitle("Learned room readouts; shared topology stayed unchanged")
        fig.tight_layout()
        fig.savefig(args.output / "calibration.svg")
        fig.savefig(args.output / "calibration.png", dpi=150)
        plt.close(fig)

    online = read(args.online / "curve.json")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat, 1):
        room = f"H{i}"
        ax.plot(
            [r["step"] / 288 for r in online],
            [100 * r["per_stream"][room + "/0"]["balanced_accuracy"] for r in online],
            marker=".",
        )
        ax.set_title(room)
        ax.set_xlabel("Selected days observed")
        ax.set_ylabel("Last-day available-class recall (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    fig.suptitle("Live empty start: predictions scored before their labels arrive")
    fig.tight_layout()
    fig.savefig(args.output / "live-learning.svg")
    fig.savefig(args.output / "live-learning.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    x = [r["step"] / 288 for r in online]
    for name in ("experts", "boards", "heads", "hyperedges"):
        axes[0].plot(x, [r["profile"][name] for r in online], marker=".", label=name)
    axes[0].set_ylabel("Active count")
    axes[0].legend(ncol=4)
    axes[1].plot(
        x, [r["profile"]["parameters"] for r in online], marker=".", label="parameters"
    )
    axes[1].set_xlabel("Observed selected days per room")
    axes[1].set_ylabel("Active parameters")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Actual online architecture: refusals are not drawn as adopted growth")
    fig.tight_layout()
    fig.savefig(args.output / "evolution.svg")
    fig.savefig(args.output / "evolution.png", dpi=150)
    plt.close(fig)

    accept = read(args.acceptance / "acceptance.json")
    fig, ax = plt.subplots(figsize=(13, 5))
    rooms = list(accept["comparison"])
    names = [
        ("random_initial", "Untrained"),
        ("pretrained_frozen", "Pretrained frozen"),
        ("scratch", "Scratch bootstrap"),
        ("online", "Live empty start"),
    ]
    width = 0.2
    for j, (name, label) in enumerate(names):
        y = [100 * accept["comparison"][r]["scores"][name] for r in rooms]
        ax.bar(np.arange(6) + (j - 1.5) * width, y, width, label=label)
    ax.set_xticks(np.arange(6), rooms)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Future balanced accuracy (%)")
    ax.set_title("Frozen future test: no labels or rewards returned to the models")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / "future.svg")
    fig.savefig(args.output / "future.png", dpi=150)
    plt.close(fig)

    layouts = [
        ("initial-layout.json", "before.svg", "Before: random liquid graph"),
        ("mid-layout.json", "during.svg", "During: actual saved online graph"),
        ("final-layout.json", "after.svg", "After: actual saved online graph"),
    ]
    for source, target, title in layouts:
        if (args.online / source).exists():
            layout(args.online / source, args.output / target, title)
    rows = "".join(
        f"<tr><td>{html.escape(room)}</td>"
        + "".join(f"<td>{100 * entry['scores'][name]:.2f}%</td>" for name, _ in names)
        + "</tr>"
        for room, entry in accept["comparison"].items()
    )
    (args.output / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>36-stream liquid model evidence</title>
<style>body{font:17px system-ui;max-width:1200px;margin:32px auto;padding:0 16px;color:#182a3a}img{width:100%;border:1px solid #ddd;margin:12px 0}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #bbb}p{max-width:850px}</style>
<h1>36-stream liquid model evidence</h1><p>Six feeds per room share one recorded timeline. They test execution and state isolation, not independent data. Cold bootstrap and live online learning are separate protocols. Scores below use untouched later data.</p>
<table><tr><th>Room</th><th>Untrained</th><th>Pretrained frozen</th><th>Scratch bootstrap</th><th>Live empty start</th></tr>"""
        + rows
        + """</table>
<h2>Learning</h2><img src="learning.png" alt="Validation learning curves for each room"><img src="live-learning.svg" alt="Live delayed-label prediction scores per room"><img src="future.svg" alt="Frozen future accuracy comparison">
<h2>Actual evolution</h2><p>A flat architecture means no topology change was accepted. Candidate attempts are recorded in receipts; they are not presented as growth.</p><img src="evolution.svg" alt="Actual active shape counts over time"><img src="calibration.svg" alt="Accepted room calibration updates and final correction norms">
<h2>Before, during and after</h2><img src="before.svg" alt="Initial liquid network"><img src="during.svg" alt="Saved intermediate liquid network"><img src="after.svg" alt="Final liquid network">
<p>The matching JSON layouts preserve exact vertices, memberships and tensor aliases. See the repository documents and acceptance.json for support, protocol and pass/fail flags.</p>"""
    )

    # Preserve SVG geometry while avoiding generated trailing whitespace.
    for path in args.output.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("cold", "online", "acceptance", "output"):
        cli.add_argument(name, type=Path)
    run(cli.parse_args())
