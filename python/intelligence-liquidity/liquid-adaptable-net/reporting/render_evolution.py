"""Render verified model histories as figures and an offline inspector."""

import argparse
import json
from pathlib import Path

from evolution import ROOT, SEEDS, digest, validate_trace


def save_figure(fig, stem):
    fig.savefig(str(stem) + ".png", dpi=150)
    fig.savefig(str(stem) + ".svg", metadata={"Date": None})
    p = Path(str(stem) + ".svg")
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")


def plot_topology(record, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    phases = record["phases"]
    states = [phases[0]["active"], phases[2]["candidate"], phases[4]["active"]]
    initial = states[0]["layout"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), layout="constrained")
    for ax, state, title in zip(
        axes,
        states,
        (
            "Before: original network",
            "During: fitted shadow candidate",
            "After: adopted and monitored",
        ),
    ):
        ax.set(xlim=(0, 10), ylim=(0, 12))
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=15)
        layout = state["layout"]
        profile = state["profile"]
        ax.text(
            5,
            11.65,
            f"{len(layout['fields'])} inputs  |  {profile['parameters']:,} parameters",
            ha="center",
            fontsize=10,
        )
        positions = {board: (10.8 - i * 5.9) for i, board in enumerate(layout["cores"])}
        for board, core in layout["cores"].items():
            top = positions[board]
            board_height = 2.5 + 0.51 * core["experts"]
            base = initial["cores"].get(board, {})
            is_new = board not in initial["cores"]
            ax.add_patch(
                FancyBboxPatch(
                    (0.2, top - board_height),
                    9.6,
                    board_height - 0.1,
                    boxstyle="round,pad=.05",
                    facecolor="#fff5e9" if is_new else "#f1f6fb",
                    edgecolor="#d67b1b" if is_new else "#53799b",
                    lw=1.5,
                )
            )
            ax.text(
                0.45,
                top - 0.35,
                f"{board}: {core['experts']} expert(s), encoding {core['encoding']}",
                fontsize=10,
                weight="bold",
            )
            nodes = []

            def walk_attention(data, path="root"):
                nodes.append(f"{path}: {len(data['identities'])}h")
                for key, child in data["branches"].items():
                    walk_attention(child, path + ":" + key)

            walk_attention(core["attention"])
            ax.text(
                0.45,
                top - 0.8,
                "Delta attention  " + " → ".join(nodes),
                fontsize=9,
                color="#6a488e",
            )
            widths = [len(c["units"]) for c in core["liquid"]]
            for i, expert in enumerate(core["ids"]):
                y = top - 1.25 - i * 0.51
                ax.text(
                    0.45,
                    y,
                    expert,
                    fontsize=9,
                    weight="bold",
                    color="#d67b1b" if expert not in base.get("ids", []) else "#244b6c",
                )
                for j, (layer, width) in enumerate(zip(core["layer_ids"], widths)):
                    changed = layer not in base.get("layer_ids", []) or width != 8
                    x = 1.25 + j * (7.8 / max(4, len(widths)))
                    ax.add_patch(
                        FancyBboxPatch(
                            (x, y - 0.16),
                            1.65,
                            0.37,
                            boxstyle="round,pad=.03",
                            facecolor="#ffe0b6" if changed else "white",
                            edgecolor="#d67b1b" if changed else "#53799b",
                        )
                    )
                    ax.text(
                        x + 0.82,
                        y + 0.02,
                        f"{layer} · {width}n",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    if j + 1 < len(widths):
                        ax.text(x + 1.75, y + 0.02, "→", va="center", fontsize=9)
            removed = set(base.get("ids", [])) - set(core["ids"])
            if removed:
                ax.text(
                    0.5,
                    top - board_height + 0.55,
                    "Removed: " + ", ".join(sorted(removed)),
                    fontsize=9,
                    color="#b63838",
                )
            gates = state["gates"][board]
            ax.text(
                0.45,
                top - board_height + 0.2,
                "Layer gains: " + ", ".join(f"{g:.2f}" for g in gates["layers"]),
                fontsize=8,
            )
        for key, edge in layout["edges"].items():
            parent, expert = key.split("/")
            x, y = 5, positions[parent] - 5.15
            ax.annotate(
                "",
                xy=(5, positions[edge["child"]] + 0.05),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="#6a488e", lw=1.8),
            )
            ax.text(
                5.25,
                y - 0.35,
                f"{key} → {edge['child']} via {edge['attention']}",
                fontsize=8,
                color="#6a488e",
            )
        if len(layout["cores"]) == 1:
            ax.text(
                0.4,
                4.95,
                "Every row is an expert's liquid-stage chain.\nn = recurrent neurons; h = attention heads.\nOrange = added board, stage, expert or wider state.",
                fontsize=10,
                va="top",
                linespacing=1.7,
            )
            extra = state["gates"]["b0"]["attention"]
            if extra:
                ax.text(
                    0.4,
                    3.6,
                    "Attention branch gains: "
                    + str({k: round(v, 3) for k, v in extra.items()}),
                    fontsize=9,
                )
        ax.text(
            0.3,
            0.2,
            f"{profile['experts']} experts · {profile['boards']} boards · {profile['attention_nodes']} attention nodes · {profile['heads']} heads\nWork proxy: {profile['work_units']:,} (not measured FLOPs)",
            fontsize=9,
            linespacing=1.6,
        )
    fig.suptitle(
        f"Seed {record['seed']}: {record['action']} — observed controlled replay",
        fontsize=16,
    )
    save_figure(fig, output / f"topology-{record['seed']}")
    plt.close(fig)


def render(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [
        validate_trace(json.loads((output / f"trace-{seed}.json").read_text()))
        for seed in SEEDS
    ]
    for r in records:
        plot_topology(r, output)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), layout="constrained")
    for ax, r in zip(axes.flat, records):
        xs = [s["revealed_labels"] for s in r["samples"]]
        ax.step(
            [0] + xs,
            [r["phases"][0]["active"]["profile"]["parameters"]]
            + [s["active"]["profile"]["parameters"] for s in r["samples"]],
            where="post",
            label="Serving model",
            color="#245d91",
        )
        candidate = [s for s in r["samples"] if s["candidate"] is not None]
        ax.plot(
            [s["revealed_labels"] for s in candidate],
            [s["candidate"]["profile"]["parameters"] for s in candidate],
            "--o",
            color="#d67b1b",
            label="Shadow candidate",
            ms=3,
        )
        for x, label in ((48, "Fit"), (96, "Adopt"), (144, "Retain")):
            ax.axvline(x, color="#777777", lw=0.8, ls=":")
            ax.text(
                x,
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
            )
        ax.set(
            title=f"{r['seed']}: {r['action']}",
            xlabel="Labels revealed (not epochs)",
            ylabel="Parameter count",
            xlim=(0, 151),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Serving topology stays unchanged while a shadow candidate is fitted and checked",
        fontsize=15,
    )
    save_figure(fig, output / "lifecycle")
    plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(15, 10), layout="constrained")
    house = []
    for ax, (home, seed) in zip(
        axes, ((home, seed) for home in ("H5", "H6") for seed in (73, 83, 97))
    ):
        folder = ROOT / f"evidence/seed-{seed}"
        record = json.loads((folder / "results.json").read_text())
        item = record["homes"][home]
        for path, expected in record["artifact_sha256"].items():
            if digest(folder / path) != expected:
                raise ValueError("House artifact changed: " + path)
        rows = json.loads((folder / f"{home}-predictions.json").read_text())
        start = rows[0]["stamp"]
        events = []
        for receipt in item["adaptive_receipts"]:
            x = (
                receipt.get("check_last", receipt.get("fit_last", start)) - start
            ) / 86400
            ax.scatter(
                x,
                0,
                c=(
                    "#b24a41"
                    if receipt["status"] == "rejected_measured_gain"
                    else "#9c851c"
                ),
                s=45,
            )
            ax.annotate(
                receipt.get("action", "wait"),
                (x, 0),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
            events.append(
                dict(
                    day=x,
                    action=receipt.get("action"),
                    status=receipt["status"],
                    accepted=receipt["accepted"],
                )
            )
        ax.axhline(0, color="#777777", lw=0.7, zorder=-1)
        ax.set(
            xlim=(0, (rows[-1]["stamp"] - start) / 86400 + 0.6),
            ylim=(-0.8, 1.1),
            yticks=[],
            ylabel=f"{home} / {seed}",
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        house.append(
            dict(
                home=home,
                seed=seed,
                events=events,
                final_profile=item["final_profile"],
                results_sha256=digest(folder / "results.json"),
            )
        )
    axes[-1].set_xlabel("Elapsed days since first observation")
    fig.suptitle(
        "House runs: proposals checked, none adopted\nRed = measured-gain rejection; amber = insufficient checking classes",
        fontsize=14,
    )
    save_figure(fig, output / "house-proposals")
    plt.close(fig)
    (output / "house-events.json").write_text(json.dumps(house, indent=2) + "\n")
    render_inspector(records, house, output)
    print(
        "Verified and rendered six controlled histories and all six house proposal histories"
    )


def render_inspector(records, house, output):
    template = Path(__file__).with_name("inspector.html").read_text()
    payload = json.dumps(
        dict(controlled=records, houses=house), separators=(",", ":"), allow_nan=False
    ).replace("<", "\\u003c")
    (output / "inspector.html").write_text(
        template.replace("/* TRACE_DATA */", payload)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evolution")
    render(parser.parse_args().output)
