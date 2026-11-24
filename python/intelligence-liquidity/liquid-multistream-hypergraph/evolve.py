"""A disclosed controlled experiment: request a hyperedge, then require fresh labels."""

import argparse
import json
from pathlib import Path

import torch

from control import Guards
from device import probe
from graph import Graph
from inspection import inspect, mermaid
from ledger import Ledger, StorageLimits
from policy import ACTIONS
from provenance import Sensor
from reference import Board, FIELDS, REFERENCE, sha
from runtime import Runtime, Settings
from session import Session


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(73)
    base = Board(features=17)
    initial = REFERENCE / "evidence/seed-73/board.pt"
    base.load_state_dict(torch.load(initial, weights_only=True))
    runtime = Runtime(
        Graph(base),
        seed=73,
        settings=Settings(
            window=48,
            history=96,
            minority=16,
            exploration=0,
            fit_rate=0.01,
            fit_passes=8,
            contextual=False,
        ),
        guards=Guards(minimum_life=48),
    )
    # This is a mechanism test, not evidence of autonomous preference for hyperedges.
    for policy in runtime.graph.cores["b0"].policies:
        policy.action[-1].bias.data[ACTIONS.index("connect")] = 10
    ledger = Ledger(
        output / "events.sqlite",
        StorageLimits(
            max_bytes=128 * 1024 * 1024, retained_revisions=3, max_events=1000
        ),
    )
    session = Session(ledger, runtime)
    ledger.pin("initial", 1)
    snapshots = [dict(label="before", at=0, model=inspect(session.runtime.graph))]
    predictions = []
    fields = [*FIELDS, "new:cue"]
    sensors = {
        n: Sensor(
            n,
            "controlled-generator",
            "first",
            "synthetic-room",
            "v1",
            "identity",
            "identity",
        )
        for n in fields
    }
    for i in range(145):
        truth = (i // 3) % 2
        values = dict.fromkeys(FIELDS, 0.0)
        values["new:cue"] = 2.0 * truth - 1
        prediction = session.predict(
            f"o{i}",
            300.0 * (i + 1),
            300.0 * (i + 1),
            values,
            sensors,
            context="phase-" + str((i // 12) % 2),
        )
        session.label(
            f"l{i}",
            f"o{i}",
            truth,
            300.0 * (i + 1) + 1,
            authority="controlled-generator",
        )
        predictions.append(dict(index=i, truth=truth, **prediction))
        if i in (47, 95, 143):
            runtime = session.runtime
            model = runtime.shadow if runtime.shadow is not None else runtime.graph
            label = {
                47: "fitted candidate",
                95: "prospective decision",
                143: "monitor decision",
            }[i]
            snapshots.append(dict(label=label, at=i + 1, model=inspect(model)))
            ledger.pin("stage-" + str(i + 1), session.revision)
    runtime = session.runtime
    # Explicit construction exercises overlapping neuron membership and edge lineage.
    # These are separate from the accepted serving model and never reported as learned.
    example = runtime.graph.fork()
    vertices = [
        dict(expert="e0", layer="l0", unit=0),
        dict(expert="e1", layer="l1", unit=1),
        dict(expert="e2", layer="l2"),
        dict(expert="e3"),
    ]
    h = example.connect(
        "b0", ["e0", "e1", "e2", "e3"], ["e4"], FIELDS[:2], vertices=vertices
    )["hyperedge"]
    snapshots.append(
        dict(label="constructed shared vertices", at=145, model=inspect(example))
    )
    children = example.split_edge(h)["children"]
    snapshots.append(
        dict(label="constructed edge split", at=145, model=inspect(example))
    )
    example.merge_edge(*children)
    snapshots.append(
        dict(label="constructed edge merge", at=145, model=inspect(example))
    )
    result = dict(
        kind="controlled-hypergraph-v1",
        initial_checkpoint_sha256=sha(initial),
        intervention="connect action value biased to 10; all quality gates unchanged; final three views explicitly constructed",
        snapshots=snapshots,
        predictions=predictions,
        receipts=list(runtime.receipts),
        monitor=list(runtime.monitor_audit),
        device=probe(),
        ledger_revision=session.revision,
        ledger_bytes=(output / "events.sqlite").stat().st_size,
    )
    ledger.close()
    (output / "evolution.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for i, snapshot in enumerate(snapshots):
        # Mermaid views are generated from exact layouts through the live graph restore.
        layout = snapshot["model"]
        (output / f"stage-{i}.json").write_text(json.dumps(layout, indent=2) + "\n")
    (output / "final.mmd").write_text(mermaid(runtime.graph))
    render(output, result)
    print(
        json.dumps(
            dict(
                receipts=[
                    dict(
                        action=r.get("action"),
                        accepted=r["accepted"],
                        reason=r["status"],
                    )
                    for r in runtime.receipts
                ],
                monitor=[
                    dict(rollback=r["rollback"], supported=r["supported"])
                    for r in runtime.monitor_audit
                ],
                revision=session.revision,
                ledger_bytes=result["ledger_bytes"],
            ),
            indent=2,
        )
    )
    return result


def render(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    for i, snapshot in enumerate(result["snapshots"]):
        layout = snapshot["model"]["layout"]
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = {}
        for j, (board, core) in enumerate(layout["cores"].items()):
            for k, expert in enumerate(core["ids"]):
                positions[board + "/" + expert] = (k, j * 3)
                ax.scatter(
                    k, j * 3, s=600, color="#d9eef5", edgecolors="#174758", zorder=3
                )
                ax.text(
                    k, j * 3, board + "/" + expert, ha="center", va="center", fontsize=8
                )
        for j, (identity, edge) in enumerate(layout["hyperedges"].items()):
            x, y = j * 1.2 + 1, 1 + j * 0.5
            ax.scatter(
                x, y, marker="D", s=450, color="#ffe6a6", edgecolors="#805900", zorder=3
            )
            ax.text(x, y, identity, ha="center", va="center")
            for source in edge["sources"]:
                a = positions[edge["board"] + "/" + source]
                ax.add_patch(
                    FancyArrowPatch(
                        a,
                        (x, y),
                        arrowstyle="->",
                        mutation_scale=12,
                        color="#315b68",
                        shrinkA=18,
                        shrinkB=14,
                    )
                )
            for target in edge["targets"]:
                b = positions[edge["board"] + "/" + target]
                ax.add_patch(
                    FancyArrowPatch(
                        (x, y),
                        b,
                        arrowstyle="->",
                        mutation_scale=12,
                        color="#986800",
                        shrinkA=14,
                        shrinkB=18,
                    )
                )
        ax.set_title(
            f"{snapshot['label']} — {snapshot['at']} labels\n"
            + str(snapshot["model"]["profile"])
        )
        ax.set_xlim(-1, 6)
        ax.set_ylim(-1, 5)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output / f"stage-{i}.svg")
        plt.close(fig)
    rows = result["predictions"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for start in (0, 48, 96):
        part = rows[start : start + 48]
        correct = sum(r["truth"] == r["answer"] for r in part) / len(part)
        axes[0].plot(
            [start, start + len(part)],
            [correct, correct],
            label=f"{start}–{start + len(part)}: {correct:.1%}",
        )
    axes[0].set_ylabel("Accuracy on issued predictions")
    axes[0].legend()
    axes[1].plot(
        [r["index"] for r in rows], [r["probability"] for r in rows], label="prediction"
    )
    axes[1].set_ylabel("Occupied probability")
    axes[1].set_xlabel("Prediction before its label")
    for axis in axes:
        for boundary in (48, 96, 144):
            axis.axvline(boundary, color="gray", linestyle=":")
    fig.tight_layout()
    fig.savefig(output / "trend.svg")
    plt.close(fig)
    options = "".join(
        f'<option value="{i}">{s["label"]}</option>'
        for i, s in enumerate(result["snapshots"])
    )
    payload = json.dumps(result["snapshots"]).replace("<", "\\u003c")
    (output / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Liquid hypergraph evolution</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;background:#f4f6f8;color:#17252e}img{width:100%}pre{white-space:pre-wrap;max-height:40em;overflow:auto;background:white;padding:1em}select{font:inherit}</style>
<h1>Liquid hypergraph evolution</h1><p>The first four views follow the controlled labeled run. The last three are explicitly constructed lifecycle examples.</p>
<label>Stage <select id="stage">"""
        + options
        + """</select></label><img id="picture" alt="Expert and hyperedge memberships"><p>Inspect every neuron, tensor alias and higher-arity relation:</p><pre id="detail"></pre>
<script>const states="""
        + payload
        + """;function show(){let i=Number(document.querySelector('#stage').value);document.querySelector('#picture').src='stage-'+i+'.svg';document.querySelector('#detail').textContent=JSON.stringify(states[i],null,2)}document.querySelector('#stage').onchange=show;show()</script>"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)
