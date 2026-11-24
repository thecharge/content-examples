"""Freeze inspection views, rebuild constructed examples and draw readable diagrams.

Inspection contains live layout references. Freeze each view before further edits.
The low-level evolution runner's three constructed layouts are rebuilt here from
the committed serving model, so later mutations cannot rewrite earlier views.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(output, existing=False):
    if not existing:
        subprocess.run(
            [sys.executable, str(ROOT / "worker.py"), "evolve.py", str(output)],
            check=True,
        )
    from inspection import inspect, mermaid
    from ledger import Ledger, StorageLimits
    from session import Session
    import sqlite3

    db = sqlite3.connect(output / "events.sqlite")
    limits = StorageLimits(
        **json.loads(
            db.execute("SELECT value FROM metadata WHERE key='limits'").fetchone()[0]
        )
    )
    db.close()
    ledger = Ledger(output / "events.sqlite", limits)
    try:
        session = Session(ledger)
        example = session.runtime.graph.fork()
        record = json.loads((output / "evolution.json").read_text())
        snapshots = deepcopy(record["snapshots"][:4])
        vertices = [
            dict(expert="e0", layer="l0", unit=0),
            dict(expert="e1", layer="l1", unit=1),
            dict(expert="e2", layer="l2"),
            dict(expert="e3"),
        ]
        edge = example.connect(
            "b0",
            ["e0", "e1", "e2", "e3"],
            ["e4"],
            example.fields[:2],
            vertices=vertices,
        )["hyperedge"]
        snapshots.append(
            dict(
                label="constructed shared vertices",
                at=145,
                model=deepcopy(inspect(example)),
            )
        )
        children = example.split_edge(edge)["children"]
        snapshots.append(
            dict(
                label="constructed edge split", at=145, model=deepcopy(inspect(example))
            )
        )
        example.merge_edge(*children)
        snapshots.append(
            dict(
                label="constructed edge merge", at=145, model=deepcopy(inspect(example))
            )
        )
        record["snapshots"] = snapshots
        record["view_builder_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        record["view_method"] = (
            "Deep-copy each inspection immediately. Rebuild constructed views from committed revision 291; do not reuse live layout references."
        )
        for i, snapshot in enumerate(snapshots):
            profile, layout = snapshot["model"]["profile"], snapshot["model"]["layout"]
            if profile["hyperedges"] != len(layout["hyperedges"]):
                raise ValueError("Snapshot incidence differs from profile")
            (output / f"stage-{i}.json").write_text(
                json.dumps(snapshot["model"], indent=2) + "\n"
            )
        (output / "evolution.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
        (output / "final.mmd").write_text(mermaid(session.runtime.graph))
        render(output, record)
    finally:
        ledger.close()
    print("Built seven independent evolution views from committed state")


def render(output, record):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    for i, snapshot in enumerate(record["snapshots"]):
        model = snapshot["model"]
        layout, profile = model["layout"], model["profile"]
        fig, ax = plt.subplots(figsize=(11, 6))
        positions = {}
        for j, (board, core) in enumerate(layout["cores"].items()):
            for k, expert in enumerate(core["ids"]):
                positions[board + "/" + expert] = (k, j * 3)
                ax.scatter(
                    k, j * 3, s=1100, color="#d9eef5", edgecolors="#174758", zorder=3
                )
                ax.text(
                    k,
                    j * 3,
                    board + "/" + expert,
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        memberships = []
        for j, (identity, edge) in enumerate(layout["hyperedges"].items()):
            point = (0.5 + j * 1.5, 1.5 + j * 0.3)
            ax.scatter(
                *point,
                marker="D",
                s=800,
                color="#ffe6a6",
                edgecolors="#805900",
                zorder=3,
            )
            ax.text(*point, identity, ha="center", va="center", fontsize=11)
            for source in edge["sources"]:
                a = positions[edge["board"] + "/" + source]
                ax.add_patch(
                    FancyArrowPatch(
                        a,
                        point,
                        arrowstyle="->",
                        mutation_scale=14,
                        color="#315b68",
                        shrinkA=23,
                        shrinkB=18,
                        connectionstyle="arc3,rad=.05",
                    )
                )
            for target in edge["targets"]:
                b = positions[edge["board"] + "/" + target]
                ax.add_patch(
                    FancyArrowPatch(
                        point,
                        b,
                        arrowstyle="->",
                        mutation_scale=14,
                        color="#986800",
                        shrinkA=18,
                        shrinkB=23,
                        connectionstyle="arc3,rad=.05",
                    )
                )

            def vertex(v):
                return (
                    v["expert"]
                    + ("/" + v["layer"] if "layer" in v else "")
                    + ("/u" + str(v["unit"]) if "unit" in v else "")
                )

            memberships.append(
                identity
                + ": "
                + ", ".join(vertex(v) for v in edge["vertices"])
                + " → "
                + ", ".join(edge["targets"])
            )
        summary = f"{profile['experts']} experts · {profile['hyperedges']} hyperedges · {profile['parameters']:,} parameters · {profile['work_units']:,} work units"
        ax.set_title(
            snapshot["label"] + f" — {snapshot['at']} labels\n" + summary, pad=20
        )
        ax.set_xlim(-0.8, 5.4)
        ax.set_ylim(-1.1, 4)
        ax.axis("off")
        fig.text(
            0.05,
            0.04,
            "\n".join(memberships) or "No computational hyperedges yet.",
            fontsize=9,
            va="bottom",
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.23)
        fig.savefig(output / f"stage-{i}.svg")
        plt.close(fig)
    options = "".join(
        f'<option value="{i}">{s["label"]}</option>'
        for i, s in enumerate(record["snapshots"])
    )
    payload = json.dumps(record["snapshots"]).replace("<", "\\u003c")
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
    parser.add_argument("--existing", action="store_true")
    args = parser.parse_args()
    build(args.output, args.existing)
