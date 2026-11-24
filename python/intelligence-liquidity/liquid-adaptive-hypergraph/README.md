# Liquid adaptive hypergraph

This separate experiment extends [Liquid Adaptable Net](../liquid-adaptable-net/README.md).
Several groups can use the same expert, subexpert, liquid layer or neuron.
Higher-arity messages connect these memberships during the actual neural forward pass.
The earlier experiment remains unchanged. [BASELINE.json](BASELINE.json) pins its source.

The model retains liquid recurrent cells, mixture-of-experts voting, independent
checkers, hierarchical delta attention, forgetting, and guarded changes to width and depth.
The extension adds overlapping liquid messages, measured split/merge nominations,
device provenance, durable revisions and collection of unused tensor payloads.

Read [RESULTS.md](RESULTS.md) for measured outcomes, including costs and refusals.
An accepted change passes measured checks; this cannot guarantee future accuracy.
[Verification record](reporting/verification.json) and [file fingerprints](MANIFEST.json)
make the delivered evidence inspectable.

## Inspect the model

- [Architecture and equations](ARCHITECTURE.md): what runs, what is shared, and what stays private.
- [Evolution views](evidence/evolution/index.html): download this folder and open the HTML locally.
- [Before](evidence/evolution/stage-0.svg), [fitted candidate](evidence/evolution/stage-1.svg),
  [prospective decision](evidence/evolution/stage-2.svg), [monitoring](evidence/evolution/stage-3.svg).
- [Overlapping neuron memberships](evidence/evolution/stage-4.svg),
  [edge split](evidence/evolution/stage-5.svg), [edge merge](evidence/evolution/stage-6.svg).
  These last three views are explicitly constructed examples, not autonomous discoveries.
- [Observed prediction trend](evidence/evolution/trend.svg).
- [Full evolution records](evidence/evolution/evolution.json): neurons, tensor aliases,
  hyperedges, measurements, predictions and decisions.

The graph pictures show the expert overview. Their matching `stage-N.json` files
include every recurrent neuron and tensor alias. A tensor hash denotes equal content.
The `canonical` field denotes actual sharing of a parameter object.

## Run locally

Use Python 3.12 and the pinned CPU dependencies in [requirements.txt](requirements.txt).
Plots also need Matplotlib. No GPU or GitHub Actions is required.

```bash
python -m pip install -r requirements.txt
python -m pip install matplotlib==3.11.2
python -m unittest test_hypergraph test_adaptive test_scaling -q
python device.py
python reporting/build_evolution.py /tmp/hypergraph-evolution
```

For a durable inference session:

```bash
python worker.py experiment.py create /tmp/room.sqlite
python worker.py experiment.py feed /tmp/room.sqlite --input readings.jsonl
python worker.py experiment.py inspect /tmp/room.sqlite
```

See [OPERATIONS.md](OPERATIONS.md) for the event format, sensor replacement,
delayed labels, rewards, scratch training, fine-tuning, corrections and recovery.
See [CONFIGURATION.md](CONFIGURATION.md) for graph, memory, disk and work limits.

## What the experiment establishes

It implements a bounded computational hypergraph on the existing liquid runtime.
An expert or neuron can feed several hyperedges; several parent experts can call
the same child board; exact layers and whole liquid kernels can share parameters.
Each calling context keeps its own recurrent state.

Correlation and representation similarity are additional controller evidence.
They cannot certify a merge, create ground-truth labels or bypass prospective checks.
The policy learns action values from measured outcomes and optional delayed rewards.
It is a contextual bandit, not a multi-step reinforcement-learning algorithm.

The new durable entry point is `Session` in [session.py](session.py).
The inherited JSONL workflow remains available for comparison and offline training.
Its event replay has weaker crash and numerical guarantees than the SQLite session.

The evaluation has two separate protocols: online adaptation with delayed labels,
and completed training followed by frozen inference on unseen later data.
During the latter, labels are used only by the scorer. Weights and architecture
are checked before and after inference; only recurrent state can change.

## Why the onboard AMD GPU is not used

The machine exposes a Radeon 780M through PCI inspection. The current execution
environment has `torch==2.7.1+cpu`, no HIP runtime, and neither `/dev/kfd` nor `/dev/dri`.
The model also inherits CPU tensor allocations and a CPU-only runtime guard.
Changing that guard alone would not produce a working device port.

[GPU.md](GPU.md) records the probe, AMD documentation, and the remaining device work.
`--device rocm` fails explicitly instead of silently running the model on CPU.
No driver or host configuration was changed.
