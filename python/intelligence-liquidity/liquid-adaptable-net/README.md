# Liquid Adaptable Net

A separate CPU experiment copied from [adaptive-liquid-board](../adaptive-liquid-board/README.md) at [66e994f](https://github.com/thecharge/content-examples/commit/66e994fe49a59dfbaad0e2a719cd7507d732686c). Both earlier experiments remain intact. The starting prediction and recurrent behavior match the supplied [verified liquid board](../verified-liquid-board/README.md).

The new model can change its liquid layers, recurrent neuron counts, delta-attention hierarchy and child-network graph, as well as its experts, encoding neurons, attention heads and input fields. A single bounded controller selects changes throughout the graph. There are no fixed left and right roles or predetermined growth schedule.

- [Architecture, equations and Mermaid diagrams](ARCHITECTURE.md)
- [Every operating setting and resource boundary](CONFIGURATION.md)
- [Training, online feedback, snapshots and recovery](OPERATIONS.md)
- [Measured accuracy, topology and CPU performance](RESULTS.md)
- [Before, during and after: visual evolution and offline inspector](EVOLUTION.md)
- [Requirements and test evidence](TESTING.md)
- [Delivery and verification record](TASK.md)

## What changed

| Capability | Previous adaptive board | This experiment |
| --- | --- | --- |
| Liquid layers | Three fixed stages per expert | Insert and retire stages at any position, within limits |
| Recurrent liquid neurons | Eight fixed coordinates | Split and prune conductance neurons per stage; preserve an eight-coordinate interface |
| Encoding neurons | Split only | Split and prune |
| Delta attention | Variable heads in one memory | A branching hierarchy of memories, each with variable heads and forgetting |
| Child networks | Expert-owned nested boards | Attention-routed nested boards; fork shared networks or retire network edges separately from experts |
| Connections | Learned child gain and input grouping | Also learned attention gates, routes and recurrent readouts; low routes skip child execution during inference |
| Adoption | Prospective paired quality check | Also a subsequent monitoring block with automatic rollback |
| Resource protection | Topology bounds and allocation estimate | Adds global attention/work bounds, reservations, and an isolated worker with hard address-space and CPU ceilings |
| Offline training | Fixed-topology fit | First runs the adaptive stage loop on training observations, then fits the accepted topology with early stopping |
| Reinforcement credit | Measured contextual-bandit return | Also accepts explicit delayed environment reward and journals it |

Growth starts with function-preserving transformations and then learns. Contraction must maintain measured quality while saving parameters or estimated work. Default growth checks allow no class-recall or quorum loss on their grading block. Later monitoring can restore the preceding graph. These finite checks cannot guarantee unchanged accuracy on every future input.

The reward learner is a neural contextual bandit for architecture decisions. This is not an actor-critic agent or a general reward-only task trainer. External rewards shape structural credit; measured binary outcomes remain required to certify prediction accuracy. Inference without outcomes evolves liquid state and routing but does not invent accuracy evidence or permanently adopt ungraded changes.

## Run locally

Use Python 3.12 and the pinned CPU dependencies. Commands run from this folder:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python -m unittest discover -s ../verified-liquid-board -v
```

Use the worker for hard process limits:

```sh
python worker.py --receipt evidence/worker-73.json evaluate.py \
  /path/to/hpdmobile-archives evidence/seed-73 --seed 73
python worker.py verify.py evidence/seed-73 --archives /path/to/hpdmobile-archives
python worker.py workflow.py inspect evidence/seed-73/H5-adaptive.pt
```

Repeat the comparison for seeds 83 and 97. The evaluator checks all twelve original archive hashes, the original checkpoint, and the previous experiment's source and artifact fingerprints. It runs the new learner from day one. The frozen, original contextual and previous adaptive predictions are reused from the verified previous receipts; their weights are not retrained. Every comparator must have the same timestamps and truths. This avoids repeating already recorded baseline computation. `compare.py` retains the more expensive option that reruns the original contextual and frozen models.

Controlled streams exercise actual adoption of the new mechanisms:

```sh
python worker.py demonstrate.py evidence/controlled-60.json --seed 60 --windows 145
python worker.py demonstrate.py evidence/controlled-62.json --seed 62 --windows 145
python worker.py demonstrate.py evidence/controlled-65.json --seed 65 --windows 145
```

These seeds illustrate layer addition, recurrent-neuron splitting and attention branching. They were selected to cover mechanisms in an artificial, perfectly informative cue stream. They are not house-data model selection or evidence that those structures are necessary for higher accuracy. Neither actions nor checking predictions are forced. Each command saves a running model beside its receipt.

## Component map

| File | Responsibility |
| --- | --- |
| `graph.py` | Transactional boards, stable identities, dynamic stages, network edges, sharing, state alignment and topology reservations |
| `liquid.py` | Variable-width conductance LTC, neuron transformations and public readout |
| `hierarchy.py` | Recursive delta-memory tree, gates, node routes and memory remapping |
| `attention.py` | Individual delta-head transformations and forgetting equations |
| `policy.py` | Learned action values, feature grouping and bounded reinforcement replay |
| `control.py` | Error/drift feedback, gradient probes, cooldown, lifecycle eligibility and allocation estimates |
| `runtime.py` | Ordered predictions and outcomes, candidate fitting, prospective decisions, monitoring and full checkpoints |
| `adaptation.py` | The original contextual learner replayed through the complete graph |
| `workflow.py` | Scratch training, fine-tuning, journaled reward/outcome events and point-in-time recovery |
| `worker.py` | Isolated process, OS resource limits, timeout and cancellation cleanup |
| `evaluate.py`, `verify.py` | Same-data comparison and independent receipt verification |
| `test_adaptive.py`, `test_scaling.py` | Inherited behavior and new mechanisms, including failure paths |
| `reporting/` | Separate report generation and isolated inference profiling |

No GitHub Actions, GPU trainer or external service is needed. Model limits belong to this CPU research experiment; a direct Python import does not install OS limits on its caller. Use the worker when a hard process ceiling is required.
