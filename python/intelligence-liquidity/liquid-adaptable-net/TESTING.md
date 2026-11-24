# Requirements and evidence

Run these commands from this experiment with Python 3.12 and its pinned CPU dependencies:

```sh
python -m unittest discover -v
python -m unittest discover -s ../adaptive-liquid-board -v
python -m unittest discover -s ../verified-liquid-board -v
```

Observed locally: 57 new-experiment tests pass, 36 previous-experiment tests pass, and 20 original tests pass. The earlier folders are unchanged. Tests cover meaningful transformations and failure paths; they do not establish universal prediction accuracy.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Remain a liquid network | `liquid.py`, `graph.py` | Native conductance parity and recurrent sequence tests |
| Add fields over time | `Graph.expand`, runtime schema handling | New cue learning, missing inputs, bounded inputs and old-row learning-rate tests |
| Expert growth, retirement and sharing | Graph peer/depth mutations, ties | Function preservation, prospective adoption, retirement and shared-state tests |
| Child networks and independent forks | Graph edges, fork/retire methods | Exact context preservation, parameter independence, shared-kernel detachment and descendant lifecycle tests |
| Vertical layer adaptation | Stable stage identities and residual gates | Insert every position, sequence/state preservation, learnable gate and checkpoint tests |
| Recurrent neuron adaptation | Conductance split and prune | Sequence preservation, state identity, restore and lower-bound tests |
| Encoding width adaptation | Encoding split/prune transformations | Native projection parity, contraction restore and learned split tests |
| Branching delta attention | `hierarchy.py` recursive memories | Node/head alignment, differentiable child routing, gate learning and bounds tests |
| Head growth, shrink and forgetting | `attention.py`, node mutations | Split equivalence, pruning floor, identity remapping and elapsed-time decay tests |
| Control and reinforcement | `control.py`, `policy.py` | Error hysteresis, lifecycle, gradient probes, learned action selection, action masks, bounded replay and external reward tests |
| No unsupported accuracy adoption | Runtime paired gates and monitor | Strict class gates, insufficient evidence, forced rollback, finite receipts and reward-only rejection tests |
| CPU resource envelope | Graph reservations and `worker.py` | Refuse before deepcopy, recursive work accounting, batch/width bounds, OS allocation rejection and child cleanup tests |
| Training and recovery | `workflow.py` | Scratch/fine-tune, adaptive selection, early-stop restoration, exact pending resume, journal tamper detection and reward PITR tests |
| Same datasets and baselines | `evaluate.py`, `verify.py` | Twelve archive hashes, timestamp/label parity, baseline source/artifact fingerprints and independent rescoring |

The controlled saved runs show actual policy-selected layer addition (seed 60), recurrent splitting (62), attention branching (65) and expert retirement (88). They use a synthetic cue and selected illustration seeds. They do not replace house comparisons or show that each extra structure is necessary for learning.

To verify saved house evidence with original archives:

```sh
python worker.py verify.py evidence/seed-73 --archives /path/to/archives
python worker.py verify.py evidence/seed-83 --archives /path/to/archives
python worker.py verify.py evidence/seed-97 --archives /path/to/archives
```

The evaluation restores every final model, checks its exact layout and profile, verifies trial decisions and recomputes scores. `--rerun` performs the longer deterministic new-model replay and compares prediction hashes. Baseline learning is reused from fingerprinted earlier evidence.

## Rebuild the report

Install `reporting/requirements.txt`. The committed `reporting/inputs.json` contains the same first 160 H5 inputs for every timing run. For each seed (73, 83, 97) and checkpoint house (H5, H6), profile the old and new model in separate sequential processes:

```sh
python worker.py --receipt reporting/worker-profile-old-73-H5.json \
  reporting/profile_model.py ../adaptive-liquid-board \
  ../adaptive-liquid-board/evidence/seed-73/H5-adaptive.pt \
  reporting/inputs.json reporting/profile-old-73-H5.json
python worker.py --receipt reporting/worker-profile-new-73-H5.json \
  reporting/profile_model.py . evidence/seed-73/H5-adaptive.pt \
  reporting/inputs.json reporting/profile-new-73-H5.json
python reporting/build.py
```

The report builder independently checks house receipts and controlled-model hashes, restores controlled checkpoints, checks completed monitoring, generates PNG/SVG figures and writes `reporting/manifest.json`. The manifest covers sources, models, receipts, docs and figures; it excludes itself and the delivery log. Reporting depends on Matplotlib only; inference does not.

Re-running measured commands overwrites their output, so use a separate evidence folder when retaining the published measurements. A checksum proves artifact identity, not model correctness. Hard process limits require the worker; direct imports only apply model-level controls.

## Visual evolution verification

`python reporting/evolution.py verify` checks all six controlled histories: model-source and observer hashes, causal phase boundaries, exact published prediction/receipt parity for the four existing seeds, final tensor digests and model layouts, monitoring decisions and independently recomputed monitoring metrics. The two additional illustrative seeds use the identical stream and settings. No house experiment is rerun for this documentation update.

`python reporting/render_evolution.py` verifies the traces before drawing. It also verifies each house artifact hash before deriving the 51 completed proposal events. Figures and the self-contained HTML inspector are generated from these exports. Root model Python/configuration files and earlier evidence are unchanged. See [EVOLUTION.md](EVOLUTION.md) for the commands and the limits of intermediate topology exports.

Observed browser verification: `python reporting/check_inspector.py --screenshot evolution/inspector-preview.png` passed 116 checks in local headless Chromium: all 108 run/phase/role combinations, six house event lists and both phase navigation controls. The viewer and checker hashes are recorded in `evolution/inspector-check.json`. The preview was visually inspected. Four additional corruption probes rejected changed chronology, altered predictions, a wrong model digest and invented monitoring metrics.
