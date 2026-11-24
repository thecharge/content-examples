# Verification

## Local checks

The implementation uses local verification. GitHub Actions is disabled on the repository.

```bash
python -m unittest test_hypergraph test_adaptive test_scaling -q
ruff check .
ruff format --check .
python -m compileall -q .
```

The observed regression run passed 78 tests. The 21 new tests cover hyperedge
execution and the durable session; the other 57 preserve the preceding model's checks.
The controlled encoding-split test now uses seed 80, because the expanded action
space changes sampling. It still requires an actual learned split, prospective
accuracy gain and retained encoding growth. It does not force acceptance.

The unequal-weight sharing fixture changes every liquid layer. This preserves its
original refusal check while allowing the newly implemented exact sharing of an
unchanged individual layer.

## New failure and mechanism checks

| Check | What must happen |
| --- | --- |
| Zero-gain edge over a sequence | Probabilities remain exactly equal before fitting. |
| Nonzero liquid message | Gradients reach the message and liquid source parameters. |
| Repeated neuron membership | Two hyperedges name the same stable neuron; save/load retains it. |
| Delta branch | A message uses the selected branch and gradients reach its query. |
| Edge split and merge | Parents remain in lineage; live edge count stays bounded. |
| Work ceiling | A rejected connection allocates no message module. |
| Partial layer sharing | Parameters alias; recurrent states stay private; growth detaches. |
| Expanded action policy | Eligible hyperedge actions can be selected and limits mask them. |
| Perfect Pearson, wrong calibration | Correlation alone cannot nominate a merge. |
| Repeated observations | Duplicate support IDs do not inflate evidence count. |
| Commit failure | Serving runtime, event cursor and context remain unchanged. |
| Retry | Identical event input returns once; conflicting input is rejected. |
| Stale writer | Another writer's committed revision cannot be overwritten. |
| Corrupt payload | Recovery fails before exposing a runtime. |
| Correction | Descendants invalidate; predictions stop until clean recovery. |
| Deduplicated tensors | Loading does not alias independent mutable tensors. |
| Root retention | A pin preserves an old payload; releasing its last root permits collection. |
| Storage exhaustion | Objects and event insertion roll back together. |
| Missing feature evidence | No definition or model state commits. |

## Recorded controlled run

```bash
python reporting/build_evolution.py /tmp/hypergraph-evolution
```

The action-value bias explicitly requests `connect`. Fitting, quality checks and
monitoring remain active. The synthetic cue is a perfect target indicator.
This tests implementation and lifecycle; it cannot establish real-data usefulness
or prove the new edge is necessary for learning the cue.

The first 48 labels fit, the next 48 predictions check, and another 48 monitor.
The stored SQLite ledger has 291 revisions after 145 predictions and labels.
Its snapshot roots retain the initial state and inspected stages.

The last three views explicitly construct overlapping neuron vertices and then
split and merge their edge. They test construction and inspection, not autonomous
architectural decisions. Do not include them in counts of learned mutations.

The view builder freezes each inspection before another graph edit. Inspection
layouts are live references until copied. It rebuilds the constructed views from
the committed model and checks their incidence count against the recorded profile.

## Same-data comparison

```bash
python worker.py evaluate.py /path/to/archives /tmp/seed-73 --seed 73
python worker.py evaluate.py /path/to/archives /tmp/seed-83 --seed 83
python worker.py evaluate.py /path/to/archives /tmp/seed-97 --seed 97
python worker.py verify.py /tmp/seed-73 --archives /path/to/archives
```

Verification checks the original twelve archive fingerprints, saved comparator
sources and artifacts, input chronology, prediction counts, recomputed scores,
acceptance gates and restored model profiles. Repeat verification for every seed.
The raw predictions retain refusals and incorrect answers.

The neural benchmark excludes SQLite per-event persistence. Its timings were observed
while separate seed runs shared this host. They are not isolated latency comparisons.
[RESULTS.md](RESULTS.md) separates model quality from this execution context.

## Completed training, then unseen inference

The stricter test trains the new model and preceding model separately on each
room's first 14 selected days. It then freezes topology and all learned tensors.
No test-period label or reward is passed to either model. The scorer reads those
labels separately. Recurrent state can continue evolving from sensor readings.

```bash
python reporting/strict_holdout.py . /path/to/archives /tmp/new-73 --seed 73
python reporting/strict_holdout.py ../liquid-adaptable-net /path/to/archives /tmp/previous-73 --seed 73
python reporting/strict_holdout.py . /path/to/archives /tmp/new-73 --seed 73 --verify
python reporting/strict_holdout.py ../liquid-adaptable-net /path/to/archives /tmp/previous-73 --seed 73 --verify
```

Repeat for seeds 83 and 97. Each record includes the training cutoff, first test
timestamp, training/test row counts, model/source/archive hashes, actual training
receipts and before/after weight fingerprints. An unfinished shadow is excluded;
a candidate still under monitoring uses its retained fallback.

The verifier reloads both snapshots, confirms identical topology and learned tensor
fingerprints, checks strict temporal separation and recomputes every score.
This measures unseen future data in rooms with an earlier adaptation prefix.
It does not claim those rooms were entirely absent from training.

After all six paired runs and three online runs have been verified, regenerate
the published tables and chart from their records:

```bash
python reporting/results.py
```

The reporter reads this folder's `evidence` paths. Its chart compares balanced
accuracy; raw records also retain ordinary accuracy, class counts and Brier scores.

## Limits of the checks

The tests do not prove unchanged accuracy on future sensor distributions.
They do not establish automatic semantic label discovery, distributed consensus,
multi-step reinforcement learning, or GPU execution.
The ledger is designed for one owning session per process and optimistic coordination
between processes. Cross-ledger correction propagation remains an operator task.
Recovery refuses changed source or framework fingerprints until an explicit migration.
