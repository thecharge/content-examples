# Results for the 36-stream experiment

The 36-stream execution target passes. The all-room accuracy target fails.
The live empty-start model improves over its untrained starting point in all six
rooms and beats the frozen pretrained baseline in five. H4 misses that baseline.
Offline scratch training also beats it in five rooms, but misses H3.

[Acceptance flags and dataset checks](evidence/acceptance-seed-73/acceptance.json)
retain these failures. Nothing was tuned after reading the future scores.

## Frozen future accuracy

All numbers below are balanced accuracy percentages, with seed 73.

| Room | Untrained | Pretrained frozen | Equally retrained | Scratch bootstrap | Live empty start |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 50.36 | 64.26 | 68.10 | 69.11 | 66.72 |
| H2 | 61.30 | 63.20 | 73.85 | 74.11 | 73.19 |
| H3 | 61.98 | 52.67 | 43.02 | 48.51 | 71.88 |
| H4 | 41.26 | 50.68 | 56.81 | 56.12 | 49.57 |
| H5 | 46.20 | 73.19 | 82.97 | 89.86 | 78.26 |
| H6 | 56.40 | 63.32 | 64.06 | 64.51 | 63.61 |

The live H4 score is 49.57%, versus 50.68% frozen. Offline H3 is 48.51%, versus
52.67% frozen. Positive average gains do not satisfy the per-room requirement.
Neither run establishes guaranteed convergence or accuracy without loss.

The preceding per-room hypergraph's same-seed future scores were 73.73% for H5 and
62.54% for H6. Live shared learning reaches 78.26% and 63.61%. Offline bootstrap
reaches 89.86% and 64.51%. These comparisons share the exact future rows; their
training and starting-state differences are listed below.

## Matched XGBoost comparison

The added CPU baseline uses the original reference tree recipe: one shared
histogram booster, 150 rounds, depth 3, eta 0.04, seed 73, one thread,
inverse-frequency fitting weights and a fixed 0.5 threshold. It receives the
same 17 causal features without room identity. Neither validation nor future
labels fit the trees. No tree settings were selected from the scores.

| Room | XGBoost all prefix | XGBoost exact blocks | Liquid scratch bootstrap | Liquid live empty start |
| --- | ---: | ---: | ---: | ---: |
| H1 | 68.18 | 69.79 | 69.11 | 66.72 |
| H2 | 72.32 | 62.30 | 74.11 | 73.19 |
| H3 | 73.16 | 73.70 | 48.51 | 71.88 |
| H4 | 56.99 | 57.30 | 56.12 | 49.57 |
| H5 | 82.07 | 79.35 | 89.86 | 78.26 |
| H6 | 61.34 | 62.20 | 64.51 | 63.61 |
| Equal-room mean | **69.01** | **67.44** | **67.04** | **67.21** |

These are balanced accuracy percentages on exactly joined future timestamps and
truths. The all-prefix booster fits 20,718 unique rows; the block control fits
exactly the 20,088 rows used by offline liquid gradient fitting, omitting the
same incomplete contiguous blocks. Six replica streams never multiply training
rows. Saved-model reload and six-replica tree predictions are exactly equal.

The strict block comparison puts XGBoost 0.40 percentage points above offline
scratch on average; each wins three rooms. All-prefix XGBoost beats live liquid
by 1.81 points and wins four rooms; liquid wins H2 and H6. This is neither an
all-room liquid accuracy win nor a demonstration of XGBoost's best achievable
score. It is one fixed reference recipe and one seed.

The live liquid model also learns from the validation-period prefix as time
passes; the offline model selects and calibrates there. Trees use that period
only for reporting. Liquid future results were already known when this new tree
protocol was written. No liquid fitting or tree tuning followed these scores.
H4 validation has one class; its reported validation balanced score therefore
cannot assess both recalls. Every future room has both classes. H5 has only
12 vacant examples, so its 89.86% liquid result is not a six-room result.

Reproduce with the existing dataset cache and pinned environment:

```sh
python reporting/xgboost_compare.py
python reporting/verify_xgboost.py
```

[Protocol](reporting/xgboost-protocol.json),
[results, versions, timing and hashes](evidence/xgboost-seed-73/results.json),
[independent row joins and metric checks](evidence/xgboost-seed-73/verification.json),
and saved boosters, unique fitting-row lists and timestamped predictions in the
same evidence folder make the comparison inspectable. The recorded tree fit
wall times exclude Python import and preprocessing; they are not an end-to-end
speed comparison against liquid training or serving. No efficiency win is claimed.

## What adapted

The live run accepted **11 room-calibration updates**. Each room learned at least
one correction. It accepted **zero shared supervised-weight or topology proposals**:
nine prospective proposals failed room gates, and eight fitting opportunities
waited for class evidence. Reinforcement policy credit still updated from outcomes.
The active graph stayed at one board, five experts, three liquid layers per expert,
eight neurons per layer, two root delta-attention heads and zero hyperedges.

The architecture can grow and shrink in controlled lifecycle tests. This recorded
data run does not demonstrate successful structural evolution. Its flat evolution
plot is an observed result. Future scores also show that finite calibration gates
cannot guarantee accuracy on new inputs.

- [Validation learning](figures/learning.svg) and [live prediction learning](figures/live-learning.svg).
- [Future comparison](figures/future.svg), [active architecture](figures/evolution.svg), and [room corrections](figures/calibration.svg).
- Actual saved layouts: [before](figures/before.svg), [during](figures/during.svg), [after](figures/after.svg).
- [Visual report](figures/index.html), viewable after cloning, and [final correction coefficients](evidence/live-seed-73/learning-state.json).
- [Independent online receipt audit](evidence/live-seed-73/audit.json).

## Model artifacts

[Full live runtime](evidence/live-seed-73/final.pt) contains room corrections, all
36 private states, histories, bounds, policy and pending trial state. It reloads
with zero pending labels. Use that artifact to continue adaptive execution.
[Active graph only](evidence/live-seed-73/trained-graph.pt) excludes room corrections
and temporal state; it is not the whole learned multistream model.

[Calibrated offline scratch graph](evidence/acceptance-seed-73/scratch/calibrated.pt)
is selected using validation. [Equally retrained graph](evidence/acceptance-seed-73/pretrained-retrained/calibrated.pt)
provides the same-prefix comparison. Initial and intermediate checkpoints remain
in their run directories. Prediction records and exact executed-source archives
sit alongside the models.

## Stream fidelity and costs

Independent one-room runs, six-stream runs and 36-stream runs produce identical
answers for both new models. Maximum probability differences are 0.0000001193 for
scratch and 0.0000000597 for live. Frozen tensor fingerprints stay unchanged and
future feedback calls are zero.

The scratch replay took 18.19 seconds for 8,056 logical predictions across six
streams and 75.35 seconds for 48,336 across 36 streams. Counted private-state tensors
increased from 4,704 to 28,224 bytes. Those counts can repeat aliased correction
storage and exclude model weights, Python objects and process memory. There is
one shared 35,249-parameter graph, rather than one weight copy per stream.

The live worker completed in 1,841.054 seconds with 742,456 KiB peak resident memory.
Its own future replay and the independent replay are separate recorded checks.
The scores above reproduce exactly between them.

## Protocol

One randomly initialized model serves six room contexts with six logical streams
per room. Each room's six feeds repeat the same recorded timeline. They test
execution and private state; they are not independent observations or new data.

The experiment uses the original twelve HPDmobile environmental and ground-truth
archives. [Dataset records](evidence/seed-73/dataset.json) retain archive hashes,
row counts, class counts and exact chronological boundaries. The first twelve
selected days fit weights; the next two select checkpoints and fixed policies.
Later rows form the untouched future test. Selected days can have calendar gaps.

Offline fitting uses 558 complete, contiguous blocks of 36 rows: 20,088 gradient
examples from 20,718 prefix rows. Short segment tails do not enter block fitting.
The scratch model selected epoch 8 and stopped after epoch 12. The equally retrained
pretrained model selected epoch 3 and stopped after epoch 7. Both use AdamW, bounded
backpropagation windows and validation early stopping. Offline fitting keeps the
architecture fixed. It does not demonstrate online structural adaptation.

The separate live run starts empty directly under all 36 streams. Delayed external
labels arrive after one to three intervening samples; duplicate room feeds train
once. Rewards grade issued answers. Shared candidates must pass prospective checks
in every room. Room calibration also has availability and class checks. Future
inference returns no labels or rewards to either model.

The policy grid was fixed in [protocol.json](reporting/protocol.json). Policy and
probability-scale selection sees validation only. Future scores did not change the
training, policy grid, gates or selected checkpoints.

## Data limits

H4 validation has 576 occupied examples and no vacant examples. Its validation
score measures the available class, not two-class balanced accuracy. Future H4
contains both classes. H5 future has only 12 vacant and 276 occupied examples;
its high score has limited minority-class support.

All future scores use balanced accuracy: the average of vacant recall and occupied
recall. The records also give ordinary accuracy, class counts, Brier error and
quorum coverage. A Brier error measures squared probability error; lower is better.
A quorum means enough experts agreed under the model's voting rules.

The original pretrained checkpoint learned H1–H3 and used H4 for selection. The new
shared fitting prefixes include all six rooms. The equally retrained comparison
uses that same new prefix and validation. Earlier hypergraph H5/H6 checkpoints were
trained separately for fourteen days and retain their historical state. Their
future rows match, but their training and starting-state protocols differ.

## Execution and provenance

[Original-checkpoint fidelity](reporting/reference-fidelity.json) reproduces the
preceding H5 and H6 trained models through 36 copied historical contexts. Decisions
and quorum are exact. Maximum probability difference is 0.0000001192092896. This
is execution evidence; it does not prove six-room shared learning.

The live run's archived root source files match the delivered root source files.
The offline run started before later calibration and scheduling fixes. Its exact
executed source archive is retained. Five files differ: adaptation.py, cold_start.py,
multistream.py, stream_replay.py and test_multistream.py. Offline fitting disables
feedback and topology changes. The independent acceptance check serves its saved
weights through the delivered code; the source difference remains disclosed.

The preceding experiment is unchanged. BASELINE.json checks all 31 pinned source
and configuration files. Full source and artifact hashes belong in MANIFEST.json.

## Local checks

- 94 regression tests passed in 79.206 seconds.
- Four additional lifecycle and causality tests passed in 2.321 seconds.
- CLI scratch creation, batch prediction, reversed stream feedback and separate
  process reload passed; all pending labels drained.
- Ruff checks passed. Final artifact and document checks are recorded separately.

The lifecycle fixtures exercise real adoption, all-stream state migration,
monitoring rollback, schema expansion and pending-label restore. Controlled tests
are distinct from topology changes observed in the recorded-data experiment.

## Costs and limits

The offline worker completed in 1,599.148 seconds, with 420,768 KiB peak resident
memory. These are local process measurements, not cloud prices or token costs.
The worker caps address space at 4 GiB, CPU at 7,200 seconds, wall time at 7,500
seconds and individual files at 256 MiB. Tensor reservations are separate estimates.

Private state and compute are not free. State grows with open streams and active
paths; batching can share a forward call but does not remove replicated work.
Benchmark timings include Python coordination and are not durable SQLite service
throughput. Concurrent local jobs also affect timing; this is not an isolated speed
benchmark. Superseded development runs are excluded from acceptance costs and
retained outside this delivery; the final-run measurements are not total project
costs.

The host exposes an AMD Phoenix3 adapter in PCI inventory, but this environment has
CPU-only PyTorch and no compute/render device access. Actual GPU execution and
CPU/GPU parity remain unverified. No drivers or system settings were changed.
No GitHub Actions were used.
