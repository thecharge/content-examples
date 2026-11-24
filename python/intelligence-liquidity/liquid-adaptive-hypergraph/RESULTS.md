# Measured results

The new experiment adds overlapping liquid memberships and durable revisions.
The following protocols answer different questions. Their scores are kept separate.

## Online adaptation on the original streams

Every prediction precedes its label. Later labels can update the model before subsequent predictions.
The same twelve HPDmobile archives, original seed checkpoints, features and chronology were checked.
The preceding model's predictions were reused after source/artifact fingerprint verification.

| Room | Previous, seeds 73 / 83 / 97 | Hypergraph, seeds 73 / 83 / 97 | Mean previous / new |
| --- | --- | --- | --- |
| H5 | 82.61% / 87.50% / 89.86% | 82.61% / 87.50% / 89.86% | 86.65% / 86.65% |
| H6 | 62.54% / 64.57% / 62.64% | 62.78% / 64.57% / 62.64% | 63.25% / 63.33% |

Completed structural/weight proposals: 50. Accepted: 1.
All six final models retained five experts, one board, two attention heads and zero computational hyperedges.
Seed 73 in H6 accepted a neuron contraction: 34,754 parameters and 70,492 work units at the end.
Its later balanced accuracy rose from 62.54% to 62.78%. The other five comparisons matched the preceding model.
This dataset run demonstrates one accepted contraction, but no useful learned hyperedge growth.
The controlled run below establishes the new mechanism separately.

## Completed training, then unseen inference

Both models were trained separately on the first 14 selected days of each room.
The later chronological block was held out. No label or reward from that block entered the model.
Learned tensors and topology were frozen and verified unchanged after inference.
Only recurrent/delta state could evolve from sensor readings. A still-monitored candidate used its retained fallback.

| Room | Previous, seeds 73 / 83 / 97 | Hypergraph, seeds 73 / 83 / 97 | Mean previous / new |
| --- | --- | --- | --- |
| H5 | 73.73% / 73.01% / 87.14% | 73.73% / 73.01% / 87.14% | 77.96% / 77.96% |
| H6 | 62.54% / 64.57% / 62.64% | 62.54% / 64.57% / 62.64% | 63.25% / 63.25% |

These are unseen future observations in rooms with an earlier adaptation prefix.
They are not entirely unseen rooms. Raw training/test timestamps and both frozen snapshots are retained under `evidence/strict`.

## Cost

The initial model and five final models use 35,249 parameters versus 34,749 previously: 1.44% more.
The operation proxy remains 72,832 units for those unchanged serving topologies; the accepted contraction reduces it to 70,492.
The additional parameters belong to the larger action policy. This is added flexibility with a parameter cost, not a measured speedup.
Neural benchmark timings exclude SQLite persistence and were measured while independent seed runs shared the host.
Do not use them as an isolated old/new latency comparison.

## Controlled hyperedge lifecycle

The controlled cue directly determines the class. The action-value bias explicitly requests a connection.
The first 48 labels fit; the following 48 predictions check; another 48 monitor.
The connection was accepted with balanced-accuracy gain 0.500 and Brier gain 0.227536.
Monitoring rollback: False. The ledger retains 291 committed revisions.
The serving model changed from zero to one hyperedge. Its fitted schema also included the new cue.
This is a disclosed mechanism test. It does not show that the hyperedge was necessary or selected autonomously.
The final three pictures explicitly construct overlapping neuron memberships, edge splitting and edge merging.
They are excluded from counts of learned mutations.

## Inspect the evidence

- [Accuracy chart](reporting/accuracy.svg) and [machine-readable summary](reporting/summary.json).
- [Controlled evolution](evidence/evolution/index.html), [prediction trend](evidence/evolution/trend.svg), and full stage JSON files.
- Online records: [seed 73](evidence/seed-73/results.json), [seed 83](evidence/seed-83/results.json), [seed 97](evidence/seed-97/results.json).
- Strict inference records and snapshots: `evidence/strict/{new,previous}-seed-{73,83,97}`.
- [Verification instructions](TESTING.md) and [full vision mapping](VISION.md).

## Limits of these results

H5 contributes only 288 later observations, including 12 vacant observations; H6 contributes 2,016 with 1,030 vacant and 986 occupied.
The three seeds reuse the same two rooms. They are not three independent deployments.
No result guarantees future accuracy, useful growth on every stream, or AMD GPU performance.
No GitHub Actions run was used.
