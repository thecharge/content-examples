# Measured learning and structural outcomes

These are CPU runs of the supplied HPDmobile protocol. All 12 archive hashes, original checkpoints and frozen reference scores are checked. The original contextual learner is actually rerun, and its later scores must match the published receipts. H1–H3 supply the original 12,087 fitting rows; H4 supplies 2,016 selection rows. Those initial fits are reused.

The new model carries the original contextual learner forward. Its structural proposals are an additional learning path. The weight-only control also retains contextual learning, while disabling every structural action. No settings or checkpoint were selected using the new test scores.

## Learning over time

![Trailing balanced accuracy across the stream](reporting/learning-trends.svg)

Curves use only the trailing 576 observations, with at least 16 examples of each class. Lines show the mean of seeds 73, 83 and 97; shading shows their range. Dotted lines mark the end of the original first week and the start of scoring after day 14. The index counts observed windows, not uninterrupted calendar time. Gaps reset recurrent states. Overlapping curves indicate matching behavior, not missing series. The separate range below contains only 12 vacant H5 examples, so its final score cannot be read directly from the class-filtered rolling curve.

## Exact comparison ranges

| Range | Home | Rows | Vacant | Occupied |
| --- | --- | ---: | ---: | ---: |
| First week | H5 | 2010 | 784 | 1226 |
| First week | H6 | 2016 | 463 | 1553 |
| After day 14 | H5 | 288 | 12 | 276 |
| After day 14 | H6 | 2016 | 1030 | 986 |

Each runtime begins on day one. Previous answers arrive before the next prediction. The later range therefore follows the same 14-day online warmup as the article. The final current answer remains unrevealed.

## First-week balanced accuracy

Values are percentages. Balanced accuracy gives vacant and occupied recall equal weight. Raw accuracy, class counts, Brier error, quorum and paired intervals remain in each receipt.

| Seed | Home | Frozen | Original adaptive rerun | New adaptive | Context + weights only | Saved XGBoost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 73 | H5 | 64.12 | 64.12 | 64.12 | 64.12 | 56.20 |
| 73 | H6 | 74.30 | 74.30 | 74.30 | 74.30 | 77.54 |
| 83 | H5 | 59.09 | 59.09 | 59.09 | 59.09 | 56.20 |
| 83 | H6 | 74.68 | 77.06 | 77.06 | 77.06 | 77.54 |
| 97 | H5 | 66.24 | 66.24 | 66.24 | 66.24 | 56.20 |
| 97 | H6 | 74.25 | 74.25 | 74.25 | 74.25 | 77.54 |

## Balanced accuracy after day 14

Values are percentages. Balanced accuracy gives vacant and occupied recall equal weight. Raw accuracy, class counts, Brier error, quorum and paired intervals remain in each receipt.

| Seed | Home | Frozen | Original adaptive rerun | New adaptive | Context + weights only | Saved XGBoost |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 73 | H5 | 73.73 | 82.61 | 82.61 | 82.61 | 44.38 |
| 73 | H6 | 63.42 | 62.54 | 62.54 | 62.54 | 57.75 |
| 83 | H5 | 73.01 | 87.50 | 87.50 | 87.50 | 44.38 |
| 83 | H6 | 60.53 | 64.57 | 64.57 | 64.57 | 57.75 |
| 97 | H5 | 78.99 | 89.86 | 89.86 | 89.86 | 44.38 |
| 97 | H6 | 62.74 | 62.64 | 63.08 | 63.08 | 57.75 |

## Architecture decisions

The six streams completed 51 prospective adaptive trials and retained 1. Contextual updates run separately and are recorded under `contextual_receipts`. A rejected structural proposal therefore does not mean the running model stopped learning.

| Action | Completed | Retained |
| --- | ---: | ---: |
| learn | 27 | 1 |
| spawn | 7 | 0 |
| deepen | 2 | 0 |
| split | 3 | 0 |
| grow_heads | 0 | 0 |
| prune_heads | 4 | 0 |
| retire | 8 | 0 |
| share | 0 | 0 |

Each growth trial first checks function preservation and the candidate's resource bounds. Its earlier fit and validation data precede the subsequent check predictions. Both paired graphs continue receiving the same accepted contextual correction. A quality gate measures a finite block; it does not guarantee that every later window improves monotonically.

| Seed | Home | Accepted contextual updates | Retained trial actions | Final experts | Final root heads |
| ---: | --- | ---: | --- | ---: | ---: |
| 73 | H5 | 1 | None | 5 | 2 |
| 73 | H6 | 1 | None | 5 | 2 |
| 83 | H5 | 1 | None | 5 | 2 |
| 83 | H6 | 1 | None | 5 | 2 |
| 97 | H5 | 2 | None | 5 | 2 |
| 97 | H6 | 0 | learn | 5 | 2 |

Do not interpret ordinary contextual gains as evidence that extra experts or heads were necessary. The tables show separately which changes were retained. Inspect each prospective receipt for its own paired scores and class-specific effects.

## Controlled adaptation

These artificial streams are separate from occupancy accuracy. All original inputs are zero; a new input supplies `2 * truth - 1`, with `truth = (index // 3) % 2`. Labels arrive after prediction. The first 48 labels fit a candidate and the next 48 grade it. No action or candidate prediction is forced.

| Seed | Retained first action | Active / candidate balanced accuracy | Final experts |
| ---: | --- | ---: | ---: |
| 73 | deepen | 50.0% / 100.0% | 6 |
| 83 | split | 50.0% / 100.0% | 5 |
| 97 | retire | 50.0% / 100.0% | 4 |

Separate behavioral tests check learned proposals for attention growth and pruning, the minimum head count, preservation across recurrent sequences, replayed reinforcement credit, native contextual parity, early stopping, snapshot fine-tuning and point-in-time restoration. A constructed perfect cue is not evidence of universal occupancy accuracy.

## Inspect and reproduce

- [Seed 73 receipt](evidence/seed-73/results.json)
- [Seed 83 receipt](evidence/seed-83/results.json)
- [Seed 97 receipt](evidence/seed-97/results.json)
- [Architecture and Mermaid diagrams](ARCHITECTURE.md)
- [Model inspection, training and recovery commands](OPERATIONS.md)

Each seed directory contains raw H5/H6 predictions, full active and weight-only CPU checkpoints, and recoverable mutation boundaries. `verify.py` recomputes scores, intervals, structural gates and chronology; checks source/archive/checkpoint/artifact fingerprints; and restores the final topology. `--rerun` performs the complete computation again and compares prediction hashes.

Local verification passed 36 adaptive behavioral tests and 20 unchanged reference tests. House records are independently checked by the verifier before this report is generated. No GitHub Actions were used.

Regenerate this report with `python reporting/generate.py` after installing `reporting/requirements.txt`. Plot/report fingerprints and the generator's own fingerprint are saved in `reporting/manifest.json`.

These are two previously studied homes and three seeds. Later H5 has only 12 vacant examples. Finite check blocks, shared research history and one configured controller limit the conclusions. The reported outcomes establish behavior on these streams; they do not guarantee monotonic accuracy, unlimited adaptation or better scores from every larger graph.
