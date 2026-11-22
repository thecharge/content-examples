# Adaptive liquid results

This records the original run before the upstream-commit defect was found.
The original revision counts describe the internal replay path. A later test
showed that the returned earlier proposal and its check could remain stale.
[Corrected model](../adaptive-liquid-v2/README.md)

The measured results challenge the proposed cross-site and sample-efficiency gains. At 25% and 50% fitting budgets the full model scored 50% balanced accuracy and accepted zero windows on each occupancy test site. At 100% it achieved the highest mean balanced accuracy on UCI test 1, while stronger baselines and ablations led on UCI test 2, the Danish apartments and both Graz rooms. For heating, the full model improved temperature error over the neural baselines and increased thermal-energy error. The bucket mean had the lowest error on all four measured heating quantities.

All 123 frozen conditions completed on CPU. The 99 occupancy fits and 21 neural heating fits stopped early under the configured validation patience. The three mean controls had no optimizer steps. All 120 neural runs verified exact checkpoint restoration on their validation fixture. The seed-41 full occupancy snapshot also reproduced metrics and traces on all five holdouts.

## Occupancy

The table gives mean all-row balanced accuracy in percent over seeds 41, 42 and 43. Each row includes every eligible example, even when the model abstains. The tests contain 2,658 UCI test 1, 9,745 UCI test 2, 10,656 Danish, 28,515 Graz Room A and 9,783 Graz Room B windows. [Accuracy, balanced accuracy, coverage and timing by seed](metrics.json)

| Model | Budget | Parameters | UCI 1 | UCI 2 | Danish | Graz A | Graz B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 25% | 49 | 85.75 | 79.74 | 57.79 | 72.91 | 71.77 |
| gru | 25% | 345 | 85.50 | 79.49 | 58.94 | 64.90 | 74.37 |
| library | 25% | 407 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| liquid | 25% | 385 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| attention | 25% | 394 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| attention_aux | 25% | 421 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| adaptive | 25% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| no_feedback | 25% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| delta_only | 25% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| hard_only | 25% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| no_delta | 25% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| current | 50% | 49 | 85.83 | 79.47 | 55.96 | 68.96 | 68.01 |
| gru | 50% | 345 | 85.53 | 79.01 | 59.79 | 64.72 | 75.54 |
| library | 50% | 407 | 84.10 | 72.57 | 60.73 | 69.77 | 81.55 |
| liquid | 50% | 385 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| attention | 50% | 394 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| attention_aux | 50% | 421 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| adaptive | 50% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| no_feedback | 50% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| delta_only | 50% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| hard_only | 50% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| no_delta | 50% | 6266 | 50.00 | 50.00 | 50.00 | 50.00 | 50.00 |
| current | 100% | 49 | 86.21 | 76.57 | 54.77 | 64.19 | 63.61 |
| gru | 100% | 345 | 85.72 | 78.07 | 58.84 | 62.99 | 73.91 |
| library | 100% | 407 | 85.28 | 71.01 | 55.79 | 67.32 | 79.62 |
| liquid | 100% | 385 | 85.91 | 76.58 | 61.80 | 73.17 | 75.40 |
| attention | 100% | 394 | 85.62 | 76.20 | 58.72 | 70.80 | 73.01 |
| attention_aux | 100% | 421 | 86.12 | 74.65 | 58.05 | 69.47 | 71.34 |
| adaptive | 100% | 6266 | 87.18 | 65.67 | 52.19 | 55.42 | 60.12 |
| no_feedback | 100% | 6266 | 86.44 | 66.18 | 52.33 | 56.33 | 62.23 |
| delta_only | 100% | 6266 | 86.46 | 66.18 | 52.32 | 56.34 | 62.28 |
| hard_only | 100% | 6266 | 86.42 | 66.15 | 52.32 | 56.28 | 62.09 |
| no_delta | 100% | 6266 | 86.28 | 73.19 | 66.25 | 74.51 | 80.00 |

The full-budget model's fresh Graz balanced accuracy was 55.42% in Room A (54.73–56.43% across seeds) and 60.12% in Room B (57.50–62.42%). The no-delta ablation reached 74.51% and 80.00%. Its higher scores on these two rooms and the previously explored Danish site challenge a benefit from the delta path in this configuration. The 25% current-row baseline reached 72.91% and 71.77% on Graz and used 49 parameters. UCI test 1 favored the full model at 87.18%. The same model scored 65.67% on UCI test 2, where the 25% current-row baseline reached 79.74%.

Each cell below gives mean coverage / mean accepted-only balanced accuracy, both in percent. A dash means no seed had accepted examples of both classes. Where balanced accuracy exists for only some seeds, the mean uses those seeds, with exact coverage and unavailable values retained in the receipts. Every baseline accepted every eligible row, so its accepted-only scores equal its all-row scores.

| Model | Budget | UCI 1 | UCI 2 | Danish | Graz A | Graz B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive | 25% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| no_feedback | 25% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| delta_only | 25% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| hard_only | 25% | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 |
| no_delta | 25% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| adaptive | 50% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| no_feedback | 50% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| delta_only | 50% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| hard_only | 50% | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 | 100.00 / 50.00 |
| no_delta | 50% | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — | 0.00 / — |
| adaptive | 100% | 100.00 / 87.18 | 100.00 / 65.67 | 100.00 / 52.19 | 100.00 / 55.42 | 100.00 / 60.12 |
| no_feedback | 100% | 87.41 / 74.65 | 92.10 / 59.66 | 98.96 / 51.93 | 97.46 / 53.47 | 96.46 / 56.94 |
| delta_only | 100% | 87.41 / 74.67 | 92.10 / 59.65 | 98.96 / 51.92 | 97.47 / 53.48 | 96.46 / 57.02 |
| hard_only | 100% | 100.00 / 86.42 | 100.00 / 66.15 | 100.00 / 52.32 | 100.00 / 56.28 | 100.00 / 62.09 |
| no_delta | 100% | 100.00 / 86.28 | 100.00 / 73.19 | 100.00 / 66.25 | 100.00 / 74.51 | 100.00 / 80.00 |

The full model's 25% and 50% Graz runs changed an upstream proposal and replayed a dependent stage 57,030, 57,030 and 28,515 times per seed in Room A. Corresponding Room B counts were 19,566, 19,566 and 9,783. These are stage events, so a window can contribute two. Coverage stayed at zero. The 100% full model accepted all Graz rows and made zero revisions there. The forced-disagreement check changed an upstream prediction by 0.07904 and a downstream replayed output by 0.00585 in model units, with the affected stage still abstaining. [Forced and natural route evidence](verification.json) · [Run receipts](receipts)

## Heating

All rows below score the same 532 logged outcomes, with 244 off and 288 on. The shared training-support gate passed both actions on 380 rows. Errors are seed means and use the recorded action. The reward error refers to the physical utility computed from observed energy and temperature discomfort. The learned bonus applies only to ranking model candidates.

| Model | Parameters | Temperature MAE °C | Thermal MAE kWh | Discomfort MAE °C·h | Physical reward MAE | Picks by seed 41 / 42 / 43 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive | 6649 | 0.44606 | 0.32526 | 0.00020 | 0.32490 | 380 / 0 / 0 |
| attention | 468 | 0.47548 | 0.26778 | 0.00023 | 0.26739 | 380 / 380 / 380 |
| delta_only | 6649 | 0.44552 | 0.32525 | 0.00020 | 0.32489 | 380 / 0 / 0 |
| hard_only | 6649 | 0.44547 | 0.32526 | 0.00020 | 0.32489 | 380 / 380 / 380 |
| liquid | 459 | 0.47520 | 0.23862 | 0.00020 | 0.23826 | 380 / 380 / 380 |
| mean | 0 | 0.31398 | 0.03073 | 0.00018 | 0.03109 | 380 / 380 / 380 |
| no_delta | 6649 | 0.44616 | 0.32547 | 0.00020 | 0.32510 | 380 / 0 / 0 |
| no_feedback | 6649 | 0.44549 | 0.32526 | 0.00020 | 0.32489 | 380 / 0 / 0 |

The full model's temperature MAE ranged from 0.43327 to 0.45425 °C and thermal error ranged from 0.32286 to 0.32715 kWh. It chose on for all 380 supported windows at seed 41. At seeds 42 and 43 the critics vetoed both candidates on all 380 supported windows, and performed 1,064 and 2,128 total stage replays across the two candidates and all rows. Switching to hard-only routing on those same weights recovered 380 recommendations per seed. The additive learned bonus caused zero rank flips among passing candidates in the measured runs. The learned component's observed effect was therefore in routing and abstention. These offline choices cannot establish energy savings, comfort improvements, or the outcome of an unchosen action. [Same-weight gate replay](verification.json) · [Per-row outcome errors and candidate traces](receipts)

## Fitting and verification

The occupancy fits used 1,555, 3,180 and 6,493 windows at the three budgets, with a shared 1,636-window validation split. Heating used 871 fitting and 228 validation windows. Seeds 41–43 selected full occupancy checkpoints at epochs 2/2/1 at 25%, 1/1/1 at 50%, and 4/4/5 at 100%. Actual stop epochs were 4/4/3, 3/3/3 and 6/6/7, with 28/28/21, 39/39/39 and 156/156/182 optimizer updates. Full heating selected epoch 1 and stopped at epoch 3 with 12 updates for each seed. Mean full-model fitting time was 12.12, 17.38 and 62.03 seconds for occupancy and 5.55 seconds for heating. At the full occupancy budget, mean scoring time was 5.03 seconds for Graz A and 1.74 seconds for Graz B. Fit and test times reflect concurrent local CPU work.

Both the causal delta prefix test and the critic gradient isolation test passed. Invalid hard checks vetoed a candidate under deliberately high learned scores. A forced negative critic logit triggered the normal retry path without the force flag. The saved heating checkpoint reproduced candidate outcomes, score components and the hard-support veto from observations alone. [Verification receipts and population digests](verification.json)

The three seeds measure training variability on shared, overlapping test windows. Mean, range and individual seed results are descriptive. Graz A and B share one office site, so they do not constitute independent building samples. The comparison measures transfer of a classifier fitted on UCI, including changes in cadence, occupancy measurement, and the conservative sensor embargo. The report preserves every condition, including low coverage and failed comparisons. [Source and alignment audit](source-audit.md) · [Frozen criteria](freeze.json)
