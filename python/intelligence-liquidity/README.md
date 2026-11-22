# Intelligence liquidity

These are the runnable experiments behind the article and vlog. Download the data from the original publishers. The code checks the archive fingerprints before scoring, and the recorded results let you compare your run.

| Experiment | What to inspect | Recorded outcome |
| --- | --- | --- |
| [Library LTC and browser replay](library-cell/README.md) | The original model using `ncps`, both GRU comparisons, a last-reading baseline, and replay with an edited time gap | On the larger test file with a fixed row-drop mask, balanced accuracy was 92.75% for LTC, 92.96% for the original GRU, and 88.76% for the last-reading baseline. The smaller GRU was chosen after looking at test scores. |
| [Custom LTC and same-office benchmark](custom-ltc/README.md) | The implemented cell, fitted weights, recorded cadence, identical targets after a fixed row-drop mask, and Python replay | Model rankings depend on the file and on whether you count missed occupancy or false alarms. Both files come from one office. |
| [Training, resume, scoring and rollback](lifecycle/README.md) | Two saved snapshots, validation feedback, model promotion, exact resume, the Python stopping exercise, and stale-input handling | Epoch 7 stayed best, training stopped at 9, and resuming after epoch 2 reproduced the final snapshot. The promotion check kept the older model. |
| [Unseen-room transfer](external-validation/README.md) | Frozen checkpoints, prewritten input mapping, and an independent UCI archive | Both liquid models missed every one of the 1,880 occupied windows. The second room used different sensors and timing, and both humidity inputs were unavailable. |
| [Offline heating control](building-control/README.md) | Logged commands, measured heat and temperature, the reward function, the fitted action table, and a later time split | On 532 later intervals the policy recommended off 468 times and abstained 64 times. The log cannot show what would have happened had it controlled the radiator. |

The examples use Python 3.12 and CPU PyTorch 2.7.1 for the neural models. The first example also needs `ncps` and Node for its parity check. The heating-policy code needs only Python 3.11 or later.

## Data provenance

- **Occupancy Detection** by Luis Candanedo and Véronique Feldheim, [UCI DOI 10.24432/C5X01N](https://doi.org/10.24432/C5X01N), CC BY 4.0. The example CSVs and embedded readings are small extracts with normalized headers or reformatted fields. The accompanying JSON includes scores, derived intervals and model parameters.
- **Room Occupancy Estimation** by Adarsh Pal Singh and Sachin Chaudhari, [UCI DOI 10.24432/C5P605](https://doi.org/10.24432/C5P605), CC BY 4.0. Download from the publisher for the second-room check.
- **Sub-hourly measurement datasets from 6 real buildings** by Igor Sartori and colleagues, version 3, [Mendeley DOI 10.17632/xztfbtsgys.3](https://doi.org/10.17632/xztfbtsgys.3), CC BY 4.0. The heating example uses the ZEB Living Lab file and publishes calculated summaries. Download the measurements from the publisher.

The closing proposals in the article and video are future experiments. This repository contains no measurements for closed-form networks or temporal attention and no live heating trial.

## Compare with the original paper

The original LTC paper reports 94.63% mean occupancy accuracy across five runs.
Its closest comparison, a continuous-time RNN, reports 94.54%. That benchmark
uses 32-reading windows from the same source dataset. Our custom benchmark uses
eight readings, different baselines and its own split and scoring. The
second-room check also changes sensors, missing inputs and sampling cadence.
We did not measure accuracy against training-set size, so these experiments
cannot show which model learns from fewer examples.
[Original LTC benchmark and occupancy protocol](https://arxiv.org/html/2006.04439v4)

The occupancy model updates its recurrent state and predicts from the final
state. The training loop checks validation loss after each epoch, keeps the
weights from the best qualifying epoch and stops after two checks with less
than 0.005 improvement. The heating policy uses a separate table of state and
action estimates, scored by heat use and temperature. The experiments have
no attention layer and do not train an LTC with reward.
