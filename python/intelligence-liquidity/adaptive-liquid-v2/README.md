# Clock-aware liquid model and bounded occupancy retry

The correction is verified on 64 real Aalto windows from the restored V2 seed-41, 100% checkpoint. Forced stage-2 revision called the three liquid verifiers 2, 2, and 1 times. The returned stage-1 logit changed by up to 0.161327 and its verifier state by up to 0.024156. The stage-2 replay received the rechecked stage-1 state from the verifier. Assertions also cover rollback of a nonfinite recheck, invalid clock veto, and exhaustion of a one-retry budget. The pre-score test used 64 UCI windows. In a separate acceptance check, a forced invalid initial verifier score yielded zero accepts without replay and 64 accepts after a successful recheck. Forcing a stage-3 revision yielded verifier call counts of 1, 2, and 2 and the replay received the refreshed stage-2 evidence.

## Results

Each row averages nine matched Aalto runs over seeds 41–43 and 25%, 50%, and 100% fitting-data budgets. All used the same 1,409 windows. The fixed logit cutoff is zero (0.5 probability). The other cutoff maximizes balanced accuracy on UCI validation. Accuracy and balanced accuracy count abstention as incorrect in this table. AUROC and average precision score the proposals across all rows. Misses count accepted false negatives, with positive abstentions reported separately in the receipts.

| Model or route | Fixed balanced | Calibrated balanced | Calibrated accuracy | AUROC | AP | Misses | Coverage | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current observation | 68.68% | 71.83% | 74.88% | .8743 | .9187 | 100.2 | 100% | .4264 |
| GRU | 67.92% | 76.72% | 75.18% | .8617 | .9022 | 263.2 | 100% | .3169 |
| Library LTC | 62.26% | 56.61% | 59.66% | .5192 | .5740 | 227.6 | 100% | .0316 |
| Custom fused LTC | 71.90% | 78.73% | 79.43% | .8558 | .8930 | 147.0 | 100% | .4842 |
| V1 refitted | 74.25% | 76.83% | 75.96% | .8371 | .8799 | 232.7 | 100% | .3540 |
| V2 without trained correction | 76.33% | 70.86% | 73.07% | .8500 | .8979 | 145.6 | 100% | .3581 |
| V2 without delta, direct | 71.35% | 78.62% | 79.17% | .8739 | .9158 | 154.2 | 100% | .4740 |
| V2 without delta, fitted policy | 68.25% | 72.74% | 74.13% | .8738 | .9158 | 118.6 | 91.06% | .4483 |
| V2 direct | 74.31% | 72.59% | 74.36% | .8501 | .8978 | 150.6 | 100% | .3804 |
| V2 fitted policy without process bonus | 69.81% | 68.19% | 69.63% | .8501 | .8979 | 148.4 | 90.40% | .3492 |
| V2 fitted policy with process bonus | 69.91% | 68.22% | 69.66% | .8500 | .8978 | 148.4 | 90.51% | .3473 |

V2 with the process bonus averaged 390.1 attempted revisions, 12.8 helped decisions, 9.4 hurt decisions, and 53.4 positive abstentions per run. The process bonus changed 481 active action sequences across the nine runs, a mean of 53.4 per run. The raw comparison counts 486 changes. Five were stage-3 choices computed within the batch after stage-2 abstention. The abstention veto prevents any subsequent revision or accepted answer for those rows. The retained audit shows an active change on row 136 from keep to revise, with reward 0.94 versus 1.00 for the correct unrevised answer. Row 460 changes from false alarm to correct. Row 80 changes from correct to a miss.

At seed 41 and 25%, the [preserved V1](../adaptive-liquid-v1/README.md) stopped after 28 updates and predicted empty. That earlier run's Graz result was 56.52% balanced accuracy at the original cutoff and 62.95% after UCI-only calibration, versus 75.74% for its no-delta variant at the original cutoff. This sweep refits V1 with weighted loss and a longer allowance. V1 and V2 ran the full 240 steps and selected checkpoints averaging 178.7 and 101.3 steps respectively. At steps 24, 120, and 240, mean weighted validation loss was 1.581, 1.259, and 1.054 for V1, and 1.494, 1.003, and 1.235 for V2. V2 improved sooner and lost validation performance with continued training. Retained curves include every 24th step, fit-batch loss, validation loss, and the 120-step and 240-step points where reached. The other networks used the same 240-step cap, with early stopping after at least 120 updates and three validation checks without improvement. Seed-41 V2 at 25% selected step 96 after completing all 240 updates. The fit receipts include class counts, exact reload checks, and wall times. Optimizer updates were matched by allowance, with measured wall time differing across architectures. The initial scratch-directory failure briefly left two fit queues active. A clean rerun at eight fit processes and one Torch thread per process reproduced all 72 frozen checkpoint hashes and exact validation reloads. Mean network fit time in that rerun was 32.06 seconds for V1 and 19.42 seconds for V2, plus 1.04 seconds to fit the V2 policy. The per-run timings are in `timing-recheck.json`. The random revision schedule shares its seeded generator with V2 minibatch sampling, so the seed matches across models and the precise minibatch sequences can differ.

## Equations and time

For each hidden channel, write the frozen-rate equation as `C dh/dt = B - A h`. Here `h` is the normalized hidden value, `C` is positive capacitance, `A` is summed positive conductance, and `B` is the conductance-weighted sum of reversal and leak-rest values. The shared time unit is minutes, so `A/C` has units per minute. The stored timestamp encoders subtract minute timestamps and use one nominal minute for the first observation in each reset window. All capacitances and leaks in the custom cells use softplus plus a positive floor.

Algorithm 1 freezes the gates during each substep and applies the fused update `(C*h + dt*B)/(C + dt*A)`. V2 applies the exact constant-rate update for that same interval, `h + (B/A - h)*(-expm1(-dt*A/C))`, and refreshes the recurrent conductance every substep. The V2 cell uses one aggregate positive learned gate per output and one corresponding reversal value. The custom fused baseline has per-connection sigmoid gates and learned reversal values. Numerical solver comparison holds the V2 gate and inputs fixed between solvers. At 2, 4, 8, and 16 substeps, exponential maximum errors against the 2,048-substep reference were .11390, .05847, .02926, and .01455. Fused errors were .18384, .10920, .06038, and .03179. The analytic fixed-rate test expected 0.38199077. [Hasani et al., arXiv:2006.04439v4, §2 Algorithm 1, §6 and room-occupancy appendix](https://arxiv.org/html/2006.04439v4)

The delta state is a matrix per window. Each key channel decays as `exp(-rate*elapsed_minutes)` with a learned positive rate in inverse minutes. Each write uses the difference between the token value and the value retrieved from the decayed state. A learned sigmoid gate scales that write and another gates the post-write read. The raw sensor channels enter every stage alongside memory initialized to a residual scale of 0.01. Relative to Kimi Delta Attention, this is a single-head physical-clock recurrence with token-independent rates, vector write and output gates, normalized queries and keys, and a trainable scalar memory scale. It has no language-model attention stack or chunked kernel. The verification compares the recurrence to an independently expressed calculation and tests prefix equality and sensitivity to an elapsed-time change. [Kimi Team, arXiv:2510.26692, §§2.2 and 3](https://arxiv.org/html/2510.26692v1)

The paper scores ordinary accuracy. The separate paper-style UCI check uses 32 observations, five UCI streams, ordinary cross-entropy, and uncalibrated raw accuracy. The chronological split, Adam settings, window-level prediction, and compact ncps model are this experiment's settings. It selected step 168, with UCI test accuracies 97.65% and 98.95%. That check uses Light and HumidityRatio in addition to the three transfer sensors and is confined to UCI. [UCI Occupancy Detection](https://doi.org/10.24432/C5X01N)

## Retry and process targets

At stages 2 and 3, the policy observes the current proposal logit, current and prior verifier logits, and remaining retry count. A revision recomputes the prior proposal and verifier and consumes one retry. A transaction retains the revised values and replay only when all its outputs are finite. The final hard veto checks the original sensor/clock inputs and latest proposal and verifier values. The policy can abstain. The model evaluates batched proposals even when individual rows have already abstained.

On UCI fit data, the offline procedure replays the four keep/revise paths and includes terminal abstention. The fixed payoff is +1 for correct, −2 for a miss, −1 for a false alarm, −0.20 for abstention, and −0.06 per attempted revision. Ridge regressions fit stage-3 terminal Q targets and stage-2 targets of immediate payoff plus the maximum fitted continuation value. Separate regressions learn early downstream correctness change and late upstream correctness change. The process variant adds 0.10 times that fitted change to revision values, including in the continuation calculation. Verifier logits are learned correctness signals. Process estimates are fitted changes in categorical correctness. Labels enter training rewards and post hoc scoring only. The receipt records attempt costs and actions even when a revision is rejected.

## Data and freeze

The eight-step transfer input uses temperature, relative humidity, and room CO2 at every site. Aalto comes from a separate lab site with different ventilation strategies. Its 1,430 source timestamps were strictly increasing, with gaps of 1–2,395 minutes. Windows spanning gaps over five minutes were excluded. The target is recorded occupant count greater than zero at the final observation. Inputs exclude airflow and the ML count estimate. Naive timestamps are interpreted on a common UTC clock to compute elapsed differences. The packed windows contain 837 positive and 572 negative observations. Adjacent windows overlap, so the nine configurations do not supply nine independent replications. [Aalto ventilation dataset, Zenodo 10.5281/zenodo.18299691](https://doi.org/10.5281/zenodo.18299691)

Titan recorded the original freeze manifest before the scoring process opened labels. The freeze timestamp is 2026-11-22 12:41:57.768504 UTC. The scorer reported 2026-11-22 12:42:13 UTC at launch, with the frozen file SHA-256 `532d87b8a865c7adc457e3a5d75e91a8653bc49061eee48fc46e804073ac146b`. That manifest pins the source, Aalto archive digest, and all 72 checkpoints containing fitted weights, feature normalization, cutoffs, and policy parameters. The scorer checked every pin before loading fresh rows. No fitting or cutoff selection ran on Aalto. All 72 Aalto score receipts matched byte for byte in a replay from the public checkout. At seed 41 with the full fitting budget, the V2 learned route scored 51.84% and 44.73% effective balanced accuracy in Graz Rooms A and B, and 45.38% in the Danish apartment set. The corresponding no-delta learned route scored 68.22%, 77.79%, and 54.24%. These sites were already explored.

For heating, the 532 held-out logged actions produced an observed utility and a forecast from fitting-only command-conditioned bucket means. The utility is negative thermal kWh minus twice the integrated degrees outside 20–24 °C. Payoff MAE was 0.03073 in these utility units. There were 468 held-out rows in buckets with at least eight examples of each logged action. Outcomes for commands that were not taken are unobserved, and these receipts establish forecast error on logged actions. This run includes no trained V2 heating comparison or live actuation. [Six-building dataset, Mendeley v3](https://data.mendeley.com/datasets/xztfbtsgys/3)

## Reproduce on CPU

The recorded runs used Python 3.12.3, torch 2.7.1+cpu, and ncps 1.0.1.
From this directory, install the [pinned dependencies](../attention-and-reward/requirements.txt).
Save these copies in `data/`.

| Local file | Publisher |
| --- | --- |
| `occupancy-fresh.zip` | [UCI occupancy archive](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) |
| `aalto-ventilation-test.csv` | [Aalto ventilation test](https://zenodo.org/records/18299691/files/ventilation_test.csv?download=1) |

The results bundle includes the exploratory comparison and logged heating summary.
Their input archives are linked in [the earlier experiment](../adaptive-liquid-v1/README.md).
Mikala, Xu, and Huotari published the Aalto dataset under CC BY 4.0.
The archive SHA-256 is `bf0617f9c273e64f100f2e9303e4104d5346ebab1c3060fa8a9cc88850e596b2`.
[Publisher record](https://doi.org/10.5281/zenodo.18299691)

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r ../attention-and-reward/requirements.txt
mkdir -p replay
cat artifacts-*.b64 | python -m base64 -d > artifacts.tar.gz
python -m tarfile -e artifacts.tar.gz replay
python reproduce.py check replay --data data
python reproduce.py score replay --data data
python summarize.py replay
python reproduce.py fit fit-example --data data --model v2 --seed 41 --budget 100
```

The decoded bundle SHA-256 is `3d0c370eb61453b49976b370b7471067ea09462ed2aa199e562fe75dbbfa241f`.
It includes model checkpoints, fit receipts, scores by window, test traces,
and exploratory summaries. The [freeze record](freeze-content.json) uses
portable input names and preserves the checkpoint and source content hashes
from the original freeze. The score command verifies those hashes before it
scores the models. The refit command writes to a separate directory.
For a fresh evaluation, choose an untouched site and freeze the model, its
cutoff and its policy before opening that site's test labels.

## How much could a better retry policy recover?

The [route ceiling](route-ceiling.json) enumerates the four keep or revise
routes through the trained stages on the Aalto windows. For each window it
asks whether any of those routes predicts the observed label. It uses that
label after scoring, so this is an analysis of possible corrections, not a
policy score. Across nine full-model checkpoints, the direct prediction
averaged 72.59% balanced accuracy. Perfect selection among the existing
routes would reach 74.81% and could fix about 28 of 1,409 windows per
checkpoint. The fused LTC averaged 78.73% in the same comparison. The
no-delta ablation went from 78.62% to an 81.24% ceiling. The issue is also
what the existing stages can predict and what they can change when they
revise. Lowering the retry cost alone cannot make a missing correction
available.

```sh
python reproduce.py route-ceiling replay --data data
```

## Check what changes when the input columns change

The [single-reading check](prior/paper_feature_check.py) fits the same type
of linear classifier on UCI twice. It first uses temperature, relative
humidity and CO₂, then adds light and humidity ratio. The two same-office
test accuracies go from 87.77% to 97.30% and from 75.30% to 87.79%.
[Recorded comparison](prior/paper-feature-diagnostic.json)

```sh
python prior/paper_feature_check.py \
  data/occupancy-fresh.zip
```

This isolates the input-column change in a simple model on UCI. It does
not measure a liquid solver effect or estimate how well those added
signals would work at another site.

## Compare decisions at the same coverage

When the direct model is forced to skip the same windows as the learned
route, its all-window balanced accuracy is 67.68%. The learned revisions
lift this to 68.22%. Among answered windows, the change is 74.96% to 75.67%.
The model answered 90.51% of the windows. Without those omissions the
direct model reached 72.59% across all windows. The revisions helped
slightly on the windows the learned route answered. Its omissions account
for the larger drop across the full set. The learned route also earned
less reward than the direct model under its own payoff table.

```sh
python compare_coverage.py replay
```

The comparison uses the original per-window predictions and the same
acceptance decisions for both models. [Results by checkpoint](compare-coverage.json)
