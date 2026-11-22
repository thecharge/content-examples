# Liquid network occupancy benchmark

A single liquid cell reads the history in the normal model. The mixture has a
shared liquid core and three liquid experts. Its router wakes one expert per
reading and preserves each expert's state and elapsed time. The delta variant
adds a board that fades older associations and writes the error between the
value it reads and the value it expects. Ordinary prediction uses the trained
router as usual. Optional review gives the final proposal to an independent
liquid checker, can try a different expert once, and commits or discards its
proposed state and board together.

Run from this directory with CPU PyTorch 2.7.1. Set DATA to the directory
containing these exact downloads. Use separate fresh output directories for
smoke and the main fit.

- Save the [UCI archive](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) as `occupancy-fresh.zip`.
- Save the [Aalborg CSV](https://zenodo.org/records/10673763/files/dataset__2023_09_02__2023_10_04.csv) as `dataset__2023_09_02__2023_10_04.csv`.

The archive fingerprints and feature choices are in [the protocol](liquid-moe-protocol.md).

```sh
python benchmark.py prepare --data "$DATA"
python benchmark.py fit --data "$DATA" --out "$SMOKE" --steps 1 --seed 41
python benchmark.py fit --data "$DATA" --out "$RUN"
python benchmark.py freeze --data "$DATA" --out "$RUN"
```

Record the SHA-256 printed by freeze in the runner's independent record before releasing the Aalborg CSV to the scorer. Transfer the fit receipt, frozen manifest and checkpoint files to the scorer with the matching source files. Set FROZEN to that independently recorded digest.

```sh
python benchmark.py score --data "$DATA" --out "$RUN" --frozen-sha256 "$FROZEN"
```

Preparation validates the UCI archive before opening it and reports usable windows. Training repeats that validation and splits the original training file after raw row 6500. Each room or partition is sorted by timestamp. For every observed endpoint, take eight distinct observations as of trailing five-minute offsets over 35 minutes. A selected reading must be less than five minutes old at its offset. Missing sensor values and observation gaps over 7.5 minutes reset history. The only sensor inputs are CO2 and temperature, standardized using fit windows. The clock gives the actual elapsed minutes between selected observations.

Each seed uses a separately seeded common sequence of 64-window batches and 144 Adam updates at a constant learning rate of 0.003. Save the checkpoint with lowest unweighted UCI validation log loss among steps divisible by 24 and the final step. Ties keep the earlier snapshot. Choose a logit cutoff from −8 to 8 in increments of 0.25 by validation balanced accuracy, then prefer the closest to zero and then the smaller cutoff. The receipt stores normalization, source and checkpoint fingerprints, batch sequence fingerprints and selection details.

Only scoring opens the external CSV. It verifies the freeze, fit, source, checkpoints and publisher fingerprints. People count provides the binary score target. The two UCI test files are reported separately. Frozen replay requires the same source, input bytes and checkpoint bytes. Refit for independent reproduction with the recorded PyTorch version and seeds. Compare receipts and per-tensor checkpoint values since fit timings can vary.

Read the pooled and six per-room results in scores.json. Compare full-coverage all-window balanced accuracy across the normal liquid, routed liquid, delta memory and reviewed delta variants for each seed. The optional reviewed route also reports accuracy among decisions it keeps. Its full-coverage counterpart uses the ordinary delta logit for abstentions. Payoff gives +1 for a correct decision, −2 for missed occupancy, −1 for false alarms, −0.20 for abstention and −0.06 per attempted revision. Undefined rates are null. Fit and inference times report wall-clock seconds. Expert trace counts and parameter counts describe compute separately. The 80% and 90% figures are expectations for the comparison, and the six rooms are from one building.

## What the experiment found

All 15 checkpoints and their UCI validation cutoffs were frozen before scoring.
Aalborg contributed 53,418 labeled windows across six rooms in one building,
including 6,351 occupied windows. The table averages three seeds. The balanced
score weighs occupied and empty recall equally and counts abstentions as misses
for their respective class.

| Model or path | Balanced accuracy | Raw accuracy | Coverage | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| Current reading | 51.53% | 39.94% | 100% | −0.2406 |
| GRU | 62.65% | 76.71% | 100% | 0.4679 |
| Normal liquid | 61.90% | 65.61% | 100% | 0.2612 |
| Routed liquid experts | 57.97% | 58.54% | 100% | 0.1199 |
| Routed liquid experts with delta memory, ordinary prediction | 62.05% | 71.07% | 100% | 0.3623 |
| Same delta weights with learned review and abstention | 52.38% | 55.27% | 81.98% | 0.1541 |
| Same review with an ordinary prediction for each abstention | 61.16% | 70.72% | 100% | 0.2933 |

The normal liquid network reached 85.91% and 78.13% balanced accuracy on the two
separate UCI test files. The ordinary delta MoE reached 84.66% and 71.61%. Those
tests came from the training room. On the Aalborg test, always guessing empty
would give 88.11% raw accuracy and 50% balanced accuracy.

### What changed and where it failed

The delta board improved the routed model by 4.09 points on the external
test. Delta's 62.05% was close to the normal liquid network's 61.90%. The MoE's
dominant expert handled 93.6% to 99.5% of ordinary expert calls, depending on the
seed. The models had three experts, a shared liquid core and real top-one
execution. The fitted routers rarely used that extra capacity. The plain
liquid network had 726 parameters, the MoE 4,440 and the delta MoE with its
checkers 8,990. These are comparisons under the same data and optimizer
schedule with different inference costs.

The learned process estimates and the finite-state, clock, retry-budget and
challenger-identity checks are separate. The release trace checked those rules
on real windows. It records the original expert and score, distinct challenger,
checker estimates, estimated gain, returned decision and remaining budget.
The review policy nevertheless attempted a challenger on every usable window in
all three seeds. Across all 55,621 usable windows per seed, seed 41 changed
5,807 decisions, helping 2,546 and hurting 3,108 with labels, with the remaining
changes on windows without ground truth. The other two seeds changed zero
binary decisions despite committing alternative expert state. Seed 43
abstained on 30,023 usable windows, including windows without ground truth.

A post-score cutoff diagnostic found the best threshold per seed with the
Aalborg labels. It capped the ordinary delta MoE at 76.00% balanced accuracy on
average and 80.72% in the best seed. This is a hindsight calculation, never a
deployment cutoff. It shows that threshold transfer matters and that a better
threshold alone cannot produce 90% from these weights.

The older fixed V2 result was rechecked on previously explored Aalto data.
Three pairs shared the initial core weights, sampled batches, scheduled revision
draws, 240-step allowance and checkpoint rule. Delta averaged 79.49% balanced
accuracy versus 78.12% without delta. It improved one pair and hurt two.
This diagnostic changes the earlier conclusion about the size and direction of
the average delta effect under this fitting protocol.

### Replay and inspect

The [frozen directory](frozen/) includes the weights, training receipt, input
and source hashes, score record, and separate threshold and review diagnostics.
Use a fresh directory for replay and copy in the weights, fit receipt and
frozen manifest. The published manifest digest is in
[frozen/label-release.txt](frozen/label-release.txt).

```sh
python test_model.py
python benchmark.py score --data "$DATA" --out "$RUN" --frozen-sha256 "$FROZEN"
python audit_trace.py --data "$DATA" --out "$RUN" --frozen-sha256 "$FROZEN"
python threshold_diagnostic.py --data "$DATA" --out "$RUN" --frozen-sha256 "$FROZEN"
```

For the older Aalto diagnostic, add the original
[ventilation data](https://doi.org/10.5281/zenodo.18299691) to DATA as
`aalto-ventilation-test.csv` and run
`python recheck_v2.py --data "$DATA" --output v2-recheck.json`.

A clean replay matched all 63 score records except inference wall time. Altering
the seal or a checkpoint was rejected before opening the external data. The
eight unit checks covered sparse execution, memory update, training gradients,
causality, the retry budget, commit, and rollback. The scoring receipt also
contains per-room results. In the early benchmarks the `expert_counts` field
groups batch count values. Use `ordinary_expert_calls_by_index` in the threshold
diagnostic for actual calls per expert.

[Original LTC equations](https://arxiv.org/html/2006.04439v4) ·
[Sparse routing in Switch Transformers](https://www.jmlr.org/beta/papers/v23/21-0998.html) ·
[Kimi Delta Attention](https://arxiv.org/html/2510.26692v1) ·
[UCI data](https://doi.org/10.24432/C5X01N) ·
[Aalborg data](https://zenodo.org/records/10673763)
