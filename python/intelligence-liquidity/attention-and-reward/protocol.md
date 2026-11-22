# Freeze the attention experiment

Can a liquid network with attention and feedback reach higher occupancy
accuracy using fewer training labels? This is a hypothesis. Score every
comparison on the same unseen rows and report losses as well as wins.

## Occupancy

Fit on UCI Occupancy Detection `datatraining.txt`. Use the first 6,500 rows as
the fitting pool and the remaining rows as validation. Use 25%, 50%, and
100% of the same chronological fitting pool, evenly spread across contiguous
blocks if needed to retain both classes. Record exactly which rows and windows
enter each budget. Keep all preprocessing and auxiliary targets within that
budget. Use only temperature, humidity and CO₂, shared with the Danish
apartment dataset. Supply explicit elapsed minutes. Construct windows within
one split and one room and reject windows crossing long gaps. Fix every setting
before scoring the Danish data. Do not select checkpoints, thresholds, feature
mappings, architecture or budgets from either holdout. Use logged 0.5 decision
threshold, independent UCI publisher test files, and the independent five
Danish apartments (Zenodo 10761326, CC BY 4.0 if confirmed by metadata).
Occupancy is measured by logbooks in the apartments. Sampling is 15 minutes
there and about one minute in the source office. Do not claim matched cadence.

Compare the last-reading network, GRU, library LTC, custom LTC with a final
state, custom LTC with attention, and custom LTC with attention and
self-supervised next-reading prediction. Use three seeds, equal maximum epochs,
identical batch sizes and validation-driven early stopping with patience two;
report actual parameters, epochs and updates and fit windows. The occupancy
readout scores the current window. Auxiliary targets come from the next reading
of the fitting block and never enter the forward input at inference. Report
accuracy, balanced accuracy, false positives and false negatives, apartment-level
scores and validation learning curves. Compare smaller-budget candidate against
full-budget baselines on the same untouched external room windows. Keep the
older UCI external room as a separate historical stress test where humidity is
unavailable, unless a single prespecified mapping can be applied to all models.

## Heating

Learn an offline, one-step action-conditioned outcome model from ZEB Living Lab
published commands and observations. Use measured indoor and outdoor history,
predict near-future indoor temperature, heat and accumulated comfort deviation
for a candidate action, and use predicted reward to rank actions at inference.
Use a training-only action-support gate and abstain when either action lacks
coverage. Both control alternatives must pass the same gate. Reserve March 7
and later for the chronological holdout, keeping raw histories and targets
within their respective split. Choose validation from the tail of training.
Report all compared networks on those same held-out logged actions, including
temperature/reward prediction errors, on/off support, recommended commands,
and abstentions. Include a conservative simpler baseline. Report measured
prediction error only for the logged action, with no counterfactual savings or
human-preference claims. Reward is minus thermal kWh minus two times accumulated
degree-hours outside 20–24 °C. No connection to a live controller.

## Verification

Save a snapshot containing configuration and weights, and prove reload gives
the same output. Include inference traces showing attention weights across
time and actual reward components and action ranking. A small fixed-seed test
must catch future-target leakage, invalid attention weights and unsupported
action recommendations. Capture fixed commands, input hashes and raw evaluation
receipts. Keep end results as measured even when no model improves.
