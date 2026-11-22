# Intelligence liquidity

A three-layer liquid mixture of experts for occupancy replay and a separate
heating replay. Each layer has three specialists, one liquid checker and two
heads of delta attention on the shared record.

## Run a saved model

Use Python 3.12 and the dependency versions in `requirements.txt`. For a CPU
PyTorch wheel, follow the [PyTorch install guide](https://pytorch.org/get-started/locally/).

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python serve.py evidence < examples/readings.jsonl
python serve.py evidence --live < examples/readings.jsonl
```

The three rows are a teaching example. The second thermometer value is 40 °C
to provoke a refusal. Each output shows the expected temperature and error,
the expert that ran and which earlier layer was retried. A null decision means
the model withheld its answer. Temperature is a witness for the check.
The actual occupancy answer comes from the independent occupancy record.
[workflow.py](workflow.py) contains the model. [serve.py](serve.py) reads and
writes one JSON object per line. A new process starts with the saved settings.

## What the model changes

- Six integration steps move the hidden state. The sensor and current state
  set the strength of the pulls in the differential equation.
- The attention board fades past data one channel at a time. Each layer then
  writes a fraction of the difference between its new value and the recalled
  value. The board is saved only if the whole path passes.
- The checker can send an objection back one layer. The model replays from its
  saved state and tries the remaining specialists. It accepts only paths that
  pass all measured checks, and uses the learned process score to rank those
  paths. If all paths fail, it keeps the original states and board.
- With live updates on, the next CO₂ reading grades the previous forecast.
  One small update to the learned settings follows the emitted decision. The
  previous answer stays fixed. A gap over 450 seconds starts a fresh state and
  breaks the link between forecasts and feedback.
- The heat model uses the same architecture with separate weights. In the log,
  the executed action earns a cost from measured energy and time outside the
  comfort band. The policy uses that outcome for reward learning once the next
  decision has been made. A log gives one measured outcome per decision.

## Fit and score

Get the fitting and heating archives from the links in the
[previous experiment](../liquid-workflow/README.md). Give each archive the
filename in that guide. The fresh test uses only two published files.

```sh
mkdir -p h6-data
curl -fL https://ndownloader.figshare.com/files/27443933 -o h6-data/H6_ENVIRONMENTAL.zip
curl -fL https://ndownloader.figshare.com/files/27443936 -o h6-data/H6_GROUNDTRUTH.zip
python experiment.py fit /path/to/fit-data run-output --epochs 20
python experiment.py score /path/to/fit-data run-output --external h6-data
python experiment.py control /path/to/fit-data control-output --epochs 20
```

The fit command saves a best checkpoint and a resume file after each epoch.
An epoch is one pass through the fitting blocks. To resume, run the same
command in the same output folder. The next epoch starts from the recorded
optimizer and model settings. The 20-epoch cap and four-epoch patience both
apply. Use a new directory for a clean fit. [fit.json](evidence/fit.json) has
the validation curves and checkpoint hashes.

The transfer test takes the first seven dates with a full day of labels and a
matching BS4 file from [HPDmobile H6](https://www.nature.com/articles/s41597-021-01055-x).
It takes the first row in each five-minute window with both sensor readings
and its matching occupancy label. It uses the logged CO₂-equivalent value
as the input, temperature as a separate check and the study's binary label
only for scoring. All models use the same fitting blocks and test readings.
The GRU, upstream `ncps` liquid model and XGBoost use the CO₂ channel.
The workflow can also use temperature to veto or reroute the work.
The test has one home and one sensor channel.

[evidence/h6.json](evidence/h6.json) reports correct answers, missed people,
false alarms, withheld answers, forecast error and time. Accuracy counts a
withheld answer as zero. Balanced accuracy gives occupied and empty readings
equal weight. Raw traces keep the claims from rejected paths. The small,
medium and full fits test 25%, 50% and 100% of the fitting blocks. One seed
estimates each fit. These data do not identify a minimum data requirement.

[evidence/control.json](evidence/control.json) reports closed replay of the
executed commands. The housing test scores occupancy. The heating test
scores action values against recorded outcomes. The heat replay does not
operate an actuator or measure the return from running the proposed policy.

## References

- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439), conductance model and solver.
- [Kimi Linear, section 3](https://arxiv.org/html/2510.26692v1#S3), channel forgetting and delta writes.
- [Test-Time Training](https://proceedings.mlr.press/v119/sun20b.html), learning during inference.
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779), the action-value penalty used in the heater fit.
- [HPDmobile environmental readings](https://doi.org/10.6084/m9.figshare.14368703) and [occupancy labels](https://doi.org/10.6084/m9.figshare.14368706).
- [Measured residential building logs](https://data.mendeley.com/datasets/xztfbtsgys/3), logged heater commands, temperature and energy.

## Compare a local thermometer reference

The absolute check compares a warm house with the temperatures learned in
the fitting rooms. The second mode starts with the local thermometer and
updates that reference only after the current decision. It ignores abrupt
changes above the tolerance. The first row sets the reference and gets no
answer. A long gap starts a new reference.

```sh
python serve.py evidence --local-reference --live < examples/readings.jsonl
mkdir -p h5-data
curl -fL https://ndownloader.figshare.com/files/27443894 -o h5-data/H5_ENVIRONMENTAL.zip
curl -fL https://ndownloader.figshare.com/files/27443882 -o h5-data/H5_GROUNDTRUTH.zip
python score_h5.py evidence h5-data
python benchmark.py evidence h6-data local-cpu.json
```

The reference change came from inspecting temperature coverage in H6.
The separate test uses H5 channel RS1 and the same selection rule as H6.
The H5 filenames and rule are in [transfer.py](transfer.py), and the
checkpoint and archived results are in [h5.json](evidence/h5.json).
Compare absolute, local and local with live updates using the same weights.

- [H5 sensor archive](https://doi.org/10.6084/m9.figshare.14368670)
- [H5 occupancy archive](https://doi.org/10.6084/m9.figshare.14368658)
