# Custom LTC occupancy experiment

The runnable entry point is [model.py](model.py). It has a custom PyTorch cell, a conventional GRU and a final-row baseline. It uses Python 3.12.3, PyTorch 2.7.1+cpu, CPU float32, one thread and deterministic PyTorch algorithms. The dependency-free [replay.py](replay.py) uses the same saved LTC values. The archive stays outside the repository.

## Reproduce the follow-up experiments

- [Training, snapshots and rollback](../lifecycle/README.md) replays the training decisions and frozen checkpoints from the article.
- [Second-room transfer test](../external-validation/README.md) uses unchanged checkpoints on independently collected occupancy data.
- [Offline radiator policy](../building-control/README.md) fits a separate policy from real heating commands and measured outcomes.

## Sources and data

- Hasani, Lechner, Amini, Rus and Grosu, *Liquid Time-constant Networks*, AAAI 2021, DOI 10.1609/aaai.v35i9.16936. The experiment implements the conductance form and fused semi-implicit solver from equation 2 and Algorithm 1, with a learned capacitance, per-connection gates and a supervised readout. See Section 2, Table 3 and the occupancy appendix for the paper’s model and benchmark. [AAAI paper](https://doi.org/10.1609/aaai.v35i9.16936) · [Author manuscript v4](https://arxiv.org/html/2006.04439v4)
- Candanedo and Feldheim, *Occupancy Detection*, UCI Machine Learning Repository, 2016, DOI 10.24432/C5X01N, CC BY 4.0. The benchmark labels are based on time-stamped photographs. The training script verifies the publisher archive before reading the CSV files. The sample CSV omits the record IDs and labels and keeps the measured values. The full measured readings and labels in the six inference windows are in [results.json](results.json). [Dataset](https://doi.org/10.24432/C5X01N) · [Publisher archive](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Archive SHA-256 is `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`. The archive has 20,560 measured data rows and 4,750 positives. Row IDs and timestamps serve different purposes: IDs locate the measured row, and timestamps supply elapsed time. No timezone is provided. Feature order is Temperature (°C), Humidity (% RH), Light (lux), CO2 (ppm), HumidityRatio (kg water vapour per kg dry air).

| Data | Rows | Positive rows | Scored targets | Positive targets |
| --- | ---: | ---: | ---: | ---: |
| datatraining.txt, fitting | 6,500 | 1,141 | 6,493 | 1,134 |
| datatraining.txt, validation | 1,643 | 588 | 1,636 | 588 |
| datatest.txt | 2,665 | 972 | 1,857 | 669 |
| datatest2.txt | 9,752 | 2,049 | 6,820 | 1,425 |

The checked window runs from record 143 to 150 in `datatest.txt`. Its last raw record is:

```csv
"150","2015-02-02 14:29:00",23.745,26.445,481.5,815.25,0.00480888622067716,1
```

## Protocol frozen before evaluation

The architecture and scoring are in the initial version of [model.py](model.py). This project previously inspected both supplied test files and compared library-based LTC and GRU models. The current training reads only `datatraining.txt`. It uses its first 6,500 chronological rows for fitting and population scaling, and its final 1,643 for checkpoint selection. Histories never span these boundaries. The first gap is assigned one minute for each eight-row window. Other gaps are timestamp differences in seconds divided by 60. The paper used the same dataset with overlapping 32-step windows and a different validation protocol. These figures measure the custom eight-row experiment.

- LTC, 449 parameters. Sensors `[B,8,5]`, gaps `[B,8]`, state `[B,8]`, sensory edges `[5,8,4]`, recurrent edges `[8,8,4]` with self-connections. Each edge stores raw strength, slope, midpoint and reversal. Bias, raw leak and raw capacitance are each `[8]`. Softplus makes strengths positive and the leak and capacitance have a positive floor of `1e-6`. Each step holds sensors fixed and uses six substeps with recurrent gates recalculated from the old state. The linear readout has weight `[1,8]` and bias `[1]`; it produces `[B]` logits.
- GRU, 551 parameters. Ten units and an input of `[B,8,6]` with natural log elapsed minutes appended to the same five scaled features. The last state feeds a linear readout.
- Final row, 225 parameters. A five-to-32-to-one ReLU network that sees the last row of the same window.
- Twelve epochs per model, seeds 41, 42, 43, batch size 256, seeded shuffled windows, binary cross-entropy on logits without class weights, and Adam implemented with Torch tensors and autograd. Adam uses learning rate 0.01, betas 0.9 and 0.999, bias correction and epsilon `1e-8`. Select the first epoch with minimum validation BCE at original cadence. Selected epochs in seed order were LTC 7/12/11, GRU 5/4/6, last-row 1/1/1. The benchmark example uses seed 41 by the predefined seed order.
- During evaluation drop zero-based offsets 2, 3 and 4 modulo 10 in each test file. Evaluate every surviving target from the eighth survivor onward in both modes for all models. Original mode builds each history from original rows. Missing mode uses retained rows only. The target label stays unchanged. The decision threshold is 0.5. Two independently reloaded models produced equal test logits in each case.

For one destination, the dynamics use capacitance times state derivative equal to leak conductance times (bias minus state), plus each active conductance times (reversal minus state). Which connections pull hardest changes with the current source values. In each substep set `total = leak + sum(active)` and `pull = leak * bias + sum(active * reversal)`. With capacitance `C` and substep `dt` minutes, the code advances to `(C * state + dt * pull) / (C + dt * total)`. Expanding the same fraction by dividing both sides by `dt` shows why the previous state carries weight `C/dt`.

## Measured results

Means over the three seeds. False alarms and misses are average counts per seed. Balanced accuracy averages occupied and empty recall. BCE is mean binary cross-entropy on logits. Per-seed counts, scores and example windows are in [results.json](results.json).

| File | History | Model | Accuracy % | Balanced % | BCE | False alarms | Misses |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| datatest.txt | original | ltc | 97.25 | 97.49 | 0.1038 | 40.0 | 11.0 |
| datatest.txt | original | gru | 97.34 | 97.73 | 0.1122 | 43.3 | 6.0 |
| datatest.txt | original | current | 96.37 | 97.04 | 0.1660 | 63.3 | 4.0 |
| datatest.txt | missing | ltc | 97.25 | 97.49 | 0.1038 | 40.0 | 11.0 |
| datatest.txt | missing | gru | 97.00 | 97.34 | 0.1174 | 46.0 | 9.7 |
| datatest.txt | missing | current | 96.37 | 97.04 | 0.1660 | 63.3 | 4.0 |
| datatest2.txt | original | ltc | 98.88 | 98.95 | 0.0651 | 63.7 | 13.0 |
| datatest2.txt | original | gru | 90.09 | 92.25 | 0.2292 | 618.7 | 57.3 |
| datatest2.txt | original | current | 88.79 | 88.76 | 0.2493 | 603.7 | 161.0 |
| datatest2.txt | missing | ltc | 98.87 | 98.95 | 0.0652 | 64.0 | 13.0 |
| datatest2.txt | missing | gru | 89.82 | 91.90 | 0.2375 | 630.0 | 64.3 |
| datatest2.txt | missing | current | 88.79 | 88.76 | 0.2493 | 603.7 | 161.0 |

The original-cadence GRU on the first file had fewer misses and better balanced accuracy, while the LTC had fewer false alarms and lower BCE. The LTC had higher balanced accuracy under the row-drop pattern on that file and fewer errors of both kinds on the second file. Adjacent windows overlap, and all rows come from one office. Seed variation is captured in the raw scores. A new time block with independent labels, and other offices if the claim extends to them, could challenge the ranking. The deterministic mask tests missing whole readings under one pattern.

## Train, evaluate, infer and roll back

Run from this directory. Use Python 3.12 and CPU PyTorch 2.7.1. The commands below use the PyTorch CPU index for Linux. Training and evaluation verify the archive’s SHA-256 before reading it.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
curl --fail --location \
  https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip \
  --output occupancy.zip
```

```bash
d=.
py=python
archive=occupancy.zip
"$py" "$d/model.py" train "$archive" > candidate.json
"$py" "$d/model.py" evaluate "$archive" candidate.json
"$py" "$d/model.py" infer "$d/checkpoint.json" --archive "$archive"
"$py" "$d/model.py" infer "$d/checkpoint.json" --csv "$d/sample.csv" \
  --now '2015-02-02 14:30:00'
"$py" -B "$d/parity.py"
python3 "$d/replay.py"
```

The two inference commands return 0.9206820726394653 and occupied for the sample. Archive mode is a historical replay and checks age only when `--now` is supplied. For your own CSV, use the header from [sample.csv](sample.csv), exactly eight strictly increasing ISO timestamps on one local clock, five finite numerical readings on each row, and `--now` on that same clock. A gap over five minutes, stale or future newest reading and malformed data exit 2, emit a diagnostic on stderr and yield no occupancy result.

In a live shadow trial the caller logs rejected windows as unknown and leaves the existing schedule in control. After an outage, accumulate eight valid rows again. Choose a threshold on local validation evidence from the relative costs of missed occupancy and false alarms. Freeze the candidate and its policy, then evaluate both on a later labelled block, including unknowns and the dropout condition. Confirm or revise freshness limits from the intended site. Keep every approved checkpoint with its schema, feature order, scaling, threshold and fallback rules. To restore a prior custom-ltc-v1 checkpoint, pass its JSON path to the same command. The demonstration threshold is fixed in the executable; carry the matched executable or configured policy with the saved file.

```bash
"$py" "$d/model.py" infer /path/to/previous-approved.json --csv "$d/sample.csv" \
  --now '2015-02-02 14:30:00'
```

## Store and run the model on Hugging Face

Create a private model repository to store the checkpoint and its guide. The
checkpoint is JSON. It contains the weights, the order and scale of the inputs,
the model schema and the training record. The code in this directory loads it.
[Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```bash
python -m pip install huggingface_hub
hf auth login
MODEL=YOUR_HF_ACCOUNT/intelligence-liquidity
mkdir -p model-upload
cp README.md model.py replay.py parity.py checkpoint.json results.json parity.json sample.csv model-upload/
hf repos create "$MODEL" --private
hf upload "$MODEL" model-upload .
```

Use the commit SHA from that upload to fetch the exact version you approved.
Check the downloaded file against the checkpoint hash below before use.

```bash
MODEL_REVISION=PASTE_APPROVED_COMMIT_SHA
hf download "$MODEL" checkpoint.json --revision "$MODEL_REVISION" --local-dir downloaded
shasum -a 256 downloaded/checkpoint.json
python model.py infer downloaded/checkpoint.json --csv sample.csv --now '2015-02-02 14:30:00'
```

To run the replay on the web, create a separate Gradio Space. This example uses
the Python replay and the same embedded weights that `parity.py` checks against
the checkpoint. Upload the paired replay again when you change the model.
[Gradio Space guide](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)

```bash
mkdir -p space-upload
cp replay.py space-upload/
cat > space-upload/README.md <<'EOF'
---
title: Intelligence liquidity
sdk: gradio
app_file: app.py
---
Change the gap between the last two readings and replay the saved LTC model.
Data: UCI Occupancy Detection (CC BY 4.0), Candanedo and Feldheim.
https://doi.org/10.24432/C5X01N
EOF
cat > space-upload/app.py <<'EOF'
import gradio as gr
from replay import GAPS, ROWS, probability


def score(final_gap):
    return f"{probability(ROWS, [*GAPS[:-1], final_gap]):.4%}"


if __name__ == "__main__":
    gr.Interface(score, gr.Slider(1, 5, step=1, label="Final gap in minutes"),
                 gr.Textbox(label="Occupancy score"), title="Intelligence liquidity").launch()
EOF
SPACE=YOUR_HF_ACCOUNT/intelligence-liquidity-demo
hf repos create "$SPACE" --repo-type space --sdk gradio --private
hf upload "$SPACE" space-upload . --repo-type space
```

In the Space, check that the scores at one and four minutes match the replay
below. The Space changes one gap in a fixed sample. Use `model.py infer` to
score new CSV input with validation and a freshness check.

## Execution record

- Both training runs exited 0. The second produced a byte-identical JSON checkpoint. The JSON holds tensor arrays, feature order, fit/validation summaries, runtime versions, architecture, schema version and data provenance.
- Evaluation of the saved checkpoint was repeated and its JSON was byte-identical to [results.json](results.json).
- The standalone benchmark and valid CSV returned identical probabilities. Stale CSV and a malformed header returned exit 2 with the recorded diagnostics.
- The replay is 12,375 bytes. The parity script compared six archive windows at their original final gaps and at edited final gaps of two and four minutes. All 18 passed both the 1e-6 probability tolerance and equal 50% decisions. Maximum absolute probability error was 5.842647177178861e-8. [Raw parity results](parity.json)
- Replay on the eight measured rows returns 92.0682% at one minute and 92.0438% at four minutes. The four-minute gap is an edit to the example.
- The checkpoint SHA-256 is `bfd5704418c51fcb84454a21d76dca9091d0187ddb489d9791cf8a8a85e4c1a4`. The model source SHA-256 is `ce38c32849ca9c6705a63877b8476880dc3e214f9d7022729a0f0a193f536b16`. The benchmark JSON SHA-256 is `3d7b4643333792f27aaad03d94c47be26e90dc9bee77454fa91ef4e9103f86e9`. Run `sha256sum` on these files when copying them.
