# Intelligence liquidity

Train a real liquid time-constant network on office sensor readings. Then replay
the saved model and see what changes when a report arrives late.

## Get the exact data

Luis Candanedo and Véronique Feldheim’s [UCI Occupancy Detection dataset](https://doi.org/10.24432/C5X01N)
contains 20,560 readings from one office. Each row gives the timestamp, temperature,
humidity, light, CO₂, and a derived humidity ratio. Time-stamped photos supplied
the occupied or empty labels. The data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The browser example uses a small attributed extract from the data.

| UCI file | Rows | How this example uses it |
| --- | ---: | --- |
| `datatraining.txt` | 8,143 | First 6,500 rows fit the weights. Last 1,643 choose when to stop training. |
| `datatest.txt` | 2,665 | Held out from training. 1,857 matched predictions scored. |
| `datatest2.txt` | 9,752 | Held out from training. 6,820 matched predictions scored. |

Run these commands from this example’s directory. The Python commands were
tested on Linux with Python 3.12 and CPU PyTorch. The browser replay only needs a
modern browser.

```sh
curl --fail --location 'https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip' --output occupancy.zip
python3 -m zipfile -l occupancy.zip
shasum -a 256 occupancy.zip
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The archive hash is `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`.
The training and evaluation commands check it before reading any rows.

Here is row 150 from `datatest.txt`. This is the last reading in the first
browser example. The final `1` means someone was in the room. Run the command
to see the same row in your copy of the archive.

```sh
unzip -p occupancy.zip datatest.txt | grep -F '2015-02-02 14:29:00'
```

```csv
"150","2015-02-02 14:29:00",23.745,26.445,481.5,815.25,0.00480888622067716,1
```

## Train and use the model

The network processes eight readings at a time. Five sensor values and the
time since the last reading enter an eight-unit LTC. The unit states change
as the readings come in. A final linear layer uses the last state to give one
occupancy score. Training changes the weights. Replay holds the weights fixed.

```sh
.venv/bin/python occupancy.py train occupancy.zip > candidate.json
.venv/bin/python occupancy.py evaluate occupancy.zip candidate.json > candidate-results.json
.venv/bin/python occupancy.py infer occupancy.zip checkpoint.json
.venv/bin/python parity.py
```

Open `replay.html` and choose the first recorded window. The saved seed 41
checkpoint scores it at **90.1295%**. Edit the final gap from one to four minutes
and the score becomes **90.2958%**. The reading and recorded label stay the
same. The four-minute interval is a replay edit. `parity.py` uses Node.js 24 and
compares the browser code with PyTorch for 18 measured and edited-gap cases.

In the larger held-out file with rows deliberately dropped, mean balanced
accuracy across three seeds was **92.75% LTC**, **92.96% 11-unit GRU**, and
**88.76% current-reading model**. Balanced accuracy gives occupied and empty
readings equal weight. The LTC raised fewer false occupancy alarms and missed
more occupied minutes than the primary GRU. The test files come from one room.
The result measures this replay policy. A later comparison with a nine-unit
GRU was chosen after seeing test scores and is exploratory. Both comparisons,
ordinary accuracy and mistake counts are in `results.json`.

Reproduce the original 11-unit comparison without changing the default model.

```sh
cp occupancy.py occupancy-primary.py
patch occupancy-primary.py primary-gru.patch
.venv/bin/python occupancy-primary.py train occupancy.zip > primary-candidate.json
.venv/bin/python occupancy-primary.py evaluate occupancy.zip primary-candidate.json > primary-results.json
```

The rerun matched all 12 original score records across the two files, two
timing conditions and three seeds. Its checkpoint bytes differ from the
archived hash. Both hashes and the rerun scores are in
`primary-gru-reproduction.json`. The saved checkpoint from this rerun is
`primary-gru-retrained.json`.

## Put the weights on the Hugging Face Hub

The Hub stores downloadable files and their revisions. This checkpoint is JSON
with weights for the models, feature order, normalization and training metadata.
The loader is `occupancy.py`. Upload the checkpoint with a model card and link
back to this guide. Use a private model repository for the first upload and
choose visibility and a weight license when ready. The dataset has its own license.
[Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload) ·
[Model cards](https://huggingface.co/docs/hub/en/model-cards)

```sh
.venv/bin/python -m pip install --upgrade huggingface_hub
. .venv/bin/activate
hf auth login
MODEL=YOUR_HF_ACCOUNT/intelligence-liquidity
hf repos create "$MODEL" --private
hf upload "$MODEL" checkpoint.json checkpoint.json
hf upload "$MODEL" README.md README.md
```

Copy the commit SHA from the final upload. Pin it when downloading an approved
version. Keep the SHA-256 for the file separately so you can verify its bytes.
The checkpoint included with this example has SHA-256
`bdb75939a613c0cd26c2ca66df77187a60e2e3e073054d4d90d35a15c4c26cc1`.

```sh
MODEL_REVISION=PASTE_APPROVED_COMMIT_SHA
hf download "$MODEL" checkpoint.json --revision "$MODEL_REVISION" --local-dir downloaded
shasum -a 256 downloaded/checkpoint.json
python occupancy.py infer occupancy.zip downloaded/checkpoint.json
```

## Host the browser replay

Use a [Hugging Face Static HTML Space](https://huggingface.co/docs/hub/en/spaces-sdks-static)
for this demo. It runs the trained seed 41 weights in the visitor’s browser and
needs no Python server. The Space and model repository have independent versions.
Update the replay and checkpoint together when promoting a new demonstration.

```sh
SPACE=YOUR_HF_ACCOUNT/intelligence-liquidity-demo
mkdir -p space-upload
cp replay.html space-upload/index.html
cat > space-upload/README.md <<'EOF'
---
title: Intelligence liquidity
sdk: static
app_file: index.html
---
Replay a trained LTC using measured office data.
EOF
hf repos create "$SPACE" --repo-type space --sdk static --private
hf upload "$SPACE" space-upload . --repo-type space
```

Open the Space and repeat the one-minute and four-minute example before making
it public. The same `index.html` also works with a static file host. A live
service that accepts room readings needs its own service and input validation.

## Replace a running model

Start with a shadow trial. Run predictions beside the existing schedule while
the existing controller keeps authority. Log elapsed time, missing sensors,
the model version, and the two kinds of mistake on later labelled data. Treat
stale or invalid readings as unknown and let the existing schedule take over.
Choose a threshold and promotion rule before scoring the next held-out period.
Train into a new checkpoint, keep the old version, and roll back by reloading
the last approved checkpoint, feature schema and threshold together.

The [original LTC paper](https://arxiv.org/html/2006.04439v4) uses the same
UCI dataset. This smaller teaching run uses
[`ncps`](https://github.com/mlech26l/ncps) with its own split and scoring rules.
