# Intelligence liquidity

Train a liquid time-constant model on measured office data, restart its run from an early snapshot, and check which model should stay active. Everything runs locally on CPU.

## Reproduce the measured run

The archived result used Linux x86-64, Python 3.12.3 and PyTorch 2.7.1+cpu. Run these commands from the `intelligence-liquidity` directory. Use the same versions to compare exact bytes. Other environments can need a numerical tolerance.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.7.1'
curl --fail --location 'https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip' -o occupancy.zip
out="$(mktemp -d)/run"
.venv/bin/python -B lifecycle/model.py verify occupancy.zip "$out" --expect lifecycle/receipt.json
```

The program checks the archive hash before reading the ZIP. It prints fitting loss, validation loss, the saved epoch, how many passes have missed the gain threshold, and the stop reason. In the recorded run, the first batch loss dropped from 0.76858 to 0.75130. The liquid model saved epoch 7 and stopped at epoch 9. Restarting from epoch 2 reproduced the final snapshot byte for byte. [Full measured receipt](receipt.json)

Use the saved epoch 2 snapshot to check just the restart. Inference takes a separate, smaller candidate file and checks its SHA-256 before opening the saved weights.

```bash
resume="$(mktemp -d)/run"
.venv/bin/python -B lifecycle/model.py resume occupancy.zip lifecycle/ltc-02.json "$resume"
cmp lifecycle/ltc-09.json "$resume/ltc-09.json"

.venv/bin/python -B lifecycle/model.py infer lifecycle/candidate.json lifecycle/sample.csv \
  --sha256 8fd2e03f0e1dc482e34ec9e7a83b3bb6ee2cd67e2ec4f7c732c4dc1c380832e2 \
  --now '2015-02-02 14:30:00'
```

The score is 0.920682 for the eight recorded rows. Change the clock to 14:35 and the input is refused as stale. Run `python3 lifecycle/replay.py` and change `PATIENCE` from 2 to 1 to see why stopping sooner can leave the same weights selected.

## How to read the evidence

The first 6,500 training rows update the weights and set feature scaling. The last 1,643 provide validation feedback and decide whether to promote. The publisher's two other files were held out for this run. The model missed 17 and 22 occupied targets in those files at recorded cadence. The GRU missed 7 and 134, and a model using only the final row missed 2 and 121. The candidate and the previous liquid model missed the same counts. The receipt also shows false alarms and a separate simulated dropout replay on matching targets.

The reported validation gain was about 0.00000002. The release check required 0.005 and kept the previous checkpoint. The local replay separately exercises a trial switch, a stale-input error and a rollback to the previous SHA-256. This is evidence for one measured office and a checked local model lifecycle. A live installation requires fresh data, a tested fallback and monitoring of stale inputs and both kinds of error.

The example uses Luis Candanedo and Véronique Feldheim's UCI *Occupancy Detection* dataset under CC BY 4.0. The label comes from time-stamped photographs. `sample.csv` contains eight source rows from `datatest.txt` with a normalized header. The source ZIP SHA-256 is `4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a`. [UCI dataset](https://doi.org/10.24432/C5X01N) · [Original LTC paper, sections 2 and 3](https://arxiv.org/html/2006.04439v4)

## Store an approved revision on Hugging Face

Use a private model repository for the approved checkpoint, scorer and receipt. The commands below use the old checkpoint because that is the one this run kept active. Keep epoch snapshots in training artifact storage if they have no role in serving. [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

```bash
.venv/bin/python -m pip install huggingface_hub
.venv/bin/hf auth login
model=YOUR_HF_ACCOUNT/intelligence-liquidity
upload="$(mktemp -d)"
mkdir -p "$upload/lifecycle" "$upload/custom-ltc"
cp lifecycle/README.md "$upload/README.md"
cp lifecycle/model.py lifecycle/receipt.json lifecycle/sample.csv "$upload/lifecycle/"
cp custom-ltc/model.py custom-ltc/checkpoint.json "$upload/custom-ltc/"
.venv/bin/hf repos create "$model" --private
.venv/bin/hf upload "$model" "$upload" .
```

Keep the exact commit printed by the upload command. Download that revision and verify the checkpoint fingerprint while scoring the known input. Hugging Face stores the files. A running service still needs a Python application using the scorer and an environment with the pinned dependencies.

```bash
revision=PASTE_APPROVED_COMMIT_SHA
.venv/bin/hf download "$model" --revision "$revision" --local-dir downloaded
.venv/bin/python -B downloaded/lifecycle/model.py infer downloaded/custom-ltc/checkpoint.json \
  downloaded/lifecycle/sample.csv \
  --sha256 bfd5704418c51fcb84454a21d76dca9091d0187ddb489d9791cf8a8a85e4c1a4 \
  --now '2015-02-02 14:30:00'
```

### Run the same replay in a private Space

This Space replays the public eight-row sample against the approved checkpoint.
Its clock slider is a historical replay. It shows both a successful score and
the stale-input refusal. [Gradio Spaces guide](https://huggingface.co/docs/hub/en/spaces-sdks-gradio)

```bash
space_files="$(mktemp -d)"
cp -R downloaded/lifecycle downloaded/custom-ltc "$space_files/"
cat >"$space_files/README.md" <<'CARD'
---
title: Intelligence liquidity
sdk: gradio
python_version: "3.12"
app_file: app.py
---
Replay measured UCI Occupancy Detection records 143–150 (CC BY 4.0).
https://doi.org/10.24432/C5X01N
CARD
cat >"$space_files/requirements.txt" <<'REQ'
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.7.1+cpu
REQ
cat >"$space_files/app.py" <<'PY'
import hashlib
import json
from pathlib import Path

import gradio as gr

from custom_ltc import model

ROOT = Path(__file__).resolve().parent
approved = (ROOT / "custom-ltc/checkpoint.json").read_bytes()
fingerprint = hashlib.sha256(approved).hexdigest()
if fingerprint != "bfd5704418c51fcb84454a21d76dca9091d0187ddb489d9791cf8a8a85e4c1a4":
    raise RuntimeError("The approved model file does not match its fingerprint.")
checkpoint = json.loads(approved)
rows = model.csv_rows(ROOT / "lifecycle/sample.csv")


def score(minute):
    try:
        result = model.infer(checkpoint, rows, f"2015-02-02 14:{int(minute):02d}:00")
    except ValueError as error:
        return {"status": "unknown", "reason": str(error)}
    return result


if __name__ == "__main__":
    gr.Interface(
        fn=score,
        inputs=gr.Slider(30, 35, step=1, value=30, label="Historical clock minute after 14:00"),
        outputs=gr.JSON(label="Score or refusal"),
        title="Replay eight office readings",
    ).launch()
PY
mkdir -p "$space_files/custom_ltc"
mv "$space_files/custom-ltc/model.py" "$space_files/custom_ltc/"
space=YOUR_HF_ACCOUNT/intelligence-liquidity-demo
.venv/bin/hf repos create "$space" --repo-type space --sdk gradio --private
.venv/bin/hf upload "$space" "$space_files" . --repo-type space
```

Minute 30 should score about 0.920682. Minute 35 should return `unknown`.
The example keeps the approved file fingerprint in the app. For the next
approved model, update and test the app and checkpoint together.
