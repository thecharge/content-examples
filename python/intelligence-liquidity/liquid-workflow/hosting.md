# Host a liquid occupancy checkpoint

A Hugging Face model repository can hold the frozen checkpoint and its evaluation record. Running the checkpoint still needs the Python code in this directory. To let the model learn while serving, run the stream on a separate CPU process with persistent, private storage.

## Prepare the files

Choose the checkpoint and its matching cutoff from `evidence/fit.json`. Record the exact Git revision that produced it. Start a folder for the model card and upload files.

```sh
mkdir model-upload
cp evidence/41-100.pt evidence/fit.json evidence/scores.json model-upload/
sha256sum model-upload/41-100.pt
```

Write `model-upload/README.md` with the code revision, checkpoint hash, input units, cutoff 0.76 for this example, training data links, and held-out scores. State that the checkpoint estimates occupancy from temperature and carbon dioxide, and that a checker can withhold the estimate. Link the Python runtime and note that scores vary by room. See the [Hugging Face model card guide](https://huggingface.co/docs/hub/model-cards).

## Upload and replay

Install `huggingface_hub` in the virtual environment and use a Hugging Face account with write access. Replace `YOUR_ACCOUNT` below. The first command creates a private model repository.

```sh
python -m pip install huggingface_hub
hf auth login
hf repos create YOUR_ACCOUNT/liquid-occupancy --type model --private
hf upload YOUR_ACCOUNT/liquid-occupancy model-upload .
hf download YOUR_ACCOUNT/liquid-occupancy --local-dir downloaded-model
python serve.py downloaded-model/41-100.pt --cutoff .76
```

Check the downloaded checkpoint against the hash before loading it. Supply readings as described in the [runtime instructions](README.md#send-readings-to-an-inference-process). The upload commands follow the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

For a private live service, run one process per room or independently managed stream. Add `--live --snapshot state.pt` on persistent storage and restrict both inputs and snapshots to the service account. This example exposes a local JSON-lines process. A web deployment needs its own authenticated adapter and per-stream request ordering. Keep the original checkpoint for a restart with fresh state.
