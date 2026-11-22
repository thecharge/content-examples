# Host a saved liquid model

Bundle `model.py`, the selected `.pt`, the fit and score receipts, and a model card. Describe the input order, normalization and units. Specify whether to reset the recurrent state or carry it for each room. For occupancy, normalize CO₂ as `(ppm - 500) / 200` and temperature as `(celsius - 22) / 3`. Apply the cutoff from the manifest to the sigmoid output. The heat checkpoints use different inputs and have different heads.

Give the exact commit and checkpoint hash. Link the original data and their terms. Leave raw sensor archives at their publisher URLs. Report the held-out results and the limits of logged action replay. Label the weights as a research example for offline use.

To upload to the Hugging Face Hub, authenticate with a write token kept outside the bundle. Run these commands from the directory containing the bundle. Choose a model repository you own.

```sh
python -m pip install huggingface_hub
hf auth login
hf repos create YOUR_ACCOUNT/liquid-blackboard --repo-type model --private
hf upload YOUR_ACCOUNT/liquid-blackboard . . --repo-type model
```

Download a pinned revision to test it. Compare the checkpoint's SHA-256 with the model card. Use `torch.load(path, weights_only=True)` and load the resulting state dictionary into the matching architecture. Keep adaptation separate per stream. A restart from the original checkpoint gives you a reference for diagnosing drift. Hosting the files makes the experiment available for others to replay. A hosted inference service also needs a runtime that keeps each stream's state and updates isolated.

[Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
