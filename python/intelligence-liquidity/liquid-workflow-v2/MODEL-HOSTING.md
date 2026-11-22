# Host the saved liquid model

Start with the verified local replay in [README.md](README.md). The file
`evidence/workflow.pt` contains PyTorch tensors. The architecture is in
`workflow.py`; a generic language-model endpoint cannot load this file.

## Put a checkpoint in a model repository

1. Create a private Hugging Face **model** repository.
2. Add a model card that links to the source version, describes the CO₂
   and thermometer inputs, and links to the frozen transfer receipt.
3. Upload the weights and fit receipt. Keep source archives outside the
   model repository and link to the publishers instead.

```sh
python -m pip install huggingface_hub
hf auth login
hf upload YOUR_NAME/liquid-workflow evidence/workflow.pt workflow.pt
hf upload YOUR_NAME/liquid-workflow evidence/fit.json fit.json
hf upload YOUR_NAME/liquid-workflow evidence/h6.json h6.json
```

Record the Hub revision. Download from that revision into a fresh location,
then run the same three readings. The replay checks the hash before it calls
`torch.load(..., weights_only=True)`.

```sh
hf download YOUR_NAME/liquid-workflow workflow.pt fit.json \
  --revision YOUR_COMMIT_SHA --local-dir downloaded
python serve.py downloaded < examples/readings.jsonl
```

## Run the model

A small Python service can wrap the replay protocol. Give each room a
separate model session and require readings in time order. Keep the frozen
checkpoint available to restart the session. Enable live learning only for
that session after replaying the local validation data. Watch withheld
answers, sensor errors and occupancy errors separately.

A Hugging Face Space can run this code in a Docker image. Use the same Python
and dependency versions. Pin the weights and replay the saved input during
health checks. Use a private Space for data that identifies a building or
its occupants.

The example was run locally. The Hub steps describe hosting and were not
run against an account.

- [Hugging Face model upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [Hugging Face Docker Spaces guide](https://huggingface.co/docs/hub/en/spaces-sdks-docker)
