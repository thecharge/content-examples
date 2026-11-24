# Run, recover and retrain a model

Use the pinned environment from [README.md](README.md). Commands below run from this folder. All work runs on CPU.

## Choose configuration

`configuration.json` describes live adaptation and the house comparison. `training-configuration.json` describes offline scratch training and fine-tuning. The offline defaults allow 20 passes, a 20% chronological validation tail, patience two and restoration of the best validation weights.

Make a copy before changing settings. Pass that file explicitly:

```sh
python compare.py /path/to/hpdmobile-archives ./comparison-73 \
  --seed 73 --configuration ./my-live-configuration.json
python workflow.py train scratch ./training.jsonl ./scratch.pt \
  --configuration ./my-training-configuration.json
```

The three sections have different roles:

| Section | Values to review |
| --- | --- |
| `learning` | Label window, retained history, class minimum, exploration, fitting rate and passes, validation, policy replay updates and quality gates |
| `guards` | Target error, hysteresis, drift and gradient thresholds, expert lifetime, cooldown, memory estimate, parameter ceiling and warmup |
| `limits` | Maximum experts, hierarchy depth, named fields, encoding width, minimum/maximum delta heads and per-policy replay capacity |

The `limits` section constructs a new graph. A loaded checkpoint retains its own graph limits and topology. Its learning and controller settings are preserved when resuming. Fine-tuning explicitly uses the supplied training settings on a copied graph.

The original contextual learner retains the reference optimizer and bounded-correction protocol. Those values are explained in the original [`contextual.py`](../verified-liquid-board/contextual.py), rather than relabeled as newly learned controls. The wider architecture and operating ceilings are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Inspect a saved network

```sh
python workflow.py inspect evidence/seed-73/H5-adaptive.pt
```

The output lists named inputs, expert identities, child edges, shared kernels, encoding widths and head identities. It also lists learning settings, controller guards, pending answers and recent contextual update decisions.

The checkpoint is the complete running model, rather than only a weight array:

```python
from runtime import Runtime

model = Runtime.load("evidence/seed-73/H5-adaptive.pt")
print(model.graph.layout())
print(list(model.contextual_audit))
print(list(model.receipts))
```

Both pending labels and an unfinished structural trial are retained. Loading does not invent the final unrevealed answer.

## Keep an event journal and take snapshots

Use `Journal` when point-in-time recovery is required:

```python
from runtime import Runtime
from graph import Graph
from workflow import Journal

runtime = Runtime(Graph())  # Fresh clock for this example.
session = Journal(runtime, "branch-a.events.jsonl")
issued = session.predict(300., {"temp_c": 0.2, "window_open": 1.})
session.snapshot("snapshots/after-prediction.pt")
session.observe(issued["token"], truth=1, revealed_at=301.)
session.predict(600., {"temp_c": 0.3, "window_open": 0.})
```

Each successful call writes and flushes a numbered event. A snapshot includes a `.snapshot.json` sidecar with its checkpoint hash, cursor and journal-prefix hash. Keep the checkpoint, sidecar and original journal together. `session.reset_state()` and `session.discard(token)` also record events.

Call prediction and observation methods through the journal wrapper. Direct calls to `runtime` are not recorded. Snapshot after an event has been written, through `session.snapshot`. Automatic mutation-boundary checkpoints are useful for stage recovery; they do not substitute for a journaled snapshot and its sidecar.

To continue an existing journal after restarting, load its latest matching snapshot and recover the subsequent events first. The wrapper rejects a file whose last sequence is ahead of the supplied cursor. This prevents appending an unrelated future to a restored model.

## Restore to a chosen time

```sh
python workflow.py recover snapshots/after-prediction.pt branch-a.events.jsonl \
  restored-at-600.pt --until 600
```

This restores the snapshot, checks the earlier journal prefix and replays events through time 600. The prediction at 600 is retained as pending. Its later answer is excluded. Times use the caller's numeric clock; benchmark clocks are Unix seconds.

The restored checkpoint receives its own snapshot sidecar. Future events belong in its new `restored-at-600.pt.events.jsonl` journal. Recovery cannot move earlier than the selected snapshot or recover events missing from the journal.

```python
from workflow import recover, Journal

runtime, cursor = recover("snapshots/after-prediction.pt",
                          "branch-a.events.jsonl", until=600.)
branch = Journal(runtime, "branch-b.events.jsonl", cursor=cursor)
```

Missing or duplicate sequences and changes to the saved prefix cause recovery to fail. The behavioral test also checks pending-label identity and the next prediction after recovery.

## Train from scratch

Training input is a JSON Lines file with one externally labeled observation per line:

```json
{"stamp":300,"values":{"temp_c":0.2,"window_open":1},"truth":1}
{"stamp":600,"values":{"temp_c":0.1,"window_open":0},"truth":0}
```

Supply enough chronological rows of both classes for fitting and validation. Missing numeric inputs use `null`. Input names and magnitudes obey the same schema and resource guards as the runtime. Existing reference field names must use the reference normalization.

```sh
python workflow.py train scratch training.jsonl scratch.pt
```

Scratch training creates random liquid weights and trains the new graph. It does not load the original checkpoint. It writes `scratch.pt` and `scratch.pt.training.json`. The receipt records data and checkpoint hashes, training range, configuration, validation history and final graph.

## Fine-tune a snapshot

```sh
python workflow.py train finetune training.jsonl tuned.pt \
  --checkpoint restored-at-600.pt --until 600
```

This copies the restored graph and fits only rows through time 600. It retains the snapshot's topology, named parameters and contextual correction. It starts fresh inference states and a fresh optimizer for the new fitting pass. Existing weights use the configured lower learning rate; newly added inputs can learn at the new-weight rate.

The command writes a separate model and receipt. It does not overwrite the source snapshot or automatically adopt the result into a live stream. Assess it on later data before deployment. The `--until` cutoff concerns training observations; it is distinct from event-log recovery.

## Check evidence

```sh
python -m unittest discover -v
python -m unittest discover -s ../verified-liquid-board -v
python verify.py evidence/seed-73 --archives /path/to/hpdmobile-archives
python verify.py evidence/seed-83 --archives /path/to/hpdmobile-archives
python verify.py evidence/seed-97 --archives /path/to/hpdmobile-archives
```

Add `--rerun` to reproduce complete house predictions and compare their saved hashes. This performs the training again and takes longer. The normal verifier recalculates scores and structural gates from saved predictions and checks all recorded artifacts.

No command starts GitHub Actions. The model, data-derived predictions, receipts and documents are ordinary repository files for inspection.
