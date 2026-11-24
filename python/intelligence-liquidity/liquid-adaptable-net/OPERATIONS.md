# Train, run and recover

Use Python 3.12 with [requirements.txt](requirements.txt). All examples run from this folder. Use [configuration.json](configuration.json) for live learning, [training-configuration.json](training-configuration.json) for offline fitting and [worker-configuration.json](worker-configuration.json) for hard process limits. See [configuration](CONFIGURATION.md) for every key.

## Inspect models and reproduce comparisons

```sh
python worker.py workflow.py inspect evidence/seed-73/H5-adaptive.pt
python worker.py --receipt evidence/worker-73.json evaluate.py \
  /path/to/hpdmobile-archives evidence/seed-73 --seed 73
python worker.py verify.py evidence/seed-73 --archives /path/to/hpdmobile-archives
```

Repeat for seeds 83 and 97. Normal verification recalculates scores and trial gates, verifies baselines and artifact hashes, and restores the saved graph. Add `--rerun` to the verifier for a complete new computation and comparison of prediction hashes. It takes substantially longer.

The archive directory follows the [original download instructions](../liquid-workflow-v3/README.md#reproduce-the-fits). No data is downloaded automatically. The comparison reuses the original H1–H3 training and H4 selection; it starts each H5/H6 runtime on day one and scores the original first-week and after-day-14 ranges. The last answer remains pending. Reset recurrent state on missing five-minute windows.

`evaluate.py` reuses the previously published baseline predictions only after checking their complete source and artifact fingerprints. It checks each timestamp and label against the parsed data. It does not rerun baseline training. `compare.py` offers the additional, expensive original-contextual rerun protocol.

## Ordered online learning and rewards

```python
import torch
from graph import Graph
from reference import Board, REFERENCE
from runtime import Runtime
from workflow import Journal

base = Board(features=17)
base.load_state_dict(torch.load(
    REFERENCE / "evidence/seed-73/board.pt", weights_only=True))
runtime = Runtime(Graph(base))
session = Journal(runtime, "events.jsonl")
issued = session.predict(300., {"temp_c": 0.2, "new:sound": 0.7})
session.reward(issued["token"], value=0.6, revealed_at=301.)  # Optional.
session.observe(issued["token"], truth=1, revealed_at=302.)
session.snapshot("snapshots/at-302.pt")
```

This API example runs in its caller's process. To install hard limits, put the host loop inside a worker-managed application or extend the worker's explicit entry-point list for that application. Merely importing `Runtime` does not change the host application's process limits.

Use stable names and explicitly scaled, finite numeric inputs. Missing evidence uses `None`/JSON `null`. Names beginning `parent:` are internal. The reference normalization remains required for original fields. New names are queued until a safe boundary; their zero-initialized projections remain unsupported until a fitted candidate earns adoption.

Tokens identify issued predictions. Reveal outcomes once, in order, strictly after issuance and without moving event time backwards. An optional scalar reward must be finite and within `[-1, 1]`, can be attached once, and must also be delayed. It shapes structural reinforcement credit; it does not replace the binary truth needed by the accuracy gate. Rewards and truths can arrive at the same later timestamp. Discard an unavailable outcome explicitly through `session.discard(token)`.

The network evolves liquid state, delta memory and routing during inference. It can execute a proposal at a boundary prepared by earlier outcomes. It does not refit weights on an unknown current answer. After adoption it keeps the predecessor for a further monitoring block; proposals pause during monitoring. Later harm or insufficient class evidence restores the predecessor.

Inference without labels eventually reaches `maximum_pending`; consume or explicitly discard tokens. Pending storage is bounded. This classification experiment is not a reward-only environment agent; see [architecture](ARCHITECTURE.md) for the scope of its contextual-bandit reinforcement learner.

## Full checkpoints and point-in-time recovery

```python
runtime.save("running.pt")
resumed = Runtime.load("running.pt")
print(resumed.graph.layout())
print(resumed.graph.profile())
print(list(resumed.monitor_audit))
```

A checkpoint stores active and candidate models, a monitoring predecessor when present, all private states, attention and neuron identities, sharing, policies/replay, schema, pending outcomes, queued fields and RNG state. It also preserves unfinished checks. It does not pretend pending answers are known. This experiment uses its own checkpoint schema; previous-experiment runtime snapshots use their own loader. The supplied original weight checkpoint is the common comparison starting point.

`Runtime.save` atomically replaces its target. A stage also writes a durable boundary checkpoint before mutation, plus a configured number of numbered boundaries. For point-in-time recovery, use the journal wrapper and a matching snapshot sidecar:

```sh
python worker.py workflow.py recover snapshots/at-302.pt events.jsonl \
  restored.pt --until 600
```

The sidecar fingerprints the checkpoint, cursor and journal prefix. Recovery rejects tampering, missing/duplicated sequences and backward clocks. It replays predictions, rewards, outcomes, resets and discards through the requested time. Keep the snapshot, sidecar and original journal together. A bare mutation-boundary checkpoint does not replace this journal evidence.

Resume on the restored branch journal, or create a new one:

```python
from workflow import recover, Journal
runtime, cursor = recover("snapshots/at-302.pt", "events.jsonl", until=600.)
session = Journal(runtime, "branch.events.jsonl", cursor=cursor)
```

A worker memory, CPU or wall limit can end a run. Restart from the latest durable checkpoint; use the event journal to recover later completed events. A worker failure cannot guarantee preservation of unjournaled work. Cancellation terminates its child rather than leaving training running in the background.

## Scratch training and fine-tuning

Input is chronological JSON Lines data, with an optional environment reward:

```json
{"stamp":300,"values":{"temp_c":0.2,"new:sound":0.7},"truth":1,"reward":0.6}
{"stamp":600,"values":{"temp_c":0.1,"new:sound":0.1},"truth":0}
```

Supply enough rows of both classes for fitting, validation and prospective structural checks.

```sh
python worker.py workflow.py train scratch training.jsonl scratch.pt
python worker.py workflow.py train finetune training.jsonl tuned.pt \
  --checkpoint restored.pt --until 600
```

Scratch training uses random initial weights. Fine-tuning copies the loaded topology and weights and applies the supplied training envelope. Incompatible limits cause refusal before the copy. Both modes run the same adaptive stage-selection loop over earlier and subsequent training observations, then fit the last accepted graph at a fixed shape. The receipt lists structural decisions and an unfinished selection trial, if any. That unfinished candidate is excluded from the final fit.

The final fit uses a fresh AdamW optimizer, chronological validation, early stopping and best-weight restoration. Default offline maximum is 20 passes; live stages default to one. New weights warm up while old updates use the configured lower rate. States and optimizer momentum start fresh for the offline fork. This is snapshot fine-tuning, not optimizer-step continuation halfway through an interrupted backward pass.

Outputs are the model and `.training.json` receipt with source-data/checkpoint hashes, range, settings, selection decisions and validation history. They do not overwrite the source snapshot or automatically replace a live model. The `--until` cutoff limits training rows; journal recovery controls which events existed at a chosen time. Keep those operations distinct.

## Verify mechanisms and performance

```sh
python -m unittest discover -v
python -m unittest discover -s ../verified-liquid-board -v
python worker.py demonstrate.py evidence/controlled-60.json --seed 60 --windows 145
python worker.py demonstrate.py evidence/controlled-62.json --seed 62 --windows 145
python worker.py demonstrate.py evidence/controlled-65.json --seed 65 --windows 145
```

The controlled streams exercise layer growth, recurrent splitting and attention branching. Their cue is constructed to be perfectly informative. Other fixed illustrative seeds are 63 for encoding splits, 68 for child growth and 88 for expert retirement. They establish mechanism behavior, not superiority on occupancy data.

`reporting/profile_model.py` measures graph inference in a fresh one-thread process, with a warmup pass and three measured passes on identical inputs. It excludes fitting and feedback. The report separately gives parameter count, checkpoint size and process peak RSS. CPU timing on a shared host is noisy; inspect the method and raw samples before treating differences as an architectural speedup. Report generation and profile fingerprints are separate from model-source fingerprints.

No command uses GitHub Actions. Source, model snapshots, predictions and receipts are ordinary files on main for independent inspection.
