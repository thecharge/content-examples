# Run and recover an experiment

## Choose the entry point

`experiment.py` uses the transactional SQLite session. Use it for recorded live
inference, delayed feedback, corrections and exact state recovery.

`workflow.py` retains the preceding offline training interface. It can train from
scratch, fine-tune a new experiment checkpoint, and write inspected training receipts.
Its JSONL journal re-executes operations during recovery. That path does not provide
the new ledger's atomic event/state commit or exact tensor recovery.

Run entry points through `worker.py` when testing resource limits.
The worker applies address-space, CPU-time, wall-time and per-file limits.
An operating-system refusal kills that worker without replacing a committed session.

## Start from the verified liquid checkpoint

From this experiment directory:

```bash
python worker.py experiment.py create /tmp/room.sqlite
```

The default seed is 73. The initial event records the original checkpoint hash.
For a fresh model, add `--from-scratch`. This creates random initial weights; they
need training before their predictions mean anything.

To import a runtime trained in this new experiment:

```bash
python worker.py experiment.py create /tmp/tuned.sqlite --runtime /tmp/tuned.pt
```

The previous folder's runtime format has a different action-policy shape.
Do not rename an old checkpoint to the new schema. The default importer deliberately
uses the original verified board weights and constructs the new adaptive controls.

## Record a sensor reading

Each JSONL input line is one event. Here is a single reading, formatted across
lines for readability. Write it on one line in the input file.

```json
{
  "kind": "observation",
  "event_id": "room-a:reading:1",
  "event_time": 300.0,
  "arrival_time": 301.0,
  "context": "room-a:day",
  "values": {"temp_c": 0.2},
  "quality": {"temp_c": "good"},
  "sensors": {
    "temp_c": {
      "field": "temp_c",
      "source": "room-a:temperature",
      "incarnation": "installed-device-1",
      "room": "room-a",
      "schema": "environment-v1",
      "calibration": "calibration-1",
      "normalization": "training-normalization-1"
    }
  }
}
```

`values` already contains values in the model's input scale. Recording a calibration
or normalization version does not apply that transformation for you.
Keep raw-data transformation code and its version with the source data.

Feed the file:

```bash
python worker.py experiment.py feed /tmp/room.sqlite --input readings.jsonl
```

The session records both source event time and arrival time. The neural clock uses
arrival time, which must increase beyond previous predictions and label reveals.
Out-of-order source events can be recorded at their actual arrival time.
This experiment does not insert a late reading retroactively into recurrent state.

A prediction is returned only after its event and resulting state commit together.
On retry, reuse its event ID and identical input. A different input with the same ID
is rejected. After another process advances the database, refresh the session before
issuing a new operation.

## Give a later label or reward

```json
{"kind":"label","event_id":"room-a:label:1","prediction_id":"room-a:reading:1","truth":1,"revealed_at":302.0,"authority":"occupancy-meter"}
```

The target is binary occupancy. Labels arrive after their prediction, in prediction
order, and may not move the stream clock backwards. `authority` names the measured
label source. The default label schema is `occupancy-binary-v1`.

An optional reward must arrive while that prediction still awaits its label:

```json
{"kind":"reward","event_id":"room-a:reward:1","prediction_id":"room-a:reading:1","value":0.7,"revealed_at":301.5}
```

Rewards lie between -1 and 1 and apply once. They shape learned action values.
They do not create occupancy labels or override class, accuracy or quorum gates.
Without labels, recurrent state still changes, but supervised adaptation cannot
certify an improvement. The bounded pending queue eventually refuses more predictions.

## New, missing and replaced sensors

For a new field, give its provenance immediately. The runtime queues that field.
It continues predicting with fitted fields while waiting for a labeled schema stage.
The new field starts without an enabled learned contribution. After fitting, the
candidate must earn adoption on future observations.

An omitted existing field is missing. A supplied null reading is also missing.
`broken` and `stale` quality markers mask a supplied value while retaining its raw
recorded value for inspection. Missing values are not replaced with a fake zero measurement.

When replacing hardware, keep the logical field if its meaning and scale remain the
same, and change `incarnation`, calibration and normalization versions.
The model can retain the logical feature while the ledger distinguishes old and
new physical evidence. A changed meaning needs a new logical field.
The session records these facts; it does not automatically prove replacement calibration.

Use a separate session per room stream unless you intentionally train a combined
stream. Sharing parameters between boards does not mean two rooms should accidentally
share the same recurrent context.

### Two rooms at the same time

The current runtime owns one recurrent state, delta-memory state, clock and pending
feedback queue. An observation's `context` records provenance and informs similarity
measurements. It does not select an independent stream state.

Interleaving H5 and H6 into one session would therefore mix their temporal histories.
Two writers against one ledger also need revision coordination; they cannot update
one model concurrently. Loading two separate sessions creates independent learners,
even when they start from identical weights.

A jointly learning multi-stream extension needs private state and feedback queues
per stream, shared weights, coordinated mutation boundaries and state alignment.
Its acceptance checks must measure each room separately. A combined average could
hide deterioration in one room. Fair sampling must prevent a faster stream from
dominating the shared learner. Rewards must retain their stream and prediction IDs.

That extension is not implemented here. The recorded H5/H6 comparisons use separate
runtimes. They do not establish accuracy under simultaneous shared-model learning.

## Add a derived feature with lineage

```json
{"kind":"derived_feature","event_id":"room-a:feature:1","name":"combined:air","sources":["co2eq_ppm","tvoc_ppb"],"evidence_id":"room-a:label:1","operation":"mean"}
```

The evidence event must exist and remain valid. Means and two-source differences
are implemented. Sources must be distinct. Definitions cannot chain derived
features, overwrite an input, or exceed the feature/input ceilings.
Missing source values yield a missing derived value.
The model still needs labeled fitting before using the added field.

This is an explicit feature-definition API. The model does not automatically invent
semantic labels or arbitrary feature programs from a correlation score.

## Correct bad evidence

```json
{"kind":"correction","event_id":"room-a:correction:1","target":"room-a:label:1","reason":"Occupancy meter reported the wrong class"}
```

The correction leaves the old event body intact. It marks that evidence and its
dependent transitions invalid, records the affected IDs, and quarantines the model.
Further new predictions and learning are refused.

Pin a clean recovery point before a risky experiment. With the Python API:

```python
ledger.pin("before-replacement", session.revision)
```

To restore a retained clean revision, append a restore event:

```json
{"kind":"restore","event_id":"room-a:restore:1","revision":2}
```

Restoring discards the later model context in the new serving revision. It does not
delete the later audit events. Invalid or quarantined revisions cannot be blessed
as clean. If all clean payloads have expired, retrain from clean data or import a
retained clean checkpoint into a new ledger. A retained hash cannot reconstruct
deleted tensors.

## Point-in-time branches and retention

```bash
python worker.py experiment.py branch /tmp/room.sqlite --revision 2 --output /tmp/room-branch.sqlite
python worker.py experiment.py inspect /tmp/room-branch.sqlite
```

A branch records the parent revision, parent snapshot address and database hash.
Its event history begins with that imported state. Keep the parent database if you
need its earlier observation history. Cross-ledger invalidation is an operator task;
the code does not automatically propagate corrections into an offline copied branch.

Recent snapshots and explicit pins retain payloads. `ledger.unpin(name)` releases
a pin; `ledger.collect()` removes objects no longer reachable from any retained root.
The serving root cannot be unpinned. Snapshots include shadow/fallback/pending/replay
state, so these parts remain rooted while needed.

The event count, object count and database byte ceilings refuse new commits when
exhausted. Branch into a new ledger before that point. Audit event headers are not
silently truncated to make space. The SQLite rollback journal can temporarily occupy
additional space up to the changed database pages; filesystem free space remains
an operating-system concern.

## Train from scratch or fine-tune

Offline input lines contain `stamp`, `truth` and `values`.
The inherited workflow fits candidates, uses early stopping, and retains future
labels for prospective evaluation. It writes a receipt beside the saved runtime.

```bash
python worker.py workflow.py train scratch train.jsonl /tmp/scratch.pt
python worker.py workflow.py train finetune train.jsonl /tmp/tuned.pt --checkpoint /tmp/scratch.pt
```

Run `python workflow.py --help` and `python workflow.py train --help` for the exact
interface. Import the resulting runtime into a new durable session for live events.
Full feedback training and structural changes remain available after import.
The action learner is a contextual bandit; there is no PPO or distributed trainer.

## Inspect and reproduce

```bash
python worker.py experiment.py inspect /tmp/room.sqlite --revision 2
python worker.py evaluate.py /path/to/12-hpd-archives /tmp/hypergraph-seed73 --seed 73
python worker.py verify.py /tmp/hypergraph-seed73 --archives /path/to/12-hpd-archives
```

The benchmark uses the same twelve publisher archives, original seed checkpoints,
features, H5/H6 chronology and later scoring ranges as the preceding experiment.
It checks archive, source, baseline prediction and artifact hashes.
Baseline training is reused from fingerprinted saved predictions.
The benchmark measures the neural runtime; it excludes per-event SQLite durability cost.

For completed training followed by frozen inference on unseen later data, use
`reporting/strict_holdout.py`. Its test labels never enter the model.
[TESTING.md](TESTING.md) gives the paired new/previous commands and verifier.

Recovery requires the recorded Python, PyTorch and experiment source fingerprints.
Changed code requires an explicit migration rather than silent replay under a new
implementation. Exact saved-state recovery does not imply bit-identical training
across hardware or library versions.
