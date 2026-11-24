# Feeding, learning and recovering the model

## Start empty or load learned weights

```bash
python worker.py experiment.py create /tmp/rooms.sqlite --multistream --from-scratch
```

This creates random liquid weights with no trained checkpoint. An empty model may
issue a strong class bias; it is not required to predict 50% in each room. Learning
requires externally measured labels or a separate supervised training stage.
Inference alone updates liquid memory and attention, not supervised correctness.

Omitting `--from-scratch` loads the explicitly recorded verified-board checkpoint.
`--runtime checkpoint.pt` imports a saved runtime, including its graph, histories,
room calibrations, private stream states, pending tokens, bounds and RNG state.
For offline bootstrap/retraining use `cold_start.py`; its source archive, selected
epoch and checkpoint identify exactly what was fitted. The inherited `workflow.py`
remains a single-stream workflow; use multistream `Session` for multistream recovery.

## Stream and evidence identities

A stream ID names one temporal history and cannot change room. A room ID names one
calibration/history/quality group. An evidence ID names an underlying physical
observation. A sensor incarnation names one installed device. These IDs serve
different purposes.

Python callers can use the lightweight coordinator directly:

```python
runtime.open("room5/camera-a", "H5")
result = runtime.predict(
    1800000300, {"temperature": 0.2, "light": None},
    stream_id="room5/camera-a", room="H5",
    evidence_id="room5/observation/100",
)
runtime.observe(result["token"], 1, 1800000301)
```

Use the actual normalized feature names from the model for existing inputs. Unknown
names queue as new fields and remain unsupported until a labeled fitting stage.
The direct coordinator is a single-owner in-memory API. Use `Session` when operations
must roll back on errors and predictions must be durable before being returned.

For a transactional observation, `Session.predict` takes event ID, sensor event
time, arrival time, values, a `Sensor` object for every field, `context` as room ID,
`stream_id`, and `evidence_id`. The JSONL CLI accepts the same named fields. Sensor
metadata includes field, source, incarnation, room, schema, calibration and
normalization. Missing, broken and stale values become absent evidence.

`Session.predict_batch(event_id, observations)` atomically commits one microbatch.
Every item has its own observation ID, times, values, sensors, room context, stream
ID and evidence ID. A batch may contain one next observation per stream, up to the
configured batch limit. Returned predictions have tokens. Feedback names the batch
ID plus `prediction_token`; it cannot accidentally grade another batch's token.

```python
session.label("label-42", "batch-7", truth=1, revealed_at=1800000301,
              authority="measured-occupancy", prediction_token=42)
session.reward("reward-42", "batch-7", value=1.0, revealed_at=1800000301,
               prediction_token=42)
```

Labels are external binary occupancy facts. Rewards are finite values in [-1, 1]. A
reward can influence policy credit but cannot manufacture a label or bypass gates.
Repeated event IDs return the committed result when the request is identical;
conflicting retries fail. Batch event IDs are durable ledger events. Individual
observation IDs inside a batch are metadata; corrections name the batch event and
invalidate the dependent batch history.

## New, missing and replaced sensors

A new sensor's field enters the bounded schema queue. Existing streams continue
using the supported schema. At a fitting boundary, input weights expand without
changing existing contributions, private correction vectors map by feature name,
and incompatible contextual replay histories reset. Later fitting and prospective
checks decide whether the new evidence improves the model. No label vocabulary is
automatically invented.

A broken or silent sensor supplies `null` with the appropriate quality marker.
Availability masks prevent it from acting as a genuine zero reading. Liquid time
and attention forgetting still advance when another observation arrives. If a room
sends nothing, no synthetic observation is invented and other rooms continue.

A replacement records a new incarnation. Reusing a semantic field requires compatible
units, calibration and normalization supplied by the caller; a changed meaning
needs a new field. The network cannot establish device equivalence from a field
name alone. Corrections can invalidate the old evidence and its dependent model
revisions rather than erasing provenance.

## Delay, backpressure and deduplication

Prediction clocks must increase within a stream. Labels must follow that stream's
prediction order and arrive after the issued event and its recorded arrival time.
Other streams need not wait. Reward and label arrival times are validated within
the relevant stream. Shared fit/check causality uses the coordinator's availability
sequence rather than assuming different rooms have synchronized clocks.

Replicas use the same room and evidence ID. Their event time, normalized values and
truth must agree. The first revealed copy trains shared/room learning; later copies
drain their private pending tokens without inflating support. Conflicts fail and
require correction. Once a retained fingerprint expires, a room watermark rejects
older unrecognized evidence; it does not count a very late replica as a new sample.

A full stream queue receives backpressure. `drain()` can still select healthy
streams. Lack of one room's checking labels can block a **shared adoption**, not
another room's predictions. Such a trial expires under the configured limit.

## Snapshots, rollback and point-in-time recovery

The SQLite session commits event, snapshot, evidence dependencies and returned
result together. A process restart restores every stream, room calibration, pending
feedback token, queue, graph version and control state. A stale writer refreshes
rather than overwriting another committed revision.

`correct(event_id, target, reason)` invalidates dependent revisions and quarantines
learning/inference. `restore(event_id, revision)` requires a clean retained revision.
Branching creates a new database with parent snapshot provenance. Automatic
monitoring rollback restores the earlier shared graph and its private fallback
trajectories; full point-in-time restoration additionally restores the recorded
calibrations, histories and outstanding work from that database revision.

Snapshots do not replace backups. Keep the database and retained objects within
the explicit storage envelope. A changed source/environment fingerprint requires
an explicit migration rather than silently resuming under different code.
