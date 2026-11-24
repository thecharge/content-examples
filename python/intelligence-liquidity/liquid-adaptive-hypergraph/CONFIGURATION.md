# Bound the experiment

All amounts below are configured defaults, not measured speed or accuracy claims.
Change a configuration in a new run and retain its file with the results.
The controller also records the actual resource profile of every candidate.

## Graph and training limits

[configuration.json](configuration.json) contains `learning`, `guards` and `limits`.
[training-configuration.json](training-configuration.json) supplies the offline defaults.
Counts and timing controls retain the preceding experiment's meanings.

| Bound | Default | Meaning |
| --- | ---: | --- |
| Experts | 1–32 | Total live experts across boards; initial model has five. |
| Boards | 16 maximum | Shared child boards count once as parameter owners. |
| Child depth | 3 maximum | Limits recursive calls. |
| Input fields | 64 maximum | Includes fitted, queued and derived inputs. |
| Liquid layers | 3–8 per board | Stable layer IDs survive insertion and removal. |
| Encoding width | 8–32 | Expert input representation. |
| Recurrent neurons | 4–16 per liquid layer | Eight-coordinate public readout remains available. |
| Attention heads | 1–8 per attention node | Global head ceiling is 64. |
| Attention nodes | 16 maximum | Attention-tree depth is at most three. |
| Parameters | 200,000 maximum | Checked before candidate allocation and fitting. |
| Tensor training reservation | 128 MiB | A conservative model/optimizer/state reservation, not total process memory. |
| Worst-case work | 1,000,000 units | A deterministic operation proxy, not elapsed time. |
| Batch | 8 maximum | Applies to public graph batches and fitting microbatches. |
| Pending labels | 64 maximum | New predictions refuse when this queue fills. |
| Replay/history | 576 labeled rows | Fitting and monitoring windows use 288 rows by default. |

The operation proxy includes hyperedge vertex pooling, message readouts, destinations
and sensor-presence checks. A shared child still incurs work for every calling context.
Shared parameter storage does not make repeated execution free.

Growth reserves space before allocating a shadow. Candidate fitting creates a fresh
optimizer, preserves retained tensors, scales old gradients, warms new parameters,
clips gradients and uses early stopping. The hard process ceiling belongs to the
worker, rather than the tensor reservation estimate.

## Hyperedge and similarity controls

These fields are under `limits.hyper_limits`:

| Field | Default | Meaning |
| --- | ---: | --- |
| `min_edges` | 0 | Minimum live hyperedges. Initial empty topology requires zero. |
| `max_edges` | 16 | Maximum live hyperedges in the entire model. |
| `max_members` | 16 | Selected liquid vertices, destination experts and sensor fields per edge. |
| `max_mass` | 16 | Maximum multiplier on the bounded learned edge message. |
| `minimum_samples` | 16 | Distinct labeled observation IDs needed for a pair measurement. |
| `window` | 96 | Maximum labeled rows considered for similarity evidence. |
| `minimum_context` | 8 | Rows needed before reporting a context's residual difference. |
| `merge_correlation` | 0.95 | Pearson threshold for a redundancy nomination. |
| `maximum_disagreement` | 0.03 | Mean absolute prediction difference allowed for that nomination. |
| `split_residual_gap` | 0.10 | Minimum difference between populated context residual means. |

The default controller constructs an edge from up to four existing experts, one
destination, and fields from its learned group selector. It reduces membership to
fit the configured incidence ceiling. Structural requirements such as two source
experts are model definitions, rather than statistical thresholds.

The graph also accepts explicit stable layer/neuron vertices through `Graph.connect`.
Topology edits belong on an isolated candidate. Splitting needs at least four source
experts so each child retains two. A merge needs compatible board, targets, sensor
fields and attention route. The resulting membership must still fit all ceilings.

The context residual difference is a diversity measurement. Its signed pairwise
form can cancel the shared label term. It is not evidence that either expert is
more accurate. The fresh-label quality gate makes that separate decision.

## Storage and feature controls

[storage-configuration.json](storage-configuration.json) configures the durable CLI:

| Field | Default | Meaning |
| --- | ---: | --- |
| `storage.max_bytes` | 256 MiB | Database page and immutable-object budget checked before commit. |
| `storage.max_events` | 10,000 | Maximum revisions in one ledger, including initialization. |
| `storage.max_objects` | 100,000 | Immutable manifests and tensor payloads. |
| `storage.max_event_bytes` | 1 MiB | Maximum serialized request, result and evidence references. |
| `storage.retained_revisions` | 8 | Recent snapshot payloads automatically retained. |
| `maximum_features` | 8 | Maximum explicit derived-feature definitions. |

Direct Python users pass `StorageLimits` to `Ledger` and `maximum_features` to
`Session`. The ledger persists its storage limits and refuses a conflicting reopen.
The session persists feature definitions and their ceiling.

These limits do not imply permanent history. Explicit pins can preserve older payloads,
but enough pins can exhaust storage. Event headers stay within the event-count ceiling.
Start a new branch to establish a new bounded horizon. Keep the parent when its full
provenance is still needed. A branch's parent hash is a pointer, not a copy of history.

The database budget excludes sibling training-boundary files and SQLite's temporary
rollback journal. The inherited runtime retains two archived boundary checkpoints
and its current boundary. The worker separately caps each file and process memory.
Provision filesystem space for these bounded files before starting a run.

## Hard process limits

[worker-configuration.json](worker-configuration.json) sets these defaults:

- Address space: 4 GiB.
- CPU time: 7,200 seconds.
- Wall time: 7,500 seconds.
- Individual file size: 256 MiB.

An operating-system refusal can terminate the worker before it records a new result.
The last committed SQLite revision remains the recovery point.
Run a smaller workload or configure a new worker envelope before retrying.
Do not interpret an allocation refusal as an accepted structural change.
