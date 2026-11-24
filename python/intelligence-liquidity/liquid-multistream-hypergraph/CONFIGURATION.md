# Bounds and controls

[configuration.json](configuration.json) retains the preceding graph and learning
configuration. [stream-configuration.json](stream-configuration.json) adds explicit
stream bounds. The CLI loads both. Python callers can construct immutable
`Settings`, `Guards` and `StreamLimits` values.

| Stream setting | Default | Effect |
| --- | ---: | --- |
| `maximum_streams` | 36 | Maximum open logical identities |
| `maximum_rooms` | 6 | Maximum room histories/calibrators |
| `batch_size` | 8 | At most eight independent next samples in one forward call |
| `maximum_pending_per_stream` | 8 | Backpressure for one slow-label stream |
| `maximum_queue` | 72 | Ingress queue ceiling |
| `history_per_room` | 96 | Total distinct retained fitting examples per room, balanced by class |
| `maximum_deduplication` | 4096 | Retained observation fingerprints before an expiry watermark advances |
| `maximum_resident_bytes` | 1 GiB | Conservative tensor reservation for stream states and replay |
| `trial_timeout_predictions` | 20736 | Maximum logical predictions before unsupported checking expires |

The global pending-label limit is separate: 64 by default in `Settings`. The live
36-stream benchmark explicitly uses 288 to accommodate its delayed labels. Opening
streams does not reserve an unbounded queue. A batch is checked against global and
per-stream pending bounds before it runs. Closing requires draining or discarding
its outstanding work. Room identities and learned calibration remain retained and
bounded; closing a stream does not silently erase room learning.

The preceding graph defaults remain 32 experts, 16 boards, 3 levels of board nesting,
3–8 liquid layers, 4–16 liquid neurons per expert/layer, 1–8 heads per attention node,
16 attention nodes, 64 total heads, 64 input fields, 16 hyperedges and one million
estimated work units. These count different resources; a child board is not a
liquid layer and a head split is not an expert split.

The controller additionally caps parameters at 200,000 and estimated training bytes
at 128 MiB. Reservation includes simultaneous active/candidate/replay models,
optimizer buffers, replay and a conservative autograd estimate. Multistream state
adds its own reservation and can veto a candidate before fitting. These are
estimates. The worker’s 4 GiB address-space limit, 7200 CPU seconds, 7500 wall seconds
and 256 MiB per-file ceiling are separate hard OS limits. A rejected allocation is
not a promise that any requested configuration will fit; recover from the last
committed snapshot after a worker failure.

Prospective windows default to 288 distinct examples per room, with at least 16
examples of each class. Growth and weight changes retain the inherited minimum
accuracy/Brier gains and paired lower-bound check. A room veto forbids worse class
recall, Brier error or quorum coverage even when aggregate quality improves. Room
calibration uses a strict local class/quorum floor too.

Storage defaults remain 256 MiB, 10,000 events, 100,000 objects, 1 MiB per event and eight
retained revisions. Durable batch snapshots incur work beyond the neural forward
pass. Full benchmark prediction logs and checkpoints are artifacts; benchmark timing
must not be described as SQLite service throughput.

CPU is the canonical control backend. A read-only accelerator inference mirror uses
the PyTorch `cuda` device namespace, which a ROCm build also exposes. No GPU stream
implementation is hidden in commented code. The available host has a CPU-only
PyTorch build and no accessible GPU device, so accelerator parity is unverified.
