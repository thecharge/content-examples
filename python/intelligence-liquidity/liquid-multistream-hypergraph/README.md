# Liquid multistream hypergraph

One liquid model serves six rooms with six logical streams per room: **36 streams**.
Each stream has its own recurrent cells, delta-attention memory, descendant contexts,
clock and pending predictions. Weights, topology and the structural controller are
shared. Calibration is learned once per room, from deduplicated observations.

This separate experiment extends the [liquid adaptive hypergraph](../liquid-adaptive-hypergraph/README.md).
[BASELINE.json](BASELINE.json) pins the preceding source; that experiment is unchanged.
The liquid equations, experts, checker votes, hierarchical attention, shared vertices,
21 structural actions, rewards, provenance and recovery remain available.

The acceptance target is matching or improving pretrained accuracy per room after
starting from random weights. **A runnable adaptive model does not establish that target.**
Read [RESULTS.md](RESULTS.md) for the measured verdict, class support, costs and failures.
Repeated feeds establish execution and state isolation; they add no independent data.

- [Architecture](ARCHITECTURE.md): shared and private state, batching, learning and release gates.
- [Operations](OPERATIONS.md): streams, sensors, labels, rewards, snapshots and retraining.
- [Configuration](CONFIGURATION.md): graph, scheduler, memory and storage bounds.
- [Task and verification state](TASK.md): what is implemented, running and still unverified.

## Run locally

Use Python 3.12 and the pinned CPU dependencies. Run from this directory.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -q
python reporting/contract_checks.py -q
python worker.py experiment.py create /tmp/rooms.sqlite --multistream --from-scratch
python worker.py experiment.py feed /tmp/rooms.sqlite --input observations.jsonl
python worker.py experiment.py inspect /tmp/rooms.sqlite
```

`--from-scratch` imports no trained weights. Without it, creation explicitly loads
one pinned verified-board checkpoint. CPU is the verified execution backend.
An accelerator must be accessible through a compatible PyTorch build; requesting
an unavailable device raises an error. Real GPU execution has not been verified here.

## Reproduce the end tests

Supply the same twelve HPDmobile environmental and ground-truth ZIPs used by the
[reference experiment](../verified-liquid-board/README.md#data-and-evaluation). Its data section
links the archive/download-ID list. H1–H6 identify recorded houses, treated as six
room contexts. Raw archives and the derived cache stay outside this delivery; their
hashes and split metadata are retained.

```bash
python worker.py cold_start.py /path/to/archives /tmp/cold-start --seed 73
python worker.py stream_replay.py /tmp/cold-start/dataset.pt /tmp/live-streams --seed 73
```

The first command fits a randomly initialized model, selects it using validation
only, then tests frozen future inference. It also tests a pretrained model frozen
and retrained on the same new prefix. The second starts an empty model directly
under 36 feeds, delayed labels, rewards and guarded online adaptation. These are
different learning protocols and have separate results.

The cached tensors preserve the original preprocessing and archive fingerprints.
No future label enters fitting, calibration, structural rewards or early stopping.
Training curves, layouts, checkpoints, source archives and prediction records make
both successful learning and refusal to change inspectable.

While the live run is active, its daily `latest.pt` contains an actual runtime
snapshot. Capture one for the intermediate graph in another terminal:

```bash
python reporting/capture_layout.py /tmp/live-streams/latest.pt /tmp/live-streams/mid
```

Independent checks and plots use the same completed runs:

```bash
python reporting/audit_online.py /tmp/live-streams
python reporting/capture_layout.py /tmp/live-streams/final.pt /tmp/live-streams/learning
python reporting/acceptance.py /tmp/cold-start/dataset.pt /tmp/cold-start /tmp/live-streams /tmp/acceptance --archives /path/to/archives
python -m pip install -r reporting/requirements.txt
python reporting/build_report.py /tmp/cold-start /tmp/live-streams /tmp/acceptance /tmp/figures
```

The inherited `verify.py` checks the preceding single-stream experiment. The
multistream checks above additionally recompute room vetoes and compare independent
single-stream runs with six and 36 streams. No GitHub Actions are needed.
