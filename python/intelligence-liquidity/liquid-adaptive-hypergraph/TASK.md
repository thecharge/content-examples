# Liquid adaptive hypergraph

Reader: a developer reviewing or repeating this experiment.
Purpose: build and measure a separate extension without changing its baseline.

## Objective

Add real higher-arity liquid messages, provenance-backed similarity decisions,
transactional revisions, lineage, deduplication and bounded orphan collection.
Keep liquid experts, neurons, layers, hierarchical delta attention, delayed labels,
rewards, shadow fitting, prospective checks, monitoring and rollback.
Do not promise unchanged accuracy on unseen future data.

## Delivery checklist

- [x] Computational hyperedges with bounded membership and delta routing.
- [x] Similarity measurements with support IDs and guarded action nominations.
- [x] Add, split, merge and retire hyperedges through the existing quality gate.
- [x] Observation provenance, schema and device incarnation records.
- [x] Transactional event revisions, exact artifact recovery and stale-write rejection.
- [x] Label corrections and transitive evidence invalidation.
- [x] Immutable tensor deduplication and reachability-based collection.
- [x] Configured disk, record, graph and training limits.
- [x] Tests for recovery, corruption, concurrency, missing inputs and mutation guards.
- [x] Complete and verify online and strict frozen same-data comparisons.
- [x] Observed visual evolution, including frozen independent inspection views.
- [x] Architecture, operation, configuration and limitation documents.
- [x] GPU diagnosis and explicit device report.

## Constraints

CPU is available. PyTorch is 2.7.1+cpu with no HIP runtime. This session exposes
neither /dev/kfd nor /dev/dri. No driver changes or GitHub Actions are authorized.
Keep the preceding experiments unchanged.

## Current verified state

- Implemented computational hyperedges, stable expert/layer/neuron memberships, exact partial-layer sharing and four learned edge actions.
- Kept the original liquid hierarchy, full-weight fitting, attention growth and shrinkage, delayed rewards, monitoring and rollback.
- Added SQLite atomic state/event commits, tensor deduplication, corrections, quarantine, roots and bounded collection.
- Passed 78 tests, Ruff checks/format and bytecode compilation. CLI smoke covered create, observations, rewards, labels, derived features, correction, clean restore and branch.
- Recovered the controlled ledger at revision 291 with one accepted hyperedge and no monitoring rollback. Full views are generated through reporting/build_evolution.py; raw inspection layouts must be deep-copied before subsequent mutations.
- Baseline's 22 pinned source hashes remain unchanged. Actions permission API reported enabled=false. No Actions run was used.

## Completed comparisons

Online seeds 73,83,97 completed and passed verify.py. H5 mean balanced accuracy matched at 86.65%. H6 changed from 63.25% to 63.33%. Of 50 completed proposals, one neuron contraction was accepted in seed73/H6. No computational hyperedge was adopted on these streams.

User clarified that completed training must also be tested by inference on unseen data. All six independent CPU runs completed and passed their strict verifier. Each trained only on the first14 selected days, then froze learned tensors and topology without test labels or rewards. Old/new scores matched for every seed: H5 mean77.96%, H6 mean63.25%. Outputs: evidence/strict/{new,previous}-seed-SEED. Limits:4GiB address space,7200s CPU and7500s wall time.

The later multi-stream question exposed a separate boundary: context records provenance but does not isolate recurrent state. Shared simultaneous learning across H5/H6 is not implemented or verified. OPERATIONS.md explains the required private states, coordinated mutations and per-room quality gates. Do not claim the current comparisons cover that case.

## Delivery state

Implementation and local verification are complete. RESULTS.md and the chart separate both protocols, actual structure counts and costs. reporting/verification.json records observed checks. MANIFEST.json fingerprints the delivered experiment files. The main-branch delivery must preserve the earlier experiments and keep GitHub Actions disabled.
