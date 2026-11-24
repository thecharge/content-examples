# Shared multi-stream liquid hypergraph

## Objective

Implement one adaptive model with private temporal state for arbitrary bounded room streams, CPU batching and a device-neutral execution interface. Keep all existing liquid experts, hierarchical delta attention, mutations, rewards, provenance and recovery. Preserve preceding experiments.

## User acceptance targets

- Single-stream correctness matches the preceding runtime.
- More than two concurrent streams, including six rooms times six streams:36 logical streams.
- One randomly initialized model learns from data without importing pretrained weights. Record initial, intermediate and final quality on rooms1–6.
- Compare final predictions against pretrained references using the same chronology and withheld future data. Matching or improving those scores is a measured acceptance target, never an assumed guarantee.
- Keep training-prefix, validation and untouched future-test labels separate. Replicated feeds test execution; do not count them as independent evidence or independent rooms.
- Shared mutations pass per-room checks. Fast or slow-label streams must not block unrelated streams.
- Document architecture, configuration, recovery, CPU/GPU behavior, tests, learning curves and measured limitations.

## Constraints

No GitHub Actions, driver changes or deployment. CPU verification is available; real GPU verification depends on accessible hardware. No subagents. Preserve old source and artifact fingerprints.

## Plan

1. Add private stream state, fair scheduling, explicit clocks and batched gather/scatter.
2. Coordinate contextual calibration, balanced shared fitting, per-room prospective gates, atomic shape adoption and rollback.
3. Extend transactional snapshots, feedback lineage, bounds and recovery to all streams.
4. Add scratch training,36-stream evaluation, original references, frozen future inference, learning/shape visuals and independent verifiers.
5. Run regression and new concurrency/causality tests, inspect diffs, update docs and evidence.
6. Before any authorized main delivery, verify Actions disabled.

## Verified outcome (2026-11-24)

The implementation and complete CPU experiments are finished. The all-room accuracy requirement remains unmet. See RESULTS.md and the machine-readable acceptance flags. Do not describe the experiment as full convergence.

- One shared graph serves 36 private stream contexts. Frozen one/six/36 answers match. Both final workers exited zero within their limits.
- Live empty-start learning accepted 11 room calibrations and no shared weight/topology change. It beats frozen pretrained accuracy in five rooms but misses H4. Offline scratch misses H3.
- Independent checks rebuilt original data, checked chronological splits, replayed models, reproduced prior H5/H6 scores, verified frozen tensors and recomputed all nine room vetoes and causal calibration receipts.
- Local regression: 94 tests passed in 79.206 seconds. Four extra lifecycle/causality tests passed in 2.321 seconds. CLI scratch/batch/reward/label/reload passed.
- Models, initial/intermediate/final layouts, prediction records, learning/evolution plots and exact executed-source archives are retained. The earlier experiment is unchanged. Raw archives/cache and superseded development artifacts are excluded.
- GPU execution is unverified. No driver changes, deployment, GitHub Actions or subagents were used.
- Final lint, document links, artifact manifest and delivery checks are recorded in reporting/verification.json. Git history identifies the authorized main delivery.

## Research limitation

The six recorded room timelines and one seed do not prove universal convergence or no future loss. Any next attempt to improve the missed rooms needs a declared development protocol and fresh untouched evaluation data. Do not select changes using these exposed future scores and then call this same test unseen.
