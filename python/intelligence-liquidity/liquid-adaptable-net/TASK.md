# Liquid Adaptable Net delivery record

Objective: deliver a separate CPU experiment copied from adaptive-liquid-board at 66e994fe49a59dfbaad0e2a719cd7507d732686c, with adaptation vertically and horizontally, full documentation, tests, saved models and same-data comparisons. Preserve both earlier experiments. Deliver to main without GitHub Actions.

Implemented: variable-width conductance LTC with recurrent split/prune and learned public readouts; liquid-stage insertion at every position and stage retirement; branching delta trees with per-node head scaling and forgetting; attention-routed child boards, shared-network forks and independent edge retirement; existing expert and encoding mutations; new input fields; global topology/work reservations; explicit delayed reward events; post-adoption monitoring and rollback; adaptive selection during scratch/fine-tune workflows; early stopping and snapshot/journal point-in-time recovery. See ARCHITECTURE.md, CONFIGURATION.md and OPERATIONS.md.

Resource review fixed two issues before final evidence: detaching one shared kernel now copies only that member and preserves other ties; cancelling a worker now terminates and reaps its child. Initial partial comparisons were stopped and excluded from delivery evidence. Final runs use frozen source hashes and isolated workers with address-space, CPU, wall and file limits.

Observed local verification:

- New experiment: `python -m unittest -v` passed 57 tests in 70.077 seconds.
- Previous adaptive experiment: 36 tests passed in 57.165 seconds.
- Original verified experiment: 20 tests passed in 0.884 seconds.
- Full H5/H6 causal replay completed for seeds 73, 83 and 97, using all twelve matching archives and the original trained checkpoints. Every baseline timestamp and truth matches. Previously published baseline predictions were reused after checking their source and artifact hashes; baseline learning was not repeated.
- Each final comparison passed independent source/artifact/archive checks, rescoring, trial-gate verification and full model restoration.
- Four controlled runs completed adoption and subsequent monitoring: layer growth, recurrent splitting, attention branching and expert retirement. Saved checkpoints and source/model hashes are independently checked by reporting/build.py.
- Twelve isolated timing runs compare old/new models for both houses and all seeds, using identical inputs, one CPU thread, one warmup and three measured passes.

Measured outcome: H5 later balanced accuracy matches the previous adaptive model for all seeds. H6 matches for seeds 73 and 83; seed 97 is 62.64% versus 63.08% previously. The preceding accepted update reduced checking quorum, which the new zero-loss default disallows. No house run accepted structural growth. The controlled new-cue streams establish mechanism behavior, not occupancy superiority. The new initial model has 6.38% more parameters and measured forward overhead. RESULTS.md records the full table and limitations.

Delivery checks passed: both earlier experiment folders and workflow files remain unchanged; Black checks all 19 Python files; all local documentation links and 104 manifest entries verify; staged whitespace checks pass; every artifact is below the GitHub file-size ceiling. The trend figure was visually inspected. The staged scope contains only this new experiment and two repository index links. GitHub Actions is disabled and is checked again before push. No GPU, distributed trainer, CI execution, external service or deployment is part of this delivery.

Limits: finite gates cannot guarantee every future accuracy window. Reward is contextual-bandit architecture credit, not a general reward-only agent. Direct imports do not install OS limits. Work units and tensor memory are estimates; worker.py supplies a hard process ceiling. Existing native eight-coordinate public interfaces, four-coordinate attention heads and checker protocol remain documented interfaces.

## Visual inspection update

Added EVOLUTION.md and reproducible, observed before/during/after views for six independent controlled runs. Four reproduce the published predictions, trial receipts and final parameter/buffer tensors exactly; two add the same-stream encoding-widening and child-board examples. Captures include original, pre-fit and post-fit shadow graphs, adoption, completed monitoring and final state. Root model sources/configuration and earlier evidence remain unchanged.

Added PNG/SVG topology panels, lifecycle capacity curves, all 51 completed house-proposal events, raw trace JSON and a self-contained HTML inspector. The inspector exposes serving/candidate/predecessor roles, stable identities, actual gates and resource counts. Existing accuracy graphs remain embedded in RESULTS.md; links now connect architecture, results and visual evolution. The intermediate exports are topology/gate records, not full intermediate runtime snapshots.

Observed verification: all six traces pass source/model hash checks, exact final checkpoint restoration, causal phase boundaries, checking decisions and recomputed monitoring metrics. Four corruption probes reject wrong chronology, altered labels, a wrong model digest and invented monitoring metrics. The local Chromium checker passes 116 cases: 108 run/phase/role views, six house-event lists and next/previous controls. PNG and browser views were visually inspected. The report builder verifies both the traces and browser artifact hashes, then refreshes the complete manifest. No model behavior changes or house retraining were needed. No Actions execution is authorized; confirm Actions remains disabled before pushing this update.

Final visual-update checks: all 22 Python files pass Black, all local Markdown links resolve, all 137 manifest entries verify, and whitespace checks pass. GitHub Actions remains disabled. The update is limited to documentation, reporting tools and derived visual/replay artifacts.
