# The full experiment and its boundaries

The intended model is a small liquid adaptive hypergraph with bounded growth.
It can share a part without copying its role into every expert.
It can also separate a shared part when independent learning earns the cost.
This file maps that vision to the implementation and the evidence to inspect.

| Requested behavior | Implementation | Inspection or check |
| --- | --- | --- |
| Inputs expand over time | Queued schema stage and bounded explicit derived features | Schema-growth regressions; feature/provenance tests |
| Broken, stale or missing sensors | Quality markers mask readings; original recorded value remains inspectable | Delayed-label and replacement test |
| Replacement keeps logical meaning and new origin | Stable field plus source incarnation, calibration and normalization versions | `provenance.Sensor`; event payload |
| Experts grow and retire | Existing learned policy, lifetime, gradient/error controls and future-label gates | Inherited actual growth/retirement tests |
| Experts form subexperts | Nested liquid boards with bounded depth and learned sensor groups | Child-board tests and graph layout |
| A subexpert belongs to several parents | Several edges point to one child board identity | Shared-child state and fork tests |
| Layers and recurrent neurons adapt | Existing guarded layer and neuron split/prune operations | Preservation, fitting and checkpoint tests |
| A neuron/layer belongs to several categories | Hyperedges name the same stable liquid vertex in several memberships | Shared-vertex test; full stage inspection JSON |
| Individual layers and tensors are shared | Exact layer/module aliasing and whole-kernel sharing | Partial-layer test; tensor `canonical` aliases |
| Shared parts can become independent | Detachment before structural edits and explicit child-network fork | Fork, alignment and layer-detachment tests |
| Attention grows, shrinks and forgets | Existing hierarchical multi-head delta memory and branch/head controls | Attention-tree tests and bounded message route test |
| Relations themselves evolve | Connect, split, merge and retire hyperedges | Policy masks, edge lifecycle and controlled accepted connection |
| Correlation informs the controller | Provenance-bearing paired prediction, disagreement, representation and context measurements | Calibration counterexample; duplicate-support test |
| Growth cannot exceed its envelope | Graph reservation, tensor/work ceilings and OS-limited worker | Allocation rejection and worker tests |
| Observed quality constrains changes | Independent fitting, prospective class/Brier/quorum checks, then monitoring | Recorded controlled and same-data receipts |
| Learning can use rewards | Delayed reward shapes action values without overriding external labels | Reward uniqueness, bounds and recovery tests |
| Retrain from scratch or fine-tune | Existing offline workflow with early stopping and saved runtime | Scratch/fine-tune regressions |
| Recover or branch at a recorded point | Exact committed snapshots, expected revisions and parent snapshot references | Recovery, branch CLI and stale-writer checks |
| Evidence has provenance | Observation/label IDs, device versions, model revisions, metric/config/code versions | Event payloads, trial evidence and session execution fingerprints |
| Bad evidence can be corrected | Append correction, invalidate dependent records and quarantine serving state | Correction and clean-restore test |
| Deduplicate without accidental parameter tying | Immutable dtype/shape/content-addressed tensor storage | Mutable-copy test |
| Retired parts can be collected safely | Reachability from serving, recent and pinned snapshots, including nested runtime roots | Pin/release and expired-recovery tests |
| Evolution is visible | Actual lifecycle snapshots, full neural inspection, SVGs, Mermaid and local HTML | `evidence/evolution` |

## Guarantees the model does not claim

Bounded capacity can be checked. Perfect future accuracy cannot.
The controller can refuse an unsupported merge and roll back observed deterioration.
It cannot guarantee that an unseen distribution will never reduce accuracy.

Correlation is evidence of association. It is not causal proof or permission to
merge unequal weights. Signed residual diversity is also not proof of relative
accuracy. The independent quality check retains that authority.

The current new-feature API defines explicit bounded transforms. It does not
autonomously invent semantic labels or arbitrary programs. External occupancy labels
and their authority remain separate from latent context groups.

Reward learning is a contextual bandit. Multi-step policy gradients and environment
control are outside this implementation. A reward cannot replace a missing label.

The computational hyperedges execute inside a board context. Shared child boards
connect board contexts through the existing hierarchy. This preserves finite
execution; it is not arbitrary recursive evaluation of a cyclic global graph.

Disk bounds imply a finite retained recovery horizon. A pin can preserve an old
payload, but cannot create more storage. Cross-ledger invalidation is an operator
responsibility, and changed source/framework versions require explicit migration.

The available environment executes the model on CPU. AMD GPU execution needs an
accessible compute device, a compatible ROCm build and a verified tensor-device port.
The hardware investigation is recorded in [GPU.md](GPU.md).
