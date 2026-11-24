# A liquid board that can change its own structure

This is a separate extension of the [verified liquid board](../verified-liquid-board/README.md) at commit [`4977126`](https://github.com/thecharge/content-examples/tree/4977126369c64c7a42df2977446e442a5d404121/python/intelligence-liquidity/verified-liquid-board), the reference supplied for this experiment. The original implementation and evidence stay intact. The extension starts with its five experts, three liquid layers per expert, three checkers, four retry rounds and two-head delta memory. Before adaptation, it produces exactly the same predictions and recurrent states.

The extension accepts additional named numeric inputs, learns which groups matter to an expert's errors, and gives every expert a small neural policy. That policy proposes keeping the structure and learning, splitting an expert into peers, creating a child board, splitting an encoding neuron, retiring an expert, sharing matching children, or growing and pruning delta-attention heads. A proposal changes a shadow copy first. Later measured answers decide whether it becomes the running network. Failed proposals teach the policy a negative return.

The [live article](https://sndv.xyz/en/blog/intelligence-liquidity) describes both a first-week evaluation and a later evaluation after 14 selected days. This folder compares those ranges separately, using the same archives, parser, feature calculations and original checkpoints. Working structural adaptation does not establish an accuracy gain. The original contextual learner is also rerun, and its later scores must match its published receipts. See [measured results](RESULTS.md) for the outcomes and limits.

The [architecture guide](ARCHITECTURE.md) contains Mermaid diagrams, equations, the reinforcement credit loop, a component map and recovery details. The original contextual correction learner remains active alongside structural learning.

## Run and inspect

See the [configuration reference](CONFIGURATION.md) for every setting and [operations](OPERATIONS.md) for configuration, model inspection, journals, point-in-time recovery, scratch training and snapshot fine-tuning.

Use Python 3.12 and the pinned CPU dependencies. From this folder:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python ../verified-liquid-board/test_model.py
python ../verified-liquid-board/test_online.py
python ../verified-liquid-board/test_replay.py
python compare.py /path/to/hpdmobile-archives ./new-run-73 --seed 73
python verify.py ./new-run-73 --archives /path/to/hpdmobile-archives
```

A separate controlled stream demonstrates actual end-to-end retention of a new child after a previously unknown input becomes predictive:

```sh
python demonstrate.py evidence/new-input-seed-73.json --seed 73
python demonstrate.py evidence/new-input-seed-83.json --seed 83
python demonstrate.py evidence/new-input-seed-97.json --seed 97
```

It uses artificial labels and is explicitly separate from house accuracy. With these fixed seeds, exploration selects child creation (73), a gradient-directed encoding-neuron split (83), and retirement (97). Each learns the added input and passes quality conditions on 48 subsequent predictions. The next stage demonstrates the cooldown. Actions and checking predictions come from the running models; neither is forced by the demonstration.

Repeat the occupancy comparison with seeds 83 and 97. To replay the complete recorded computation as well as checking the evidence:

```sh
python verify.py evidence/seed-73 --archives /path/to/hpdmobile-archives --rerun
```

The [original download instructions](../liquid-workflow-v3/README.md#reproduce-the-fits) identify all 12 HPDmobile archives. Nothing downloads or schedules a job automatically. The original H1–H3 fits and H4 checkpoint selection are reused, rather than trained again. All 12 archive hashes must match the original receipt before scoring. No GitHub Actions are used.

## Supply inputs and delayed answers

```python
import torch
from graph import Graph
from reference import Board, REFERENCE
from runtime import Runtime

base = Board(features=17)
base.load_state_dict(torch.load(
    REFERENCE / "evidence/seed-73/board.pt", weights_only=True))
network = Runtime(Graph(base), seed=73)

issued = network.predict(300., {
    "temp_c": 0.2,       # Use the reference's normalization for its named fields.
    "sound_level": 0.7,  # Additional numeric evidence can arrive at any time.
    "window_open": 1.,
    "scheduled_event": None,
})
# A separate source supplies the actual answer after the prediction.
network.observe(issued["token"], truth=1, revealed_at=600.)
network.save("running-board.pt")
network = Runtime.load("running-board.pt")
```

The original names represent normalized values, changes, recent-history offsets and circular clock coordinates; they are listed in `reference.FIELDS`. Additional names may describe sensors or other context. The caller defines their units, scaling and causal calculation. The network does not infer units or read unprovided information. Values are finite numbers, `None` or NaN for missing evidence. `parent:` is reserved for internal context. Names remain stable even when dictionary order changes. Omitting a field marks it missing; it does not erase learned weights. Infinite values, magnitudes above the configured input bound (10,000 by default), and excess resource use are rejected before prediction. Scale new evidence deliberately.

The runtime queues new names until a stage boundary with no unresolved predictions or existing candidate. Shapes remain fixed during a fitting pass and its prospective check. At the boundary, registering a name extends every relevant native projection, including value and missingness columns, the shared-memory query, the three checkers, and group heads. Existing columns retain their meaning. New columns start at zero and new external fields remain unsupported in the serving model, so even a previously empty observation cannot gain a vote from an untrained field. Candidate fitting enables those fields in its isolated copy and can learn their weights from delayed outcomes. The new input becomes usable in the serving model only on candidate adoption. Gradient probes enable unsupported fields only in a probe copy. Inputs do not need to have existed when the checkpoint was fitted.

Each prediction receives a token. Reveal tokens once, in order, at a time after their prediction and no earlier than the latest event. `discard(token)` explicitly expires an unavailable answer. Predictions do not wait for a label, but the pending-label budget prevents unbounded storage. `reset_state()` starts fresh recurrent states after a stream break; it retains weights, learned policies and the ongoing prospective trial. A deployment must decide when two observations belong to different streams. The benchmark resets on every missing five-minute window.

## What decides structure

Every expert owns a `Policy` with a neural action-value head and a feature-group head. The action input combines its eight private liquid-state coordinates with measured error, measured credit, age, newly registered fields, whether it has a child, depth, activity and confidence. The head estimates returns for `learn`, `spawn`, `deepen`, `retire`, `share`, `split`, `grow_heads` and `prune_heads`.

The runtime samples which expert may submit a proposal from its masked neural utilities. That expert chooses its action by its learned values, with 35% exploration by default. Unavailable actions are masked: no children beyond the depth limit, no growth beyond the expert limit, no second child on an occupied parent edge, no retirement of the last root, and no sharing without matching weights. There is no cycle of predetermined operations. Exploration remains possible after unsuccessful proposals. Choosing an operation and accepting it are separate decisions.

A peer split copies the proposing expert's liquid weights, projections, readouts, policy, recurrent state and complete child context into a new stable identity. Its voting mass divides equally between the donor and the new peer. The total voting mass and initial function stay the same. The board requires strictly more than half of total mass, uses mass-weighted confidence and memory contributions, and preserves the native checks and retries. With the original five unit masses this is the original three-of-five rule. Independent learning rates allow the peers to diverge during fitting. Retirement removes a selected peer and any branch with no remaining parent; the remaining total mass determines the new strict majority.

A neuron split duplicates one selected SiLU encoding unit. It copies that unit's incoming weights and bias, halves its outgoing rows and gives the other half to the new unit. Liquid state is unchanged. The other experts receive a zero extra encoding unit. Thus widening initially preserves the represented function. The encoding can grow from 16 to at most 32 units by default; each expert still has three width-eight liquid state layers. This implements a function-preserving encoding split, not an untested change to the LTC solver's state dimension. Expert splits duplicate the full liquid dynamics as well.

Gradient probes on earlier labels determine which encoding unit has the greatest loading in the leading right singular vector of its incoming-weight gradient matrix. The probe accumulates at most eight rows at a time. This is a gradient-informed location heuristic inspired by [GradMax](https://arxiv.org/abs/2201.05125); it is not a reproduction of that paper's new-neuron optimization. The duplicate-and-divide transformation follows the function-preservation principle in [Net2Net](https://arxiv.org/abs/1511.05641). Its application here is to the SiLU encoding and weighted expert protocol, not to a Transformer.

The group head learns the absolute correlation between each available input and that expert's prediction residual, using only revealed fitting history. This is a simple relevance signal, not proof of causation. Its top half of physical/context fields forms a child's view. A `learn` proposal can also regroup existing children. Every field remains available to the root. A later field can therefore enter an existing child's group after its relevance is learned.

## An expert can become a smaller biased blackboard

`Core` subclasses the original `Board`; it does not replace its decision procedure. Creating a child copies the selected expert into a one-expert board and adds eight input coordinates for its parent's current private state. A child can create peers and its own children through the same policies. It has its own three liquid layers per expert, independent checkers, voting threshold, retries, private states and adaptive delta memory, initially copied from its parent. The zero initial gain makes a new branch function-preserving for its parent; training can open that connection. A one-expert board initially needs that expert's one surviving vote; after peer growth it needs the corresponding strict majority. Allowing fewer than five experts is an intentional extension of the original constructor's fixed minimum. The checker and commitment rules still run.

A child sees the selected group and its parent's hidden context. Other inputs are masked. Parent context is also masked when no selected input is available or the parent is idle, so latent state cannot manufacture new sensor evidence. On a successful child quorum, a learned tanh gain, initialized to zero, scales the child's log odds and adds them to that parent's final liquid-layer margin. The parent's normal checker can still challenge that amended claim. Its normal quorum can still fail.

Retries begin from the same previously accepted state, including children. Child state is kept only where its parent acted, survived its checks and belonged to a successful parent round. A failed parent round rolls back child cells and shared memory. Per-row float64 timestamps preserve elapsed time for children that were skipped, including real Unix dates where float32 clocks would round away seconds. State is addressed by the complete parent path, so two parents can use the same child weights without sharing their running context.

## Share liquid weights without mixing memories

The `share` operation supports two cases:

- Entire compatible child boards have equal parameters and equal descendant structure. References are redirected to one board. Existing states are remapped to the canonical board under their original parent paths.
- Children have equal liquid-layer weights but different readouts, checkers or policies. Their liquid modules become the same parameter objects while those other modules remain separate. A shared kernel therefore accumulates gradients from both uses during later fitting.

Complete-board candidates are found by SHA-256 and then checked with exact tensor equality. Liquid-only sharing also requires exact tensor equality, including tensor shapes. Approximate similarity is insufficient. Different expert counts cannot share an entire liquid array. Resizing a tied board first detaches the affected component; other users keep their parameters and expert axes. Checkpoints preserve ties and separate states. A share proposal is not refitted before its prospective check, so a storage-only change can be judged as such.

## Learn, test, retain or retire

`Runtime` separates predictions from answers. It collects at most 576 earlier revealed examples. Both classes need at least 16 examples before fitting. It considers a stage after 288 new answers; after fitting, it runs the active and shadow graphs on the same future observations and saves both predictions before their answers arrive. At least 288 subsequent answers grade the proposal. The original contextual correction learner keeps running and applies the same accepted correction to both graphs. Unlike the original retrospective candidate replay, these candidate predictions were actually issued prospectively. Fit labels must precede every check prediction.

Candidate weight fitting uses the original `offline_loss`, including liquid stage errors, independent falsifiers, participation and measured rewards. It trains all visited boards with 12-window truncated gradients, resets at data gaps, clips the gradient norm to one, and uses one pass of fresh AdamW. New weights warm up to learning rate 0.002 over four optimizer steps. Existing weights use one tenth of that rate. Mixed old/new tensor rows use a corresponding scaled update, including weight decay, rather than relying on gradient scaling that Adam could cancel. New field columns and new child connections are treated as new weights. Optimizer momentum is deliberately restarted at each boundary; parameter identities and shapes are checked before every update. Dormant children have no visited-board loss. These are fixed settings for this comparison, not values selected for the two test homes. Expert weights stay fixed while a candidate is being checked. The contextual readout continues to learn. A chronological validation tail from earlier fitting history controls early stopping and restoration of the best fitting checkpoint. It never uses the future check block.

Learning or growth must pass the original quality requirements: at least one percentage point of balanced-accuracy gain, at least 0.002 balanced Brier improvement, positive lower end of the paired block-bootstrap gain, and no loss over two points for either class recall or quorum. The complexity-adjusted return must also be positive. It is balanced Brier improvement plus 0.002 times the fraction of parameters saved, so growth pays a small cost.

Retirement, sharing and attention-head pruning must strictly save parameters, with no measured deterioration in Brier error, either class recall, or quorum on the later block. This lets an unused redundant expert retire without requiring an artificial accuracy gain. No age threshold alone kills an expert. Age and decayed external credit inform the neural proposal; independent outcomes decide whether the removal survives.

Successful proposals teach action values a positive measured return of at least 0.01. Failed or unsupported proposals teach a negative return of at least 0.01 in magnitude. Each policy retains a bounded replay of observations, actions and measured returns. That replay trains the selected action values. The same measured experience trains all current policy heads, including policies in the candidate. This shares lifecycle experience; each policy still evaluates its own current context. A failed candidate is discarded with its inference weights and state. A successful candidate becomes active with the state accumulated during its prospective test. Every completed trial records the selected expert, action, neural value, feature grouping or mutation, fitting range, checking range, paired predictions, quality gains and decision.

## Guarded feedback control

[`configuration.json`](configuration.json) supplies live learning, quality gates, policy update settings, graph limits and controller guards. Use `compare.py --configuration path.json` to replace them. `--guards path.json` optionally replaces only the controller section. [`training-configuration.json`](training-configuration.json) supplies offline fitting defaults. Every comparison records its actual settings.

At a stage boundary, the controller calculates balanced Brier error on revealed history, compares its earlier and recent halves, and updates a bounded integral: `I = clip(0.9 I + 0.1 (error - target), 0, 1)`. The default target is 0.10, with a 0.02 deadband. Error plus this integral can latch growth on when improvement has plateaued; error below target minus the deadband clears the latch. A drift above 0.02 or newly registered fields can also request a capacity probe. A fresh gradient above `1e-7` is still required for that expert's growth action.

Growth is therefore data-driven and can only execute at discrete stages. A two-stage cooldown prevents repeated structural attempts from churning the model. Rejected growth raises the next error threshold, up to four extra deadbands. Retirement needs at least 288 measured observations of the expert by default, plus either low utilization or errors above target. Neural policy values select among operations that pass these controls; later external outcomes still decide adoption. A pressure signal is permission to test capacity, not permission to accept it.

Before allocating a schema expansion or candidate, the controller checks a conservative parameter ceiling. Before fitting it checks the exact candidate count, with defaults of 200,000 parameters, 256 context-specific expert uses and an estimated 128 MiB tensor-training budget. The estimate includes active weights, candidate weights/gradients/Adam state, mixed-row update copies and padded activation work, including repeated contexts for shared children. Probe microbatches are bounded separately. This is an allocation estimate for this small CPU experiment, not a measurement of total Python/PyTorch process RSS. Receipts expose it for inspection.

Expert, child, attention-head and encoding-neuron growth must also pass a pre-training preservation replay: answers and commitment decisions must match, and probability differences must stay within `1e-6`. Separate sequence tests check recurrent memory and child contexts. Preservation is assessed in inference mode; fitting retains the original stochastic participation policy. Each stage writes a boundary checkpoint, makes at most one mutation, rebuilds AdamW, warms up new weights, fits at fixed shape, then checks on later predictions. Retirement and regrouping are intentionally tested changes, not function-preserving growth claims.

The experts' recurrent dynamics remain liquid throughout. Delta memory, independent checkers, neural growth heads and projections remain distinct learned components around those liquid experts. This work is the CPU reference experiment; it introduces no distributed trainer.

## Liquid dynamics, attention and forgetting

The native conductance-based `LiquidODE` is imported unchanged. `AdaptiveDelta` retains the original delta and forgetting equations while allowing a variable head count. Every expert retains three width-eight liquid layers. At each solver substep, state follows the original drive divided by leak-plus-conductance update. A quiet expert carries elapsed time into its next active call.

Every board starts with two delta-attention heads and a 4-by-4 memory table per head. It can split or prune heads within configured bounds, defaulting to one through eight. A learned readout combines all heads into the same eight coordinates for the experts. Splitting copies donor memory and divides its readout, preserving the initial function. Pruning proposes the least readout mass and needs an independent no-harm check. For each head, `past = memory * exp(-softplus(forget) * minutes / 15)`. The new memory is `past + key ⊗ rate * (target - keyᵀ past)`, with normalized keys and a learned rate. Thus the update writes a prediction error and forgets old content according to elapsed time. Only a successful quorum writes it. Lifecycle statistics use a configured exponential retention factor, defaulting to 0.95. Fitting history and pending labels are bounded.

## Resource limits and recovery

Defaults allow 32 unique experts across all boards, three child levels below the root, 64 external input fields, 32 encoding neurons per expert, one through eight attention heads per board, 64 pending labels and 128 retained trial receipts. They are explicit constructor/settings arguments, not learned accuracy claims. Shared calls can use different context paths, so the number of live states can exceed the unique parameter count. Retired-expert statistics and obsolete birth mappings are removed. These limits bound in-memory adaptation; increasing them needs a caller decision and more compute.

`Runtime.save` atomically replaces a versioned checkpoint. It contains the named schema, graph, stable identities, full and liquid-only sharing, policy weights, active and shadow states, pending labels, fitting history, partial check block, receipts, decayed statistics and random-generator states. `Runtime.load` uses `weights_only=True`, restores both models in inference mode and resumes pending work. A durable boundary checkpoint is written before every candidate mutation. Pass `stage_directory` to choose its location; otherwise a temporary directory is used. The runtime also retains a configured number of numbered boundary snapshots, defaulting to two. Restoring a boundary checkpoint completes that same stage before the next prediction, without repeating controller assessment or group fitting. Other checkpoints restore the candidate and pending test directly. The provided runtime and training code target a single ordered CPU stream. The graph's native operations and masked child commits support batches; this is not a distributed stream service or an optimizer-state resume midway through a fitting pass.

## Requirement-to-evidence map

| Requested behavior | Implementation | Behavioral evidence in `test_adaptive.py` |
| --- | --- | --- |
| Preserve original liquid board | `Core(Board)`, native imports | `test_exact_native_sequence_including_missing_and_retries` |
| Additional inputs over time | `Graph.add_fields`, `Core.expand`, candidate fitting | `test_schema_expansion_preserves_every_native_projection`, `test_new_field_receives_gradient_and_changes_fitted_weights`, `test_new_input_causes_learned_child_growth_and_prospective_acceptance` |
| Network chooses peer growth/removal | `Policy.action`, `Runtime.choose`, prospective gate | `test_neural_action_values_learn_external_payoff`, `test_runtime_neural_policy_drives_peer_and_depth_candidates`, objective lifecycle tests |
| Neuron and expert splits with guarded stages | `Core.split_neuron`, weighted expert splits, `Controller`, boundary recovery | `test_growth_preserves_recurrent_function_before_fitting`, controller guard tests, exact boundary resume, actual prospective neuron-split test |
| Dynamic nested biased boards | `Graph.deepen`, `Core.stages`, parent-context inputs | `test_nested_depth_and_resource_limits`, `test_child_is_full_board_and_parent_rollback_is_atomic` |
| Group inputs by learned relevance | `Policy.group`, `Runtime.teach_groups`, regroup proposals | `test_feature_group_is_learned_from_errors` |
| Share matching liquid weights | `Graph.share_equal`, path-specific states | `test_identical_children_share_parameters_but_keep_context_state`, `test_equal_liquid_weights_share_even_when_readouts_differ`, unequal-weight refusal |
| Objective expert lifecycle | `Runtime.finish`, recorded external answers | `test_objective_lifecycle_retires_only_without_measured_harm`, strict gain and insufficient evidence tests |
| Dynamic multi-head delta and forgetting | `AdaptiveDelta`, `Graph.change_heads`, learned actions | `test_attention_split_preserves_sequence_and_prunes_to_minimum`, `test_neural_policy_proposes_attention_growth_and_pruning`, native forgetting and rollback tests |
| Preserve contextual adaptation | Original `ContextFeedback` through whole-graph replay | `test_contextual_continuation_matches_native_delayed_feedback`, original adaptive rerun in `compare.py` |
| Early stopping and retraining | Chronological validation, best-weight restoration, training CLI | `test_early_stop_restores_best_earlier_validation_weights`, `test_scratch_and_finetune_write_reusable_snapshot_receipts` |
| Point-in-time recovery | Durable event journal and fingerprinted snapshots | `test_journal_point_in_time_restores_pending_and_rejects_tampering` |
| Reinforcement credit and replay | Neural contextual bandit with delayed measured returns | `test_reinforcement_replay_is_bounded_and_restored`, learned external-payoff test |
| Safe causal continuation | Token ordering, versioned checkpoint | `test_delayed_labels_order_and_exact_pending_checkpoint_resume` |
| Same data accuracy comparison | `compare.py`, original archive/checkpoint fingerprints | `verify.py`, raw H5/H6 predictions and three-seed receipts |

Controlled tests establish whether a mechanism works under known conditions. The occupancy evidence establishes how often the learned policies proposed and retained those changes on this stream. Neither establishes that every real-world problem needs more experts, deeper boards, or a topology change at all.
