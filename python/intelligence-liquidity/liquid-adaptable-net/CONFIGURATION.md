# Configuration reference

Both configuration files are complete JSON objects with `learning`, `guards` and `limits` sections. Defaults below come from `configuration.json`. `training-configuration.json` changes `fit_passes` to 20 for scratch training and fine-tuning. Every experiment records its actual values.

These values bound and guide adaptation. They are caller choices, not learned claims of universal performance. Learned action values decide among operations allowed by these guards. Independent outcomes decide adoption.

## Learning

| Key | Live default | Meaning |
| --- | ---: | --- |
| `window` | `288` | Revealed observations per live stage and per prospective check block. |
| `history` | `576` | Maximum recent fitting rows; also the contextual queue capacity per class. |
| `minority` | `16` | Minimum observations of each class required to fit or grade a trial. |
| `exploration` | `0.35` | Probability of sampling an allowed action instead of its highest learned value. |
| `fit_rate` | `0.002` | Peak learning rate for newly added trainable weights. |
| `fit_passes` | `1` | Maximum fitting passes. Live default is one; offline training default is 20. |
| `complexity_rate` | `0.002` | Reward coefficient multiplying the fraction of parameters saved. |
| `maximum_pending` | `64` | Maximum predictions awaiting an external answer. |
| `maximum_receipts` | `128` | Maximum retained structural and contextual audit entries. |
| `contextual` | `true` | Run the original contextual margin-correction learner. |
| `full_weight_learning` | `true` | Enable candidate fitting and architecture stages; false leaves contextual-only learning. |
| `early_stop_patience` | `2` | Stop fitting after this many passes without enough validation improvement. |
| `early_stop_min_delta` | `0.0001` | Minimum balanced Brier reduction that counts as validation improvement. |
| `validation_fraction` | `0.2` | Chronological tail of earlier fitting history reserved for validation. |
| `gradient_window` | `12` | Observations per truncated recurrent-gradient update. |
| `gradient_clip` | `1.0` | Maximum global gradient norm during weight fitting. |
| `statistic_retention` | `0.95` | Previous-statistic weight in expert credit, error and activity averages. |
| `checkpoint_retention` | `2` | Numbered mutation-boundary snapshots retained, in addition to boundary.pt. |
| `minimum_accuracy_gain` | `0.01` | Required balanced-accuracy gain for weight learning and growth; 0.01 means one percentage point. |
| `minimum_brier_gain` | `0.002` | Required balanced Brier reduction for weight learning and growth. |
| `maximum_class_loss` | `0.0` | Maximum permitted recall loss for either class in a growth trial. |
| `maximum_quorum_loss` | `0.0` | Maximum permitted fraction-of-rows quorum loss in a growth trial. |
| `policy_rate` | `0.02` | Action-value optimizer learning rate. |
| `policy_updates` | `8` | Action-value optimizer steps per completed trial, using retained replay. |
| `group_rate` | `0.02` | Input-group head optimizer learning rate. |
| `group_updates` | `12` | Input-group optimizer steps at each stage boundary. |
| `group_fraction` | `0.5` | Fraction of eligible external fields selected for a child, rounded up. |
| `compute_rate` | `0.002` | Reward coefficient multiplying the fraction of graph work units saved. |
| `external_reward_scale` | `0.1` | Weight of mean explicit environment reward in structural action credit; never overrides the quality gate. |
| `minimum_policy_reward` | `0.01` | Smallest magnitude assigned to accepted or rejected trial returns. |

## Guards

| Key | Live default | Meaning |
| --- | ---: | --- |
| `max_abs_input` | `10000.0` | Largest accepted absolute numeric input. Caller scaling remains explicit. |
| `target_brier` | `0.1` | Target balanced Brier error used by the feedback controller. |
| `error_deadband` | `0.02` | Hysteresis width around the target; rejected growth raises the entry threshold. |
| `drift_threshold` | `0.02` | Recent-minus-earlier balanced Brier increase that can request a growth probe. |
| `plateau_tolerance` | `0.002` | Maximum earlier-minus-recent improvement treated as a plateau. |
| `minimum_gradient` | `1e-07` | Minimum encoding-weight gradient norm needed to permit growth for that expert. |
| `minimum_life` | `288` | Measured expert observations required before retirement or head-pruning proposals. |
| `cooldown_stages` | `2` | Minimum controller stages between structural attempts, including rejected attempts. |
| `max_parameters` | `200000` | Maximum candidate trainable parameter count. |
| `max_training_bytes` | `134217728` | Maximum estimated tensor-training allocation, in bytes; not total process RSS. |
| `probe_rows` | `96` | Maximum earlier observations used in a gradient probe. |
| `probe_batch` | `8` | Maximum rows per probe microbatch. |
| `max_context_experts` | `256` | Maximum expert uses counted across candidate context paths, including shared calls. |
| `old_gradient_scale` | `0.1` | Existing-weight update rate relative to the new-weight rate, applied after Adam. |
| `max_batch` | `8` | Largest permitted probe batch ceiling; the graph also enforces its own batch bound. |
| `replay_states` | `4096` | State slots reserved by the allocation estimate; runtime rejects `3*history + 2*window + maximum_pending` above this count. |
| `warmup_steps` | `4` | Optimizer updates over which the fitting learning rate reaches its peak. |

## Limits

| Key | Live default | Meaning |
| --- | ---: | --- |
| `max_experts` | `32` | Maximum unique experts across all boards. |
| `max_depth` | `3` | Maximum number of child levels below the root. |
| `max_fields` | `64` | Maximum registered external numeric input names. |
| `max_encoding` | `32` | Maximum encoding width in a board; recurrent liquid width has separate limits. |
| `min_heads` | `1` | Minimum delta-attention heads in each attention-tree node. |
| `max_heads` | `8` | Maximum delta-attention heads in each attention-tree node. |
| `min_experts` | `1` | Minimum root experts; optional child networks can be retired entirely. |
| `max_boards` | `16` | Maximum unique liquid boards; the root is always retained. |
| `min_layers` | `3` | Minimum liquid stages per expert, retaining the native multi-checker protocol. |
| `max_layers` | `8` | Maximum liquid stages per expert in each board. |
| `min_liquid` | `4` | Minimum recurrent conductance neurons per stage. |
| `max_liquid` | `16` | Maximum recurrent conductance neurons per stage. |
| `min_encoding` | `8` | Minimum encoding neurons per expert. |
| `max_attention_nodes` | `16` | Maximum attention-tree nodes across all boards, including roots. |
| `max_attention_depth` | `3` | Maximum child-attention levels below each attention root. |
| `max_total_heads` | `64` | Maximum heads across all attention trees. |
| `max_work` | `1000000` | Maximum deterministic work proxy per graph prediction, counting nested calls and retries. This is not measured FLOPs or wall time. |
| `max_batch` | `8` | Maximum batch rows in a direct graph call; larger batches must be divided. |
| `route_floor` | `0.05` | Inference skips a child-network call below this non-root attention route. Training retains differentiable routing. |
| `policy_capacity` | `64` | Maximum observations, actions and returns retained by each policy replay buffer. |

## Hard worker envelope

`worker-configuration.json` is separate from graph configuration. `worker.py --configuration path.json` applies it before model imports. The default is one CPU thread for Torch, OpenMP, MKL and OpenBLAS.

| Key | Default | Meaning |
| --- | ---: | --- |
| `memory_mib` | `4096` | Hard worker address-space ceiling in MiB, including libraries and allocator mappings. It is not an RSS target. |
| `cpu_seconds` | `7200` | Hard total CPU-time ceiling for one worker run. |
| `wall_seconds` | `7500` | Parent-enforced elapsed-time timeout for one worker run. |
| `file_mib` | `256` | Maximum size of an individual file written by the worker. |

Core dumps are disabled in the worker. Timeout and cancellation terminate and reap the child process. These limits affect that worker, not other applications. A limit can cause a run to fail; resume from its last durable checkpoint. They do not guarantee successful fitting in an undersized envelope.

`max_training_bytes` reserves estimated tensor work; `memory_mib` is the OS-enforced process limit. `max_work` is a model complexity proxy; `cpu_seconds` and `wall_seconds` bound the actual run. Shared parameters count once in parameter limits, but their repeated context calls count in work. Global limits can reject growth before a local maximum is reached.

## Reference constants and learned values

The inherited model starts with five experts, three liquid layers per expert, width-eight private state and two delta heads. It retains the original checker calculations, retry protocol, conductance equations and contextual optimizer. New layer gates and recurrent readouts initially preserve the reference function. Those define the reference model being extended. They are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and the original source.

Delta heads retain four coordinates each and the original elapsed-time forgetting equation. Encoding and recurrent widths, liquid-stage count, expert count, network hierarchy, attention hierarchy and head count can change within the supplied limits. The eight-coordinate public interface and four-coordinate attention heads remain fixed interfaces. Parameter values, contextual correction, participation, input groups, attention readouts and action values learn from data.

The original paired bootstrap keeps its reference block length and resampling settings. The feedback controller retains its documented bounded-integral update and failure-threshold rule. These are algorithm definitions, while the table exposes experiment schedules, operating bounds and quality targets.

See [OPERATIONS.md](OPERATIONS.md) for loading configurations and changing training or recovery modes.

For a new graph, bounds must contain the starting reference: five experts, three layers, eight recurrent neurons, sixteen encoding neurons and two root attention heads. Fine-tuning applies the supplied limits to the loaded topology and rejects an incompatible envelope before copying it. Ordinary resume preserves the snapshot settings.

The same `window` grades both a prospective trial and its subsequent monitoring block. A class-unsupported monitoring block rolls back. The previous graph is retained for this one block, not forever. The contextual learner retains the native reference acceptance protocol; the zero-loss defaults above govern the new structural and weight-candidate checks.
