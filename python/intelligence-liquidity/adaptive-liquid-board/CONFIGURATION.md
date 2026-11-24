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
| `maximum_class_loss` | `0.02` | Maximum permitted recall loss for either class in a growth trial. |
| `maximum_quorum_loss` | `0.02` | Maximum permitted fraction-of-rows quorum loss in a growth trial. |
| `policy_rate` | `0.02` | Action-value optimizer learning rate. |
| `policy_updates` | `8` | Action-value optimizer steps per completed trial, using retained replay. |
| `group_rate` | `0.02` | Input-group head optimizer learning rate. |
| `group_updates` | `12` | Input-group optimizer steps at each stage boundary. |
| `group_fraction` | `0.5` | Fraction of eligible external fields selected for a child, rounded up. |
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
| `warmup_steps` | `4` | Optimizer updates over which the fitting learning rate reaches its peak. |

## Limits

| Key | Live default | Meaning |
| --- | ---: | --- |
| `max_experts` | `32` | Maximum unique experts across all boards. |
| `max_depth` | `3` | Maximum number of child levels below the root. |
| `max_fields` | `64` | Maximum registered external numeric input names. |
| `max_encoding` | `32` | Maximum encoding width in a board; liquid state width remains eight. |
| `min_heads` | `1` | Minimum delta-attention heads allowed in each board. |
| `max_heads` | `8` | Maximum delta-attention heads allowed in each board. |
| `policy_capacity` | `64` | Maximum observations, actions and returns retained by each policy replay buffer. |

## Reference constants and learned values

The inherited model starts with five experts, three liquid layers per expert, width-eight private state and two delta heads. It retains the original checker calculations, retry protocol, conductance solver and contextual optimizer. Those define the reference model being extended. They are documented in [ARCHITECTURE.md](ARCHITECTURE.md) and the original source.

Delta heads retain four coordinates each and the original elapsed-time forgetting equation. Encoding width, expert count, hierarchy and head count can change within the supplied limits. Parameter values, contextual correction, participation, input groups, attention readouts and action values learn from data.

The original paired bootstrap keeps its reference block length and resampling settings. The feedback controller retains its documented bounded-integral update and failure-threshold rule. These are algorithm definitions, while the table exposes experiment schedules, operating bounds and quality targets.

See [OPERATIONS.md](OPERATIONS.md) for loading configurations and changing training or recovery modes.
