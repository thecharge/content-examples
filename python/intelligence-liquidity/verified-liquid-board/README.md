# A liquid board with self-selecting experts

The board estimates whether a house was occupied during a five-minute sensor
window. It fits and evaluates that same target. The separate heating experiments
in this repository use a different data source, target and reward.

## An expert reads the board and decides whether to act

The board posts all available sensor features. Every expert can read the same
fields: carbon dioxide equivalent, volatile organic compounds, temperature,
humidity and light, plus their changes and missing-value indicators. The parser
combines readings from the home's hubs into one value per field per window.
A missing reading stays marked missing. See the
[HPDmobile study](https://www.nature.com/articles/s41597-021-01055-x) and
[the original archive IDs](../liquid-workflow-v3/README.md#reproduce-the-fits).

Every expert has its own learned weights, participation policy and three
liquid time-constant layers. Its participation policy reads the shared
fields, its private state and feedback from the previous attempt. A low
score leaves its liquid computation idle. There is no central router, fixed
quota or sensor assignment. Different experts can act on the same reading
or decline it for different reasons.

An active expert can still decline to submit a vote. It adds a correction
at each layer. Training puts extra weight on cases the previous layer
struggled with, borrowing the sequential correction idea from boosting.
The comparison also fits XGBoost itself on the same fields and training
homes. [Liquid time-constant equations](https://arxiv.org/abs/2006.04439) ·
[XGBoost paper](https://arxiv.org/abs/1603.02754)

Separate classifiers challenge competing claims at each layer. They see
the same sensor fields and can make correlated errors. The board requires
a strict majority of all its experts, which is three with five experts.
If it lacks that majority, it posts the objections and tries again. After
four rounds it returns an unresolved answer and retains the last accepted
state. A majority saves the working state and updates the board's shared
delta memory. An inactive expert carries its own elapsed time to the next
call. [Gated delta memory](https://arxiv.org/abs/2412.06464)

## The next recorded answer grades the previous decision

The simultaneous votes can settle where to read and what to publish.
They cannot grade accuracy. The recorded occupancy answer for each decision
is revealed at the next observation in the labeled replay. Each submitted
vote receives credit or a penalty from that answer. A failed later attempt
costs more. Wrong answers receive no reward for agreement. No correctness
label for the current decision enters its prediction.

The update loop proposes candidate expert weights from earlier revealed
answers. It tests those weights against a later, separate block of revealed
answers. For each expert, an update must lower its log loss by at least 0.001 and preserve or improve classification
accuracy on that block. A block with too few examples of either class
freezes all weights. A candidate that fails freezes the affected expert.
The checker and delta memory parameters stay fixed during this experiment.
The board's memory state continues to evolve when it commits.

The live replay needs a source of delayed ground truth. The benchmark
has the recorded answer labels. It gives no evidence that a real deployment
could obtain labels after every window.

## Reproduce the fit

Use Python 3.12, PyTorch 2.7.1 CPU and XGBoost 3.1.3. Download the twelve
HPDmobile archives listed above into one directory.

```sh
python test_model.py -v
python benchmark.py /path/to/archives ./new-run --epochs 16
python check_replay.py /path/to/archives ./evidence
```

Homes 1 through 3 are for fitting and home 4 selects the checkpoint. Homes 5
and 6 score the frozen model and the separate replay with delayed answers.
Those homes were excluded from the fit and checkpoint selection. The delayed
replay can learn from earlier outcomes in each of those homes. The frozen
run measures accuracy with no such adaptation. Both runs start from the same
checkpoint independently for each home. The receipt records input hashes,
checkpoint hashes, each expert's participation, and the evidence used in
weight decisions.


## What happened on the replay homes

These scores give occupied and vacant windows equal weight. They include
the answers returned after an exhausted round.

| Model | Home 5 | Home 6 |
| --- | ---: | ---: |
| Checks off, frozen weights | 49.0% | 50.6% |
| Checks on, frozen weights | 49.0% | 50.6% |
| Checks on, delayed answers can justify updates | 48.8% | 50.6% |
| XGBoost trained on the same homes and fields | 48.6% | 60.6% |
| Always predict vacant | 50.0% | 50.0% |

The fit ran for five epochs. Home 4 chose the epoch 1 copy. On the frozen
replay, three votes agreed in 1,388 of 2,010 decisions on Home 5 and
98 of 2,016 on Home 6. Across 13,574 frozen rounds, there were rounds
with zero, one, two, three and four acting experts. None had all five.
Enabling checks left the scores and majority counts unchanged.

The delayed replay accepted six expert updates across four checks on
Home 5. It accepted none on Home 6. Too few examples of one class
blocked 50 of 62 update checks on Home 5 and 54 of 62 on Home 6.
The accepted updates on Home 5 improved the individual experts on
their check blocks, while the board score fell slightly across the
whole replay. On Home 5, the frozen board missed all 784 vacant windows.

The experiment demonstrates independent participation and conditional
updates. The scores are near the constant baseline. Home 6 also
shows a clear gap to XGBoost. The model needs better evidence of
occupancy before anybody relies on it.

[Replay receipt and round traces](evidence/results.json)
