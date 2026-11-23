# A liquid blackboard that answers after a bounded number of rounds

A sensor window opens a question. Five experts examine it in parallel.
The expert count is configurable, with a minimum of five. Each expert
gets a view of the readings and its own liquid state. The occupancy
views cover air, temperature and humidity, light, recent changes, or
a mix. The views overlap. Agreement is useful evidence and can still
reflect a mistake shared by the experts.

## One decision

1. The evidence stage uses a liquid time-constant cell to predict
   complementary readings. A checker compares those predictions with
   readings outside the expert's view.
2. The judgment stage uses another liquid cell to make a prediction
   and estimate confidence. It can pass. A learned willingness to try
   changes whether an expert speaks. The reported confidence stays the
   same. With no readings in its view, the expert always passes.
3. The second checker can challenge the answer using other readings.
   The learned process judge evaluates the candidate and its evidence.
   The tree variant supplies the independent answer from XGBoost in
   place of the neural counter. Both checks run before a claim counts.
4. The answer with the most surviving votes leads. Confidence breaks
   equal vote counts, then the previous accepted answer breaks an exact
   tie. Board confidence increases as more confident experts support
   the same answer. Three surviving votes are required with five experts.
5. If the board falls short, the experts start another round from the
   last accepted state with the objections and current split. This run
   allows four rounds. On exhaustion, the last round still supplies an
   answer. The trace marks it as unresolved, and the board retains the
   accepted expert state and memory. With no usable claim and no prior
   decision, the starting answer is vacant for occupancy and off for
   heat. The trace keeps every attempted round in selected examples.
6. A majority commits, including an update to delta attention. The
   memory fades old entries and learns a correction from the committed
   state. Both liquid stages use learned conductances inside their
   differential equations. Readings change the hidden state while
   later learning can change the coefficients.

## How the replay learns

A correct claim earns 0.90, with an extra 0.15 for a clean round with
a majority. A wrong claim starts at a cost of 1.10, multiplied by 1.25
for every round already used. Confidence that proves misplaced costs
additional points. A small learned process score contributes alongside
the verified outcome. For occupancy training the label verifies the
claim. For heating, only the actual recorded action supplies reward
feedback about that action.

An online update happens after the new decision has been emitted.
It compares the previous forecast against the sensor readings that have
now arrived. The occupancy replay uses only those delayed readings to
adapt. Its occupancy labels are used for scoring. The separate heating
checkpoint also learns from the logged pump and valve action and observed
reward using the discounted estimated next return. No reward is supplied
for commands the logged controller did not execute.

## Run and verify

Use Python 3.12, PyTorch 2.7.1 CPU and XGBoost 3.1.3. The
[HPDmobile archive IDs](../liquid-workflow-v3/README.md#reproduce-the-fits)
and [ZEB Living Lab download](../building-control/README.md#replay)
are in the earlier readmes. The archives are available from their
publishers. The receipts store archive and checkpoint hashes.

```sh
python test_board.py -v
python run_board.py occupancy /path/to/hpd-archives ./out/occupancy --epochs 16
python run_board.py heat /path/to/six-buildings.zip ./out/heat --epochs 16
python check_replay.py /path/to/hpd-archives ./evidence/occupancy
python check_replay.py /path/to/six-buildings.zip ./evidence/heat
```

The occupancy fit uses 14 labeled days from H1 through H3 and selects
a checkpoint on seven days of H4. H5 and H6 are excluded from fitting
and checkpoint selection and reused for comparisons with earlier runs.
Each review mode starts with the same expert weights. The live runs
start fresh from those weights for each home.

## Held-out occupancy results

Balanced accuracy gives vacant and occupied readings equal weight. An
answer is scored even if no majority forms before the limit. This makes
the fallback visible in the results.

| Decision policy | H5 | H6 | Majority reached on H5 / H6 |
| --- | ---: | ---: | ---: |
| Falsification off, weights fixed | 54.3% | 50.0% | 1 / 0 |
| Learned falsifier, weights fixed | 53.5% | 50.0% | 0 / 0 |
| Learned falsifier, learns from later readings | 53.5% | 50.0% | 0 / 0 |
| XGBoost falsifier, weights fixed | 53.5% | 50.0% | 0 / 0 |
| XGBoost falsifier, learns from later readings | 53.5% | 50.0% | 0 / 0 |
| Earlier five liquid experts with vote averaging and no falsifier | 57.5% | 54.2% | Vote averaging |
| Standalone XGBoost predicting occupancy | 55.0% | 60.1% | Single prediction |
| Always give the same answer | 50.0% | 50.0% | Single prediction |

The earlier rows use the same scored windows with independently fitted
models. See their [liquid results](../expert-decisions/debug-training/mixed-batches/README.md)
and [tree results](../expert-decisions/sensor-baseline/README.md).

The two falsifiers changed which claims survived. Neither policy reached
a majority on either occupancy replay at this checkpoint. The checks
took their full four rounds and returned a fallback answer everywhere.
On H6 each returned vacant on 2,015 of 2,016 windows. It got all 463
vacant windows right and one of the 1,553 occupied windows right.
Learning from later readings lowered forecast error without changing
the occupancy scores. On the first 500 H4 validation windows a separate
frozen diagnostic found zero to two speakers in every final round,
including with falsification off. This shows a speaking and agreement
problem in this fit. The stage checks alone cannot explain it.

## Held-out heating replay

The [logged heating data](../building-control/README.md) records the
chosen pump and valve commands, temperature and energy. The board
used 533 intervals and returned 533 answers. The learned checker
with online updates reached a majority in 518 intervals. It made 492
updates from the next sensor reading and 492 reward updates using
recorded commands and outcomes. The mean error against the one-step
reward plus discounted next estimate fell from 0.519 to 0.492. This
quantity also depends on the model's next estimate. The model
recommended heat in every held-out interval in both runs. These logs
cannot show energy savings or comfort for a new control policy.

The complete [occupancy receipt](evidence/occupancy/results.json) and
[heating receipt](evidence/heat/results.json) include per-class or
logged-action counts, fit curves, and selected full round traces.
