# A liquid board with self-selecting experts

Can five small models read the same sensors, disagree, and improve when the
right answer arrives? This example tests that idea on five-minute windows
from the [HPDmobile study](https://www.nature.com/articles/s41597-021-01055-x).
The task is to tell whether somebody was home at the end of a window.
Each answer is either vacant or occupied.

## How the board makes a decision

Five experts read the same sensor fields. Each has its own weights,
participation decision and three [liquid time-constant layers](https://arxiv.org/abs/2006.04439).
Each layer adds to the expert's proposed answer. During training, later layers
pay more attention to mistakes made earlier. An expert can stay idle, work
without voting, or submit a vote. The board has no central router.

The board reads its shared [delta memory](https://arxiv.org/abs/2412.06464)
on every round. It compares each layer's answer with a separate checker.
Two strong disagreements can remove an expert's vote if the expert's
own reliability score is low. The checker reads the same sensors as the
experts and can make the same mistake.

Three of five matching votes settle the question. If that fails, the board
shares the objection and tries again, up to four rounds. If it runs out of
rounds, it returns its best remaining answer and keeps the last accepted state.
When a round succeeds, a challenged expert's private state rolls back.
The shared memory uses the states of experts whose votes backed the decision.
A quiet expert carries elapsed time into its next call.

How certain the board sounds depends on how many experts back the answer and
how strongly they disagree. Five strong votes for the same answer contribute
more than a split vote. The reported probability is an estimate scored against
the recorded answers. The score used for an individual expert to speak has a
floor of 0.5 and should be read as vote strength.

## When the board learns

The replay reveals the recorded answer for a window when the next observation
arrives. A correct vote earns credit. A wrong vote costs more on later rounds.
Agreement earns a bonus only when the recorded answer confirms it.

The learner uses earlier revealed labels to propose a weight change for one
expert. Before keeping it, it replays the current board and the changed board
over the same later block of 288 already-revealed answers. The check runs
the actual voting, checking and retry logic. It looks at each block once.
At least 16 answers of each class are needed in both the fitting history
and the check block. The board keeps a change only when all of these hold.

- Average recall across the two classes improves by at least one percentage point.
- The lower end of the paired accuracy improvement estimate is above zero.
- The two classes' average probability error drops by at least 0.002.
- Neither class recall nor the share of results settled by quorum falls by more than two points.

This is supervised learning after a delay, using a reward to weight learning
from mistakes. The experiment has no human preference scores. An installation
would need to supply real delayed occupancy answers to use the same update loop.
A replay can finish with every expert's weights unchanged.

## Data and evaluation

The 12 original archives and their download IDs are listed in the
[data instructions](../liquid-workflow-v3/README.md#reproduce-the-fits).
The parser skips files with fewer than 8,640 ground-truth rows and ignores
nonbinary labels. The last valid label in each five-minute window is the
answer. The parser uses the latest sensor reading from each hub and combines
hubs per field. Its five fields are carbon dioxide equivalent, volatile
organic compounds, temperature, relative humidity and light. The features
include changes since the prior window, a difference from recent readings
and time of day as encoded by the archive parser. Missing values remain marked
missing. The recent-history features reset at gaps. An archive spot check
across all 63 selected days found no short days skipped, nonbinary labels,
out-of-order ground-truth timestamps or duplicate timestamps in those days.
That check does not validate label correctness.

Homes 1 through 3 supply 14 days each for fitting. Home 4 supplies seven
days for checkpoint selection and policy checks. Homes 5 and 6 each supply
seven days for replay. The frozen weights have never fitted these homes in
each new run. Those homes have been inspected in earlier experiments, so
a new result on them cannot establish performance in a new building.

Training advances recurrent state over 36 consecutive windows and limits
gradient history to 12 at a time. The code drops the short tail of a segment
and starts a new block with a new state. Replay can retain state across many
more consecutive windows, with a reset when a window is missing. How much
that difference matters remains open.

Plain accuracy asks how often the answer is right. Balanced accuracy
first scores vacant and occupied windows separately, then averages the two
scores. Always answering occupied can look good in a house that is usually
occupied, but earns 50% balanced accuracy. The Brier score averages squared
probability error with equal weight for each class. Lower is better.
The 90% ranges resample blocks of 24 windows from each home’s ordered
replay, with wraparound at the end. They describe uncertainty within that week. The two replay homes are too few to
rank close models or estimate performance across buildings.

## Reproduce the fit and replay

Use Python 3.12 and install the pinned CPU packages in an isolated environment.
Download all 12 archives into the same directory.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -v
python benchmark.py /path/to/archives ./new-run --epochs 8 --seed 73
python check_replay.py /path/to/archives ./new-run
```

Repeat with seeds 83 and 97 to compare fits. Use
`--seed 73 --no-history` to test the effect of removing the clock and
recent-history features together. The XGBoost baseline in each run sees
the same features and fitting homes as the board. Model and tree
checkpoints, archive and source hashes, counts and sample decisions are
in each receipt. The verifier checks file hashes, class counts and discrete
decisions exactly. It allows small differences in recorded floating point
scores and parameters across CPU builds.

The [original receipt](evidence/results.json) reports the older model checked
in the audit. Use commit `0453f9a` to replay it with its original code.
On Home 5 it got zero of 784 vacant windows right. Its balanced accuracy
was 49.0% and 50.6% on Homes 5 and 6. Its plain accuracy was 59.8% and
75.4%, below always occupied at 61.0% and 77.0%. The [earlier averaging experiment](../full-liquid-board/README.md)
reports 57.5% and 54.2% on those homes under an earlier recipe. Treat those
numbers as history. For a matched comparison use the baselines in each new run.

## Results on the two replay homes

| Fit | Home 5 balanced | Home 5 plain | Home 6 balanced | Home 6 plain |
| --- | ---: | ---: | ---: | ---: |
| [Board, seed 73](evidence/seed-73/results.json) | 64.1% | 63.2% | 74.3% | 61.8% |
| [Board, seed 83](evidence/seed-83/results.json) | 59.1% | 61.4% | 74.7% | 64.7% |
| [Board, seed 97](evidence/seed-97/results.json) | 66.2% | 65.6% | 74.2% | 61.3% |
| [Board without added history or clock, seed 73](evidence/no-history-seed-73/results.json) | 44.7% | 52.0% | 67.2% | 56.7% |
| XGBoost with the same added features | 56.2% | 63.2% | 77.5% | 75.2% |
| XGBoost without added history or clock | 48.6% | 58.3% | 60.6% | 77.9% |
| Always occupied | 50.0% | 61.0% | 50.0% | 77.0% |
| Always vacant | 50.0% | 39.0% | 50.0% | 23.0% |

Two of three seeds clear 60% balanced accuracy on both homes. All three
beat the matched tree on Home 5 and lose to it on Home 6. On Home 6, always
answering occupied still beats every liquid board in plain accuracy because
77% of the windows are occupied. The no-history fit shows how much the
conclusion can change when the added features are removed together.

For example, seed 97 got 543 of 784 vacant windows and 775 of 1,226
occupied windows right in Home 5. Its within-home balanced accuracy range
was 59.5% to 73.3%. In Home 6 it got 455 of 463 vacant windows and 780 of
1,553 occupied windows right, with a range of 69.4% to 79.2%.

## What changed when the board checked, voted and learned

| Seed | Home 5 majority / 2,010 | Home 6 majority / 2,016 | Home 5 votes blocked | Home 6 votes blocked |
| --- | ---: | ---: | ---: | ---: |
| 73 | 448 | 340 | 7 | 0 |
| 83 | 526 | 362 | 16 | 0 |
| 97 | 1,958 | 1,976 | 8 | 0 |

The checker blocked 31 votes in Home 5 across three seeds and none in
Home 6. The checked seed 73 answered one additional window incorrectly
on Home 5. For seed 83, checking lowered the Home 5 majority count by four
with the same class totals. The results show the checker can intervene.
They give no evidence that it improves accuracy. Expert participation
and quorum still vary widely across seeds. Seed 97 had a majority most of
the time, while the other two seeds relied mainly on fallback answers.

The delayed update gate kept zero candidates. Across the 36 completed
feedback blocks in the three main runs, 12 had too few examples of a class
or too little earlier history. The other 24 failed the whole-board gain
rule. The largest balanced gain on any candidate check block was 3.45
points. Its probability error rose by 0.00083, so the gate rejected it.
There was no measured improvement from online weight changes.

See [the audit and verification record](AUDIT.md) for the changes and
remaining limits. An independent test in more buildings is still needed. Establishing an
adaptive gain also requires a reliable source of delayed labels.
