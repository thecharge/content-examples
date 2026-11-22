# Intelligence liquidity

This experiment gives three groups of liquid specialists a shared memory. A checker can reject a draft, try another specialist, or send work back to the previous group. A live replay uses later sensor readings to adjust the model after it records its prediction. The heating experiment uses the same design with a different fit and a reward based on energy and comfort.

## Run it

Use Python 3.11 or newer. Create a virtual environment and install the tested CPU packages.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install xgboost==3.1.3
python -m unittest -v test_runtime
```

Put the eight files in the [data table](../liquid-blackboard/README.md) in one directory. Add the [Room Occupancy Estimation archive](https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip) as `independent-uci.zip`. The experiment verifies the file hashes.

```sh
python run.py fit data output --epochs 10
python compare.py data output
python run.py score data output
python heating.py data output
python latency.py data output/41-100.pt output/latency.json --cutoff .76
python inspect_rewind.py data output/41-100.pt output/rewind.json --cutoff .76
```

The occupancy fit uses seeds 41–43 and three amounts of training data. It keeps the checkpoint with the best validation ranking, and stops early after four epochs without improvement. Each `-resume.pt` retains the optimizer, latest parameters and progress. To continue an interrupted fit, rerun with the same options plus `--resume`. Graz, Munich, Porto Alegre and UCI Room Occupancy Estimation stay outside the fit and model selection.

## Results

The full model with live updates scored 56.43% in Graz, 24.18% in Munich, 50.05% in Porto Alegre and 46.82% on UCI Room Occupancy Estimation, giving both occupancy classes equal weight. XGBoost scored higher on all four. It uses the same sensor pair with features from recent readings. In Munich, the full model without updates withheld about 60% of the readings. With the checks switched off, the same saved model's equal-weight score went from 24.72% to 64.57%. The checks are part of the full model. This comparison measures their effect.

On one occupied Munich reading, the rewind brought the next sensor forecast closer to the actual reading. It also changed the answer from occupied to empty. The [paired rewind record](evidence/rewind.json) shows that a better sensor forecast can produce a worse occupancy decision.

The full model needed the full training budget to reach its best frozen score in Graz, Munich and UCI. The 25% budget scored highest among the three frozen budgets in Porto Alegre. None of the liquid runs established the smallest training budget needed to beat XGBoost across rooms.

In the heating replay, the model chose off on each of the 489 windows where it issued a choice and withheld 44. Its simulated choices tied an always-off policy. The observed logs and simulated action pairs are separate in the results.

See [all 84 comparisons](evidence/scores.json), [the tree baseline](evidence/xgboost.json), [the heating replay](evidence/heating.json), [CPU timing](evidence/latency.json) and [training resume verification](evidence/resume.json).

## Send readings to an inference process

Use the cutoff in the fit manifest for the selected checkpoint. Pass `--live` to learn after each new reading in a continuous sequence. The cutoff stays fixed. The process drops short-term memory after a gap longer than 450 seconds. Learned coefficients carry across gaps.

```sh
python serve.py evidence/41-100.pt --cutoff .76 --live --snapshot state.pt
```

Send one line at a time via standard input. Timestamps are increasing integer Unix seconds. A null occupancy answer means the required sensor check failed. The outputs include which specialist ran, any retry, and how long prediction and feedback took.

```jsonl
{"timestamp":300,"co2_ppm":510,"temperature_c":22.1}
{"timestamp":600,"co2_ppm":560,"temperature_c":22.2}
```

Restart with the same arguments and snapshot file to keep learned coefficients, the last reading and the shared memory. Keep the state file on trusted storage. A fresh checkpoint starts without updates from a prior run. To host a selected checkpoint, see [Model hosting](hosting.md).

## What changes as time passes

Every accepted stage carries its chosen specialist's state forward. Readings and elapsed time change how quickly each liquid cell moves. The next reading also grades the previous forecast during live occupancy replay. The program takes a clipped gradient step with a learning rate of 0.001. It can change the strength of sensor connections, the strength of feedback, the resting state and other learned settings. The equation and the number of experts stay fixed. Idle experts stay in memory and the sparse prediction path skips their computation. Updating the model considers alternatives, so timing that path separately matters.

The shared memory reads with a learned key, fades by channel and writes part of the difference between the incoming value and its prediction. Each stage learns a process score and has a separate check against current readings. A downstream objection can restore and rerun the preceding stage once. The single-stream inference path can rewind across stages. Training and gradient replay evaluate alternatives inside individual stages.

The occupancy label is held back until scoring. The heating path sees the recorded action and then learns from the observed energy and comfort. The historical log contains one action for each starting state. The second action in the paired comparison comes from a simulator. Its next-temperature error is measured on the observed actions. The simulation cannot establish how the actual building would have reacted to an untried action.

- [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439)
- [Kimi Linear and the gated delta memory](https://arxiv.org/abs/2510.26692)
- [Test-Time Training with Self-Supervision](https://proceedings.mlr.press/v119/sun20b.html)
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)
- [XGBoost](https://arxiv.org/abs/1603.02754)

## Same-reading CPU comparison

Use the same downloaded data folder and retained checkpoints as the full replay.

```sh
python compare_latency.py /path/to/downloads evidence /tmp/liquid-cpu-timing.json
```

Each model read the first 400 consecutive Munich Office_A rows in order. Three
fresh passes used one CPU thread, alternated model order and discarded the first
20 rows of each pass. This left 1,140 timed readings per model on the measured
Ryzen 7 PRO 8840HS. Timing includes input preparation, causal features or
state, checks and the live update where used. It excludes input/output,
checkpoint writes, loading, and training. The median is the middle reading.
The 95th percentile is the time within which 95% of readings finished.

| Model | Median | 95th percentile |
| --- | ---: | ---: |
| XGBoost | 0.14 ms | 0.16 ms |
| Frozen liquid MoE | 1.41 ms | 3.58 ms |
| Live liquid MoE | 8.12 ms | 10.59 ms |

The live path took about three times as long as the frozen path at the 95th
percentile in this run. Unused experts still occupy memory. This measurement
covers one CPU, room and checkpoint seed. It does not measure energy or memory.
The [timing receipt](evidence/cpu-timing.json) records the exact percentiles,
library versions, checkpoint hashes and sample counts.
