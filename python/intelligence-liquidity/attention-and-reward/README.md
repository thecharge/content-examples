# Temporal attention and logged rewards

This experiment adds a single temporal attention pool to a conductance-based
liquid time-constant cell. A variant learns to predict the next sensor reading
alongside occupancy. In the heating comparison, the networks learn one-step
temperature, heat and comfort outcomes from recorded commands. At inference,
the learned outcomes feed a fixed reward formula for comparing commands.

## Reproduce the measurements

Use Python 3.12 on a CPU. Download the three source files and keep the archive
filenames for these commands. Each loader checks SHA-256 against `data.py`.

- `office.zip` comes from [UCI Occupancy Detection](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip), CC BY 4.0.
- `apartments.csv` is [Dataset.csv on Zenodo](https://zenodo.org/records/10761326/files/Dataset.csv?download=1), CC BY 4.0.
- `six-buildings.zip` comes from [Mendeley versions and files](https://data.mendeley.com/datasets/xztfbtsgys/3), CC BY 4.0. Use [this version 3 archive](https://data.mendeley.com/public-files/datasets/xztfbtsgys/files/00524e03-d7db-4399-8d65-7615e3afe024/file_downloaded).

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py check
python benchmark.py freeze
python benchmark.py occupancy office.zip --danish apartments.csv --seed 41
python benchmark.py heat six-buildings.zip --seed 41 --summary
python audit.py replay heat-41.json attention six-buildings.zip --compressed
```

Repeat each benchmark for seeds 42 and 43. Each benchmark prints one JSON record
per model and budget. The checked-in JSON files group those records. The heating
row archives are gzip-compressed and base64-encoded. `audit.py replay` compares
aggregate scores and traces against the selected checkpoint receipt before
printing the row archive. Use `audit.py membership office.zip` to list the
exact endpoints permitted for fitting and next-reading targets.

`freeze.json` records the initial code hashes. The first occupancy invocation
stopped during Danish timestamp parsing before it emitted scores. The only edit
to frozen inputs before the retry was in `data.py` for the publisher's day-month-year
timestamp. `freeze-parser-repair.json` records the corrected hash. The original
receipts for the heating runs carry the first hash, because they did not use the
Danish timestamp reader.

## What the tests found

Mean balanced occupancy accuracy across three seeds on 10,656 external
apartment windows. This metric gives occupied and empty windows equal weight.
The UCI office tests, confusion counts, per-apartment counts, fitting windows,
update counts, validation losses and selected weights are in the receipts.

| Network | 25% fitting pool | 50% | 100% |
| --- | ---: | ---: | ---: |
| Latest readings | 57.79% | 55.96% | 54.77% |
| GRU | 58.94% | 59.79% | 58.84% |
| Library LTC | 50.00% | 60.73% | 55.79% |
| Custom LTC | 50.00% | 50.00% | 61.80% |
| Custom LTC with attention | 50.00% | 50.00% | 58.72% |
| Custom LTC with attention and next-reading prediction | 50.00% | 50.00% | 58.05% |

On 532 later heating intervals, all three models ranked the off command first
380 times and withheld a command 152 times. Both commands need at least eight
fitting examples in the same state bucket before ranking. The mean lookup had
mean absolute reward error 0.0311 on the command that actually ran. The custom
liquid cell had 0.2383 and temporal attention had 0.2674 across seeds. The reward
subtracts thermal energy in kWh and twice the comfort deviation in degree-hours.
The log gives one measured result per interval. It cannot tell us the energy or
comfort effect of the command that did not run.

See [the fixed protocol](protocol.md) for the complete split, feature mapping,
decision threshold, selection procedure and limits. This comparison is a
reference for later architectures with learned stage checks and adaptive replay.
