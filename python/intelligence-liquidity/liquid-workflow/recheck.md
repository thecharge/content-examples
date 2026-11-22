# Recheck the occupancy results

## Which model is which?

The [original LTC paper](https://cdn.aaai.org/ojs/16936/16936-13-20430-1-2-20210518.pdf)
reports 94.63% ordinary accuracy for its occupancy benchmark, in Table 3.
Its [released runner](https://github.com/raminmh/liquid_time_constant_networks/blob/4a8a4fa22a91c4067b6944a264819ff30a4aae9d/experiments_with_ltcs/occupancy.py)
uses temperature, humidity, light, CO₂ and humidity ratio, with the original
UCI training and two test files. Those are different inputs and splits from
this cross-building test. The 94.63% is a published result.

The three-stage workflow in [model.py](model.py) uses a custom liquid cell.
Its gate depends on the incoming reading and previous state. It takes one
solver step over each interval. It adds expert routing, a learned reviewer,
a separate reconstruction threshold, a retry and shared memory. The
[shared memory](model.py#L38) uses a gated error write inspired by
[Kimi Delta Attention](https://arxiv.org/abs/2510.26692). The workflow does
not run Kimi's token attention or its hybrid model.

The new [reference comparison](compare_reference.py) uses the upstream
[ncps PyTorch LTCCell](https://github.com/mlech26l/ncps) from the paper
authors' companion repository, pinned to release 1.0.1. It uses a single
fully connected eight-unit cell with six solver substeps per reading and a
linear occupancy output. It trains on the same 48,304 labeled rows used
by the custom full-data fit, reads the same two sensors and splits, and
tests the same untouched sites. It uses three seeds, a ten-epoch cap
and the same validation-based stopping and cutoff selection. Its weights are
frozen during testing. These choices give a plain upstream-cell comparison
under this protocol. They do not reproduce the paper's five-sensor experiment
or isolate every effect of the custom architecture.

## Recompute the charts

From this directory, with the environment and data listed in the
[run instructions](README.md):

```sh
python -m pip install ncps==1.0.1
python compare_reference.py /path/to/data /path/to/new-reference-run
python audit_scores.py
```

The [reference receipt](evidence/reference-ltc.json) records validation curves,
each selected epoch and cutoff, data and model hashes, and per-site counts.
The [chart receipt](evidence/correctness.json) comes from independently
recomputing both accuracy measures from 108 saved count rows. It also records
the percentage of occupied readings caught and the percentage withheld.

For a checkpoint-level check of Munich, copy `evidence` to a new writable
directory, then point the scorer at the copy. It will verify all checkpoint
hashes and write a new score file.

```sh
cp -a evidence /path/to/recheck
python run.py score /path/to/data /path/to/recheck --sites munich --variants frozen no-check
```

The full model's occupancy cutoff was selected on Aalto using batched
predictions without cross-stage rewinds. The fixed reconstruction check then
governed abstentions in single-reading testing. Its threshold was never tuned
for correctness or coverage by room. The early check is therefore a suspect
when the room changes. In Munich, the full frozen workflow withheld 59.73%
of readings and caught 2.42% of occupied readings. Turning the checks off
with the same saved weights raised the equal-weight score from 24.72% to
64.57%. That change also turns off the refusal and repair path. It measures
how much the result depends on that policy.

Use a separate validation set to select the check and compare correctness
at similar answer rates before concluding how much the policy helps.
Treat these four sites as recorded results once they have informed a design
change. A new claim about unfamiliar rooms needs fresh test data.
