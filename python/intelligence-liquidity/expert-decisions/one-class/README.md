# Can one kind of room reading teach the model both?

We trained five liquid experts twice. One fit saw only readings taken
when a home was vacant. The other saw only readings taken when it was
occupied. Each expert learned to forecast the next five-minute reading
from its own sensor view. At least three experts needed a usable
forecast. Their median forecast error was the score. An error above
the cutoff meant the model predicted the other class.

H1–H3 supplied 3,036 vacant rows for the first fit and 9,051 occupied
rows for the second. We discarded changes that crossed a class boundary
in those rows. Both runs used the same learning rate of 0.003 and
selected epoch 24, the last allowed epoch. The first half of H4 selected
each checkpoint. The second half set each cutoff at the 90th percentile
of the forecast errors for its known class. Calibration also used only
the known class. H5 and H6 were held out of fitting and calibration and
are reused benchmark homes. Replay carried state across consecutive
readings and used the labels only to score its decisions.

## Held-out results

| Training | H5 vacant correct | H5 occupied correct | H5 balanced accuracy | H6 vacant correct | H6 occupied correct | H6 balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vacant only | 773 / 784 | 47 / 1,226 | 51.2% | 453 / 463 | 41 / 1,553 | 50.2% |
| Occupied only | 27 / 784 | 1,158 / 1,226 | 48.9% | 9 / 463 | 1,493 / 1,553 | 49.0% |
| Both classes | 645 / 784 | 388 / 1,226 | 57.0% | 461 / 463 | 138 / 1,553 | 54.2% |

Balanced accuracy averages the two class-specific percentages so an
always-occupied or always-vacant answer scores 50%. Both one-class runs
withheld six H5 decisions and one H6 decision. The table counts them as
misses. The [both-class comparison](../debug-training/mixed-batches/README.md)
used the same windows and the cutoff selected on H4. It trained at
0.008 to predict occupancy directly, so this comparison changes the
training objective as well as which rows the model sees.

Both one-class models mostly predicted the class they had seen in
training. The score did a poor job of ranking occupancy even before the
cutoff on H6. Its area under the ranking curve was 0.504 for vacant-only
and 0.509 for occupied-only, with 0.5 representing chance ranking. In
this experiment, how hard a sensor reading was to forecast was a poor
signal for whether the home was occupied.

## Reproduce

Use the [dependencies and HPDmobile archives](../README.md#run-it). Run
from the parent expert-decisions directory.

```sh
python one-class/replay.py /path/to/archives /path/to/output/vacant 0
python one-class/replay.py /path/to/archives /path/to/output/occupied 1
```

The receipts include the epoch curves, frozen cutoffs, results and
SHA-256 digests. Each checkpoint contains the trained weights and
the settings needed to score a stream.

- [Vacant-only receipt and checkpoint](evidence/vacant/)
- [Occupied-only receipt and checkpoint](evidence/occupied/)
