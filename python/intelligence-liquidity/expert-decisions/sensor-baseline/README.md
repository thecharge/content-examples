# Check the labels and train a tree baseline

The [HPDmobile dataset](https://www.nature.com/articles/s41597-021-01055-x)
records whether the whole home was occupied. We checked that each
five-minute window used the last recorded label in that window. All
18,129 matched, and the last label was always recorded within 10
seconds of the decision time. Fifteen labeled windows had no
environmental data and were skipped by the existing loader.

In 142 scored windows, occupancy changed inside the five-minute
window. The end label tells us whether anyone was home at the decision
time. A sensor may still reflect the earlier state, for example if
carbon dioxide remains high after someone leaves. The audit checks
how recorded labels are joined. It cannot tell whether the recorded
arrival and departure times match the actual events.

## Compare on the same home splits

The baseline has 100 shallow XGBoost trees. It sees the five
environmental readings and their changes since the previous reading.
Each prediction uses one feature row. The training set is all 12,087
available H1–H3 rows, with inverse-frequency class weights. The
[liquid comparison](../debug-training/mixed-batches/README.md) uses
the same source splits and feature columns, with overlapping training
sequences and separate views for its experts. This compares two
complete training approaches.

The first half of H4 selects the tree cutoff. The second half checks
whether that cutoff still works later in the same home. The cutoff
stays fixed at 0.8383 on H5 and H6, which are reused benchmark homes.

| Split | Vacant correct | Occupied correct | Tree balanced accuracy | Best prior mixed-batch liquid balanced accuracy | Tree score ranking (AUC) |
| --- | ---: | ---: | ---: | ---: | ---: |
| H4 first half | 134 / 239 | 591 / 769 | 66.5% | — | 0.726 |
| H4 second half | 182 / 272 | 409 / 736 | 61.2% | — | 0.618 |
| H5 | 547 / 784 | 494 / 1,226 | 55.0% | 57.5% | 0.530 |
| H6 | 455 / 463 | 339 / 1,553 | 60.1% | 54.2% | 0.799 |

Balanced accuracy gives the two classes equal weight. AUC measures
how often a randomly chosen occupied row ranks above a randomly
chosen vacant row, with half credit for ties. The trees answered
every scored window. On H6 the scores ranked the classes reasonably
well, but the H4 cutoff missed 1,214 occupied windows. On H5 the
ranking itself was close to chance, so a cutoff alone has limited
room to help. Of the 32 windows with a recorded change within a
bucket in H5 and H6, the trees got 18 right. Those few windows
cannot explain the low scores across the two streams.

## Reproduce

Use the [existing dependencies and datasets](../README.md#run-it)
from the expert-decisions directory.

```sh
python sensor-baseline/evaluate.py /path/to/hpd-archives ./sensor-baseline-run
```

The [results](evidence/results.json) contain per-home audit counts,
both cutoffs and SHA-256 digests of the public archives, source and
[fitted tree model](evidence/trees.json).
