# Mix short and long training sequences

The previous trainer shuffled within each sequence-length group. Every epoch ended with one batch of seven unoccupied sequences and two batches of one occupied sequence each. Nine of the 1,008 sequences accounted for three of 12 optimizer steps.

This run keeps the readings in time order inside each sequence. It pads shorter sequences with masked rows, shuffles all sequences together and splits them into eight equal batches. Each loss averages over valid expert readings. The 1,008 sequences and the counts of valid expert readings remain the same: 15,020 vacant and 45,155 occupied. The model, archive selection, no-offset calibration and early-stop patience match the preceding debug run. Falsification stays off.

## Reproduce

Run from the parent expert-decisions directory with its Python dependencies and [HPDmobile archives](../../README.md#run-it).

```sh
python debug-training/mixed-batches/train_mixed.py /path/to/archives /path/to/output/slow --lr .001
python debug-training/mixed-batches/train_mixed.py /path/to/archives /path/to/output/control --lr .008
```

The receipt keeps per-epoch training and validation losses, label mix and gradient norms per batch, the checkpoint and archive hashes, counts at both cutoffs, and example raw predictions. Each checkpoint contains weights, the calibration scales and the fitted cutoff.

## Results

| Batching and learning rate | Checkpoint epoch | Stop epoch | H4 validation loss, first → last | H5 balanced accuracy | H6 balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous groups, 0.001 | 1 | 13 | 0.6801 → 0.6813 | 52.9% | 53.3% |
| Mixed, 0.001 | 18 | 30 | 0.6792 → 0.6656 | 57.5% | 54.2% |
| Previous groups, 0.008 | 1 | 13 | 0.6767 → 0.6822 | 51.4% | 49.4% |
| Mixed, 0.008 | 3 | 15 | 0.6732 → 0.6560 | 57.0% | 54.2% |

These scores use each run's cutoff selected on H4. Balanced accuracy gives vacant and occupied windows equal weight. A withheld decision earns no point. Checkpoint selection uses the average expert ranking score on the first half of H4. The first epoch stopped being the best, and validation loss ended lower at both learning rates.

H6 still shows a decision-cutoff problem. At 0.008, the board score ranked occupied windows above vacant ones with an AUC of 0.725. With the 0.576 cutoff selected on H4, it caught 138 of 1,553 occupied windows and 461 of 463 vacant windows. With the fixed 0.50 reference cutoff, it caught 999 occupied and 317 vacant windows, or 66.4% balanced accuracy. H5 dropped from 57.0% to 52.1% at that same reference cutoff. Neither home contributed labels to fit the model or choose the cutoff. Both have been reused in earlier replays.

- [Mixed batches at 0.001](evidence/slow/mixed-results.json)
- [Mixed batches at 0.008](evidence/control/mixed-results.json)
- [Earlier rate-controlled debug run](../README.md)
