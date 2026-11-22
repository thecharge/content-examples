# Debug the five liquid experts

This copy keeps the original expert architecture and sensor inputs. It trains new weights with the calibration intercept fixed to zero. It fits a positive scale per expert on the second half of H4 and picks the final board cutoff on the first half of H4. H5 and H6 only score the result. The replay keeps the original rules for speaking and passing and leaves falsification off so the comparison isolates training and cutoff changes.

Use the [same twelve public archives](../../liquid-workflow-v3/README.md#reproduce-the-fits) and the parent example's Python dependencies.

```sh
python debug-training/train_debug.py /path/to/hpd-archives /path/to/output/slow --lr .001
python debug-training/train_debug.py /path/to/hpd-archives /path/to/output/control --lr .008
```

Both runs use the same seed, training windows, binary loss and 12-epoch early-stop patience. The training command saves the best weights, the scales and the decision cutoff. The receipt includes every epoch's losses, gradients and parameter movement from the tiny-batch test, five raw prediction rows per replay, and archive and artifact hashes.

## What happened during training

| Learning rate | Selected epoch | Stopped after | Training loss, first → last | H4 validation loss, first → last | H4 expert ranking score, first → last |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 1 | 13 | 0.70818 → 0.70799 | 0.68009 → 0.68129 | 0.655 → 0.538 |
| 0.008 | 1 | 13 | 0.70974 → 0.70393 | 0.67674 → 0.68219 | 0.666 → 0.526 |

The training loss moved at both rates. After epoch one, the validation loss increased and the scores became worse at sorting occupied windows above vacant ones. Reducing the rate slowed the deterioration. It didn't make a later checkpoint win.

On the five consecutive training rows, the true labels were occupied, occupied, occupied, vacant, vacant. Every probability began at 0.5 because the readout starts at zero. After 100 steps, 0.001 got 3/5 right and lowered loss from 0.693 to 0.669. The 0.02 control got 5/5 right in every expert and lowered loss to 0.182 on exactly the same rows. The receipts also show that gradients reach the liquid cell after the first readout update. The 0.02 result measures memorization of five rows.

On the first H6 row, the slower model's five raw logits were -0.00786, -0.02260, -0.00598, -0.00224 and -0.02610. They map to probabilities of 0.4980, 0.4943, 0.4985, 0.4994 and 0.4935 before the fitted scales. The true label was occupied.

## Held-out results

Balanced accuracy gives occupied and vacant windows equal weight. A withheld call counts as a miss.

| Decision policy | H5 balanced accuracy | H6 balanced accuracy | H6 occupied caught | H6 vacant caught |
| --- | ---: | ---: | ---: | ---: |
| Original five-expert run | 50.0% | 50.0% | 1,553 / 1,553 | 0 / 463 |
| Offset-free, learning rate 0.008 | 51.4% | 49.4% | 335 / 1,553 | 358 / 463 |
| Offset-free, learning rate 0.001 | 52.9% | 53.3% | 140 / 1,553 | 452 / 463 |

In H6, the slow board averaged 0.49730 on vacant readings and 0.49750 on occupied readings. The score distributions overlap and the transferred cutoff of 0.49806 misses most occupied windows. Lowering the rate helped the equal-weight score slightly but both test homes remain close to 50%. The two new checkpoints are for debugging.

- [Slow run receipt](evidence/slow/debug-results.json)
- [Matched rate-control receipt](evidence/control/debug-results.json)
- [Original replay receipt](../evidence/results.json)
