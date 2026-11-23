# What the occupancy audit changed

The audit found a board that almost always answered “occupied.” Its checkers
rarely intervened and the tests passed if the checker was removed. Its update
gate graded individual experts on different decision logic from the actual
board. The old receipt remains in [evidence](evidence/results.json).

| Gap | Repair and remaining question |
| --- | --- |
| A constant predictor looked competitive | Report ordinary and balanced accuracy and the actual counts for each class, alongside both constant answers and a matched XGBoost fit. |
| Removing the checker or speaking rule passed the suite | Test that opposing evidence can block a quorum, challenged state rolls back even if three other experts agree, and an active expert can decline to speak. Removing either rule now fails its test. Real benchmark outcomes with checks on and off are reported together. |
| The update gate graded a small batch of first-round experts | Grade one candidate against the unchanged board over 288 later observations. It has to improve both balanced accuracy and probability error. Record rejected candidates and single-class blocks. The test suite exercises the gradient path and rejection by the real board gate. |
| A fallback counted like consensus | Record quorum and exhaustion separately. The accuracy denominator still includes exhausted decisions. |
| Different act rates were treated as specialisation | Record each expert's first-round class counts when assessed, acting, speaking and counted. A different participation rate alone is insufficient evidence of specialisation. |
| Imports depended on execution order and the receipt missed shared code | Load the shared modules from explicit paths. Hash the parser, liquid recurrence and shared memory code, fit and replay scripts, and CPU dependency file. Pin the direct packages. The replay checks categorical fields exactly and permits small variation in floats. Transitive packages are not locked. |
| A few windows could change the ranking | Fit three recorded seeds. Give within-home uncertainty ranges. These homes were examined previously and still count as only two independent buildings. |
| Training and replay carried state for different lengths | Raise training segments from 12 to 36 windows with gradient truncation every 12. Test that blocks and derived history reset on gaps. Long uninterrupted replay still carries state further than training. |
| The window label and data exclusions were unclear | Define the target as the last valid recorded label in a five-minute bucket and test the parser. Inspect the selected archives for discarded days, invalid labels and timestamp order. |
| Vote strength was called confidence | Report a board probability using both the strength and direction of the experts' votes. Select its scale on Home 4 and score its errors on Homes 5 and 6. |
| Thresholds lacked checks | Include Home 4 trials for vote offset, checker threshold, speaking threshold and probability scale. Compare the frozen board with checks disabled in every replay. Run one fit with the added history and clock features disabled. The exploration schedule, retry penalties and loss weights still need separate comparisons. |

The replay commands start from saved weights and regenerate the predictions,
counts, update decisions and final weights. The benchmark commands train
from the published archive inputs. Neither uses a live building-control
system. The old receipt was also replayed from its pinned source revision.
