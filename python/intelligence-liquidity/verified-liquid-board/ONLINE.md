# Can the board improve after it answers?

The original online learner trained one expert on first-round predictions. Its
gate asked whether the resulting change improved the full board, including
membership, abstentions, vetoes, retries and memory. That objective mismatch
made the proposed changes a poor fit for the gate.

Two follow-up replays keep the same gate and start from saved weights. Both
use actual later occupancy labels.

1. **One shared bias.** Try adding a small constant to each expert's final
   voting score. Choose among four shifts by replaying the entire board on
   up to 576 previously revealed observations. This can correct an overall
   tendency to predict one class. It can also trade one class's mistakes for
   the other's. Score days 8 through 14 in each replay home.
2. **A correction that depends on the readings.** Fit a regularised linear
   correction from current sensor features and missing-value indicators.
   The training target is a recorded occupancy answer, and the starting
   estimate is the current board's probability. Test three strengths on the
   earlier answers through the full board. Then take one candidate through
   the same approval gate. Score usable rows on days 15 through 21, a
   different range from the first follow-up. The parser yielded 288 rows
   in Home 5 and 2,016 in Home 6 in that range. The effect on a proposed vote is bounded to 0.8 in
   either direction. Inner liquid equation parameters remain fixed.

The learner consumes the previous answer before it predicts the next
observation. It can reuse past checking blocks for later fitting, after
their outcomes become available. Each candidate only sees one approval
block. Frozen and adaptive boards start with identical weights and see
identical inputs. The frozen board still uses its liquid state and memory.
Only the adaptive board can change voting parameters.

Acceptance requires a one-point balanced accuracy gain, a positive lower
paired estimate for that gain, a 0.002 reduction in class-balanced squared
probability error, and no decline larger than two points in either class's
recall or in the rate of majority decisions. A block with fewer than 16
observations of either class is too small for a decision. The public
receipt includes all completed checks and both replay scores. The interval
in the receipt resamples adjacent windows from the same home.

These replays evaluate weeks within the same two buildings. A seed change
offers a different fit and keeps the same buildings. The two follow-up
methods use different weeks, so comparing their absolute scores would
confound the methods with the changed data. Compare each adaptive board
against its frozen copy on the same week. This is supervised learning with
delayed labels. The experiment includes no reward for a building-control
action.

```sh
python -m unittest discover
python online.py /path/to/archives evidence/seed-73/board.pt /tmp/bias-73.json
python contextual.py /path/to/archives evidence/seed-73/board.pt /tmp/context-73.json
```

Repeat with seeds 83 and 97. Input archives and pinned CPU packages are
described in [the main README](README.md).

The exact code for the archived first follow-up is at revision `656390d`.
The first replays kept zero changes across all three seeds and both homes.
A single direction of adjustment could raise the combined accuracy while
causing an unacceptable increase in errors for a class.

## Sensor-dependent replay results

| Fit | Home 5 frozen | Home 5 adaptive | Changes | Home 6 frozen | Home 6 adaptive | Changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 73 | 73.7% | 82.6% | 1 | 63.4% | 62.5% | 1 |
| Seed 83 | 73.0% | 87.5% | 1 | 60.5% | 64.6% | 1 |
| Seed 97 | 79.0% | 89.9% | 2 | 62.7% | 62.6% | 1 |

Home 5 had 12 vacant observations in the scored slice, too few for the
paired interval. All three fits improved on this one day of usable rows.
On Home 6 the effect varied by fit. Its paired intervals include zero for
all three fits. Check counts in the table cover the entire chronological
replay. Scores cover only the later slice.

[Bias receipts](evidence/online) and sensor-dependent receipts record the
starting checkpoint, source and archive hashes, class counts, approval
blocks and final learned correction. Use the pinned revision above to
match the bias receipt's source hash.
