# What a radiator can teach an offline policy

The [ZEB Living Lab measurements](https://data.mendeley.com/datasets/xztfbtsgys/3), described in [the data paper](https://doi.org/10.1016/j.dib.2023.109149), record radiator pump and valve setpoints, thermal power, and indoor and outdoor temperature in a real experimental house. Human preferences and occupant comfort ratings were not collected for this example. The temperature band below is a teaching choice.

## Replay

Run with Python 3.11 or newer. There are no third-party Python dependencies.

```bash
curl -fL 'https://data.mendeley.com/public-files/datasets/xztfbtsgys/files/00524e03-d7db-4399-8d65-7615e3afe024/file_downloaded' -o six-buildings.zip
python3 replay.py six-buildings.zip --expect receipt.json
```

The script checks the publisher's SHA-256 for the archive and uses its one-minute ZEB Living Lab CSV. It forms complete 15-minute transitions and keeps two actions only. Off means both pump and valve were commanded to 0%. On means both were commanded to 100%. The experiment cannot separate those effects. Partial or changing commands are skipped. Power is measured thermal power, not electricity use or a bill. Sum the minute samples in watts and divide by 60,000 for approximate thermal kWh. Temperature cost is the number of degrees outside 20–24 °C, integrated over those minutes. The reward subtracts the heat plus twice that temperature cost. Changing the band or weight would define a new experiment.

The state is the living-room temperature rounded down to a whole degree, the outside temperature divided into 4 °C bands, and whether the room cooled by more than 0.2 °C over the previous 15 minutes. The script estimates the reward and next-state distribution of recorded actions and runs discounted value iteration. It returns a recommendation only for states where both commands occurred at least eight times during fitting. An unsupported next state ends the look-ahead. Observations nearby in time can be correlated, so eight records are a coverage rule, not eight independent experiments.

## What the logs say

| | Fitting | Held out from 7 March 2019 UTC |
| --- | ---: | ---: |
| Complete transitions for these two commands | 1,101 | 532 |
| Outside the chosen temperature band at any minute | 34 | 3 |

The learned policy said **off 468 times** and **abstained 64 times**. It never said on. The training-frequency baseline recommended off 245 times, on 223 times and abstained 64 times. The logged controller actually used on 288 times and off 244 times.

On 211 held-out intervals, the learned recommendation matched the exact recorded action. Its outcome fields describe only those 211 logged intervals. A log shows what happened under the controller that was running. It does not show what would have happened under a different sequence of commands. Differences in timing, weather, other equipment and what happened earlier can all affect the comparison. These logs do not establish energy savings or comfort under the learned policy.

The resulting policy is an offline tabular baseline. It shares no weights with the occupancy classifier and has no attention layer or human-preference model. The replay does not connect to building equipment.
