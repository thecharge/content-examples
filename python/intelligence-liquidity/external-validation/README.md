# Replay the frozen model in a second room

The [Room Occupancy Estimation dataset](https://doi.org/10.24432/C5P605) supplies measurements and human-recorded counts for another room. The evaluator runs the existing [liquid time-constant network](https://arxiv.org/html/2006.04439v4) and same-office baselines without retraining or changing the cutoff.

From this directory, with Python and CPU PyTorch 2.7.1 installed:

```bash
curl -fL 'https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip' -o room-occupancy.zip
python evaluate.py room-occupancy.zip --expect receipt.json
```

See `protocol.json` for the mapping and frozen input conditions. The test retains the original timing, uses temperature and light from sensor 1 and carbon dioxide from sensor 5, and fills missing humidity and humidity ratio with the training means. Windows that cross gaps over five minutes are excluded. The receipt pins the archive, CSV, protocol, model source and exact metrics.

The new room supplied 10,101 scored windows. The candidate found none of the 1,880 occupied windows. The reported 81.39% accuracy comes from calling every window empty and corresponds to 50% balanced accuracy. This checks transfer under a different room, instrumentation, sampling and missing channels. It cannot tell which change caused the failure or how the same model would handle a second room with all of its inputs.
