# First adaptive liquid model

This is the recorded first adaptive model and its results. An independent
replay exposed a defect: stage two could change its own output, while the
returned stage-one proposal and its checker still used the old values.
The recorded scores describe this version. Use the [corrected model](../adaptive-liquid-v2/README.md)
to see the upstream commit and recheck.

[Protocol](protocol.md) · [Data and alignment](source-audit.md) ·
[Results](results.md) · [Metrics](metrics.json) · [Inference checks](verification.json)

On 38,298 unseen windows in two Graz rooms, the full model reached 56.52%
balanced accuracy with the full fitting pool. It reached 50% and accepted no
predictions with smaller pools. The same model with delta memory set to zero
reached 75.74% with the full pool. Balanced accuracy gives occupied and empty
windows equal weight. These runs did not show that the full model can learn
from less data. See the results for the matched baselines, budgets and costs.

## Download the evidence

Use Python 3.12 and the CPU dependencies in the earlier
[attention experiment](../attention-and-reward/README.md). Create a local
`data` directory and save these files from their publishers. The benchmark
checks the archive hashes before using them.

| File | Publisher |
| --- | --- |
| `office.zip` | [UCI office readings](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) |
| `apartments.csv` | [Zenodo Danish apartments](https://zenodo.org/records/10761326/files/Dataset.csv?download=1) |
| `graz.zip` | [Zenodo Graz room measurements v2](https://zenodo.org/records/11573783/files/occupancy-detection-dataset.zip?download=1) |
| `six-buildings.zip` | [Mendeley building archive v3](https://data.mendeley.com/public-files/datasets/xztfbtsgys/files/00524e03-d7db-4399-8d65-7615e3afe024/file_downloaded) |

The publisher archives and raw photographs are not included in this repository.
The Graz record identifies its license as AGPL-3.0-or-later. The repository's
license covers the original code in this example.

## Run one comparison and restore its checkpoint

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r ../attention-and-reward/requirements.txt
python reproduce.py check
python reproduce.py --data data graz-audit
python receipts.py review
python reproduce.py --data data occupancy \
  --model adaptive --seed 41 --budget 100 \
  --mode all --freeze freeze.json
python receipts.py extract receipts/occupancy-adaptive-41-100.json \
  > checkpoint.json
python reproduce.py --data data occupancy \
  --model adaptive --seed 41 --budget 100 \
  --checkpoint checkpoint.json --mode all --freeze freeze.json
```

`reproduce.py` routes the local archive filenames to the frozen experiment
without modifying the hashed files. The restored checkpoint skips fitting.
Model scores and routing repeat. Timing depends on the machine and load.
Use `heat` for the logged heating experiment. To train every comparison, use
the model names, seeds and budgets in [the protocol](protocol.md). Keep the
frozen file with the reported receipt.

The [123 receipts](receipts) contain the fitted weights, validation steps,
checks and test results. Decode a receipt with `receipts.py extract`.
Heating receipts keep the predicted energy, comfort and check scores for
each action. They measure forecast error only for the command that ran.
The model does not connect to a heater.
