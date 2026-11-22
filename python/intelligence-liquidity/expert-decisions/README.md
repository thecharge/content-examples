# Five liquid experts that can pass

Five experts read one five-minute window in parallel. Each keeps its own
liquid state and sees a different slice of the sensors: air quality,
temperature and humidity, light, recent changes, or mixed readings. A
window always gives each expert the choice to pass or offer an occupancy
claim and its estimated confidence. The trace records the readings that
were available and the rule behind the decision. It provides an audit of
the gate and the inputs available to it. It does not estimate which input
caused a neural prediction.

An expert without readings passes. With readings, the normal gate needs
55% confidence to cover a cost of 0.10 points. A rotating trial slot can
lower the gate by adding up to 0.15 points to expected payoff. It leaves
reported confidence alone. After a recorded answer arrives, a correct
claim earns +0.90, a wrong claim earns −1.10 and a pass earns zero.
A separate quadratic score rewards honest confidence. The offline fit
learns from labeled sequences. Replay scores rewards without updating
weights. A reward from this run measures occupancy claims, so there are
no heating actions or live control outcomes here.

## Run it

Python 3.12, PyTorch 2.7.1 (CPU), XGBoost 3.1.3. Download the twelve
[HPDmobile archives and follow the original naming instructions](../liquid-workflow-v3/README.md#reproduce-the-fits).

```sh
python -m pip install torch==2.7.1 xgboost==3.1.3
python -m unittest -v
python experiment.py /path/to/hpd-archives ./run-output --epochs 15
```

Use `run-output/results.json` to read losses, selected epoch, replay
counts, confidence bins and example gate decisions. It includes SHA-256
digests of the archives, source and snapshots. The committed
[receipt and snapshots](evidence/) are from one CPU run. H1–H3 supplied
12,087 training rows. The first half of H4 selected the checkpoint and
the second half fitted the confidence conversion. H5 and H6 were
excluded from both steps. These two homes are reused comparisons
from earlier experiments, with the same window selection as the
three-expert board. The recorded occupancy label covers the whole home.

## Compare the same experts with and without review

In both modes the board averages the probabilities of experts who spoke.
With review enabled, a separate XGBoost tree ensemble sees only the
columns outside that expert's view. It can veto a conflicting claim if
it has sensor evidence and puts at least 75% probability on the other
answer. If the board runs out of claims, it withholds its answer. This
isolates a veto without adding retry, delta attention or online weight
updates. The [previous board](../liquid-workflow-v3/README.md) has those
separate mechanisms.

| Home | Review | Correct | Vacant windows correct | Both classes count equally | Answered |
| --- | --- | ---: | ---: | ---: | ---: |
| H5 | Off | 1,226 / 2,010 | 0 / 784 | 50.0% | 100% |
| H5 | On | 1,226 / 2,010 | 0 / 784 | 50.0% | 100% |
| H6 | Off | 1,553 / 2,016 | 0 / 463 | 50.0% | 100% |
| H6 | On | 1,553 / 2,016 | 0 / 463 | 50.0% | 100% |

The board called every window occupied. Both runs matched an always-occupied
baseline. There were seven expert passes, zero bonus-driven claims and
zero vetoes over 20,130 opportunities. The tests exercise low-confidence
trials and complementary-evidence vetoes. Those paths did not fire in the
real replays. In H5, the 10,036 claims in the 70–80% confidence band averaged
73.0% confidence and were correct 60.9% of the time. H6's 10,079 claims
averaged 72.8% confidence and were correct 77.0% of the time. Confidence
is a fitted estimate and a score of 1 is never treated as proof.

[Training debug copy](debug-training/README.md) compares two learning rates,
records raw predictions and overfits a five-row batch as a control.
