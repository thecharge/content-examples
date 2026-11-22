# How well do liquid models work in a new building?

The strongest model changed with the building.
Plain liquid had the best balance in Graz. A simple CO₂ reading won in Munich.
Delta memory helped the mixture in both places. Review added about 0.2 points in each.

## Compare both buildings

Balanced accuracy gives the same weight to catching occupied rooms and recognizing empty rooms.
An empty room is easier to guess on average because most readings come from empty rooms.
The Brier score measures errors in the predicted probabilities. Lower is better.

| Model | Graz balanced accuracy | Munich balanced accuracy | Munich Brier score |
| --- | ---: | ---: | ---: |
| CO₂-only baseline | 70.6% | **64.3%** | 0.281 |
| Linear sensor baseline | 84.1% | 60.7% | **0.246** |
| Gated recurrent unit (GRU) | 84.3% | 56.1% | 0.281 |
| Plain liquid | **85.7%** | 56.7% | 0.267 |
| Wider plain liquid | 79.8% | 63.8% | 0.254 |
| Liquid mixture of experts | 81.3% | 58.7% | 0.288 |
| Mixture with delta memory | 84.2% | 59.8% | 0.288 |
| Mixture with memory and review | 84.4% | 60.0% | 0.288 |

These are means across three runs with different random seeds.
[Graz scores](frozen/graz-scores.json) cover 29,694 windows in two offices.
[Munich scores](frozen/munich-scores.json) cover 28,100 windows in two offices, including 8,938 occupied windows.
The windows overlap. Each test measures transfer to one building.

## What worked and where it broke

The experts look at current readings, changes over time, or both together.
Every expert learns and contributes according to the weights assigned by the model.
In Graz, each expert received about 28% to 40% of the total weight.
In Munich, one expert received up to 75%. The training rule did not prevent this imbalance on new data.

The memory keeps a running summary and updates it when a step differs from what it predicts.
Memory improved the average by 2.9 points in Graz and 1.1 points in Munich.
Most of the gain came from one seed in each test.

When the mixture is close to its cutoff, a second opinion can compare the experts' majority vote with its answer.
A separate linear sensor model estimates whether changing the answer would reduce the cost of mistakes.
The rule charges a missed occupant twice as much as a false alarm.
The Aalto check enabled review in two runs. The third kept its first answer.
Review fixed 249 Graz decisions and broke 106 across three runs.
In Munich it fixed 232 and broke 234. One run sent 59% of windows to review.
Its small gain in balanced accuracy comes from how that score weighs mistakes in each class.

This test predicts occupancy. It does not measure the quality of heating or ventilation decisions.
The liquid cell adjusts how fast it responds to each step. Here it uses a compact update with a state-dependent rate.
The [original liquid time-constant paper](https://arxiv.org/html/2006.04439v4) describes a different cell and occupancy test.

## Which data each stage can see

Every model uses carbon dioxide (CO₂) and temperature.
Each window contains eight readings taken about five minutes apart over the preceding 35 minutes.
The cameras or manual counts supply the answers for scoring.

| Stage | Data |
| --- | --- |
| Learn model weights | [UCI office](https://doi.org/10.24432/C5X01N) and [Aalborg office building](https://zenodo.org/records/10673763) |
| Fit probabilities, select checkpoints and cutoffs, check review | Three separate recordings from the [Aalto ventilation lab](https://zenodo.org/records/18299691) |
| Repeat check | [Two Graz offices with camera counts](https://zenodo.org/records/11573783) |
| New test | [Two Munich offices with manually recorded occupancy](https://doi.org/10.17632/v7kw9ycccp.2) |

The run used 450 updates per model and saved candidate weights at steps 150, 300, and 450.
The first Aalto recording fit the probability scale.
The second selected weights by how well they ranked readings and set the decision cutoff.
The third selected review rules that fixed more errors than they introduced.
Both tests ran after the weights, cutoffs, and review rules were saved and hashed.
The project had used Graz in an earlier experiment. Munich was new to the project.
The Munich evaluator keeps every fifth minute and restarts the history on each recorded day.
The raw readings come from the publishers and stay outside this repository.

## Replay the test

Use Python 3.12 and the PyTorch CPU version recorded in [the frozen manifest](frozen/frozen.json).
Keep the source files in a local data folder. The test script checks their hashes.

| Source file | Save as |
| --- | --- |
| [UCI archive](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) | `occupancy-fresh.zip` |
| [Aalborg September–October file](https://zenodo.org/records/10673763/files/dataset__2023_09_02__2023_10_04.csv?download=1) | `dataset__2023_09_02__2023_10_04.csv` |
| [Aalto first recording](https://zenodo.org/records/18299691/files/training_data_1.csv?download=1) | `aalto-1.csv` |
| [Aalto second recording](https://zenodo.org/records/18299691/files/training_data_2.csv?download=1) | `aalto-2.csv` |
| [Aalto ventilation recording](https://zenodo.org/records/18299691/files/ventilation_test.csv?download=1) | `aalto-ventilation-test.csv` |
| [Munich archive](https://data.mendeley.com/public-api/zip/v7kw9ycccp/download/2) | `munich.zip` |
| [Graz archive](https://zenodo.org/records/11573783/files/occupancy-detection-dataset.zip?download=1) | `graz.zip` |

Replay needs only the archive for the site you choose.

```sh
python evaluate.py /path/to/data frozen 0e02e15af443d17c1230c6f17d22ee5ea41461c13db233d8fe0e0f2a14cc675a --site munich
```

Use `--site graz` to replay the repeat check.

To fit the models again, use `python experiment.py /path/to/data new-frozen`.
Pass the hash printed by that command to `evaluate.py` and replace `frozen` with `new-frozen`.
