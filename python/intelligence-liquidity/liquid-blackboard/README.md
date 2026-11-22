# Intelligence liquidity

Three layers of liquid experts share a delta memory. Each layer has its own liquid checker. The checker can repair the representation it received. The network learns from the next sensor reading while it is in use. A separate heating fit also learns from the rewards of recorded actions.

The full model did not win at every site. The [score receipt](frozen/scores.json) records all seeds, controls, budgets and individual heating decisions. The [fit manifest](frozen/frozen.json) records the code, checkpoint and training-input hashes, the chosen epoch, the validation curve and the occupancy cutoff for each fit.

## Reproduce the run

Use Python 3.12 and the CPU build of PyTorch from the manifest. The data stays in a separate local folder.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v
```

Save the publishers' files under these names.

| Save as | Publisher file |
| --- | --- |
| `occupancy-fresh.zip` | [UCI Occupancy Detection](https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip) |
| `dataset__2023_09_02__2023_10_04.csv` | [Aalborg office](https://zenodo.org/records/10673763/files/dataset__2023_09_02__2023_10_04.csv?download=1) |
| `aalto-1.csv` | [Aalto first recording](https://zenodo.org/records/18299691/files/training_data_1.csv?download=1) |
| `aalto-2.csv` | [Aalto second recording](https://zenodo.org/records/18299691/files/training_data_2.csv?download=1) |
| `graz.zip` | [Graz offices](https://zenodo.org/records/11573783/files/occupancy-detection-dataset.zip?download=1) |
| `munich.zip` | [Munich offices](https://data.mendeley.com/public-api/zip/v7kw9ycccp/download/2) |
| `porto-alegre.csv` | [CONSEG office, Porto Alegre](https://zenodo.org/api/records/20548374/files/final_dataset_csv.csv/content) |
| `six-buildings.zip` | [Six Swedish buildings, version 3](https://data.mendeley.com/public-api/zip/xztfbtsgys/download/3) |

To replay the saved run, copy the checkpoints into a new output folder first. Scoring writes a new receipt there.

```sh
cp -R frozen replayed
python experiment.py score /path/to/data replayed --workers 4 \
  --freeze f5bffb3986cac4eb5ab05b2ff8fffa83b46f1c6c65668b84e385608eb4be3f17
```

To train from scratch before scoring, run `python experiment.py fit /path/to/data refit --epochs 24 --workers 4`. Use its printed freeze hash and the `refit` folder in the score command. Keep the fit manifest and score output together. Report the hardware and Torch build if you compare runtimes. Publish the code revision along with the archive.

## What happens inside

1. The cell uses a bounded conductance update. The strength and direction of each connection and the time gap determine how the state moves. This follows the approach in the [liquid time-constant paper](https://arxiv.org/html/2006.04439v4).
2. Each layer routes between three liquid experts with different sensor views. Each layer has a separate liquid checker that uses the incoming proposal, the new proposal, memory and the previous forecast error. The learned process score estimates the reduction in next-reading forecast error. A positive score permits the blend to include a repair to the incoming representation. The next layer receives that blend in the same forward pass.
3. The shared memory learns per-channel decay. It writes the difference between the new value and what the faded memory already predicts. This uses the delta idea from [Kimi Linear](https://arxiv.org/html/2510.26692v1).
4. Streaming occupancy inference emits a prediction before using the current sensor reading to evaluate the previous forecast. It only adjusts the four input-adapter parameters. The occupancy labels only enter scoring. The heating model starts from a separate fit. It records its recommendation before using the historical command and the observed transition for a discounted reward update. The heating loss uses a conservative penalty and a small verified forecast-improvement bonus alongside energy and comfort.
5. Validate using the subsequent records and select a checkpoint. This benchmark allows up to 24 epochs and stops after eight epochs without a new best result. In an application, keep a reference checkpoint and start each new room with fresh recurrent state. Compare adaptation with frozen inference, so you can detect drift and reload the reference checkpoint.

## What the split measures

Occupancy training uses UCI and Aalborg. Aalto chooses the checkpoint and cutoff. Graz and Munich had appeared in this project before. Porto Alegre supplied a new site for this comparison. Each test uses the first 2,016 prepared observations per room. Each full-data configuration uses seeds 41, 42 and 43. The 25%, 50% and 100% comparison uses seed 41 and 1,127, 2,252 and 4,503 training blocks. It tests three budgets for one seed. A smaller dataset did better in some rooms. The experiment leaves the global optimum open.

The balanced-accuracy table gives an empty room and an occupied room equal weight. Each entry averages three seeds and uses the validation cutoff.

| Model | Graz | Munich | Porto Alegre |
| --- | ---: | ---: | ---: |
| Plain liquid, no state carry | 75.78% | 54.50% | 71.74% |
| Plain liquid, state carry | 63.81% | 60.57% | 52.41% |
| Liquid experts | 60.44% | 67.15% | 50.04% |
| Experts and delta | 55.58% | 70.96% | 50.00% |
| Experts, delta and checkers | 66.88% | 68.25% | 52.48% |
| Full model with online input adaptation | 57.22% | 71.29% | 50.23% |

Clipping constrains adaptation. Some trained starting adapter values lie outside the inference bounds, so the first update can snap them to a bound. The online comparison includes that jump as well as subsequent learning. It cannot determine how much of the change came from either part.

The held-out heating replay starts on 7 March 2019 UTC. All six fits recommend off for all 380 cases with enough training support, and make no recommendation on 153 cases. The online updates change the values but preserve those decisions. The record supplies the outcome of the command that happened. It leaves the alternative's effect unknown. Use the replay to inspect what the model learns, and use a simulator or a prospective controlled study to measure control performance.

## Share a checkpoint

A model host needs the checkpoint, manifest, inference code, a usage example and a model card with the split and limitations. [Package and upload a checkpoint](MODEL-HOSTING.md).

## Check what the repair gate changes

```sh
python measure_repairs.py /path/to/data frozen
```

This [separate comparison](frozen/repairs.json) uses the same seed-41 full checkpoint, same cutoff and state carry in both passes. It disables only repairs in the second pass. The receipt counts decisions that repair fixed or broke and gives a real example of each. It evaluates a whole stream with repair on or off. Earlier repaired state and memory can influence later decisions.
