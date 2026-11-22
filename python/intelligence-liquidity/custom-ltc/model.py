import argparse
import csv
import hashlib
import io
import json
import platform
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn.functional import softplus
from math import isfinite

SHA = "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"
FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TRAIN, TEST_A, TEST_B = "datatraining.txt", "datatest.txt", "datatest2.txt"
WINDOW, CUT, EPOCHS = 8, 6500, 12
SEEDS = [41, 42, 43]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


UNITS, SUBSTEPS = 8, 6
SCHEMA = "custom-ltc-v1"


class Liquid(nn.Module):
    def __init__(self):
        super().__init__()
        self.capacitance = nn.Parameter(torch.zeros(UNITS))
        self.leak = nn.Parameter(torch.zeros(UNITS))
        self.bias = nn.Parameter(torch.zeros(UNITS))
        self.sensory = nn.Parameter(self.connections(len(FEATURES)))
        self.recurrent = nn.Parameter(self.connections(UNITS))
        self.readout = nn.Linear(UNITS, 1)

    @staticmethod
    def connections(sources):
        # Each [source, destination] stores strength, slope, midpoint, reversal.
        shape = (sources, UNITS)
        return torch.stack((torch.full(shape, -2.0), torch.ones(shape),
                            torch.zeros(shape), torch.randn(shape)), dim=-1)

    @staticmethod
    def conductance(source, connections):
        strength, slope, midpoint, reversal = connections.unbind(-1)
        gate = torch.sigmoid(slope * (source.unsqueeze(-1) - midpoint))
        active = softplus(strength) * gate
        return active.sum(dim=1), (active * reversal).sum(dim=1)

    def advance(self, sensors, state, elapsed):
        sensory, sensory_pull = self.conductance(sensors, self.sensory)
        leak = softplus(self.leak) + 1e-6
        # Multiplying through by dt gives the C/dt update without dividing by dt.
        dt = elapsed[:, None] / SUBSTEPS
        capacitance = softplus(self.capacitance) + 1e-6
        for _ in range(SUBSTEPS):
            recurrent, recurrent_pull = self.conductance(state, self.recurrent)
            total = leak + sensory + recurrent
            pull = leak * self.bias + sensory_pull + recurrent_pull
            state = (capacitance * state + dt * pull) / (capacitance + dt * total)
        return state

    def forward(self, sensors, gaps):
        state = sensors.new_zeros((len(sensors), UNITS))
        for reading, elapsed in zip(sensors.unbind(1), gaps.unbind(1)):
            state = self.advance(reading, state, elapsed)
        return self.readout(state).squeeze(-1)


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 10, batch_first=True)
        self.readout = nn.Linear(10, 1)

    def forward(self, sensors, gaps):
        inputs = torch.cat((sensors, gaps.log().unsqueeze(-1)), dim=-1)
        states, _ = self.gru(inputs)
        return self.readout(states[:, -1]).squeeze(-1)


class Current(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, sensors, gaps):
        return self.classifier(sensors[:, -1]).squeeze(-1)


MODELS = {"ltc": Liquid, "gru": Recurrent, "current": Current}
LOSS = nn.BCEWithLogitsLoss()


def open_data(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError("Unexpected archive SHA-256")
    return ZipFile(io.BytesIO(raw))


def read_rows(archive, name):
    lines = archive.read(name).decode().splitlines()
    fields = ["index", "date", *FEATURES, "Occupancy"]
    rows = [(row["date"], [float(row[key]) for key in FEATURES], int(row["Occupancy"]))
            for row in csv.DictReader(lines[1:], fieldnames=fields)]
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    return rows


def summary(rows):
    positives = sum(row[2] for row in rows)
    return dict(rows=len(rows), occupied=positives, occupied_fraction=positives / len(rows),
                start=rows[0][0], end=rows[-1][0])


def windows(rows, kept, targets, mean, scale):
    dates = [datetime.fromisoformat(rows[i][0]) for i in kept]
    gaps = torch.tensor([1.] + [(b-a).total_seconds()/60 for a, b in zip(dates, dates[1:])])
    positions = [j for j, original in enumerate(kept) if original in targets and j >= WINDOW-1]
    ids = torch.tensor(positions)[:, None] + torch.arange(1-WINDOW, 1)
    elapsed = gaps[ids]
    elapsed[:, 0] = 1.
    sensors = (torch.tensor([rows[i][1] for i in kept]) - mean) / scale
    labels = torch.tensor([rows[kept[j]][2] for j in positions], dtype=torch.float32)
    return sensors[ids], elapsed, labels, [kept[j] for j in positions]


@torch.no_grad()
def predict(model, x, g):
    return torch.cat([model(a, b) for a, b in zip(x.split(256), g.split(256))])


def tensor_state(model):
    return {key: value.tolist() for key, value in model.state_dict().items()}


def load_model(name, entry):
    model = MODELS[name]()
    model.load_state_dict({key: torch.tensor(value) for key, value in entry["weights"].items()})
    return model.eval()


def train(name, seed, fit, val):
    torch.manual_seed(seed)
    model = MODELS[name]()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for p in model.parameters() if p.requires_grad]
    first = [torch.zeros_like(p) for p in parameters]
    second = [torch.zeros_like(p) for p in parameters]
    step = 0
    order = torch.Generator().manual_seed(seed)
    best, losses, weights = float("inf"), [], None
    x, g, y, _ = fit
    vx, vg, vy, _ = val
    for epoch in range(1, EPOCHS+1):
        model.train()
        for ids in torch.randperm(len(y), generator=order).split(256):
            model.zero_grad(set_to_none=True)
            loss = LOSS(model(x[ids], g[ids]), y[ids])
            loss.backward()
            step += 1
            with torch.no_grad():
                for p, m, v in zip(parameters, first, second):
                    m.mul_(0.9).add_(p.grad, alpha=0.1)
                    v.mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                    corrected = (v / (1 - 0.999**step)).sqrt().add_(1e-8)
                    p.addcdiv_(m, corrected, value=-0.01 / (1 - 0.9**step))
        model.eval()
        value = LOSS(predict(model, vx, vg), vy).item()
        losses.append(value)
        if value < best:
            best, weights = value, tensor_state(model)
    assert all(torch.isfinite(torch.tensor(losses)))
    entry = dict(weights=weights, parameters=params, validation_bce=losses,
                 selected_epoch=losses.index(best)+1)
    return entry


def metrics(logits, actual):
    positive, occupied = logits >= 0, actual.bool()
    tp = int((positive & occupied).sum())
    tn = int((~positive & ~occupied).sum())
    fp = int((positive & ~occupied).sum())
    fn = int((~positive & occupied).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, bce=LOSS(logits, actual).item(),
                accuracy=(tp+tn)/len(actual), balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2)


def json_text(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def evaluate(archive, checkpoint):
    mean, scale = map(torch.tensor, (checkpoint["mean"], checkpoint["scale"]))
    result = {}
    examples = []
    for name in (TEST_A, TEST_B):
        rows = read_rows(archive, name)
        all_ids = list(range(len(rows)))
        kept = [i for i in all_ids if i % 10 not in (2, 3, 4)]
        targets = set(kept[WINDOW-1:])
        result[name] = dict(source=summary(rows), scored=summary([rows[i] for i in sorted(targets)]),
                            dropped=len(rows)-len(kept))
        for mode, history in (("original", all_ids), ("missing", kept)):
            x, g, y, ids = windows(rows, history, targets, mean, scale)
            assert ids == sorted(targets)
            result[name][mode] = {}
            for seed, entries in checkpoint["runs"].items():
                models = {}
                for model_name, entry in entries.items():
                    model = load_model(model_name, entry)
                    logits = predict(model, x, g)
                    assert torch.equal(logits, predict(load_model(model_name, entry), x, g))
                    models[model_name] = metrics(logits, y)
                    if name == TEST_A and seed == str(SEEDS[0]) and model_name == "ltc":
                        for ix in (0, 90, 400):
                            end = history.index(ids[ix])
                            raw = [rows[j] for j in history[end-WINDOW+1:end+1]]
                            examples.append(dict(mode=mode, rows=raw, elapsed=g[ix].tolist(),
                                                 probability=logits[ix].sigmoid().item()))
                result[name][mode][seed] = models
    return dict(test=result, examples=examples)


def infer(checkpoint, rows, now=None):
    if len(rows) != WINDOW:
        raise ValueError("Expected exactly eight readings")
    dates = [datetime.fromisoformat(row[0]) for row in rows]
    if any(date.tzinfo is not None for date in dates):
        raise ValueError("Use timezone-naive timestamps on the same local clock")
    if any(not isfinite(v) for row in rows for v in row[1]):
        raise ValueError("Sensor values must be finite")
    for earlier, later in zip(dates, dates[1:]):
        if not 0 < (later - earlier).total_seconds() <= 300:
            raise ValueError("Readings must be increasing and at most five minutes apart")
    if now is not None:
        reference = datetime.fromisoformat(now)
        if reference.tzinfo is not None:
            raise ValueError("Use a timezone-naive reference clock")
        if not 0 <= (reference - dates[-1]).total_seconds() <= 300:
            raise ValueError("Final reading is stale or in the future")
    mean, scale = map(torch.tensor, (checkpoint["mean"], checkpoint["scale"]))
    x, g, _, _ = windows(rows, list(range(WINDOW)), {WINDOW - 1}, mean, scale)
    if not torch.isfinite(x).all():
        raise ValueError("Normalized sensor values exceed numeric range")
    model = load_model("ltc", checkpoint["runs"][str(SEEDS[0])]["ltc"])
    probability = predict(model, x, g).sigmoid().item()
    if not isfinite(probability):
        raise ValueError("Model returned a nonfinite score")
    return dict(timestamp=rows[-1][0], probability=probability,
                occupied=probability >= 0.5, threshold=0.5, seed=SEEDS[0])


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["date", *FEATURES]:
            raise ValueError("Expected date followed by the checkpoint feature order")
        rows = []
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("Wrong number of CSV fields")
            rows.append((row["date"], [float(row[key]) for key in FEATURES], 0))
        return rows


def fit_checkpoint(archive):
    rows = read_rows(archive, TRAIN)
    fit_rows, val_rows = rows[:CUT], rows[CUT:]
    raw = torch.tensor([row[1] for row in fit_rows])
    mean, scale = raw.mean(0), raw.std(0, correction=0)
    fit_ids = list(range(len(fit_rows)))
    val_ids = list(range(len(val_rows)))
    fit = windows(fit_rows, fit_ids, set(fit_ids), mean, scale)
    val = windows(val_rows, val_ids, set(val_ids), mean, scale)
    runs = {str(seed): {name: train(name, seed, fit, val) for name in MODELS}
            for seed in SEEDS}
    return dict(archive_sha256=SHA, dataset_doi="10.24432/C5X01N",
                paper_doi="10.1609/aaai.v35i9.16936", features=FEATURES,
                window=WINDOW, schema=SCHEMA, units=UNITS, substeps=SUBSTEPS,
                mean=mean.tolist(), scale=scale.tolist(), runs=runs,
                versions={"torch": torch.__version__}, python=platform.python_version(),
                epochs=EPOCHS, fit=summary(fit_rows), validation=summary(val_rows),
                fit_targets=len(fit[2]), validation_targets=len(val[2]),
                fit_positives=int(fit[2].sum()), validation_positives=int(val[2].sum()))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train").add_argument("archive")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("archive")
    evaluation.add_argument("checkpoint")
    inference = commands.add_parser("infer")
    inference.add_argument("checkpoint")
    source = inference.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive")
    source.add_argument("--csv")
    inference.add_argument("--now", help="Reference time on the sensor's local clock")
    args = parser.parse_args()
    try:
        if args.command == "train":
            with open_data(args.archive) as archive:
                output = fit_checkpoint(archive)
        else:
            check = json.loads(Path(args.checkpoint).read_text())
            identity = (check["archive_sha256"], check["features"], check["window"],
                        check["schema"], check["units"], check["substeps"])
            if identity != (SHA, FEATURES, WINDOW, SCHEMA, UNITS, SUBSTEPS):
                raise ValueError("Incompatible data or model schema")
            if args.command == "evaluate":
                with open_data(args.archive) as archive:
                    output = evaluate(archive, check)
            elif args.csv:
                if args.now is None:
                    raise ValueError("CSV inference requires --now for the freshness check")
                output = infer(check, csv_rows(args.csv), args.now)
            else:
                with open_data(args.archive) as archive:
                    # The verified archive locates published record 150 at offset 10.
                    rows = read_rows(archive, TEST_A)[3:11]
                if rows[-1][0] != "2015-02-02 14:29:00":
                    raise ValueError("Unexpected benchmark target")
                output = infer(check, rows, args.now)
                output.update(record=150, file=TEST_A, recorded_label=rows[-1][2])
        print(json_text(output))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, csv.Error) as error:
        parser.exit(2, f"Cannot score request: {error}\n")


if __name__ == "__main__":
    main()
