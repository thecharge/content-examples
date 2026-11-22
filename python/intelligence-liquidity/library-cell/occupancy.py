import argparse
import csv
import hashlib
import io
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn
from ncps.torch import LTC
from ncps.wirings import FullyConnected

SHA = "4ae3f46aa98eedff564a9f6924d1635173e2fd2c816004342a9be93076d3a81a"
FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TRAIN, TEST_A, TEST_B = "datatraining.txt", "datatest.txt", "datatest2.txt"
WINDOW, CUT, EPOCHS = 8, 6500, 12
SEEDS = [41, 42, 43]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class Liquid(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltc = LTC(5, FullyConnected(8, self_connections=True), ode_unfolds=6)
        self.readout = nn.Linear(8, 1)

    def forward(self, sensors, gaps):
        state = sensors.new_zeros((len(sensors), 8))
        for i in range(sensors.size(1)):
            # Preserve [batch, 1] for broadcasting across cell units.
            out, state = self.ltc.rnn_cell(sensors[:, i], state, gaps[:, i, None])
        return self.readout(out).squeeze(-1)


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 9, batch_first=True)
        self.readout = nn.Linear(9, 1)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "infer"])
    parser.add_argument("archive")
    parser.add_argument("checkpoint", nargs="?")
    args = parser.parse_args()
    with open_data(args.archive) as archive:
        if args.mode == "train":
            rows = read_rows(archive, TRAIN)
            fit_rows, val_rows = rows[:CUT], rows[CUT:]
            raw = torch.tensor([row[1] for row in fit_rows])
            mean, scale = raw.mean(0), raw.std(0, correction=0)
            fit = windows(fit_rows, list(range(CUT)), set(range(CUT)), mean, scale)
            n = len(val_rows)
            val = windows(val_rows, list(range(n)), set(range(n)), mean, scale)
            runs = {str(seed): {name: train(name, seed, fit, val) for name in MODELS}
                    for seed in SEEDS}
            check = dict(archive_sha256=SHA, features=FEATURES, window=WINDOW, schema=2,
                         mean=mean.tolist(), scale=scale.tolist(), runs=runs,
                         versions={pkg: version(pkg) for pkg in ("torch", "ncps", "numpy")},
                         python=platform.python_version(), epochs=EPOCHS,
                         fit=summary(fit_rows), validation=summary(val_rows),
                         fit_targets=len(fit[2]), validation_targets=len(val[2]))
            print(json_text(check))
        else:
            check = json.loads(Path(args.checkpoint).read_text())
            if (check["archive_sha256"], check["features"], check["window"], check["schema"]) != (SHA, FEATURES, WINDOW, 2):
                raise ValueError("Incompatible data or model schema")
            results = evaluate(archive, check)
            if args.mode == "infer":
                results = results["examples"]
            print(json_text(results))


if __name__ == "__main__":
    main()
