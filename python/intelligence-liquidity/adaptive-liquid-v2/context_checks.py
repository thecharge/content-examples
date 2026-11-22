"""Ancillary UCI, explored transfer and logged heating checks."""
import csv
import importlib.util
from io import TextIOWrapper
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn import functional as F
from ncps.torch import LTC

import experiment as e
from data import apartments, heat_rows, heat_windows, number


def paper_style():
    fields = (2, 3, 4, 5, 6)
    def load(member):
        with ZipFile(e.UCI) as archive, TextIOWrapper(archive.open(member)) as stream:
            rows = csv.reader(stream)
            next(rows)
            return [(i, e.minute(row[1]), tuple(number(row[j]) for j in fields), int(row[-1]))
                    for i, row in enumerate(rows)]
    def windows(rows):
        for i in range(31, len(rows)):
            history = rows[i - 31:i + 1]
            gaps = [b[1] - a[1] for a, b in zip(history, history[1:])]
            if all(0 < d <= 5 for d in gaps):
                yield dict(x=[r[2] for r in history], gaps=[1, *gaps], y=rows[i][-1])
    torch.manual_seed(41)
    raw = load("datatraining.txt")
    center = e.normalizer([r[2] for r in raw[:6500]])
    train = e.pack(list(windows(raw[:6500])), center)
    val = e.pack(list(windows(raw[6500:])), center)
    cell, head = LTC(5, 8, batch_first=True, ode_unfolds=6), nn.Linear(8, 1)
    optimizer = torch.optim.Adam([*cell.parameters(), *head.parameters()], lr=0.003)
    def forward(x, t):
        state = x.new_zeros(len(x), 8)
        for signal, gap in zip(x.unbind(1), t.unbind(1)):
            result, state = cell.rnn_cell(signal, state, gap[:, None])
        return head(result)[:, 0]
    @torch.no_grad()
    def acc(data):
        return sum(int(((forward(x, t) >= 0) == y).sum()) for x, t, y in e.batches(data)) / len(data[2])
    curve, best = [], None
    for step in range(1, 241):
        ids = torch.randint(len(train[2]), (64,))
        x, dt, y = (v[ids] for v in train)
        loss = F.binary_cross_entropy_with_logits(forward(x, dt), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 24 == 0:
            accuracy = acc(val)
            curve.append([step, accuracy])
            if best is None or accuracy > best[0]:
                best = accuracy, step, [{k: v.clone() for k, v in m.state_dict().items()}
                                        for m in (cell, head)]
    for m, weights in zip((cell, head), best[2]):
        m.load_state_dict(weights)
    return dict(curve=curve, selected_step=best[1], raw_accuracy={
        member: acc(e.pack(list(windows(load(member))), center))
        for member in ("datatest.txt", "datatest2.txt")})


def heating(directory):
    archive = "/tmp/sndv-external/six-buildings.zip"
    fit, fit_skips = heat_windows(heat_rows(archive, False))
    test, test_skips = heat_windows(heat_rows(archive, True))
    samples = {}
    for row in fit:
        samples.setdefault((tuple(row["bucket"]), row["action"]), []).append(row["y"])
        samples.setdefault((None, row["action"]), []).append(row["y"])
    means = {k: torch.tensor(v).mean(0) for k, v in samples.items()}
    records = []
    for row in test:
        action = row["action"]
        key = tuple(row["bucket"]), action
        supported = all(len(samples.get((tuple(row["bucket"]), a), ())) >= 8 for a in (0, 1))
        forecast = means.get(key, means[None, action]).tolist()
        measured = row["y"]
        value = lambda result: -result[1] - 2 * result[2]
        records.append(dict(id=row["id"], logged=action, supported=supported,
                            observed=measured, forecast=forecast,
                            observed_reward=value(measured), forecast_reward=value(forecast)))
    mae = sum(abs(r["observed_reward"] - r["forecast_reward"]) for r in records) / len(records)
    result = dict(fit=len(fit), held_out=len(test), supported=sum(r["supported"] for r in records),
                  fit_skips=fit_skips, held_out_skips=test_skips, payoff_mae=mae, records=records)
    (directory / "heating.json").write_text(json.dumps(result))
    return {k: v for k, v in result.items() if k != "records"}


def explored(directory):
    spec = importlib.util.spec_from_file_location("graz", e.V1.with_name("graz.py"))
    graz = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graz)
    sets = {name: rows for name, (rows, _) in graz.load(e.INPUT / "graz.zip").items()}
    sets["Danish"] = [window for room in apartments(e.INPUT / "apartments.csv").values()
                      for window in e.occupancy_windows([room], 20, 15)]
    results = {}
    with torch.no_grad():
        for name in e.NAMES:
            saved = torch.load(directory / f"{name}-41-100.pt", weights_only=True)
            model = e.build(name)
            model.load_state_dict(saved["weights"])
            model_results = {}
            for site, rows in sets.items():
                data = e.pack(rows, saved["stats"])
                scores, covered, _, _ = e.predict(model, data, saved["policies"].get("0.1"))
                model_results[site] = dict(n=len(rows), metrics=e.metrics(
                    scores, data[2], covered, saved["cutoff"]), ranking=e.ranking(scores, data[2]))
            results[name] = model_results
    return results


if __name__ == "__main__":
    torch.set_num_threads(1)
    directory = Path(sys.argv[2])
    command = sys.argv[1]
    outcome = {"heat": lambda: heating(directory), "paper": paper_style,
               "explored": lambda: explored(directory)}[command]()
    (directory / f"{command}-summary.json").write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome))
