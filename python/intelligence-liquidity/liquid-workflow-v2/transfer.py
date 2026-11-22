"""Load the first seven complete H6 days on the fixed BS4 channel."""

import csv
from datetime import datetime, timezone
import io
from pathlib import Path
from zipfile import ZipFile

import torch

from experiment import sha

H6_ARCHIVES = {
    "H6_ENVIRONMENTAL.zip": "ff9539b61c921a9f5dcbc32b95bd6b0705f91536ecd3dbfae48ddcc36dffff58",
    "H6_GROUNDTRUTH.zip": "28ddb88f6deca3af403c0cb1d35e721614fcf02ecb523ef64d412ce7e3101574",
}


H5_ARCHIVES = {
    "H5_ENVIRONMENTAL.zip": "2052daee624cb0aff7827097d34422c536fdf5e86047734b90b858cd6bb1fe4f",
    "H5_GROUNDTRUTH.zip": "4d3de5bee5abd29d0826a70376b80fefa2f052e2c5701e2b77e575b6f0852d9b",
}


def rows(archive, name):
    with io.TextIOWrapper(archive.open(name)) as source:
        return list(csv.DictReader(source))


def house(data, home, channel, archives):
    hashes = {name: sha(data / name) for name in archives}
    if hashes != archives:
        raise ValueError(f"The {home} study archives changed")
    environmental, ground_truth = (data / n for n in archives)
    selected = {}
    with ZipFile(environmental) as sensors, ZipFile(ground_truth) as labels:
        names = sorted(n for n in sensors.namelist() if n.startswith(f"{home}_{channel}_ENV/") and n.endswith(".csv"))
        dates = [n.split("/")[-1][:10] for n in names
                 if f'{n.split("/")[-1][:10]}_{home}_groundtruth.csv' in labels.namelist()]
        for day in dates:
            truth = {r["timestamp"]: r["occupied"] for r in rows(labels, f"{day}_{home}_groundtruth.csv")}
            if len(truth) < 8640:  # predeclared full-day label coverage
                continue
            for reading in rows(sensors, f"{home}_{channel}_ENV/{day}_{channel}_{home}_env.csv"):
                time = reading[""]
                observed = truth.get(time)
                if observed not in ("0", "1") or not reading["co2eq_ppm"] or not reading["temp_c"]:
                    continue
                stamp = int(datetime.fromisoformat(time).replace(tzinfo=timezone.utc).timestamp())
                selected.setdefault(stamp // 300, (stamp, (float(reading["co2eq_ppm"]),
                                                            float(reading["temp_c"])), int(observed)))
            if len({datetime.fromtimestamp(i[0], timezone.utc).date() for i in selected.values()}) == 7:
                break
    t, x, y = map(torch.tensor, zip(*(selected[k] for k in sorted(selected)), strict=True))
    normalized = (x - torch.tensor((500., 22.))) / torch.tensor((200., 3.))
    return [(f"HPDmobile {home} {channel}", normalized, y.float(), t)], hashes



def h6(data):
    return house(data, "H6", "BS4", H6_ARCHIVES)


def h5(data):
    return house(data, "H5", "RS1", H5_ARCHIVES)
