"""Exact image-label join with a full additional five-minute sensor embargo."""
import csv
from collections import Counter
from datetime import datetime
import hashlib
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from data import WINDOW, minute, number, valid_history

SHA256 = "3bb3e7c81efcf3d81988933577ddc9e4bdc1289eeff11424747801c0c993b47c"


def load(path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SHA256:
        raise ValueError("Unexpected Graz archive")
    local = ZoneInfo("Europe/Vienna")
    results = {}
    with ZipFile(path) as archive:
        def read(member, delimiter=","):
            with TextIOWrapper(archive.open(member), encoding="utf-8-sig") as stream:
                return list(csv.DictReader(stream, delimiter=delimiter))

        for room in ("Room_A", "Room_B"):
            prefix = f"data/raw/{room}/"
            sensor = read(prefix + "sensor_data.csv")
            name, = [n for n in archive.namelist()
                     if n.lower() == (prefix + "image-based-occupancy-groundtruth.csv").lower()]
            labels = read(name, ";")
            published = {minute(r["Date"]): number(r["Image-based-occupancy"])
                         for r in read(f"data/intermi/{room}/df_occ.csv")}
            features = {}
            counts = Counter(sensor_rows=len(sensor), raw_image_rows=len(labels))
            for row in sensor:
                t = minute(row["Date"])
                if t in features:
                    raise ValueError(f"Duplicate sensor time in {room}: {t}")
                features[t] = tuple(number(row[k]) for k in ("temperature", "humidity", "co2"))
            windows = []
            seen = set()
            for row in labels:
                wall = datetime.fromisoformat(row["Date"])
                time = minute(wall.replace(tzinfo=local).isoformat())
                if time in seen:
                    counts["duplicate_label"] += 1
                seen.add(time)
            times = Counter(minute(datetime.fromisoformat(r["Date"]).replace(tzinfo=local).isoformat())
                            for r in labels)
            for row in labels:
                wall = datetime.fromisoformat(row["Date"])
                t = minute(wall.replace(tzinfo=local).isoformat())
                label = number(row["Image-based-occupancy"])
                if wall.replace(tzinfo=local, fold=0).utcoffset() != wall.replace(tzinfo=local, fold=1).utcoffset():
                    counts["ambiguous_or_nonexistent_local_time"] += 1
                    continue
                if times[t] != 1 or not (0 <= label < float("inf") and label.is_integer()):
                    counts["invalid_or_nonunique_image_label"] += 1
                    continue
                if t in published:
                    counts["intermediate_label_matches" if published[t] == label
                           else "intermediate_label_differs"] += 1
                if t not in features:
                    counts["no_exact_sensor_timestamp"] += 1
                    continue
                # Exclude t and t-5. Even if raw sensor rows denote starts of
                # five-minute bins, the last bin ends before the target time.
                history = [features.get(t - 5 * k) for k in range(WINDOW + 1, 1, -1)]
                if None in history or not valid_history(history, [5] * WINDOW, 5):
                    counts["missing_or_invalid_history"] += 1
                    continue
                windows.append(dict(id=t, time=t, x=history, gaps=[5.] * WINDOW,
                                    y=int(label > 0), next_x=None))
            counts.update(eligible=len(windows), occupied=sum(r["y"] for r in windows),
                          sensor_without_raw_label=len(features.keys() - seen))
            results[room] = (sorted(windows, key=lambda r: r["id"]), dict(counts))
    return results
