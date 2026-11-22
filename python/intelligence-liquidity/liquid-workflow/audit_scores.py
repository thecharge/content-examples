"""Verify saved correctness counts and emit the values used in the article."""
from collections import defaultdict
import json
from math import isclose
from pathlib import Path
from statistics import mean


def count_scores(row):
    occupied, empty = row["occupied"], row["empty"]
    occupied_right = occupied - row["missed"] - row["withheld_occupied"]
    empty_right = empty - row["false_alarm"] - row["withheld_empty"]
    withheld = row["withheld_occupied"] + row["withheld_empty"]
    assert 0 <= occupied_right <= occupied and 0 <= empty_right <= empty, row
    assert withheld == row["undecided"], row
    caught = occupied_right / occupied
    accuracy = (occupied_right + empty_right) / (occupied + empty)
    equal_weight = (caught + empty_right / empty) / 2
    assert isclose(accuracy, row["accuracy"], abs_tol=1e-7), row
    assert isclose(equal_weight, row["equal_weight"], abs_tol=1e-7), row
    return dict(accuracy=accuracy, equal_weight=equal_weight, caught_occupied=caught,
                withheld=withheld / (occupied + empty))


def audit(evidence):
    def read(name):
        return json.loads((evidence / f"{name}.json").read_text())

    groups = defaultdict(list)
    rows_checked = 0
    for file in ("scores", "reference-ltc", "xgboost"):
        report = read(file)
        rows = report["rows"] if file == "scores" else (r for run in report["runs"] for r in run["rows"])
        for row in rows:
            counts = count_scores(row)
            rows_checked += 1
            if file == "scores" and not row["checkpoint"].endswith("-100.pt"):
                continue
            groups[row["site"], row["variant"]].append(counts)
    output = {}
    for (room, model), runs in sorted(groups.items()):
        assert len(runs) == 3, (room, model)
        output.setdefault(room, {})[model] = {
            metric: round(100 * mean(r[metric] for r in runs), 2)
            for metric in runs[0]
        }
    return {"rows_checked": rows_checked, "percent_by_site": output}


if __name__ == "__main__":
    print(json.dumps(audit(Path(__file__).resolve().parent / "evidence"), indent=2))
