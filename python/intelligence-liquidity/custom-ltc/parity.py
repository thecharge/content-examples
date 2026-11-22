import json
from pathlib import Path

import torch

import model
import replay


def main():
    root = Path(__file__).resolve().parent
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    results = json.loads((root / "results.json").read_text())
    entry = checkpoint["runs"]["41"]["ltc"]
    assert replay.WEIGHTS == entry["weights"]
    assert replay.MEAN == checkpoint["mean"] and replay.SCALE == checkpoint["scale"]
    first = results["examples"][0]
    assert replay.ROWS == first["rows"] and replay.GAPS == first["elapsed"]
    assert len((root / "replay.py").read_bytes()) < 16_384
    net = model.load_model("ltc", entry)
    mean, scale = map(torch.tensor, (replay.MEAN, replay.SCALE))
    cases = []
    for example in results["examples"]:
        for edit in (None, 2, 4):
            gaps = example["elapsed"].copy()
            if edit is not None:
                gaps[-1] = edit
            rows = example["rows"]
            x = (torch.tensor([[row[1] for row in rows]]) - mean) / scale
            g = torch.tensor([gaps])
            expected = model.predict(net, x, g).sigmoid().item()
            observed = replay.probability(rows, gaps)
            error = abs(expected - observed)
            assert error <= 1e-6 and (expected >= 0.5) == (observed >= 0.5)
            cases.append(dict(timestamp=rows[-1][0], cadence=example["mode"],
                              final_gap=gaps[-1], torch=expected, python=observed, error=error))
    print(json.dumps(dict(cases=cases, maximum_error=max(item["error"] for item in cases)),
                     indent=2))


if __name__ == "__main__":
    main()
