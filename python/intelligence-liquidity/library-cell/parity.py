"""Compare the exact HTML replay script against PyTorch, including edited gaps."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import torch

from occupancy import load_model, predict

HERE = Path(__file__).resolve().parent
checkpoint_bytes = (HERE / "checkpoint.json").read_bytes()
checkpoint = json.loads(checkpoint_bytes)
results = json.loads((HERE / "results.json").read_text())
assert hashlib.sha256(checkpoint_bytes).hexdigest() == results["checkpoint_sha256"]
model = load_model("ltc", checkpoint["runs"]["41"]["ltc"])
mean = torch.tensor(checkpoint["mean"])
scale = torch.tensor(checkpoint["scale"])
cases, expected = [], []
for i, example in enumerate(results["examples"]):
    x = (torch.tensor([row[1] for row in example["rows"]]) - mean) / scale
    for override in (None, 0.25, 4.0):
        elapsed = example["elapsed"].copy()
        if override is not None:
            elapsed[-1] = override
        value = predict(model, x[None], torch.tensor(elapsed)[None]).sigmoid().item()
        if override is None:
            assert abs(value - example["probability"]) < 1e-6
        cases.append([i, override])
        expected.append(value)

html = (HERE / "replay.html").read_text()
script = re.search(r"<script>\n(.*?)</script>", html, re.S).group(1)
# Exercise the actual script and its input handlers in Node with a minimal DOM.
dom = """
const elements = new Map();
function element() {
  return {
    value: "0",
    textContent: "",
    events: {},
    get valueAsNumber() { return Number(this.value); },
    get validity() { return {valid: this.value !== "" && this.value >= 0.001 && this.value <= 60}; },
    add() {},
    replaceChildren() {},
    insertCell() { return {}; },
    addEventListener(name, fn) { this.events[name] = fn; },
  };
}
const document = {
  querySelector(id) {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  },
  createElement: element,
};
class Option {}
"""
checks = """
const observedScores = cases.map(([i, override]) => {
  select.value = i;
  select.events.change();
  if (override !== null) {
    gap.value = override;
    gap.events.input();
  }
  const sample = examples[i];
  const elapsed = sample.elapsed.with(-1, gap.valueAsNumber);
  const p = probability(model, sample.rows, elapsed);
  if (score.textContent !== (100 * p).toFixed(4) + "%") throw new Error("Unexpected visible score");
  return p;
});
gap.value = "";
gap.events.input();
if (!score.textContent.startsWith("Enter a gap")) throw new Error("Empty input accepted");
console.log(JSON.stringify({
  probabilities: observedScores,
  checkpoint: model.checkpoint_sha256,
  weights: model.runs["41"].ltc.weights,
  version: process.version,
}));
"""
run = subprocess.run(
    ["node", "-"], input=dom + script + "\nconst cases = " + json.dumps(cases) + ";\n" + checks,
    text=True, capture_output=True, check=True,
)
report = json.loads(run.stdout)
assert report["weights"] == checkpoint["runs"]["41"]["ltc"]["weights"]
assert report["checkpoint"] == results["checkpoint_sha256"]
errors = [abs(a-b) for a, b in zip(report["probabilities"], expected, strict=True)]
assert max(errors) <= 1e-6
print(json.dumps({
    "runner": report["version"],
    "max_absolute_probability_error": max(errors),
    "tolerance": 1e-6,
    "cases": [
        {"example": index, "edited_final_minutes": gap, "pytorch": p, "javascript": j}
        for (index, gap), p, j in zip(cases, expected, report["probabilities"], strict=True)
    ],
}, indent=2))
