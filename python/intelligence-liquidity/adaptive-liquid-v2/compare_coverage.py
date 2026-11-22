"""Compare direct and revised predictions on the same accepted windows."""
import argparse
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def balanced_accuracy(logits, labels, cutoff, covered, count_abstentions):
    class_scores = []
    for label in (False, True):
        scores = [
            accepted and (logit >= cutoff) == actual
            for logit, actual, accepted in zip(logits, labels, covered, strict=True)
            if actual == label and (count_abstentions or accepted)
        ]
        class_scores.append(mean(scores))
    return mean(class_scores)


def compare(work):
    membership = read(work / "aalto-membership.json")
    labels = membership["labels"]
    files = sorted(work.glob("v2-*-aalto.json"))
    if len(files) != 9:
        raise ValueError(f"Expected nine base checkpoints, found {len(files)}.")
    results = {}
    for file in files:
        measured = read(file)
        if measured["ids"] != membership["ids"]:
            raise ValueError(f"Different evaluation membership in {file.name}.")
        cutoff = read(file.with_name(file.name.replace("-aalto.json", ".json")))["cutoff"]
        direct, fitted = measured["routes"]["base"], measured["routes"]["0.1"]
        direct_all = balanced_accuracy(direct["logits"], labels, cutoff, direct["covered"], True)
        if abs(direct_all - measured["summary"]["base"]["selected"]["effective_balanced"]) > 1e-8:
            raise ValueError(f"Stored scores disagree with raw data in {file.name}.")
        shared = fitted["covered"]
        results[file.stem] = {
            "direct_all": direct_all,
            "direct_matched_all": balanced_accuracy(direct["logits"], labels, cutoff, shared, True),
            "revised_matched_all": balanced_accuracy(fitted["logits"], labels, cutoff, shared, True),
            "direct_matched_answered": balanced_accuracy(direct["logits"], labels, cutoff, shared, False),
            "revised_matched_answered": balanced_accuracy(fitted["logits"], labels, cutoff, shared, False),
        }
    averages = {
        key: mean(row[key] for row in results.values())
        for key in next(iter(results.values()))
    }
    return {"runs": results, "mean": averages}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(compare(arguments.work), indent=2))
