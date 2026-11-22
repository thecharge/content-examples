"""Run the frozen experiment from a checkout using explicit data paths."""
import argparse
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("check", "fit", "score"))
    cli.add_argument("work", type=Path)
    cli.add_argument("--data", type=Path, default=Path("data"))
    cli.add_argument("--model", default="v2")
    cli.add_argument("--seed", type=int, default=41)
    cli.add_argument("--budget", type=int, default=100)
    config = cli.parse_args()
    here = Path(__file__).resolve().parent
    earlier = here.parent / "attention-and-reward"
    work, archives = config.work.resolve(), config.data.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="liquid-check-") as temp:
        staged = Path(temp) / here.name
        staged.mkdir()
        for file in ("adaptive.py", "policy.py", "experiment.py", "verify.py", "context_checks.py"):
            shutil.copyfile(here / file, staged / file)
        (staged.parent / "attention-benchmark").symlink_to(earlier, target_is_directory=True)
        sys.path.insert(0, str(staged))
        experiment = importlib.import_module("experiment")
        experiment.INPUT = archives
        experiment.UCI = archives / "occupancy-fresh.zip"
        experiment.V1 = here / "prior/adaptive.py"
        experiment.torch.set_num_threads(1)
        experiment.torch.set_num_interop_threads(1)
        if config.mode == "fit":
            experiment.train(config.model, config.seed, config.budget, work)
        elif config.mode == "check":
            import runpy
            runpy.run_path(str(staged / "verify.py"), run_name="__main__")
        else:
            record = json.loads((here / "freeze-content.json").read_text())
            frozen = record["source_sha256"]
            actual = experiment.freeze_sources()
            for file, digest in actual.items():
                path = Path(file)
                if path == experiment.UCI:
                    key = "sndv-ltc-data/occupancy-fresh.zip"
                elif path.parent == experiment.V1.parent:
                    key = f"adaptive-v1-refuted/{path.name}"
                else:
                    key = f"{path.parent.name}/{path.name}"
                if digest != frozen[key]:
                    raise ValueError(f"Changed frozen source: {key}")
            record["source_sha256"] = actual
            (staged / "freeze.json").write_text(json.dumps(record))
            experiment.score(work)


if __name__ == "__main__":
    main()
