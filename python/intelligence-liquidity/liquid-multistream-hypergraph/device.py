"""Report actual accelerator access; never silently substitute CPU for ROCm."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

import torch


def probe():
    pci = shutil.which("lspci")
    adapters = []
    if pci:
        result = subprocess.run(
            [pci, "-nn"], capture_output=True, text=True, check=True
        )
        adapters = [
            r for r in result.stdout.splitlines() if "VGA" in r or "Display" in r
        ]
    return dict(
        torch=torch.__version__,
        hip=torch.version.hip,
        compute_devices=Path("/dev/kfd").exists(),
        render_devices=Path("/dev/dri").exists(),
        accelerator_available=torch.cuda.is_available(),
        devices=[
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        adapters=adapters,
        kernel=platform.release(),
        python=platform.python_version(),
        rocminfo=shutil.which("rocminfo"),
        model_runtime="cpu",
        gpu_model_execution_verified=False,
    )


def require(requested):
    if requested == "cpu":
        return torch.device("cpu")
    report = probe()
    if requested != "rocm":
        raise ValueError("Choose cpu or rocm")
    if (
        not report["hip"]
        or not report["accelerator_available"]
        or not report["compute_devices"]
        or not report["render_devices"]
    ):
        raise RuntimeError("ROCm unavailable: " + json.dumps(report, sort_keys=True))
    raise RuntimeError(
        "ROCm is visible, but this inherited CPU runtime needs a separately verified device port"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", choices=("cpu", "rocm"))
    args = parser.parse_args()
    if args.require:
        require(args.require)
    print(json.dumps(probe(), indent=2))
