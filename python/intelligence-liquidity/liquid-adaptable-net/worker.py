"""Run this experiment in a Linux worker with hard OS resource limits.

RLIMIT_AS bounds address space, not just tensors or resident memory. A failed
allocation terminates this worker; restore its last durable checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "workflow.py",
    "compare.py",
    "verify.py",
    "demonstrate.py",
    "evaluate.py",
    "reporting/profile_model.py",
}


def impose(configuration):
    for key in ("memory_mib", "cpu_seconds", "wall_seconds", "file_mib"):
        if not isinstance(configuration[key], int) or configuration[key] < 1:
            raise ValueError("Worker limits must be positive integers")
    for kind, limit in (
        (resource.RLIMIT_AS, configuration["memory_mib"] * 1024**2),
        (resource.RLIMIT_CPU, configuration["cpu_seconds"]),
        (resource.RLIMIT_FSIZE, configuration["file_mib"] * 1024**2),
        (resource.RLIMIT_CORE, 0),
    ):
        _, hard = resource.getrlimit(kind)
        limit = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(kind, (limit, limit))


def child(configuration, script, arguments):
    impose(configuration)
    if script == "allocation-test":
        try:
            bytearray(configuration["memory_mib"] * 1024**2)
        except MemoryError:
            print("OS rejected allocation beyond the address-space ceiling")
            return
        raise AssertionError("Memory ceiling did not reject allocation")
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / script), *arguments]
    runpy.run_path(str(ROOT / script), run_name="__main__")


def launch(configuration, script, arguments, receipt=None):
    if script not in SCRIPTS | {"allocation-test"}:
        raise ValueError("Worker only runs this experiment's entry points")
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    started = time.monotonic()
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            json.dumps(configuration),
            script,
            *arguments,
        ],
        env=environment,
        start_new_session=True,
    )
    timed_out = False
    try:
        code = process.wait(timeout=configuration["wall_seconds"])
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    except BaseException:
        # Cancellation must not leave a detached training worker running.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = dict(
        configuration=configuration,
        script=script,
        arguments=arguments,
        exit_code=code,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
        children_peak_rss_kib=usage.ru_maxrss,
    )
    if receipt is not None:
        path = Path(receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    return code, record


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        try:
            child(json.loads(sys.argv[2]), sys.argv[3], sys.argv[4:])
        except MemoryError:
            print(
                "Worker memory ceiling reached; recover the last checkpoint",
                file=sys.stderr,
            )
            sys.exit(42)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--configuration", type=Path, default=ROOT / "worker-configuration.json"
        )
        parser.add_argument("--receipt", type=Path)
        parser.add_argument("script", choices=sorted(SCRIPTS | {"allocation-test"}))
        parser.add_argument("arguments", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        configuration = json.loads(args.configuration.read_text())
        code, record = launch(configuration, args.script, args.arguments, args.receipt)
        print(json.dumps(record), flush=True)
        sys.exit(code if code >= 0 else 128 - code)
