"""Point the frozen experiment at a reader's local copies of publisher files."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"),
                        help="Directory with office.zip, apartments.csv, graz.zip and six-buildings.zip")
    options, command = parser.parse_known_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "attention-and-reward"))
    import experiment

    data = options.data.resolve()
    experiment.INPUT = data
    experiment.GRAZ = data / "graz.zip"
    experiment.SOURCES = {
        "occupancy": data / "office.zip",
        "heat": data / "six-buildings.zip",
    }
    sys.argv[1:] = command
    experiment.main()


if __name__ == "__main__":
    main()
