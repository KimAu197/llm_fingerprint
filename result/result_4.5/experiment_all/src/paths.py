"""experiment_all/src/*.py -> outputs default to parent experiment_all/."""

from pathlib import Path


def experiment_root() -> Path:
    return Path(__file__).resolve().parent.parent
