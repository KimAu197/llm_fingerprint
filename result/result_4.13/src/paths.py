"""Directory layout: result_4.13/src/*.py -> outputs in result_4.13/."""

from pathlib import Path


def result_root() -> Path:
    return Path(__file__).resolve().parent.parent
