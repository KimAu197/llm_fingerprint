#!/usr/bin/env python3
"""Regenerate result_4.13 artifacts: HF lineage, Tukey fence, strict/loose metrics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    scripts = [
        "build_lineage_from_hf_overlap_csv.py",
        "compute_tukey_fence_eval.py",
        "compute_tukey_outliers_strict_metrics.py",
    ]
    for name in scripts:
        subprocess.check_call([sys.executable, str(HERE / name)], cwd=str(HERE))


if __name__ == "__main__":
    main()
