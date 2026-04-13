"""
Run Phase 2 (bottom-k caches) and Phase 3 (overlap matrix) only.

Expects fingerprints.json in the same format as Phase 1 output:
    { "org/model-name": ["fingerprint string", ...], ... }

Usage:
    python run_phase2_from_fingerprints.py \\
        --fingerprints_file /path/to/fingerprints.json \\
        --csv_path /path/to/models.csv \\
        --output_dir ./phase2_out \\
        --gpu_ids 0

    # Without CSV: matrix order is sorted model ids (all keys in JSON).
    python run_phase2_from_fingerprints.py \\
        --fingerprints_file fingerprints.json \\
        --output_dir ./phase2_out \\
        --gpu_ids 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pairwise_overlap_matrix import (
    format_time,
    load_model_list,
    phase2_compute_caches,
    phase3_build_matrix,
    setup_run_logging,
)
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2+3 only: bottom-k caches and overlap matrix from fingerprints.json"
    )
    p.add_argument(
        "--fingerprints_file",
        type=str,
        required=True,
        help="Path to fingerprints.json (dict: model_id -> list of prompt strings)",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional CSV with model_id column; defines row/column order (same as full pipeline)",
    )
    p.add_argument("--output_dir", type=str, default="./phase2_from_fingerprints_out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bottom_k_vocab", type=int, default=2000)
    p.add_argument("--gpu_ids", type=str, default=None, help="GPU id, e.g. '0' or '2'")
    p.add_argument(
        "--batch_size_bottomk",
        type=int,
        default=32,
        help="Batch size for bottom-k (reduce on OOM)",
    )
    # Metadata / argparse.Namespace compatibility for phase2_compute_caches
    p.add_argument("--num_fingerprints", type=int, default=5)
    p.add_argument("--k_bottom_sampling", type=int, default=50)
    p.add_argument("--fingerprint_length", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    t_start = time.time()

    gpu_id = int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0

    fp_path = Path(args.fingerprints_file)
    if not fp_path.is_file():
        raise SystemExit(f"Fingerprints file not found: {fp_path}")

    with open(fp_path, "r", encoding="utf-8") as f:
        all_fps: dict[str, list[str]] = json.load(f)

    if args.csv_path:
        models = load_model_list(args.csv_path)
        missing = [m for m in models if m not in all_fps]
        if missing:
            print(
                f"[WARNING] {len(missing)} models in CSV have no entry in fingerprints.json"
            )
            for m in missing[:5]:
                print(f"  - {m}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
        extra = [k for k in all_fps if k not in models]
        if extra:
            print(
                f"[INFO] {len(extra)} models in JSON are not in CSV; "
                "they will be omitted from the matrix."
            )
    else:
        models = sorted(all_fps.keys())
        print("[INFO] No --csv_path: using sorted fingerprint keys as matrix order.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)

    print("=" * 70)
    print("PHASE 2+3 ONLY (from fingerprints.json)")
    print("=" * 70)
    print(f"Fingerprints file: {fp_path.resolve()}")
    print(f"Models in matrix:  {len(models)}")
    print(f"JSON entries:      {len(all_fps)}")
    print(f"Bottom-k vocab:    {args.bottom_k_vocab}")
    print(f"GPU:               cuda:{gpu_id}")
    print(f"Output:            {output_dir.resolve()}")
    print()

    caches = phase2_compute_caches(models, all_fps, args, gpu_id, output_dir)
    phase3_build_matrix(models, all_fps, caches, output_dir)

    n_fp = len(next(iter(all_fps.values()))) if all_fps else 0
    metadata = {
        "phase": "phase2_and_3_only",
        "fingerprints_source": str(fp_path.resolve()),
        "models": models,
        "num_models": len(models),
        "num_fingerprint_models_in_json": len(all_fps),
        "fingerprints_per_model_inferred": n_fp,
        "bottom_k_vocab": args.bottom_k_vocab,
        "batch_size_bottomk": args.batch_size_bottomk,
        "seed": args.seed,
        "gpu_id": gpu_id,
        "total_time_seconds": time.time() - t_start,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")
    print(f"Total time: {format_time(time.time() - t_start)}")
    print(f"Output:     {output_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
