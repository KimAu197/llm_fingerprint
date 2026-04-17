"""
run_two_target_cross_overlap.py

Compute pairwise overlap only for two target models vs every model in --csv_path
(same rules as the main Phase-2 cross pass). Loads fingerprints from JSON; any
model in the CSV that is missing or empty in JSON gets Phase-1 generation first.

Output:
  - cross_two_targets.csv: 2 rows x N overlap values, same layout as overlap_matrix.csv
    (first row header empty + model ids; each row is overlap(target_i, model_j)).
  - cross_two_targets_column_slice.csv: N rows with overlap(model_j, target_i) for pasting
    into the two target columns of a square matrix (pipeline uses distinct (i,j) and (j,i)).

Example:
  cd watermarking/test
  python run_two_target_cross_overlap.py \\
    --csv_path /path/model_200.csv \\
    --fingerprints_file /path/fingerprints.json \\
    --output_dir /path/out \\
    --target_models "google/gemma-2-2b,Qwen/Qwen2.5-Coder-0.5B" \\
    --gpu_ids 0 \\
    --batch_size_bottomk 24
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pairwise_overlap_matrix import (
    format_time,
    load_model_list,
    print_progress,
    setup_run_logging,
)
from run_pairwise_overlap_phase2_retry import (
    _dedup_all_prompts,
    _matrix_cell_mi_mj,
    _run_one_model_bottomk,
    build_phase2_namespace,
    phase1_regenerate_fingerprints_for_models,
)
from utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross overlap for exactly two targets vs full CSV list; write 2xN CSV"
    )
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--fingerprints_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Default: <output_dir>/cross_two_targets.csv",
    )
    p.add_argument(
        "--output_column_csv",
        type=str,
        default=None,
        help=(
            "Default: <output_dir>/cross_two_targets_column_slice.csv — N rows with "
            "overlap(row_model, target_k) to paste into the two target columns of a square matrix."
        ),
    )
    p.add_argument(
        "--target_models",
        type=str,
        required=True,
        help='Exactly two comma-separated Hugging Face ids, e.g. "org/a,org/b"',
    )
    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_fingerprints", type=int, default=5)
    p.add_argument("--fingerprint_length", type=int, default=64)
    p.add_argument("--k_bottom_sampling", type=int, default=50)
    p.add_argument("--bottom_k_vocab", type=int, default=2000)
    p.add_argument("--batch_size_bottomk", type=int, default=64)
    p.add_argument("--cuda_device_reset_each_model", action="store_true")
    p.add_argument("--no_cuda_device_reset_on_error", action="store_true")
    p.add_argument("--cuda_device_reset_after_oom", action="store_true")
    p.add_argument("--no_live_overlap_matrix", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    gpu_id = int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0
    print(f"[INFO] Using cuda:{gpu_id} (only first id from --gpu_ids is used)")

    targets = [s.strip() for s in args.target_models.split(",") if s.strip()]
    if len(targets) != 2:
        raise SystemExit("Provide exactly two ids in --target_models (comma-separated).")
    if targets[0] == targets[1]:
        raise SystemExit("--target_models must be two distinct models.")

    models = load_model_list(args.csv_path)
    if len(models) < 2:
        raise SystemExit("--csv_path must list at least two models.")
    for t in targets:
        if t not in models:
            raise SystemExit(f"Target not in CSV: {t}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)

    fp_path = Path(args.fingerprints_file)
    if fp_path.is_file():
        with open(fp_path, "r", encoding="utf-8") as f:
            all_fps: Dict[str, List[str]] = json.load(f)
    else:
        all_fps = {}
        print(f"[INFO] No fingerprints file at {fp_path}; starting empty.")

    missing_fp = [m for m in models if m not in all_fps or len(all_fps.get(m, [])) == 0]
    if missing_fp:
        print(
            f"[INFO] {len(missing_fp)} model(s) missing fingerprints; Phase1 for: {missing_fp}"
        )
        phase1_ns = build_phase2_namespace(args)
        all_fps = phase1_regenerate_fingerprints_for_models(
            missing_fp,
            all_fps,
            phase1_ns,
            gpu_id,
            fp_path,
            output_dir,
        )

    for t in targets:
        if t not in all_fps or not all_fps[t]:
            raise SystemExit(f"Fingerprints still missing for target after Phase 1: {t}")

    phase2_ns = build_phase2_namespace(args)
    all_prompts = _dedup_all_prompts(models, all_fps)
    n_prompts = len(all_prompts)
    print(f"[INFO] Unique prompts (dedup from JSON): {n_prompts}")
    print(f"[INFO] Targets: {targets}")

    t0, t1 = targets[0], targets[1]
    idx0, idx1 = models.index(t0), models.index(t1)

    block = np.full((2, len(models)), -1.0, dtype=np.float64)
    block[0, idx0] = 1.0
    block[1, idx1] = 1.0

    pinned: Dict[str, Dict[str, List[int]]] = {}
    t_wall = time.time()

    for ti, name in enumerate(targets):
        print(f"\n[bottom-k target {ti+1}/2] {name}")
        pinned[name] = _run_one_model_bottomk(name, all_prompts, phase2_ns, gpu_id)
        print(f"  Done ({n_prompts} prompts).")

    block[0, idx1] = _matrix_cell_mi_mj(
        t0, t1, all_fps, pinned[t0], pinned[t1]
    )
    block[1, idx0] = _matrix_cell_mi_mj(
        t1, t0, all_fps, pinned[t1], pinned[t0]
    )

    # overlap(row_model, target) for each row index — differs from block (target, col) in general
    col_block = np.full((len(models), 2), -1.0, dtype=np.float64)
    col_block[idx0, 0] = 1.0
    col_block[idx1, 1] = 1.0
    col_block[idx0, 1] = block[0, idx1]
    col_block[idx1, 0] = block[1, idx0]

    others = [m for m in models if m not in {t0, t1}]
    print(f"\n[cross] {len(others)} other model(s) (load once each, update both target rows)")
    cross_times: List[float] = []
    for step, mj in enumerate(others):
        print(f"\n[{step+1}/{len(others)}] vs {mj}")
        mt = time.time()
        cache_j = _run_one_model_bottomk(mj, all_prompts, phase2_ns, gpu_id)
        j = models.index(mj)
        block[0, j] = _matrix_cell_mi_mj(t0, mj, all_fps, pinned[t0], cache_j)
        block[1, j] = _matrix_cell_mi_mj(t1, mj, all_fps, pinned[t1], cache_j)
        col_block[j, 0] = _matrix_cell_mi_mj(mj, t0, all_fps, cache_j, pinned[t0])
        col_block[j, 1] = _matrix_cell_mi_mj(mj, t1, all_fps, cache_j, pinned[t1])
        cross_times.append(time.time() - mt)
        if cross_times:
            print_progress(step, len(others), cross_times, t_wall)

    out_csv = Path(args.output_csv) if args.output_csv else output_dir / "cross_two_targets.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + models)
        for ti, t in enumerate(targets):
            w.writerow([t] + [f"{block[ti, j]:.6f}" for j in range(len(models))])

    out_col = (
        Path(args.output_column_csv)
        if args.output_column_csv
        else output_dir / "cross_two_targets_column_slice.csv"
    )
    with open(out_col, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_id",
                f"overlap_row_vs_{t0}",
                f"overlap_row_vs_{t1}",
            ]
        )
        for j, mj in enumerate(models):
            w.writerow(
                [
                    mj,
                    f"{col_block[j, 0]:.6f}",
                    f"{col_block[j, 1]:.6f}",
                ]
            )

    meta = {
        "script": "run_two_target_cross_overlap.py",
        "csv_path": str(Path(args.csv_path).resolve()),
        "fingerprints_file": str(fp_path.resolve()),
        "targets_order": targets,
        "models_column_order": models,
        "output_csv": str(out_csv.resolve()),
        "output_column_csv": str(out_col.resolve()),
        "target_row_indices_in_csv_order": [idx0, idx1],
        "seed": args.seed,
        "note": (
            "cross_two_targets.csv: two rows are overlap(target_i, model_j) (target as mi in "
            "the main pipeline). cross_two_targets_column_slice.csv: for each row model j, "
            "overlap(model_j, target_i) for pasting into the target columns of a square matrix."
        ),
        "timestamp": datetime.now().isoformat(),
        "time_seconds": time.time() - t_wall,
    }
    with open(output_dir / "cross_two_targets_meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\n[INFO] Wrote {out_csv.resolve()}")
    print(f"[INFO] Wrote {out_col.resolve()} (column slice for pasting)")
    print(f"[INFO] Metadata -> {output_dir / 'cross_two_targets_meta.json'}")
    print(f"[INFO] Finished in {format_time(time.time() - t_wall)}")


if __name__ == "__main__":
    main()
