"""
run_two_target_cross_overlap.py

(1) Two-target mode: pairwise overlap for exactly two target models vs every model
in --csv_path (same rules as the main Phase-2 cross pass). Loads fingerprints from
JSON; any model in the CSV that is missing or empty in JSON gets Phase-1 first.

(2) Bipartite mode: one anchor model vs a list of partner models only (no full-table
cross). Computes overlap(anchor, partner) as a 1xK row and overlap(partner, anchor)
as a Kx1 column (distinct in general, same as the main matrix).

Output two-target:
  - cross_two_targets.csv: 2 rows x N overlap values, same layout as overlap_matrix.csv
  - cross_two_targets_column_slice.csv: N rows with overlap(model_j, target_i) for pasting
    into the two target columns of a square matrix

Output bipartite (default filenames under --output_dir):
  - cross_bipartite_1xK.csv: one row, K columns — overlap(anchor, partner_i)
  - cross_bipartite_Kx1.csv: K rows, one value column — overlap(partner_i, anchor)

Example (two targets vs full list):
  cd watermarking/test
  python run_two_target_cross_overlap.py \\
    --csv_path /path/model_200.csv \\
    --fingerprints_file /path/fingerprints.json \\
    --output_dir /path/out \\
    --target_models "google/gemma-2-2b,Qwen/Qwen2.5-Coder-0.5B" \\
    --gpu_ids 0 \\
    --batch_size_bottomk 24

Example (Apertus vs 11 bnb, 1x11 and 11x1 only; --csv_path is still the full model list
for the same global fingerprint / prompt pool as the big matrix run):
  python run_two_target_cross_overlap.py \\
    --csv_path /path/full_198.csv \\
    --fingerprints_file /path/fingerprints.json \\
    --output_dir /path/out \\
    --bipartite_anchor "swiss-ai/Apertus-8B-2509" \\
    --bipartite_partners_csv data/csv_unsloth_requested_direct_load.csv \\
    --gpu_ids 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set

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


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _load_bipartite_partners_from_args(args: argparse.Namespace) -> List[str]:
    raw_csv = getattr(args, "bipartite_partners_csv", None)
    raw_comma = getattr(args, "bipartite_partners", None)
    if raw_csv and str(raw_csv).strip():
        p = Path(raw_csv)
        if not p.is_file():
            raise SystemExit(f"Bipartite partners CSV not found: {p}")
        return _dedupe_preserve_order(load_model_list(str(p)))
    if raw_comma and str(raw_comma).strip():
        return _dedupe_preserve_order(
            [s.strip() for s in str(raw_comma).split(",") if s.strip()]
        )
    raise SystemExit(
        "Bipartite mode needs --bipartite_partners_csv and/or --bipartite_partners"
    )


def run_bipartite_mode(args: argparse.Namespace) -> None:
    anchor = (args.bipartite_anchor or "").strip()
    if not anchor:
        raise SystemExit("--bipartite_anchor must be non-empty")

    partners = _load_bipartite_partners_from_args(args)
    if not partners:
        raise SystemExit("Bipartite partners list is empty")
    if len(partners) != 11:
        print(f"[INFO] Bipartite partner count = {len(partners)} (expected 11 for the bnb list).")

    if anchor in partners:
        print(f"[INFO] Dropping anchor from partners list: {anchor}")
        partners = [p for p in partners if p != anchor]

    models = load_model_list(args.csv_path)
    if not models:
        raise SystemExit("Empty --csv_path model list")
    for m in [anchor] + partners:
        if m not in models:
            raise SystemExit(f"Model not in --csv_path: {m!r}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)
    set_seed(args.seed)
    gpu_id = int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0
    print(f"[INFO] Bipartite mode: cuda:{gpu_id}")
    print(f"[INFO] anchor={anchor}")
    print(f"[INFO] partners ({len(partners)}): {partners}")

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

    needed: Set[str] = {anchor, *partners}
    for m in needed:
        if m not in all_fps or not all_fps[m]:
            raise SystemExit(f"Fingerprints still missing after Phase 1: {m!r}")

    phase2_ns = build_phase2_namespace(args)
    all_prompts = _dedup_all_prompts(models, all_fps)
    n_prompts = len(all_prompts)
    print(f"[INFO] Unique prompts (dedup from full CSV + JSON): {n_prompts}")

    pinned: Dict[str, Dict[str, List[int]]] = {}
    t_wall = time.time()

    print(f"\n[bottom-k] anchor: {anchor}")
    pinned[anchor] = _run_one_model_bottomk(anchor, all_prompts, phase2_ns, gpu_id)
    print(f"  Done ({n_prompts} prompts).")

    for i, p in enumerate(partners):
        print(f"\n[bottom-k] partner {i+1}/{len(partners)}] {p}")
        pinned[p] = _run_one_model_bottomk(p, all_prompts, phase2_ns, gpu_id)
        print(f"  Done ({n_prompts} prompts).")

    o_ap_rows: List[float] = []
    o_pa_rows: List[float] = []
    for p in partners:
        o_ap_rows.append(
            _matrix_cell_mi_mj(anchor, p, all_fps, pinned[anchor], pinned[p])
        )
        o_pa_rows.append(
            _matrix_cell_mi_mj(p, anchor, all_fps, pinned[p], pinned[anchor])
        )

    out_1xk = (
        Path(args.bipartite_output_1xK)
        if args.bipartite_output_1xK
        else output_dir / f"cross_bipartite_1x{len(partners)}.csv"
    )
    with open(out_1xk, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + partners)
        w.writerow([anchor] + [f"{v:.6f}" for v in o_ap_rows])

    out_kx1 = (
        Path(args.bipartite_output_kx1)
        if args.bipartite_output_kx1
        else output_dir / f"cross_bipartite_{len(partners)}x1.csv"
    )
    with open(out_kx1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model_id",
                "overlap(partner,anchor)",
            ]
        )
        for p, v in zip(partners, o_pa_rows):
            w.writerow([p, f"{v:.6f}"])

    meta = {
        "script": "run_two_target_cross_overlap.py",
        "mode": "bipartite",
        "csv_path": str(Path(args.csv_path).resolve()),
        "fingerprints_file": str(fp_path.resolve()),
        "bipartite_anchor": anchor,
        "bipartite_partners": partners,
        "notes": {
            f"1x{len(partners)}": "row is overlap(anchor, partner); columns = partners",
            f"{len(partners)}x1": "rows = partners, value = overlap(partner, anchor)",
        },
        "output_1xK": str(out_1xk.resolve()),
        "output_Kx1": str(out_kx1.resolve()),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "time_seconds": time.time() - t_wall,
    }
    with open(output_dir / "cross_bipartite_meta.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\n[INFO] Wrote {out_1xk} (1x{len(partners)} overlap(anchor, partner))")
    print(
        f"[INFO] Wrote {out_kx1} "
        f"({len(partners)}x1 overlap(partner, anchor))"
    )
    print(f"[INFO] Metadata -> {output_dir / 'cross_bipartite_meta.json'}")
    print(f"[INFO] Finished in {format_time(time.time() - t_wall)}")


def run_two_target_mode(args: argparse.Namespace) -> None:
    if not (args.target_models and str(args.target_models).strip()):
        raise SystemExit("run_two_target_mode: missing --target_models")
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
    set_seed(args.seed)
    gpu_id = int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0
    print(f"[INFO] Two-target mode: cuda:{gpu_id} (only first id from --gpu_ids is used)")

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
        "mode": "two_target",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross overlap: (A) two targets vs every model in CSV, or (B) one anchor vs many "
            "partners (1xK and Kx1) only"
        )
    )
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--fingerprints_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--target_models",
        type=str,
        default=None,
        help=(
            'Two-target mode: exactly two comma-separated Hugging Face ids. Omitted or empty '
            "if --bipartite_anchor is set."
        ),
    )
    p.add_argument(
        "--bipartite_anchor",
        type=str,
        default=None,
        help="Bipartite mode: single model id; pair with --bipartite_partners_csv or --bipartite_partners",
    )
    p.add_argument(
        "--bipartite_partners_csv",
        type=str,
        default=None,
        help="CSV with header model_id: partner models (e.g. 11 bnb ids).",
    )
    p.add_argument(
        "--bipartite_partners",
        type=str,
        default=None,
        help="Bipartite mode: comma-separated partner ids (alternative to partners_csv)",
    )
    p.add_argument(
        "--bipartite_output_1xK",
        type=str,
        default=None,
        help="Bipartite: output path for 1xK row (default cross_bipartite_1xK.csv in output_dir)",
    )
    p.add_argument(
        "--bipartite_output_kx1",
        type=str,
        default=None,
        help="Bipartite: output path for Kx1 table (default cross_bipartite_Kx1.csv)",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Two-target: default <output_dir>/cross_two_targets.csv",
    )
    p.add_argument(
        "--output_column_csv",
        type=str,
        default=None,
        help="Two-target: default <output_dir>/cross_two_targets_column_slice.csv",
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
    p.add_argument(
        "--fourbit",
        action="store_true",
        help="Load every model in 4-bit. Default: only ids containing bnb-4bit use 4-bit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    is_bipartite = bool(
        (args.bipartite_anchor and str(args.bipartite_anchor).strip())
    )

    if is_bipartite and args.target_models and str(args.target_models).strip():
        print(
            "[INFO] --bipartite_anchor set: ignoring --target_models (use one mode at a time)."
        )

    if is_bipartite:
        run_bipartite_mode(args)
        return

    if args.target_models and str(args.target_models).strip():
        run_two_target_mode(args)
        return

    raise SystemExit(
        "Specify either --bipartite_anchor (plus --bipartite_partners_csv / --bipartite_partners) "
        "or --target_models (exactly two ids, comma-separated)."
    )


if __name__ == "__main__":
    main()
