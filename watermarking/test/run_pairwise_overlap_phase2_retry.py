"""
run_pairwise_overlap_phase2_retry.py

Two modes (same CLI):

1) Full Phase 2 (default when --phase2_models is omitted)
   Same as run_pairwise_overlap_matrix Phase 2 + Phase 3. Optionally write
   --save_bottomk_caches so you can retry individual models later.

2) Partial Phase 2 (when --phase2_models is set)
   With --bottomk_caches_file: load other models' caches from disk, re-run GPU only
   for listed ids (fast), merge into matrix.
   Without --bottomk_caches_file: one or more comma-separated ids in --phase2_models.
   For each target, bottom-k is computed once (caches kept in RAM); for every other
   model j, load j once and refresh all targets' rows/columns vs j; then fill
   target-vs-target cells. Same overlap rules as the main script. No cache JSON.

   Strict comparability with an old matrix: use the same --seed as the original
   run (check experiment_metadata.json); default is 42 only if that was the old default.

3) Regenerate fingerprints (optional --regenerate_fingerprints_models)
   If matrix -1 was due to missing Phase-1 fingerprints, regenerate for listed ids,
   merge into --fingerprints_file, then continue with Phase 2 when --phase2_models
   is also set (or stop after saving JSON if only regeneration was requested).

   When using --phase2_models, any target missing from fingerprints.json (or empty
   list) is auto-regenerated before Phase 2 unless you pre-filled the JSON.

   --existing_overlap_matrix with a .npy smaller than the CSV (new models at end):
   the matrix is expanded (new rows/cols -1, diagonal 1 for new indices). If the .npy
   is larger than the CSV, the top-left block is cropped. Use --strict_overlap_matrix_shape
   to forbid resize. Alignment is by row order: first n_old CSV ids must match the .npy
   from the original run.

Usage (export caches once):
 python run_pairwise_overlap_phase2_retry.py \\
        --csv_path models.csv \\
        --fingerprints_file fingerprints.json \\
        --output_dir ./out \\
        --gpu_ids 0 \\
        --save_bottomk_caches ./out/bottomk_caches.json

Usage (retry one row):
    python run_pairwise_overlap_phase2_retry.py \\
        --csv_path models.csv \\
        --fingerprints_file fingerprints.json \\
        --output_dir ./out \\
        --gpu_ids 0 \\
        --bottomk_caches_file ./out/bottomk_caches.json \\
        --existing_overlap_matrix ./out/overlap_matrix.npy \\
        --phase2_models "meta-llama/Meta-Llama-3.1-8B" \\
        --save_bottomk_caches ./out/bottomk_caches.json

Usage (no bottomk_caches JSON — refresh several targets' rows/columns vs everyone):
    python run_pairwise_overlap_phase2_retry.py \\
        --csv_path models.csv \\
        --fingerprints_file fingerprints.json \\
        --output_dir ./out \\
        --gpu_ids 0 \\
        --seed 42 \\
        --existing_overlap_matrix ./out/overlap_matrix.npy \\
        --phase2_models "org/a,org/b,org/c"

Usage (regenerate fingerprints for failed Phase1, then refresh one row in one run):
    python run_pairwise_overlap_phase2_retry.py \\
        --csv_path models.csv \\
        --fingerprints_file fingerprints.json \\
        --output_dir ./out \\
        --gpu_ids 0 \\
        --regenerate_fingerprints_models "meta-llama/Meta-Llama-3.1-8B" \\
        --existing_overlap_matrix ./out/overlap_matrix.npy \\
        --phase2_models "meta-llama/Meta-Llama-3.1-8B"
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pairwise_overlap_matrix import (
    _compute_all_bottomk_for_model,
    _maybe_reset_cuda_after_model,
    _RUN_LOG,
    format_time,
    load_model_list,
    phase2_compute_caches,
    phase3_build_matrix,
    print_progress,
    save_matrix_artifacts,
    setup_run_logging,
    update_overlap_matrix_for_new_cache,
)
from utils import (
    load_hf_model,
    overlap_ratio,
    sample_fingerprint_prompt,
    set_seed,
    unload_hf_model,
)


def _align_square_npy_to_model_count(
    arr: np.ndarray,
    n_csv: int,
    path_label: str,
) -> np.ndarray:
    """
    Resize a square .npy overlap matrix to ``n_csv`` models.

    - Expand (n_old < n_csv): top-left block is the saved matrix; new rows/cols are -1 with diagonal 1 for new indices. **Requires** the first ``n_old`` rows of the
      current CSV to be the same models in the same order as when ``path_label`` was saved;
      **append new models only at the end** of the CSV.
    - Shrink (n_old > n_csv): keep the top-left ``n_csv`` block. **Requires** the
      current CSV lists the first ``n_old`` models in the same order, with some **removed
      from the end** (or the block still aligns); otherwise overlaps will be wrong.
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path_label} must be square, got {arr.shape}")
    n_old = int(arr.shape[0])
    if n_old == n_csv:
        return arr.astype(np.float64, copy=True)
    if n_old < n_csv:
        print(
            f"[INFO] Expanding {path_label}: {n_old}x{n_old} -> {n_csv}x{n_csv}. "
            f"Top-left is the old matrix; new rows/cols default to -1 (diag 1 for new). "
            "Append new `model_id`s at the **end** of --csv_path only."
        )
        mat = np.full((n_csv, n_csv), -1.0)
        mat[:n_old, :n_old] = arr.astype(np.float64)
        for i in range(n_old, n_csv):
            mat[i, i] = 1.0
        return mat
    print(
        f"[WARNING] Cropping {path_label}: {n_old}x{n_old} -> {n_csv}x{n_csv} "
        "(top-left block). CSV must list the same leading model order as the .npy; "
        "typically you removed trailing models from the CSV."
    )
    return arr[:n_csv, :n_csv].astype(np.float64, copy=True)


def load_overlap_matrix_from_path(
    path: Path,
    models: List[str],
    *,
    strict_shape: bool = False,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        n = len(models)
        if arr.shape == (n, n):
            return arr.astype(np.float64, copy=True)
        if strict_shape:
            raise ValueError(
                f"{path} shape {arr.shape}, expected ({n}, {n}). "
                "Omit --strict_overlap_matrix_shape to auto expand/crop .npy to CSV size."
            )
        return _align_square_npy_to_model_count(arr, n, str(path))
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_labels = header[1:]
            row_data: Dict[str, List[float]] = {}
            for row in reader:
                if not row:
                    continue
                row_data[row[0]] = [float(x) for x in row[1:]]
        col_index = {m: j for j, m in enumerate(col_labels)}
        mat = np.full((len(models), len(models)), -1.0)
        for i, mi in enumerate(models):
            if mi not in row_data:
                continue
            vals = row_data[mi]
            for j, mj in enumerate(models):
                if mj not in col_index:
                    continue
                cj = col_index[mj]
                if cj < len(vals):
                    mat[i, j] = vals[cj]
        return mat
    raise ValueError(f"Use .npy or .csv, got: {path}")


def phase1_regenerate_fingerprints_for_models(
    model_ids: List[str],
    all_fps: Dict[str, List[str]],
    args: argparse.Namespace,
    gpu_id: int,
    fingerprints_out: Path,
    output_dir: Path,
) -> Dict[str, List[str]]:
    """
    Regenerate fingerprints for ``model_ids`` only; merge into ``all_fps`` and
    write ``fingerprints_out`` after each success (same generation settings as main pipeline).
    """
    print("\n" + "=" * 70)
    print("PHASE 1 (REGENERATE FINGERPRINTS, SUBSET)")
    print("=" * 70)

    merged: Dict[str, List[str]] = dict(all_fps)
    errors: Dict[str, str] = {}
    device = f"cuda:{gpu_id}"
    times: List[float] = []
    t0 = time.time()

    fingerprints_out.parent.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(model_ids):
        print(f"\n[{i+1}/{len(model_ids)}] fingerprints: {name}")
        mt = time.time()
        model = None
        tok = None
        model_failed = False
        cleanup_failed = False
        last_exc: Optional[BaseException] = None

        try:
            model, tok, _ = load_hf_model(name, device_map={"": device})
            fps: List[str] = []
            for fi in range(args.num_fingerprints):
                fp = sample_fingerprint_prompt(
                    model,
                    tok,
                    device=device,
                    l_random_prefix=8,
                    total_len=args.fingerprint_length,
                    k_bottom=args.k_bottom_sampling,
                )
                fps.append(fp)
                print(f"  fp {fi+1}/{args.num_fingerprints} generated")
            merged[name] = fps
            with open(fingerprints_out, "w", encoding="utf-8") as outf:
                json.dump(merged, outf, indent=2)
            print(f"  [SAVED] {fingerprints_out}")
        except Exception as e:
            model_failed = True
            last_exc = e
            print(f"  [ERROR] {e}")
            errors[name] = traceback.format_exc()
            _RUN_LOG.exception("Phase 1 regenerate failed for %s", name)
        finally:
            try:
                if model is not None or tok is not None:
                    unload_hf_model(model, tok)
            except Exception as cleanup_err:
                cleanup_failed = True
                print(f"  [WARNING] Cleanup failed: {cleanup_err}")
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                cleanup_failed = True
            _maybe_reset_cuda_after_model(
                gpu_id,
                args,
                model_failed=model_failed,
                cleanup_failed=cleanup_failed,
                last_exception=last_exc,
            )

        times.append(time.time() - mt)
        print_progress(i, len(model_ids), times, t0)

    if errors:
        err_path = output_dir / "fingerprint_regenerate_errors.json"
        with open(err_path, "w", encoding="utf-8") as ef:
            json.dump(errors, ef, indent=2)
        print(f"\n[WARNING] {len(errors)} model(s) failed regeneration -> {err_path}")

    print(
        f"\nPhase 1 subset done: {len(model_ids) - len(errors)}/{len(model_ids)} OK "
        f"({format_time(time.time() - t0)})"
    )
    return merged


def ensure_fingerprints_for_targets(
    targets: List[str],
    all_fps: Dict[str, List[str]],
    cli_args: argparse.Namespace,
    gpu_id: int,
    fingerprints_out: Path,
    output_dir: Path,
) -> Dict[str, List[str]]:
    """
    For any ``targets`` missing from ``all_fps`` or with an empty list, run Phase 1
    and merge into ``all_fps`` / ``fingerprints_out``.
    """
    missing = [t for t in targets if t not in all_fps or len(all_fps.get(t, [])) == 0]
    if not missing:
        return all_fps
    print(
        f"\n[INFO] {len(missing)} model(s) in --phase2_models lack fingerprints in JSON; "
        f"generating now (seed={cli_args.seed}): {missing}"
    )
    phase1_ns = build_phase2_namespace(cli_args)
    merged = phase1_regenerate_fingerprints_for_models(
        missing,
        all_fps,
        phase1_ns,
        gpu_id,
        fingerprints_out,
        output_dir,
    )
    still = [t for t in targets if t not in merged or len(merged.get(t, [])) == 0]
    if still:
        err_log = output_dir / "fingerprint_regenerate_errors.json"
        raise SystemExit(
            f"Phase 1 could not create fingerprints for: {still}\n"
            f"Fix Hugging Face repo ids (see https://huggingface.co/models), use "
            f"`hf auth login` or HF_TOKEN for gated/private models, or paste prompts "
            f"into {fingerprints_out} manually.\n"
            f"Details: {err_log if err_log.is_file() else '(no error file)'}"
        )
    return merged


def _dedup_all_prompts(models: List[str], all_fps: Dict[str, List[str]]) -> List[str]:
    all_prompts: List[str] = []
    for m in models:
        if m in all_fps:
            all_prompts.extend(all_fps[m])
    seen = set()
    unique_prompts: List[str] = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    return unique_prompts


def _matrix_cell_mi_mj(
    mi: str,
    mj: str,
    all_fps: Dict[str, List[str]],
    cache_i: Dict[str, List[int]],
    cache_j: Dict[str, List[int]],
) -> float:
    """Same rule as fill_overlap_matrix_inplace: average over mi's fingerprints."""
    if mi not in all_fps:
        return -1.0
    scores: List[float] = []
    for fp in all_fps[mi]:
        if fp in cache_i and fp in cache_j:
            scores.append(overlap_ratio(cache_i[fp], cache_j[fp]))
    if scores:
        return float(sum(scores) / len(scores))
    return -1.0


def _run_one_model_bottomk(
    name: str,
    all_prompts: List[str],
    args: argparse.Namespace,
    gpu_id: int,
) -> Dict[str, List[int]]:
    device = f"cuda:{gpu_id}"
    model = None
    tok = None
    model_failed = False
    cleanup_failed = False
    last_exc: Optional[BaseException] = None
    try:
        model, tok, _ = load_hf_model(name, device_map={"": device})
        bottomk_lists = _compute_all_bottomk_for_model(
            model,
            tok,
            all_prompts,
            args.bottom_k_vocab,
            device,
            args.batch_size_bottomk,
        )
        return {fp: bk for fp, bk in zip(all_prompts, bottomk_lists)}
    except Exception as e:
        model_failed = True
        last_exc = e
        raise
    finally:
        try:
            if model is not None or tok is not None:
                unload_hf_model(model, tok)
        except Exception:
            cleanup_failed = True
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            cleanup_failed = True
        _maybe_reset_cuda_after_model(
            gpu_id,
            args,
            model_failed=model_failed,
            cleanup_failed=cleanup_failed,
            last_exception=last_exc,
        )


def phase2_cross_targets_without_caches(
    models: List[str],
    all_fps: Dict[str, List[str]],
    target_models: List[str],
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
    existing_matrix_path: Optional[Path],
) -> None:
    """
    For each target in ``target_models``: compute and pin bottom-k cache, then for
    every non-target model j load j once and update matrix[target,j] and [j,target];
    finally fill target-vs-target pairs using pinned caches only.
    """
    print("\n" + "=" * 70)
    print("PHASE 2 (MULTI TARGET, NO STORED CACHES)")
    print("=" * 70)

    seen: set = set()
    targets: List[str] = []
    for t in target_models:
        if t not in seen:
            seen.add(t)
            targets.append(t)

    for t in targets:
        if t not in models:
            raise SystemExit(f"target not in CSV: {t}")
        if t not in all_fps:
            raise SystemExit(f"target has no fingerprints in JSON: {t}")

    all_prompts = _dedup_all_prompts(models, all_fps)
    n_prompts = len(all_prompts)
    print(f"[INFO] Unique prompts (all models): {n_prompts}")
    print(f"[INFO] Target model(s) ({len(targets)}): {targets}")

    target_idx = {t: models.index(t) for t in targets}
    target_set = set(targets)

    live = not args.no_live_overlap_matrix
    if live:
        if existing_matrix_path and existing_matrix_path.is_file():
            matrix = load_overlap_matrix_from_path(
                existing_matrix_path,
                models,
                strict_shape=getattr(
                    args, "strict_overlap_matrix_shape", False
                ),
            )
            print(f"[INFO] Seeded matrix from {existing_matrix_path}")
        else:
            matrix = np.full((len(models), len(models)), -1.0)
            print("[INFO] Starting from empty matrix (-1); pass --existing_overlap_matrix to merge")
    else:
        matrix = np.full((len(models), len(models)), -1.0)

    pinned: Dict[str, Dict[str, List[int]]] = {}
    print(f"\n--- Bottom-k for {len(targets)} target(s) (cached in RAM) ---")
    for ti, t in enumerate(targets):
        print(f"\n[targets {ti+1}/{len(targets)}] {t}")
        pinned[t] = _run_one_model_bottomk(t, all_prompts, args, gpu_id)
        print(f"  Done ({n_prompts} prompts).")
        matrix[target_idx[t], target_idx[t]] = 1.0

    others = [m for m in models if m not in target_set]
    n_others = len(others)
    times: List[float] = []
    t0 = time.time()

    print(f"\n--- Cross each of {n_others} non-target model(s) vs all targets ---")
    for step, mj in enumerate(others):
        print(f"\n[{step+1}/{n_others}] vs {mj}")
        mt = time.time()
        try:
            cache_j = _run_one_model_bottomk(mj, all_prompts, args, gpu_id)
            for t in targets:
                it, ij = target_idx[t], models.index(mj)
                matrix[it, ij] = _matrix_cell_mi_mj(t, mj, all_fps, pinned[t], cache_j)
                matrix[ij, it] = _matrix_cell_mi_mj(mj, t, all_fps, cache_j, pinned[t])
            if live:
                save_matrix_artifacts(matrix, models, output_dir, sync=True)
                with open(output_dir / "phase2_progress.json", "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "mode": "cross_targets_no_caches",
                            "targets": targets,
                            "last_cross_with": mj,
                            "timestamp": datetime.now().isoformat(),
                        },
                        pf,
                        indent=2,
                    )
                print("  [CHECKPOINT] overlap_matrix updated")
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            _RUN_LOG.exception("cross-row failed at j=%s", mj)
        times.append(time.time() - mt)
        if times:
            print_progress(step, n_others, times, t0)

    if len(targets) >= 2:
        print("\n--- Target vs target (pinned caches only) ---")
        for t1, t2 in itertools.combinations(targets, 2):
            i1, i2 = target_idx[t1], target_idx[t2]
            matrix[i1, i2] = _matrix_cell_mi_mj(
                t1, t2, all_fps, pinned[t1], pinned[t2]
            )
            matrix[i2, i1] = _matrix_cell_mi_mj(
                t2, t1, all_fps, pinned[t2], pinned[t1]
            )
        if live:
            save_matrix_artifacts(matrix, models, output_dir, sync=True)
            print("  [CHECKPOINT] target-target block updated")

    save_matrix_artifacts(matrix, models, output_dir, sync=True)
    if not live:
        print("[INFO] Wrote overlap matrix (no intermediate checkpoints; --no_live_overlap_matrix)")

    status_path = output_dir / "phase2_cache_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "cross_targets_no_caches",
                "targets": targets,
                "num_models": len(models),
                "num_prompts": n_prompts,
            },
            f,
            indent=2,
        )
    print(
        f"\n[INFO] Wrote {status_path.name}. Phase 3 skipped "
        f"(only rows/columns for targets updated)."
    )


def phase2_partial_retry(
    models: List[str],
    all_fps: Dict[str, List[str]],
    phase2_only: List[str],
    bottomk_caches_path: Path,
    existing_matrix_path: Path,
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
) -> Dict[str, Dict[str, List[int]]]:
    print("\n" + "=" * 70)
    print("PHASE 2 (PARTIAL RETRY)")
    print("=" * 70)

    all_prompts: List[str] = []
    for m in models:
        if m in all_fps:
            all_prompts.extend(all_fps[m])
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    all_prompts = unique_prompts

    n_prompts = len(all_prompts)
    batch_size = args.batch_size_bottomk
    device = f"cuda:{gpu_id}"

    with open(bottomk_caches_path, "r", encoding="utf-8") as cf:
        caches: Dict[str, Dict[str, List[int]]] = json.load(cf)
    print(f"[INFO] Loaded caches for {len(caches)} models from {bottomk_caches_path}")

    missing = [m for m in phase2_only if m not in models]
    if missing:
        raise SystemExit(f"--phase2_models not in CSV list: {missing}")
    for m in phase2_only:
        if m not in all_fps:
            raise SystemExit(f"{m} has no fingerprints in JSON")

    phase2_set = set(phase2_only)
    failures: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    times: List[float] = []
    t0 = time.time()

    live = not args.no_live_overlap_matrix
    overlap_matrix: Optional[np.ndarray] = None
    if live:
        overlap_matrix = load_overlap_matrix_from_path(
            existing_matrix_path,
            models,
            strict_shape=getattr(
                args, "strict_overlap_matrix_shape", False
            ),
        )
        print(
            f"[INFO] Live matrix from {existing_matrix_path} "
            "(updates after each successful retry model)"
        )

    for i, name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {name}")

        if name not in phase2_set:
            if name not in caches:
                msg = "No cache entry (need full Phase 2 export in --bottomk_caches_file)"
                print(f"  [SKIP] {msg}")
                skipped[name] = msg
            else:
                print("  [SKIP] Not in --phase2_models; keeping loaded cache")
            continue

        if name not in all_fps:
            skipped[name] = "No fingerprints"
            continue

        mt = time.time()
        model = None
        tok = None
        model_failed = False
        cleanup_failed = False
        last_exc: Optional[BaseException] = None

        try:
            model, tok, _ = load_hf_model(name, device_map={"": device})
            bottomk_lists = _compute_all_bottomk_for_model(
                model,
                tok,
                all_prompts,
                args.bottom_k_vocab,
                device,
                batch_size,
            )
            caches[name] = {fp: bk for fp, bk in zip(all_prompts, bottomk_lists)}
            print(f"  Computed bottom-k for {n_prompts} fingerprints")

            if live and overlap_matrix is not None:
                update_overlap_matrix_for_new_cache(
                    overlap_matrix, models, all_fps, caches, name
                )
                save_matrix_artifacts(overlap_matrix, models, output_dir, sync=True)
                prog = output_dir / "phase2_progress.json"
                with open(prog, "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "caches_ok_models": list(caches.keys()),
                            "num_caches_ok": len(caches),
                            "last_completed": name,
                            "timestamp": datetime.now().isoformat(),
                            "partial_retry": True,
                        },
                        pf,
                        indent=2,
                    )
                print(f"  [CHECKPOINT] matrix + {prog.name}")

        except Exception as e:
            model_failed = True
            last_exc = e
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failures[name] = traceback.format_exc()
            _RUN_LOG.exception("Partial Phase2 failed for %s", name)

        finally:
            try:
                if model is not None or tok is not None:
                    unload_hf_model(model, tok)
            except Exception as cleanup_err:
                cleanup_failed = True
                print(f"  [WARNING] Cleanup failed: {cleanup_err}")
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                cleanup_failed = True
            _maybe_reset_cuda_after_model(
                gpu_id,
                args,
                model_failed=model_failed,
                cleanup_failed=cleanup_failed,
                last_exception=last_exc,
            )

        times.append(time.time() - mt)
        print_progress(i, len(models), times, t0)

    status_path = output_dir / "phase2_cache_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "failures": failures,
                "skipped": skipped,
                "partial_retry": True,
                "phase2_models": phase2_only,
                "num_caches_ok": len(caches),
                "num_models": len(models),
            },
            f,
            indent=2,
        )

    if args.save_bottomk_caches:
        sp = Path(args.save_bottomk_caches)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as sf:
            json.dump(caches, sf, separators=(",", ":"))
        print(f"\n[INFO] Wrote merged caches ({len(caches)} models) -> {sp.resolve()}")

    if live and overlap_matrix is not None:
        save_matrix_artifacts(overlap_matrix, models, output_dir, sync=True)
        print(f"[INFO] Final matrix -> {output_dir / 'overlap_matrix.csv'}")

    return caches


def build_phase2_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        bottom_k_vocab=args.bottom_k_vocab,
        batch_size_bottomk=args.batch_size_bottomk,
        cuda_device_reset_each_model=args.cuda_device_reset_each_model,
        no_cuda_device_reset_on_error=args.no_cuda_device_reset_on_error,
        cuda_device_reset_after_oom=args.cuda_device_reset_after_oom,
        no_live_overlap_matrix=args.no_live_overlap_matrix,
        num_fingerprints=args.num_fingerprints,
        fingerprint_length=args.fingerprint_length,
        k_bottom_sampling=args.k_bottom_sampling,
        strict_overlap_matrix_shape=getattr(
            args, "strict_overlap_matrix_shape", False
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full Phase 2 + optional cache export, or partial Phase 2 retry"
    )
    p.add_argument("--csv_path", type=str, required=True)
    p.add_argument("--fingerprints_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./pairwise_retry_out")
    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "RNG seed for Phase 1 fingerprint sampling (and any stochastic ops). "
            "For strict comparability with an existing matrix, use the same seed as "
            "the original experiment_metadata.json / CLI (default 42 only matches "
            "runs that also used 42)."
        ),
    )
    p.add_argument(
        "--regenerate_fingerprints_models",
        type=str,
        default=None,
        help=(
            "Comma-separated model ids: run Phase 1 fingerprint generation only for these, "
            "merge into --fingerprints_file. Use when -1 was due to missing fingerprints. "
            "Combine with --phase2_models to regenerate then refresh overlap in one run."
        ),
    )
    p.add_argument("--num_fingerprints", type=int, default=5)
    p.add_argument("--fingerprint_length", type=int, default=64)
    p.add_argument("--k_bottom_sampling", type=int, default=50)
    p.add_argument("--bottom_k_vocab", type=int, default=2000)
    p.add_argument("--batch_size_bottomk", type=int, default=64)
    p.add_argument(
        "--cuda_device_reset_each_model",
        action="store_true",
    )
    p.add_argument("--no_cuda_device_reset_on_error", action="store_true")
    p.add_argument("--cuda_device_reset_after_oom", action="store_true")
    p.add_argument("--no_live_overlap_matrix", action="store_true")
    p.add_argument(
        "--save_bottomk_caches",
        type=str,
        default=None,
        help="Write merged bottom-k caches JSON after this run (large file)",
    )
    p.add_argument(
        "--phase2_models",
        type=str,
        default=None,
        help="Comma-separated model ids for partial retry only. Omit for full Phase 2.",
    )
    p.add_argument(
        "--bottomk_caches_file",
        type=str,
        default=None,
        help=(
            "Partial retry with caches: JSON from a prior full run. "
            "If omitted with --phase2_models, runs no-cache cross mode for all listed ids."
        ),
    )
    p.add_argument(
        "--existing_overlap_matrix",
        type=str,
        default=None,
        help=(
            "Seed matrix (.npy/.csv). For .npy, if size differs from CSV, default is to "
            "expand (pad -1, new diag 1) or crop top-left; see --strict_overlap_matrix_shape."
        ),
    )
    p.add_argument(
        "--strict_overlap_matrix_shape",
        action="store_true",
        help=(
            "Require .npy shape to match len(CSV) exactly. Default: auto align .npy to "
            "CSV (expand when adding models at end of CSV; crop when CSV is shorter)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(
        f"[INFO] RNG seed = {args.seed} "
        "(use original experiment_metadata.json seed for strictly comparable new fingerprints)"
    )
    gpu_id = int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0

    fp_path = Path(args.fingerprints_file)
    if fp_path.is_file():
        with open(fp_path, "r", encoding="utf-8") as f:
            all_fps: Dict[str, List[str]] = json.load(f)
    else:
        all_fps = {}
        print(f"[INFO] No existing file at {fp_path}; starting empty fingerprints dict.")

    models = load_model_list(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)

    phase2_ns = build_phase2_namespace(args)
    t0 = time.time()

    regen_raw = args.regenerate_fingerprints_models
    regen_ids = (
        [s.strip() for s in regen_raw.split(",") if s.strip()]
        if regen_raw and regen_raw.strip()
        else []
    )
    if regen_ids:
        unknown = [m for m in regen_ids if m not in models]
        if unknown:
            print(
                f"[WARNING] These ids are not in --csv_path list (still attempting load): "
                f"{unknown[:5]}{'...' if len(unknown) > 5 else ''}"
            )
        all_fps = phase1_regenerate_fingerprints_for_models(
            regen_ids,
            all_fps,
            phase2_ns,
            gpu_id,
            fp_path,
            output_dir,
        )

    partial = bool(args.phase2_models and args.phase2_models.strip())
    if regen_ids and not partial:
        meta = {
            "script": "run_pairwise_overlap_phase2_retry.py",
            "fingerprints_only": True,
            "regenerated": regen_ids,
            "fingerprints_file": str(fp_path.resolve()),
            "seed": args.seed,
            "time_seconds": time.time() - t0,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "phase2_retry_metadata.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)
        print(
            f"\nFingerprints-only run finished. Updated: {fp_path.resolve()}\n"
            f"Re-run with --phase2_models to refresh the overlap row/column."
        )
        return

    if not partial:
        if not all_fps:
            raise SystemExit(
                "Full Phase 2 needs a non-empty fingerprints.json "
                "(use --regenerate_fingerprints_models or provide an existing file)."
            )

    if partial:
        targets = [s.strip() for s in args.phase2_models.split(",") if s.strip()]
        if not targets:
            raise SystemExit("Empty --phase2_models")

        all_fps = ensure_fingerprints_for_targets(
            targets,
            all_fps,
            args,
            gpu_id,
            fp_path,
            output_dir,
        )

        if args.bottomk_caches_file:
            if not args.existing_overlap_matrix:
                raise SystemExit(
                    "Partial retry with caches requires --existing_overlap_matrix"
                )
            caches = phase2_partial_retry(
                models,
                all_fps,
                targets,
                Path(args.bottomk_caches_file),
                Path(args.existing_overlap_matrix),
                phase2_ns,
                gpu_id,
                output_dir,
            )
            all_cached = all(m in caches for m in models)
            if all_cached:
                phase3_build_matrix(models, all_fps, caches, output_dir)
            else:
                print(
                    "\n[INFO] Skipping Phase 3: not all models have caches. "
                    "Matrix on disk reflects partial retry + seeded file."
                )
        else:
            ex = Path(args.existing_overlap_matrix) if args.existing_overlap_matrix else None
            phase2_cross_targets_without_caches(
                models,
                all_fps,
                targets,
                phase2_ns,
                gpu_id,
                output_dir,
                ex,
            )
    else:
        caches = phase2_compute_caches(
            models, all_fps, phase2_ns, gpu_id, output_dir
        )
        if args.save_bottomk_caches:
            sp = Path(args.save_bottomk_caches)
            sp.parent.mkdir(parents=True, exist_ok=True)
            with open(sp, "w", encoding="utf-8") as sf:
                json.dump(caches, sf, separators=(",", ":"))
            print(
                f"\n[INFO] Wrote bottom-k caches ({len(caches)} models) -> {sp.resolve()}"
            )
        phase3_build_matrix(models, all_fps, caches, output_dir)

    meta = {
        "script": "run_pairwise_overlap_phase2_retry.py",
        "partial": partial,
        "seed": args.seed,
        "regenerated_fingerprints": regen_ids,
        "no_cache_cross_row": bool(
            partial and not getattr(args, "bottomk_caches_file", None)
        ),
        "models": models,
        "time_seconds": time.time() - t0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "phase2_retry_metadata.json", "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(f"\nDone in {format_time(time.time() - t0)}. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
