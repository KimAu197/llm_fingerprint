"""
run_pairwise_overlap_matrix.py

Optimized 2-phase pairwise fingerprinting experiment.

Phase 1 - Generate fingerprints:
    Load each model ONCE, generate N fingerprints, save to JSON, unload.

Phase 2 - Compute overlap matrix:
    Load each model ONCE, compute bottom-k vocab for ALL fingerprints, unload.
    Then build n x n overlap matrix from cached results (pure computation).

Each model is loaded exactly 2 times total (vs. n+1 times in naive approach).

Usage:
    # Full run (both phases):
    python run_pairwise_overlap_matrix.py --csv_path models.csv --gpu_ids 2

    # Skip Phase 1 if fingerprints already generated:
    python run_pairwise_overlap_matrix.py --csv_path models.csv --gpu_ids 2 \
        --fingerprints_file results/fingerprints.json
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    set_seed,
    load_hf_model,
    unload_hf_model,
    sample_fingerprint_prompt,
    compute_bottomk_vocab_batch,
    overlap_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def load_model_list(csv_path: str) -> List[str]:
    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            mid = row.get('model_id', '').strip()
            if mid:
                models.append(mid)
    return models


def save_matrix_csv(matrix: np.ndarray, models: List[str], path: Path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([''] + models)
        for i, m in enumerate(models):
            w.writerow([m] + [f"{v:.6f}" for v in matrix[i]])


def print_progress(idx: int, total: int, times: List[float], t0: float):
    if not times:
        return
    avg = sum(times) / len(times)
    remaining = total - (idx + 1)
    eta = avg * remaining
    pct = (idx + 1) / total * 100
    finish = datetime.now() + timedelta(seconds=eta)

    print(f"\n{'='*70}")
    print(f"  Progress:  {idx+1}/{total} ({pct:.1f}%)")
    print(f"  Last:      {format_time(times[-1])}  |  "
          f"Avg: {format_time(avg)}  |  "
          f"Fast: {format_time(min(times))}  |  "
          f"Slow: {format_time(max(times))}")
    print(f"  Elapsed:   {format_time(time.time() - t0)}  |  "
          f"ETA: {format_time(eta)}  |  "
          f"Finish: {finish.strftime('%m-%d %H:%M')}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Phase 1 – Generate fingerprints (one model load per model)
# ---------------------------------------------------------------------------

def phase1_generate_fingerprints(
    models: List[str],
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
) -> Dict[str, List[str]]:
    """Load each model once, generate fingerprints, return dict."""
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATE FINGERPRINTS")
    print("=" * 70)

    all_fps: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    times: List[float] = []
    t0 = time.time()
    device = f"cuda:{gpu_id}"

    for i, name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {name}")
        mt = time.time()
        
        model = None
        tok = None

        try:
            model, tok, _ = load_hf_model(name, device_map={"": device})

            fps = []
            for fi in range(args.num_fingerprints):
                fp = sample_fingerprint_prompt(
                    model, tok,
                    device=device,
                    l_random_prefix=8,
                    total_len=args.fingerprint_length,
                    k_bottom=args.k_bottom_sampling,
                )
                fps.append(fp)
                print(f"  fp {fi+1}/{args.num_fingerprints} generated")

            all_fps[name] = fps

        except Exception as e:
            print(f"  [ERROR] {e}")
            errors[name] = traceback.format_exc()
        
        finally:
            # Always unload, even if error occurred
            if model is not None or tok is not None:
                unload_hf_model(model, tok)
            # Extra cleanup after unload
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        elapsed = time.time() - mt
        times.append(elapsed)
        print_progress(i, len(models), times, t0)

        # Autosave every 5 models
        if (i + 1) % 5 == 0 or (i + 1) == len(models):
            fp_path = output_dir / "fingerprints.json"
            with open(fp_path, 'w') as f:
                json.dump(all_fps, f, indent=2)
            print(f"  [AUTOSAVE] {fp_path}")

    if errors:
        err_path = output_dir / "fingerprint_errors.json"
        with open(err_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\n[WARNING] {len(errors)} models failed fingerprint generation")

    print(f"\nPhase 1 done: {len(all_fps)}/{len(models)} models OK  "
          f"({format_time(time.time() - t0)})")

    return all_fps


# ---------------------------------------------------------------------------
# Phase 2 – Compute bottom-k caches (one model load per model)
# ---------------------------------------------------------------------------

def _compute_all_bottomk_for_model(
    model, tok, all_prompts: List[str], k: int, device: str, batch_size: int = 32,
) -> List[List[int]]:
    """Compute bottom-k for many prompts, chunked to avoid OOM."""
    results = []
    for start in range(0, len(all_prompts), batch_size):
        chunk = all_prompts[start:start + batch_size]
        batch_result = compute_bottomk_vocab_batch(
            model, tok, prompts=chunk, k=k, device=device,
        )
        results.extend(batch_result)
    return results


def phase2_compute_caches(
    models: List[str],
    all_fps: Dict[str, List[str]],
    args: argparse.Namespace,
    gpu_id: int,
) -> Dict[str, Dict[str, List[int]]]:
    """
    Load each model once, compute bottom-k for ALL fingerprints from ALL models.

    Returns:
        bottomk_caches[model_name][fingerprint_string] = [token_ids]
    """
    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTE BOTTOM-K CACHES")
    print("=" * 70)

    # Flatten all fingerprints into a single ordered list
    all_prompts: List[str] = []
    for m in models:
        if m in all_fps:
            all_prompts.extend(all_fps[m])
    # Deduplicate while preserving order (fingerprints are almost certainly unique)
    seen = set()
    unique_prompts = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    all_prompts = unique_prompts

    n_prompts = len(all_prompts)
    batch_size = getattr(args, 'batch_size_bottomk', 32)
    print(f"Total unique fingerprints to evaluate: {n_prompts}")
    print(f"Bottom-k batch size: {batch_size}")
    print()

    caches: Dict[str, Dict[str, List[int]]] = {}
    times: List[float] = []
    t0 = time.time()
    device = f"cuda:{gpu_id}"

    for i, name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {name}")

        if name not in all_fps:
            print("  [SKIP] No fingerprints (Phase 1 failed)")
            continue

        mt = time.time()
        model = None
        tok = None
        
        try:
            model, tok, _ = load_hf_model(name, device_map={"": device})

            bottomk_lists = _compute_all_bottomk_for_model(
                model, tok, all_prompts, args.bottom_k_vocab, device, batch_size,
            )

            cache = {fp: bk for fp, bk in zip(all_prompts, bottomk_lists)}
            caches[name] = cache
            print(f"  Computed bottom-k for {n_prompts} fingerprints")

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
        
        finally:
            # Always unload, even if error occurred
            if model is not None or tok is not None:
                unload_hf_model(model, tok)
            # Extra cleanup after unload
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        elapsed = time.time() - mt
        times.append(elapsed)
        print_progress(i, len(models), times, t0)

    print(f"\nPhase 2 done: {len(caches)}/{len(models)} models OK  "
          f"({format_time(time.time() - t0)})")

    return caches


# ---------------------------------------------------------------------------
# Phase 3 – Build overlap matrix (pure computation, no GPU)
# ---------------------------------------------------------------------------

def phase3_build_matrix(
    models: List[str],
    all_fps: Dict[str, List[str]],
    caches: Dict[str, Dict[str, List[int]]],
    output_dir: Path,
) -> np.ndarray:
    """Build n x n overlap matrix from cached bottom-k results."""
    print("\n" + "=" * 70)
    print("PHASE 3: BUILD OVERLAP MATRIX")
    print("=" * 70)

    n = len(models)
    matrix = np.full((n, n), -1.0)

    for i, mi in enumerate(models):
        if mi not in all_fps or mi not in caches:
            continue

        fps_i = all_fps[mi]

        for j, mj in enumerate(models):
            if mj not in caches:
                continue

            if i == j:
                matrix[i, j] = 1.0
                continue

            scores = []
            for fp in fps_i:
                if fp in caches[mi] and fp in caches[mj]:
                    scores.append(overlap_ratio(caches[mi][fp], caches[mj][fp]))

            if scores:
                matrix[i, j] = sum(scores) / len(scores)

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  Row {i+1}/{n}")

    # Save
    csv_path = output_dir / "overlap_matrix.csv"
    npy_path = output_dir / "overlap_matrix.npy"
    save_matrix_csv(matrix, models, csv_path)
    np.save(npy_path, matrix)
    print(f"\nMatrix saved: {csv_path}")
    print(f"Numpy saved:  {npy_path}")

    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    t_start = time.time()

    gpu_id = int(args.gpu_ids.split(',')[0]) if args.gpu_ids else 0

    # Load model list
    models = load_model_list(args.csv_path)
    n = len(models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAIRWISE OVERLAP MATRIX (Optimized 2-Phase)")
    print("=" * 70)
    print(f"Models:          {n}")
    print(f"Fingerprints:    {args.num_fingerprints} per model")
    print(f"Bottom-k vocab:  {args.bottom_k_vocab}")
    print(f"GPU:             cuda:{gpu_id}")
    print(f"Output:          {output_dir}")
    if args.fingerprints_file:
        print(f"Fingerprints:    {args.fingerprints_file} (skip Phase 1)")
    print()

    # --- Phase 1 ---
    if args.fingerprints_file:
        print(f"Loading pre-generated fingerprints: {args.fingerprints_file}")
        with open(args.fingerprints_file, 'r') as f:
            all_fps = json.load(f)
        print(f"Loaded fingerprints for {len(all_fps)} models")

        # Only keep models that are in both the CSV and the fingerprints file
        missing = [m for m in models if m not in all_fps]
        if missing:
            print(f"[WARNING] {len(missing)} models in CSV have no fingerprints:")
            for m in missing[:5]:
                print(f"  - {m}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")
    else:
        all_fps = phase1_generate_fingerprints(models, args, gpu_id, output_dir)

    # --- Phase 2 ---
    caches = phase2_compute_caches(models, all_fps, args, gpu_id)

    # --- Phase 3 ---
    phase3_build_matrix(models, all_fps, caches, output_dir)

    # --- Save metadata ---
    metadata = {
        'models': models,
        'num_models': n,
        'num_fingerprints': args.num_fingerprints,
        'bottom_k_vocab': args.bottom_k_vocab,
        'k_bottom_sampling': args.k_bottom_sampling,
        'fingerprint_length': args.fingerprint_length,
        'seed': args.seed,
        'gpu_id': gpu_id,
        'total_time_seconds': time.time() - t_start,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # --- Summary ---
    total = time.time() - t_start
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time:  {format_time(total)}")
    print(f"Models:      {n}")
    print(f"Output:      {output_dir}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise overlap matrix (optimized 2-phase)"
    )
    p.add_argument("--csv_path", type=str, required=True,
                   help="CSV with model list (header: model_id)")
    p.add_argument("--output_dir", type=str, default="./pairwise_results",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_fingerprints", type=int, default=5,
                   help="Fingerprints per model")
    p.add_argument("--k_bottom_sampling", type=int, default=50,
                   help="k for bottom-k sampling during generation")
    p.add_argument("--fingerprint_length", type=int, default=64,
                   help="Token length of each fingerprint")
    p.add_argument("--bottom_k_vocab", type=int, default=2000,
                   help="Bottom-k vocabulary size for overlap")
    p.add_argument("--gpu_ids", type=str, default=None,
                   help="GPU ID to use (e.g., '2')")
    p.add_argument("--fingerprints_file", type=str, default=None,
                   help="Path to pre-generated fingerprints.json (skip Phase 1)")
    p.add_argument("--batch_size_bottomk", type=int, default=32,
                   help="Batch size for bottom-k computation (reduce if OOM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
