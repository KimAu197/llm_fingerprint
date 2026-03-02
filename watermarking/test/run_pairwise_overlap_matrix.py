"""
run_pairwise_overlap_matrix.py

Pairwise fingerprinting experiment to compute overlap matrix for all models.

Design:
    Given a set of n models:
    1. For each model M (treated as "new model"):
        a. Use M to generate fingerprint prompts using random prefix + bottom-k sampling
        b. Compute M's bottom-k vocab for each fingerprint
        c. For each other model M' in the set (candidate models):
            - Compute M's bottom-k vocab using the same fingerprints
            - Calculate overlap score
    2. Output: n x n matrix where entry (i, j) = overlap score between model_i and model_j

Input CSV format:
    model_id
    model_name_1
    model_name_2
    ...
"""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    set_seed,
    load_hf_model,
    unload_hf_model,
    sample_fingerprint_prompt,
    compute_bottomk_vocab_for_model,
    compute_bottomk_vocab_batch,
    overlap_ratio,
)


# ----------------- Parallel Worker -----------------
# Must be defined at module level so multiprocessing (spawn) can pickle it.


def _candidate_worker(
    args: Tuple[str, int, List[str], int],
) -> Tuple[str, Optional[Dict[str, List[int]]], Optional[str]]:
    """
    Evaluate one candidate model in a spawned subprocess.

    Args:
        args: (candidate_name, gpu_id, fingerprints, k)
            candidate_name  - HuggingFace model ID
            gpu_id          - CUDA device index to load the model on
            fingerprints    - list of fingerprint prompt strings
            k               - bottom-k vocab size

    Returns:
        (candidate_name, cache, error_string)
        cache is None and error_string is set when loading/inference fails.
    """
    candidate_name, gpu_id, fingerprints, k = args

    # Re-import inside subprocess (spawn context has a fresh interpreter).
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from utils import load_hf_model, unload_hf_model, compute_bottomk_vocab_batch

    device = f"cuda:{gpu_id}"
    try:
        model, tok, _ = load_hf_model(candidate_name, device_map={"": device})
        bottomk_lists = compute_bottomk_vocab_batch(
            model, tok, prompts=fingerprints, k=k, device=device
        )
        cache = {fp: ids for fp, ids in zip(fingerprints, bottomk_lists)}
        unload_hf_model(model, tok)
        return candidate_name, cache, None
    except Exception:
        return candidate_name, None, traceback.format_exc()


# ----------------- Timing Utilities -----------------


def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))


class Timer:
    """Simple timer context manager."""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n[{self.name}] Started at {datetime.now().strftime('%H:%M:%S')}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"[{self.name}] Completed in {format_time(elapsed)}")
    
    @property
    def elapsed(self):
        """Get elapsed time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ----------------- Model Loading -----------------


# Using load_hf_model and unload_hf_model from utils.model_loader


# ----------------- CSV Processing -----------------


def load_model_list(csv_path: str) -> List[str]:
    """
    Load model list from CSV file.
    
    CSV format:
        model_id
        model_name_1
        model_name_2
        ...
    
    Returns:
        List of model names (in order from CSV)
    """
    models = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get('model_id', '').strip()
            if model_id:
                models.append(model_id)
    
    return models


# ----------------- Fingerprint Generation -----------------


def generate_fingerprints(
    model,
    tokenizer,
    num_fingerprints: int,
    k_bottom: int,
    total_len: int,
    device,
) -> List[str]:
    """Generate fingerprint prompts using the given model."""
    fingerprints = []
    for i in range(num_fingerprints):
        print(f"    Generating fingerprint {i+1}/{num_fingerprints}")
        fp = sample_fingerprint_prompt(
            model,
            tokenizer,
            device=device,
            l_random_prefix=8,
            total_len=total_len,
            k_bottom=k_bottom,
        )
        fingerprints.append(fp)
    return fingerprints


# ----------------- Results Saving -----------------


def save_matrix_to_csv(matrix: np.ndarray, models: List[str], output_path: Path) -> None:
    """Save overlap matrix to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([''] + models)
        # Data rows
        for i, model in enumerate(models):
            writer.writerow([model] + list(matrix[i, :]))


def update_matrix_row(matrix: np.ndarray, row_idx: int, models: List[str], output_path: Path) -> None:
    """Update CSV file with new row data (incremental save)."""
    # Read existing file if it exists
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
    else:
        # Create new file with header
        rows = [[''] + models]
        # Add empty rows for all models
        for model in models:
            rows.append([model] + ['0.0'] * len(models))
    
    # Update the specific row (row_idx + 1 to account for header)
    if row_idx + 1 < len(rows):
        rows[row_idx + 1] = [models[row_idx]] + [str(val) for val in matrix[row_idx, :]]
    
    # Write back to file
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ----------------- Bottom-k Computation -----------------


def compute_bottomk_cache(
    model,
    tokenizer,
    fingerprints: List[str],
    k: int,
    device,
) -> Dict[str, List[int]]:
    """Compute bottom-k vocab for all fingerprints in a single batched forward pass."""
    bottomk_lists = compute_bottomk_vocab_batch(
        model, tokenizer, prompts=fingerprints, k=k, device=device
    )
    return {fp: bottomk_ids for fp, bottomk_ids in zip(fingerprints, bottomk_lists)}


def compute_pairwise_overlap(
    cache_a: Dict[str, List[int]],
    cache_b: Dict[str, List[int]],
) -> float:
    """
    Compute average overlap score across all fingerprints.
    
    Returns:
        Average overlap ratio
    """
    scores = []
    for fp, vocab_a in cache_a.items():
        vocab_b = cache_b[fp]
        score = overlap_ratio(vocab_a, vocab_b)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


# ----------------- Main Experiment -----------------


def run_experiment(args: argparse.Namespace) -> None:
    """Main experiment logic."""
    set_seed(args.seed)
    
    experiment_start = time.time()
    
    # Log configuration
    print("=" * 80)
    print("PAIRWISE OVERLAP MATRIX EXPERIMENT")
    print("=" * 80)
    print(f"CSV Path: {args.csv_path}")
    print(f"Num fingerprints per model: {args.num_fingerprints}")
    print(f"Bottom-k vocab size: {args.bottom_k_vocab}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load model list
    print("=" * 80)
    print("LOADING MODEL LIST")
    print("=" * 80)
    models = load_model_list(args.csv_path)
    n_models = len(models)
    print(f"\nFound {n_models} unique models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()
    
    # Initialize results matrix
    overlap_matrix = np.zeros((n_models, n_models))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    matrix_csv_path = output_dir / "overlap_matrix.csv"
    matrix_npy_path = output_dir / "overlap_matrix.npy"
    
    # Initialize CSV file with header and empty rows
    print("Initializing output files...")
    save_matrix_to_csv(overlap_matrix, models, matrix_csv_path)
    print(f"Output CSV: {matrix_csv_path}")
    print()
    
    # Process each model as "new model"
    for i, new_model_name in enumerate(models):
        print("\n" + "=" * 80)
        print(f"NEW MODEL [{i+1}/{n_models}]: {new_model_name}")
        print("=" * 80)
        
        model_start = time.time()
        
        # Load new model
        print(f"\n[1] Loading new model...")
        with Timer(f"Load {new_model_name}"):
            try:
                new_model, new_tok, device = load_hf_model(new_model_name)
            except Exception as e:
                print(f"[ERROR] Failed to load {new_model_name}: {e}")
                continue
        
        
        skip_model = False
        try:
            # Generate fingerprints using new model
            print(f"\n[2] Generating {args.num_fingerprints} fingerprints...")
            with Timer(f"Generate Fingerprints"):
                try:
                    fingerprints = generate_fingerprints(
                        new_model,
                        new_tok,
                        args.num_fingerprints,
                        args.k_bottom_sampling,
                        args.fingerprint_length,
                        device,
                    )
                except (ValueError, Exception) as e:
                    print(f"[ERROR] Cannot generate fingerprints for {new_model_name}: {e}")
                    print(f"Skipping this model...")
                    skip_model = True
            
            if skip_model:
                # Mark entire row as error
                overlap_matrix[i, :] = -1.0
                update_matrix_row(overlap_matrix, i, models, matrix_csv_path)
                np.save(matrix_npy_path, overlap_matrix)
                continue  # Skip to next model
            
            # Compute new model's bottom-k vocab
            print(f"\n[3] Computing new model's bottom-k vocab...")
            with Timer("Compute New Model Bottom-K"):
                new_model_cache = compute_bottomk_cache(
                    new_model, new_tok, fingerprints, args.bottom_k_vocab, device
                )
            
            # new_model_cache is computed; unload new model so all GPUs are
            # available for parallel candidate evaluation.
            print(f"\n[3.5] Unloading new model to free GPU memory for candidates...")
            unload_hf_model(new_model, new_tok)
            new_model = None
            new_tok = None

            # Test against all candidate models
            n_parallel = args.num_parallel_candidates
            print(f"\n[4] Testing against {n_models} candidate models "
                  f"({n_parallel} in parallel)...")
            candidate_start = time.time()

            # Mark self-comparison
            self_idx = models.index(new_model_name)
            overlap_matrix[i, self_idx] = 1.0
            print(f"\n  [{self_idx+1}/{n_models}] {new_model_name}: "
                  f"self-comparison = 1.0000")

            # Non-self candidates with their original column indices
            candidates_to_eval = [
                (j, name) for j, name in enumerate(models)
                if name != new_model_name
            ]

            mp_ctx = mp.get_context("spawn")

            for batch_start in range(0, len(candidates_to_eval), n_parallel):
                batch = candidates_to_eval[batch_start:batch_start + n_parallel]
                short = [name.split("/")[-1] for _, name in batch]
                print(f"\n  Batch [{batch_start+1}-{batch_start+len(batch)}"
                      f"/{len(candidates_to_eval)}]: {short}")

                # Assign one GPU index per slot in this batch
                worker_args = [
                    (name, gpu_id, fingerprints, args.bottom_k_vocab)
                    for gpu_id, (j, name) in enumerate(batch)
                ]

                batch_t = time.time()

                if n_parallel == 1:
                    # Direct call — avoids subprocess spawn overhead for sequential mode
                    results = [_candidate_worker(worker_args[0])]
                else:
                    with ProcessPoolExecutor(
                        max_workers=len(batch), mp_context=mp_ctx
                    ) as executor:
                        future_map = {
                            executor.submit(_candidate_worker, wa): (j, name)
                            for wa, (j, name) in zip(worker_args, batch)
                        }
                        results = []
                        for future in as_completed(future_map):
                            j_fut, name_fut = future_map[future]
                            try:
                                ret = future.result()
                            except Exception:
                                ret = (name_fut, None, traceback.format_exc())
                            results.append((j_fut, ret))

                # Record results
                if n_parallel == 1:
                    j_col, candidate_name = batch[0]
                    _, cache, error = results[0]
                    if error:
                        print(f"    [ERROR] {candidate_name}:\n{error}")
                        overlap_matrix[i, j_col] = -1.0
                    else:
                        score = compute_pairwise_overlap(new_model_cache, cache)
                        overlap_matrix[i, j_col] = score
                        print(f"    [{j_col+1}/{n_models}] {candidate_name}: "
                              f"overlap = {score:.4f}")
                else:
                    for j_col, (ret_name, cache, error) in results:
                        if error:
                            print(f"    [ERROR] {ret_name}:\n{error}")
                            overlap_matrix[i, j_col] = -1.0
                        else:
                            score = compute_pairwise_overlap(new_model_cache, cache)
                            overlap_matrix[i, j_col] = score
                            print(f"    [{j_col+1}/{n_models}] {ret_name}: "
                                  f"overlap = {score:.4f}")

                print(f"  Batch done in {format_time(time.time() - batch_t)}")

            candidate_time = time.time() - candidate_start
            print(f"\n  Candidates tested in {format_time(candidate_time)}")

            # Save this row to CSV immediately
            print(f"\n  Saving row {i+1}/{n_models} to CSV...")
            update_matrix_row(overlap_matrix, i, models, matrix_csv_path)

            # Also save numpy array as backup
            np.save(matrix_npy_path, overlap_matrix)
            print(f"  Progress saved!")

        finally:
            # new_model may have been unloaded early in step 3.5
            if new_model is not None:
                unload_hf_model(new_model, new_tok)
        
        model_time = time.time() - model_start
        print(f"\n{'='*80}")
        print(f"NEW MODEL [{i+1}/{n_models}] COMPLETED in {format_time(model_time)}")
        print(f"{'='*80}")
    
    # Save results
    print("\n" + "=" * 80)
    print("FINAL SAVE")
    print("=" * 80)
    
    # Final save of matrix as CSV (should already be up to date)
    save_matrix_to_csv(overlap_matrix, models, matrix_csv_path)
    print(f"Final overlap matrix saved to: {matrix_csv_path}")
    
    # Final save of numpy array
    np.save(matrix_npy_path, overlap_matrix)
    print(f"Numpy array saved to: {matrix_npy_path}")
    
    # Save metadata
    metadata_path = output_dir / "experiment_metadata.json"
    metadata = {
        'models': models,
        'num_models': n_models,
        'num_fingerprints': args.num_fingerprints,
        'bottom_k_vocab': args.bottom_k_vocab,
        'k_bottom_sampling': args.k_bottom_sampling,
        'fingerprint_length': args.fingerprint_length,
        'seed': args.seed,
        'csv_path': args.csv_path,
        'total_time_seconds': time.time() - experiment_start,
        'timestamp': datetime.now().isoformat(),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {format_time(time.time() - experiment_start)}")
    print(f"Models tested: {n_models}")
    print(f"\nOverlap Matrix ({n_models} x {n_models}):")
    print()
    
    # Print matrix
    print(f"{'':30s}", end='')
    for model in models:
        short_name = model.split('/')[-1][:12]
        print(f"{short_name:>14s}", end='')
    print()
    
    for i, model in enumerate(models):
        short_name = model.split('/')[-1][:28]
        print(f"{short_name:30s}", end='')
        for j in range(n_models):
            val = overlap_matrix[i, j]
            if val < 0:
                print(f"{'ERROR':>14s}", end='')
            else:
                print(f"{val:14.4f}", end='')
        print()
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise overlap matrix experiment"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with model list (header: model_id, one model per row)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pairwise_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_fingerprints",
        type=int,
        default=10,
        help="Number of fingerprint prompts to generate per model"
    )
    parser.add_argument(
        "--k_bottom_sampling",
        type=int,
        default=50,
        help="k for bottom-k sampling during fingerprint generation"
    )
    parser.add_argument(
        "--fingerprint_length",
        type=int,
        default=64,
        help="Total length of fingerprint prompt"
    )
    parser.add_argument(
        "--bottom_k_vocab",
        type=int,
        default=2000,
        help="Size of bottom-k vocabulary for overlap computation"
    )
    parser.add_argument(
        "--num_parallel_candidates",
        type=int,
        default=1,
        help=(
            "Number of candidate models to evaluate in parallel, one per GPU. "
            "Set to the number of available GPUs (e.g. 4) for maximum throughput. "
            "Default 1 = sequential mode (original behavior)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
