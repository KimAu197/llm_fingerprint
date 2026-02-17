"""
run_base_family_experiment.py

Experiment to test fingerprinting on base model families.

Design:
  1) Positive samples: base_model vs derived_model (same family) - should have high overlap
  2) Negative samples: derived_model vs other_family_models - should have low overlap

Structure:
  - Load experiment_models_base_family.csv
  - Extract model_id (derived) and ui_base_models (base) for each model
  - Group by base model family
  - For each base model:
      * Generate fingerprints once
      * Compute base model's bottom-k vocab for each fingerprint
      * Test all derived models in same family (positive samples)
      * Test 5 random models from other families (negative samples)
  - Save results to CSV
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    set_seed,
    sample_fingerprint_prompt,
    compute_bottomk_vocab_for_model,
)


# ----------------- Timing and GPU Utilities -----------------


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - allocated,
            'device_name': torch.cuda.get_device_name(0),
        }
    return None


def log_gpu_memory(prefix=""):
    """Log GPU memory usage."""
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"  [GPU {prefix}] {gpu_info['device_name']}")
        print(f"    Allocated: {gpu_info['allocated_gb']:.2f} GB")
        print(f"    Reserved:  {gpu_info['reserved_gb']:.2f} GB")
        print(f"    Free:      {gpu_info['free_gb']:.2f} GB / {gpu_info['total_gb']:.2f} GB")
    else:
        print(f"  [GPU {prefix}] No CUDA device available")


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
        print(f"\n⏱️  [{self.name}] Started at {datetime.now().strftime('%H:%M:%S')}")
        log_gpu_memory("Start")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"⏱️  [{self.name}] Completed in {format_time(elapsed)}")
        log_gpu_memory("End")
    
    @property
    def elapsed(self):
        """Get elapsed time."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ----------------- Model Loading -----------------


def load_hf_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load HuggingFace model and tokenizer."""
    print(f"  Loading: {model_name}")
    load_start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    if device:
        model.to(device)
    model.eval()
    
    load_time = time.time() - load_start
    print(f"    ✓ Loaded in {load_time:.2f}s")
    log_gpu_memory("After Load")
    
    return model, tokenizer


def unload_hf_model(model, tokenizer) -> None:
    """Free CUDA/MPS memory."""
    print(f"    Unloading model...")
    log_gpu_memory("Before Unload")
    
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    log_gpu_memory("After Unload")


# ----------------- CSV Processing -----------------


def load_experiment_data(csv_path: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Load experiment data from CSV and group by base model.
    
    Returns:
        Dict mapping base_model -> list of {derived_model, base_model}
    """
    csv.field_size_limit(sys.maxsize)
    
    families = defaultdict(list)
    all_models = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row.get('model_id', '').strip()
            # Try ui_base_models first, then effective_base_model
            base_model = (row.get('ui_base_models') or row.get('effective_base_model', '')).strip()
            
            if not model_id or not base_model:
                continue
                
            entry = {
                'derived_model': model_id,
                'base_model': base_model,
            }
            
            families[base_model].append(entry)
            all_models.append(entry)
    
    return dict(families), all_models


# ----------------- Fingerprint Generation -----------------


def generate_fingerprints_for_base(
    model,
    tokenizer,
    num_pairs: int,
    prompt_style: str,
    k_bottom_random_prefix: int,
    total_len: int,
) -> List[Dict[str, Any]]:
    """Generate fingerprint prompts (x') using bottom-k sampling."""
    device = next(model.parameters()).device
    pairs: List[Dict[str, Any]] = []
    for i in range(num_pairs):
        print(f"    Generating fingerprint {i+1}/{num_pairs}")
        x_prime = sample_fingerprint_prompt(
            model,
            tokenizer,
            device=device,
            l_random_prefix=8,
            total_len=total_len,
            k_bottom=k_bottom_random_prefix,
        )
        pairs.append({"prompt_style": prompt_style, "x_prime": x_prime})
    return pairs


def extract_prompt_from_fingerprint(fp: Dict[str, Any]) -> str:
    """Extract the textual prompt x' from a fingerprint dict."""
    for key in ("x_prime", "prompt", "x", "input_text"):
        if key in fp:
            return fp[key]
    raise KeyError(f"Cannot find prompt text in fingerprint dict keys={list(fp.keys())}")


# ----------------- Overlap Computation -----------------


def overlap_ratio(set_a: List[int], set_b: List[int]) -> float:
    """Compute |intersection| / |set_a|."""
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    return len(sa.intersection(sb)) / float(len(sa))


def compute_model_bottomk_cache(
    model,
    tokenizer,
    fingerprints: List[Dict[str, Any]],
    k: int,
    device: str,
) -> Dict[str, List[int]]:
    """Compute bottom-k vocab for all fingerprints for a given model."""
    cache = {}
    for idx, fp in enumerate(fingerprints):
        prompt_text = extract_prompt_from_fingerprint(fp)
        bottomk_ids = compute_bottomk_vocab_for_model(
            model, tokenizer, k=k, device=device, prompt=prompt_text
        )
        cache[prompt_text] = bottomk_ids
        print(f"    Cached bottom-k for fingerprint {idx+1}/{len(fingerprints)}")
    return cache


def compute_overlap_scores(
    base_bottomk_cache: Dict[str, List[int]],
    test_bottomk_cache: Dict[str, List[int]],
) -> List[float]:
    """Compute overlap scores for all fingerprints."""
    scores = []
    for prompt_text, base_ids in base_bottomk_cache.items():
        test_ids = test_bottomk_cache[prompt_text]
        score = overlap_ratio(base_ids, test_ids)
        scores.append(score)
    return scores


# ----------------- Result Saving -----------------


def save_result(
    result: Dict[str, Any],
    csv_path: Path,
    is_positive: bool,
) -> None:
    """Save result to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "sample_type",  # "positive" or "negative"
        "base_model",
        "test_model",
        "num_pairs",
        "avg_overlap_ratio",
        "min_overlap",
        "max_overlap",
        "bottom_k_vocab_size",
        "pair_scores_json",
    ]
    
    file_exists = csv_path.exists()
    
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        scores = result['pair_scores']
        writer.writerow({
            "sample_type": "positive" if is_positive else "negative",
            "base_model": result["base_model"],
            "test_model": result["test_model"],
            "num_pairs": len(scores),
            "avg_overlap_ratio": sum(scores) / len(scores) if scores else 0.0,
            "min_overlap": min(scores) if scores else 0.0,
            "max_overlap": max(scores) if scores else 0.0,
            "bottom_k_vocab_size": result["bottom_k_vocab_size"],
            "pair_scores_json": json.dumps(scores),
        })


# ----------------- Main Experiment Logic -----------------


def run_experiment(args: argparse.Namespace) -> None:
    """Main experiment logic."""
    set_seed(args.seed)
    
    experiment_start = time.time()
    
    # Log experiment configuration
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Num fingerprints per base: {args.num_pairs}")
    print(f"Negative samples per model: {args.num_negative_samples}")
    print(f"Bottom-k vocab size: {args.bottom_k_vocab}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Log system info
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # Load experiment data
    print("=" * 80)
    print("LOADING EXPERIMENT DATA")
    print("=" * 80)
    families, all_models = load_experiment_data(args.csv_path)
    
    print(f"\nLoaded {len(families)} base model families")
    print(f"Total models: {len(all_models)}")
    for base_model, derivatives in families.items():
        print(f"  {base_model}: {len(derivatives)} derivatives")
    print()
    
    # Create output path
    output_csv = Path(args.output_dir) / f"base_family_overlap_results.csv"
    
    # Track overall statistics
    family_stats = []
    
    # Filter families if family_index is specified
    families_list = list(families.items())
    if args.family_index is not None:
        if 1 <= args.family_index <= len(families_list):
            print(f"\n🎯 Running ONLY Family {args.family_index}")
            families_to_run = [families_list[args.family_index - 1]]
        else:
            print(f"\n❌ Invalid family_index {args.family_index}. Must be 1-{len(families_list)}")
            return
    else:
        print(f"\n🎯 Running ALL {len(families_list)} families")
        families_to_run = families_list
    
    print()
    
    # Process each base model family
    for family_idx, (base_model_name, derivatives) in enumerate(families_to_run, 1):
        print("\n" + "=" * 80)
        print(f"FAMILY {family_idx}/{len(families)}: {base_model_name}")
        print("=" * 80)
        print(f"Derivatives: {len(derivatives)}")
        
        family_start_time = time.time()
        
        # 1. Load base model
        print(f"\n[1] Loading base model...")
        with Timer(f"Load Base Model ({base_model_name})"):
            try:
                base_model, base_tok = load_hf_model_and_tokenizer(base_model_name, device=args.device)
            except Exception as e:
                print(f"[ERROR] Failed to load base model {base_model_name}: {e}")
                continue
        
        try:
            # 2. Generate fingerprints
            print(f"\n[2] Generating {args.num_pairs} fingerprints...")
            with Timer(f"Generate {args.num_pairs} Fingerprints"):
                fingerprints = generate_fingerprints_for_base(
                    model=base_model,
                    tokenizer=base_tok,
                    num_pairs=args.num_pairs,
                    prompt_style=args.prompt_style,
                    k_bottom_random_prefix=args.k_bottom_random_prefix,
                    total_len=args.total_len,
                )
            
            # 3. Compute base model's bottom-k vocab
            print(f"\n[3] Computing base model bottom-k vocab...")
            with Timer("Compute Base Model Bottom-K Vocab"):
                base_bottomk_cache = compute_model_bottomk_cache(
                    base_model, base_tok, fingerprints, args.bottom_k_vocab, args.device
                )
            
            # 4. Test derived models (POSITIVE SAMPLES)
            print(f"\n[4] Testing {len(derivatives)} derived models (POSITIVE SAMPLES)...")
            positive_start = time.time()
            positive_count = 0
            
            for deriv_idx, deriv_entry in enumerate(derivatives, 1):
                derived_model_name = deriv_entry['derived_model']
                print(f"\n  [{deriv_idx}/{len(derivatives)}] Testing: {derived_model_name}")
                
                try:
                    with Timer(f"Test Positive {deriv_idx}/{len(derivatives)}"):
                        derived_model, derived_tok = load_hf_model_and_tokenizer(
                            derived_model_name, device=args.device
                        )
                        
                        # Compute derived model's bottom-k vocab
                        print(f"    Computing derived model bottom-k vocab...")
                        derived_bottomk_cache = compute_model_bottomk_cache(
                            derived_model, derived_tok, fingerprints, args.bottom_k_vocab, args.device
                        )
                        
                        # Compute overlap scores
                        scores = compute_overlap_scores(base_bottomk_cache, derived_bottomk_cache)
                        avg_score = sum(scores) / len(scores) if scores else 0.0
                        print(f"    ✓ Avg overlap: {avg_score:.4f}")
                        
                        # Save result
                        result = {
                            "base_model": base_model_name,
                            "test_model": derived_model_name,
                            "pair_scores": scores,
                            "bottom_k_vocab_size": args.bottom_k_vocab,
                        }
                        save_result(result, output_csv, is_positive=True)
                        positive_count += 1
                        
                        unload_hf_model(derived_model, derived_tok)
                    
                except Exception as e:
                    print(f"    [ERROR] Failed on {derived_model_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            positive_time = time.time() - positive_start
            print(f"\n  ✓ Positive samples: {positive_count}/{len(derivatives)} completed in {format_time(positive_time)}")
            
            # 5. Test models from other families (NEGATIVE SAMPLES)
            print(f"\n[5] Testing NEGATIVE SAMPLES (cross-family)...")
            negative_start = time.time()
            negative_count = 0
            
            # Get all models from other families
            other_family_models = []
            for other_base, other_derivs in families.items():
                if other_base != base_model_name:
                    other_family_models.extend([d['derived_model'] for d in other_derivs])
            
            # Randomly sample negative samples
            if len(other_family_models) > 0:
                # For each derived model in this family, test against N random from other families
                num_negative_per_model = min(args.num_negative_samples, len(other_family_models))
                
                for deriv_entry in derivatives:
                    derived_model_name = deriv_entry['derived_model']
                    
                    # Sample random models from other families
                    negative_samples = random.sample(other_family_models, num_negative_per_model)
                    
                    print(f"\n  Testing {derived_model_name} against {num_negative_per_model} other-family models...")
                    
                    for neg_idx, neg_model_name in enumerate(negative_samples, 1):
                        print(f"    [{neg_idx}/{num_negative_per_model}] vs {neg_model_name}")
                        
                        try:
                            neg_start = time.time()
                            neg_model, neg_tok = load_hf_model_and_tokenizer(
                                neg_model_name, device=args.device
                            )
                            
                            # Compute negative model's bottom-k vocab
                            neg_bottomk_cache = compute_model_bottomk_cache(
                                neg_model, neg_tok, fingerprints, args.bottom_k_vocab, args.device
                            )
                            
                            # Compute overlap scores
                            scores = compute_overlap_scores(base_bottomk_cache, neg_bottomk_cache)
                            avg_score = sum(scores) / len(scores) if scores else 0.0
                            neg_time = time.time() - neg_start
                            print(f"      ✓ Avg overlap: {avg_score:.4f} (took {neg_time:.1f}s)")
                            
                            # Save result
                            result = {
                                "base_model": base_model_name,
                                "test_model": neg_model_name,
                                "pair_scores": scores,
                                "bottom_k_vocab_size": args.bottom_k_vocab,
                            }
                            save_result(result, output_csv, is_positive=False)
                            negative_count += 1
                            
                            unload_hf_model(neg_model, neg_tok)
                            
                        except Exception as e:
                            print(f"      [ERROR] Failed on {neg_model_name}: {e}")
            
            negative_time = time.time() - negative_start
            print(f"\n  ✓ Negative samples: {negative_count} completed in {format_time(negative_time)}")
            
        finally:
            unload_hf_model(base_model, base_tok)
        
        # Log family completion
        family_time = time.time() - family_start_time
        family_stats.append({
            'family': base_model_name,
            'time': family_time,
            'positive_count': positive_count,
            'negative_count': negative_count,
        })
        
        print(f"\n{'='*80}")
        print(f"✓ FAMILY {family_idx} COMPLETED")
        print(f"  Time: {format_time(family_time)}")
        print(f"  Positive samples: {positive_count}")
        print(f"  Negative samples: {negative_count}")
        print(f"{'='*80}")
    
    # Log overall completion
    experiment_time = time.time() - experiment_start
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {format_time(experiment_time)}")
    print(f"Families processed: {len(family_stats)}")
    print()
    print("Per-family breakdown:")
    for i, stat in enumerate(family_stats, 1):
        print(f"\n  {i}. {stat['family']}")
        print(f"     Time: {format_time(stat['time'])}")
        print(f"     Positive: {stat['positive_count']}, Negative: {stat['negative_count']}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_csv}")
    
    # Save timing summary to JSON
    summary_path = Path(args.output_dir) / "experiment_timing_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    summary = {
        'total_time_seconds': experiment_time,
        'total_time_formatted': format_time(experiment_time),
        'num_families': len(family_stats),
        'families': family_stats,
        'config': {
            'num_pairs': args.num_pairs,
            'num_negative_samples': args.num_negative_samples,
            'bottom_k_vocab': args.bottom_k_vocab,
            'device': args.device,
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Timing summary saved to: {summary_path}")


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Base model family fingerprinting experiment"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to experiment_models_base_family.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=10,
        help="Number of fingerprint prompts to generate"
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="raw",
        choices=["raw", "oneshot", "chatml"],
        help="Prompt formatting style"
    )
    parser.add_argument(
        "--k_bottom_random_prefix",
        type=int,
        default=50,
        help="k for bottom-k sampling during fingerprint generation"
    )
    parser.add_argument(
        "--total_len",
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
        "--num_negative_samples",
        type=int,
        default=5,
        help="Number of negative samples (other-family models) per derived model"
    )
    parser.add_argument(
        "--family_index",
        type=int,
        default=None,
        help="Run only specific family (1-6). If not set, runs all families."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("=" * 80)
    print("BASE MODEL FAMILY FINGERPRINTING EXPERIMENT")
    print("=" * 80)
    print(f"CSV Path: {args.csv_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Num fingerprints: {args.num_pairs}")
    print(f"Bottom-k vocab size: {args.bottom_k_vocab}")
    print(f"Negative samples per model: {args.num_negative_samples}")
    print()
    
    run_experiment(args)


if __name__ == "__main__":
    main()
