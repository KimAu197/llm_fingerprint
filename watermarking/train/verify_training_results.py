#!/usr/bin/env python3
"""
Verify training results by comparing overlap ratios.

This script:
1. Loads the base model and fine-tuned model
2. Computes overlap using the SAME fingerprints used during training
3. Compares with the recorded overlap from training
4. Verifies the results are consistent

Usage:
    python verify_training_results.py --output_dir ./wiki_en_50steps
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils import compute_bottomk_vocab_for_model, sample_fingerprint_prompt, set_seed
    print("[info] Successfully imported utils functions")
except ImportError:
    print("[error] Could not import from utils")
    sys.exit(1)


def overlap_ratio(set_a: List[int], set_b: List[int]) -> float:
    """Compute overlap ratio between two sets."""
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    return len(sa.intersection(sb)) / float(len(sa))


def load_training_data(output_dir: str) -> Dict:
    """Load training configuration and results."""
    output_path = Path(output_dir)
    
    # Load args
    args_file = output_path / "args.json"
    if not args_file.exists():
        raise FileNotFoundError(f"args.json not found in {output_dir}")
    
    with open(args_file, 'r') as f:
        args = json.load(f)
    
    # Load fingerprints - try multiple locations
    fingerprints_file = None
    possible_locations = [
        output_path / "fingerprints.json",  # In output dir
        Path("./fingerprints.json"),  # Current directory
        Path(__file__).parent / "fingerprints.json",  # Script directory
    ]
    
    for loc in possible_locations:
        if loc.exists():
            fingerprints_file = loc
            break
    
    if fingerprints_file is None:
        # If no fingerprints.json found, try to extract from overlap_results.json
        print("[warn] fingerprints.json not found, trying to extract from training results...")
        
        results_file = output_path / "overlap_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                overlap_results_temp = json.load(f)
            
        # Check if wordlist_details contains prompts
        if overlap_results_temp and "wordlist_details" in overlap_results_temp[0]:
            # The prompts in wordlist_details are truncated to 100 chars
            # We need to get the full prompts from base_bottomk_cache keys
            print("[info] Extracting full prompts from base_bottomk_cache...")
            
            # Load base_bottomk_cache to get full prompts
            cache_file = output_path / "base_bottomk_cache.json"
            if not cache_file.exists():
                raise FileNotFoundError("base_bottomk_cache.json not found")
            
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            
            # Get full prompts from cache keys
            full_prompts = list(cache.keys())
            
            # Match truncated prompts with full prompts
            fingerprints = []
            for detail in overlap_results_temp[0]["wordlist_details"]:
                truncated_prompt = detail["prompt"]
                
                # Find matching full prompt
                matched = False
                for full_prompt in full_prompts:
                    if full_prompt.startswith(truncated_prompt):
                        fingerprints.append({"x_prime": full_prompt})
                        matched = True
                        break
                
                if not matched:
                    print(f"[warn] Could not find full prompt for: {truncated_prompt[:50]}...")
            
            print(f"[info] Extracted {len(fingerprints)} fingerprints from base_bottomk_cache")
        else:
            raise FileNotFoundError(
                "fingerprints.json not found in any of these locations:\n" +
                "\n".join(f"  - {loc}" for loc in possible_locations) +
                "\n\nPlease ensure fingerprints.json exists or training results contain prompts."
            )
    else:
        print(f"[info] Found fingerprints at: {fingerprints_file}")
        raise FileNotFoundError(
            "fingerprints.json not found and overlap_results.json not available"
        )
    
    if fingerprints_file is not None:
        with open(fingerprints_file, 'r') as f:
            fingerprints = json.load(f)
    # else: fingerprints already extracted from overlap_results.json above
    
    # Load base bottom-k cache
    cache_file = output_path / "base_bottomk_cache.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"base_bottomk_cache.json not found in {output_dir}")
    
    with open(cache_file, 'r') as f:
        base_bottomk_cache = json.load(f)
    
    # Load overlap results
    results_file = output_path / "overlap_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"overlap_results.json not found in {output_dir}")
    
    with open(results_file, 'r') as f:
        overlap_results = json.load(f)
    
    return {
        "args": args,
        "fingerprints": fingerprints,
        "base_bottomk_cache": base_bottomk_cache,
        "overlap_results": overlap_results,
    }


def verify_overlap(
    base_model_name: str,
    finetuned_model_path: str,
    fingerprints: List[Dict],
    base_bottomk_cache: Dict[str, List[int]],
    bottom_k: int,
    device: str,
) -> Dict:
    """Compute overlap and compare with training results."""
    
    print("\n[1/3] Loading models...")
    
    # Load base model
    print(f"  Loading base model: {base_model_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    base_model.to(device)
    base_model.eval()
    
    # Load fine-tuned model
    print(f"  Loading fine-tuned model: {finetuned_model_path}")
    ft_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path, trust_remote_code=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    ft_model.to(device)
    ft_model.eval()
    
    print("\n[2/3] Computing overlap...")
    
    overlap_scores = []
    for i, fp in enumerate(fingerprints):
        prompt_text = fp.get("x_prime", fp.get("prompt", ""))
        print(f"  Fingerprint {i+1}/{len(fingerprints)}...", end="\r")
        
        # Get base bottom-k from cache
        base_bottomk = base_bottomk_cache[prompt_text]
        
        # Compute fine-tuned bottom-k
        ft_bottomk = compute_bottomk_vocab_for_model(
            ft_model,
            ft_tokenizer,
            k=bottom_k,
            device=device,
            prompt=prompt_text,
        )
        
        # Calculate overlap
        overlap = overlap_ratio(base_bottomk, ft_bottomk)
        overlap_scores.append(overlap)
    
    print(f"  Computed overlap for {len(fingerprints)} fingerprints" + " " * 20)
    
    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    
    return {
        "avg_overlap": avg_overlap,
        "overlap_scores": overlap_scores,
        "min_overlap": min(overlap_scores),
        "max_overlap": max(overlap_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify training results")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Training output directory (e.g., ./wiki_en_50steps)"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--regenerate_fingerprints", action="store_true",
        help="Regenerate fingerprints if not found (uses base model)"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("=" * 80)
    print("VERIFYING TRAINING RESULTS")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load training data
    print("\nLoading training data...")
    try:
        data = load_training_data(args.output_dir)
    except FileNotFoundError as e:
        if "fingerprints.json" in str(e) and args.regenerate_fingerprints:
            print(f"\n[warn] {e}")
            print("\n[info] Regenerating fingerprints from base model...")
            
            # Load args to get model name and fingerprint config
            args_file = Path(args.output_dir) / "args.json"
            if not args_file.exists():
                print(f"[error] args.json not found in {args.output_dir}")
                return
            
            with open(args_file, 'r') as f:
                training_args = json.load(f)
            
            # Load base model
            print(f"Loading base model: {training_args['base_model_name']}")
            base_tokenizer = AutoTokenizer.from_pretrained(
                training_args['base_model_name'], 
                trust_remote_code=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                training_args['base_model_name'],
                trust_remote_code=True,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            )
            base_model.to(args.device)
            base_model.eval()
            
            # Generate fingerprints
            num_fingerprints = training_args.get('num_fingerprints', 8)
            fingerprint_len = training_args.get('fingerprint_total_len', 64)
            k_bottom = training_args.get('k_bottom_random_prefix', 50)
            
            print(f"Generating {num_fingerprints} fingerprints...")
            fingerprints = []
            for i in range(num_fingerprints):
                print(f"  Generating fingerprint {i+1}/{num_fingerprints}...", end="\r")
                x_prime = sample_fingerprint_prompt(
                    base_model,
                    base_tokenizer,
                    device=args.device,
                    l_random_prefix=8,
                    total_len=fingerprint_len,
                    k_bottom=k_bottom,
                )
                fingerprints.append({"x_prime": x_prime})
            
            print(f"✓ Generated {num_fingerprints} fingerprints" + " " * 20)
            
            # Save fingerprints
            fingerprints_file = Path(args.output_dir) / "fingerprints.json"
            with open(fingerprints_file, 'w') as f:
                json.dump(fingerprints, f, indent=2)
            print(f"Saved fingerprints to: {fingerprints_file}")
            
            # Try loading again
            try:
                data = load_training_data(args.output_dir)
            except FileNotFoundError as e2:
                print(f"\n[error] {e2}")
                return
        else:
            print(f"\n[error] {e}")
            print("\nMake sure the output directory contains:")
            print("  - args.json")
            print("  - fingerprints.json (or use --regenerate_fingerprints)")
            print("  - base_bottomk_cache.json")
            print("  - overlap_results.json")
            print("  - final_model/")
            return
    
    training_args = data["args"]
    fingerprints = data["fingerprints"]
    base_bottomk_cache = data["base_bottomk_cache"]
    overlap_results = data["overlap_results"]
    
    print(f"✓ Loaded {len(fingerprints)} fingerprints")
    print(f"✓ Loaded {len(overlap_results)} training checkpoints")
    
    # Get final training overlap
    final_training_overlap = overlap_results[-1]["avg_overlap_ratio"]
    final_step = overlap_results[-1]["step"]
    
    print(f"\nTraining results (Step {final_step}):")
    print(f"  Recorded overlap: {final_training_overlap:.4f}")
    
    # Verify overlap
    finetuned_model_path = str(Path(args.output_dir) / "final_model")
    
    verification = verify_overlap(
        base_model_name=training_args["base_model_name"],
        finetuned_model_path=finetuned_model_path,
        fingerprints=fingerprints,
        base_bottomk_cache=base_bottomk_cache,
        bottom_k=training_args["bottom_k_vocab"],
        device=args.device,
    )
    
    print("\n[3/3] Verification results...")
    
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    print(f"\nRecorded (from training):")
    print(f"  Final overlap: {final_training_overlap:.4f}")
    
    print(f"\nVerified (re-computed):")
    print(f"  Average overlap: {verification['avg_overlap']:.4f}")
    print(f"  Min overlap: {verification['min_overlap']:.4f}")
    print(f"  Max overlap: {verification['max_overlap']:.4f}")
    
    # Compare
    diff = abs(verification['avg_overlap'] - final_training_overlap)
    print(f"\nDifference: {diff:.6f}")
    
    if diff < 0.001:
        print("\n✅ VERIFICATION PASSED!")
        print("   The re-computed overlap matches the training results.")
    elif diff < 0.01:
        print("\n⚠️  VERIFICATION WARNING")
        print("   Small difference detected (< 1%). This might be due to:")
        print("   - Floating point precision")
        print("   - Random sampling in fingerprint generation")
    else:
        print("\n❌ VERIFICATION FAILED!")
        print("   Large difference detected. Possible issues:")
        print("   - Wrong model loaded")
        print("   - Different fingerprints used")
        print("   - Computation error")
    
    # Show per-fingerprint comparison
    print("\n" + "=" * 80)
    print("PER-FINGERPRINT COMPARISON")
    print("=" * 80)
    
    final_training_scores = overlap_results[-1].get("overlap_scores", [])
    
    if final_training_scores:
        print(f"\n{'FP':<4} {'Recorded':<10} {'Verified':<10} {'Diff':<10}")
        print("-" * 40)
        for i, (recorded, verified) in enumerate(zip(final_training_scores, verification['overlap_scores'])):
            diff = abs(verified - recorded)
            status = "✓" if diff < 0.01 else "⚠"
            print(f"{i:<4} {recorded:<10.4f} {verified:<10.4f} {diff:<10.6f} {status}")
    
    # Show training progression
    print("\n" + "=" * 80)
    print("TRAINING PROGRESSION")
    print("=" * 80)
    
    print(f"\n{'Step':<8} {'Overlap':<10}")
    print("-" * 20)
    
    # Show first, middle, and last few checkpoints
    checkpoints_to_show = []
    if len(overlap_results) > 10:
        checkpoints_to_show = [
            overlap_results[0],  # First
            overlap_results[len(overlap_results)//4],  # 25%
            overlap_results[len(overlap_results)//2],  # 50%
            overlap_results[3*len(overlap_results)//4],  # 75%
            overlap_results[-1],  # Last
        ]
    else:
        checkpoints_to_show = overlap_results
    
    for result in checkpoints_to_show:
        print(f"{result['step']:<8} {result['avg_overlap_ratio']:<10.4f}")
    
    if len(overlap_results) > 10:
        print(f"... ({len(overlap_results) - 5} more checkpoints)")
    
    # Summary
    first_overlap = overlap_results[0]["avg_overlap_ratio"]
    total_decrease = first_overlap - final_training_overlap
    decrease_rate = total_decrease / final_step if final_step > 0 else 0
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Initial overlap (step 1): {first_overlap:.4f}")
    print(f"Final overlap (step {final_step}): {final_training_overlap:.4f}")
    print(f"Total decrease: {total_decrease:.4f}")
    print(f"Decrease rate: {decrease_rate:.6f} per step")
    
    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
