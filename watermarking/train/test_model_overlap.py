#!/usr/bin/env python3
"""
Test overlap between a base model and a fine-tuned model.

Usage:
    python test_model_overlap.py \
        --base_model "Qwen/Qwen2.5-0.5B" \
        --finetuned_model "./quick_debug/final_model" \
        --num_fingerprints 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils import (
        compute_bottomk_vocab_for_model,
        sample_fingerprint_prompt,
        set_seed,
    )
    print("[info] Successfully imported utils functions")
except ImportError:
    print("[error] Could not import from utils")
    print("[error] Make sure utils.py is in the parent directory")
    sys.exit(1)


def overlap_ratio(set_a: List[int], set_b: List[int]) -> float:
    """
    Compute overlap ratio between two sets.
    
    Args:
        set_a: First set of token IDs (base model)
        set_b: Second set of token IDs (fine-tuned model)
        
    Returns:
        |intersection| / |set_a|
    """
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    return len(sa.intersection(sb)) / float(len(sa))


def main():
    parser = argparse.ArgumentParser(description="Test overlap between base and fine-tuned models")
    parser.add_argument(
        "--base_model", type=str, required=True,
        help="Base model name or path (e.g., 'Qwen/Qwen2.5-0.5B')"
    )
    parser.add_argument(
        "--finetuned_model", type=str, required=True,
        help="Fine-tuned model path (e.g., './quick_debug/final_model')"
    )
    parser.add_argument(
        "--num_fingerprints", type=int, default=8,
        help="Number of fingerprints to test (default: 10)"
    )
    parser.add_argument(
        "--bottom_k", type=int, default=2000,
        help="Size of bottom-k vocabulary (default: 2000)"
    )
    parser.add_argument(
        "--fingerprint_len", type=int, default=64,
        help="Length of fingerprint prompts (default: 64)"
    )
    parser.add_argument(
        "--device", type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Save results to JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("=" * 80)
    print("Testing Overlap Between Base and Fine-tuned Models")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Fine-tuned model: {args.finetuned_model}")
    print(f"Number of fingerprints: {args.num_fingerprints}")
    print(f"Bottom-k size: {args.bottom_k}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load base model
    print("\n[1/4] Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    base_model.to(args.device)
    base_model.eval()
    print(f"✓ Loaded base model: {args.base_model}")
    
    # Load fine-tuned model
    print("\n[2/4] Loading fine-tuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained(args.finetuned_model, trust_remote_code=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.finetuned_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    ft_model.to(args.device)
    ft_model.eval()
    print(f"✓ Loaded fine-tuned model: {args.finetuned_model}")
    
    # Generate fingerprints
    print(f"\n[3/4] Generating {args.num_fingerprints} fingerprints...")
    fingerprints = []
    for i in range(args.num_fingerprints):
        print(f"  Generating fingerprint {i+1}/{args.num_fingerprints}...", end="\r")
        x_prime = sample_fingerprint_prompt(
            base_model,
            base_tokenizer,
            device=args.device,
            l_random_prefix=8,
            total_len=args.fingerprint_len,
            k_bottom=50,
        )
        fingerprints.append(x_prime)
    print(f"✓ Generated {args.num_fingerprints} fingerprints" + " " * 30)
    
    # Compute overlap for each fingerprint
    print(f"\n[4/4] Computing overlap...")
    results = []
    
    for i, prompt in enumerate(fingerprints):
        print(f"  Processing fingerprint {i+1}/{args.num_fingerprints}...", end="\r")
        
        # Compute base model's bottom-k
        base_bottomk = compute_bottomk_vocab_for_model(
            base_model,
            base_tokenizer,
            k=args.bottom_k,
            device=args.device,
            prompt=prompt,
        )
        
        # Compute fine-tuned model's bottom-k
        ft_bottomk = compute_bottomk_vocab_for_model(
            ft_model,
            ft_tokenizer,
            k=args.bottom_k,
            device=args.device,
            prompt=prompt,
        )
        
        # Calculate overlap
        overlap = overlap_ratio(base_bottomk, ft_bottomk)
        
        results.append({
            "fingerprint_idx": i,
            "prompt": prompt[:100],  # First 100 chars
            "overlap": overlap,
            "base_bottomk_preview": base_bottomk[:20],  # First 20 tokens
            "ft_bottomk_preview": ft_bottomk[:20],  # First 20 tokens
        })
    
    print(f"✓ Computed overlap for {args.num_fingerprints} fingerprints" + " " * 30)
    
    # Calculate statistics
    overlaps = [r["overlap"] for r in results]
    avg_overlap = sum(overlaps) / len(overlaps)
    min_overlap = min(overlaps)
    max_overlap = max(overlaps)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average overlap: {avg_overlap:.4f}")
    print(f"Min overlap: {min_overlap:.4f}")
    print(f"Max overlap: {max_overlap:.4f}")
    print(f"Std deviation: {(sum((o - avg_overlap)**2 for o in overlaps) / len(overlaps))**0.5:.4f}")
    
    print("\n" + "-" * 80)
    print("Per-Fingerprint Results:")
    print("-" * 80)
    for r in results:
        print(f"Fingerprint {r['fingerprint_idx']:2d}: {r['overlap']:.4f}")
    
    # Show example token changes
    print("\n" + "=" * 80)
    print("EXAMPLE: First Fingerprint Token Changes")
    print("=" * 80)
    
    first_result = results[0]
    base_preview = first_result["base_bottomk_preview"]
    ft_preview = first_result["ft_bottomk_preview"]
    
    print(f"Prompt: {first_result['prompt'][:80]}...")
    print(f"Overlap: {first_result['overlap']:.4f}")
    
    print("\nBase model bottom-k (first 20 tokens):")
    for i, token_id in enumerate(base_preview):
        token_text = base_tokenizer.decode([token_id])
        print(f"  {i+1:2d}. ID={token_id:6d} Text='{token_text}'")
    
    print("\nFine-tuned model bottom-k (first 20 tokens):")
    for i, token_id in enumerate(ft_preview):
        token_text = ft_tokenizer.decode([token_id])
        print(f"  {i+1:2d}. ID={token_id:6d} Text='{token_text}'")
    
    # Check if fine-tuned model is broken
    ft_is_sequential = all(ft_preview[i] < ft_preview[i+1] for i in range(len(ft_preview)-1))
    if ft_is_sequential and ft_preview[0] < 100:
        print("\n" + "!" * 80)
        print("⚠️  WARNING: Fine-tuned model's bottom-k looks sequential (0, 1, 2, ...)")
        print("   This suggests the model's logits have collapsed.")
        print("   The model may have encountered NaN during training.")
        print("!" * 80)
    
    # Save results if requested
    if args.output_json:
        output_data = {
            "base_model": args.base_model,
            "finetuned_model": args.finetuned_model,
            "num_fingerprints": args.num_fingerprints,
            "bottom_k": args.bottom_k,
            "avg_overlap": avg_overlap,
            "min_overlap": min_overlap,
            "max_overlap": max_overlap,
            "results": results,
        }
        
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output_json}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
