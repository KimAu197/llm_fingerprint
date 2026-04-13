"""
compute_pairwise_overlap.py

Compute overlap ratio between two models.

Usage:
    python compute_pairwise_overlap.py \
        --model1 "meta-llama/Llama-2-7b-hf" \
        --model2 "meta-llama/Llama-2-7b-chat-hf" \
        --gpu_id 0
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    set_seed,
    load_hf_model,
    unload_hf_model,
    sample_fingerprint_prompt,
    compute_bottomk_vocab_batch,
    overlap_ratio,
)


def compute_overlap_between_models(
    model1_name: str,
    model2_name: str,
    device: str = "cuda:0",
    num_fingerprints: int = 5,
    k_bottom_sampling: int = 50,
    fingerprint_length: int = 64,
    bottom_k_vocab: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Compute overlap ratio between two models.
    
    Args:
        model1_name: HuggingFace model ID for base model
        model2_name: HuggingFace model ID for test model
        device: Device to use (e.g., "cuda:0")
        num_fingerprints: Number of fingerprints to generate
        k_bottom_sampling: k for bottom-k sampling during generation
        fingerprint_length: Token length of each fingerprint
        bottom_k_vocab: Bottom-k vocabulary size for overlap computation
        seed: Random seed
    
    Returns:
        dict with overlap_ratio and other stats
    """
    set_seed(seed)
    
    print(f"\n{'='*70}")
    print("PAIRWISE OVERLAP COMPUTATION")
    print(f"{'='*70}")
    print(f"Model 1:        {model1_name}")
    print(f"Model 2:        {model2_name}")
    print(f"Device:         {device}")
    print(f"Fingerprints:   {num_fingerprints}")
    print(f"Bottom-k vocab: {bottom_k_vocab}")
    print(f"{'='*70}\n")
    
    # ========== Step 1: Generate fingerprints from model1 ==========
    print(f"[1/4] Loading model1: {model1_name}")
    t0 = time.time()
    model1, tok1, _ = load_hf_model(model1_name, device_map={"": device})
    print(f"      Loaded in {time.time() - t0:.1f}s")
    
    print(f"[2/4] Generating {num_fingerprints} fingerprints from model1...")
    t0 = time.time()
    fingerprints = []
    for i in range(num_fingerprints):
        fp = sample_fingerprint_prompt(
            model1, tok1,
            device=device,
            l_random_prefix=8,
            total_len=fingerprint_length,
            k_bottom=k_bottom_sampling,
        )
        fingerprints.append(fp)
        print(f"      Generated fingerprint {i+1}/{num_fingerprints}")
    print(f"      Completed in {time.time() - t0:.1f}s")
    
    # ========== Step 2: Compute bottom-k vocab for model1 ==========
    print(f"[3/4] Computing bottom-k vocab for model1 on {len(fingerprints)} fingerprints...")
    t0 = time.time()
    bottomk_model1 = compute_bottomk_vocab_batch(
        model1, tok1,
        prompts=fingerprints,
        k=bottom_k_vocab,
        device=device,
    )
    print(f"      Completed in {time.time() - t0:.1f}s")
    
    # Unload model1
    unload_hf_model(model1, tok1)
    print("      Model1 unloaded")
    
    # ========== Step 3: Compute bottom-k vocab for model2 ==========
    print(f"[4/4] Loading model2: {model2_name}")
    t0 = time.time()
    model2, tok2, _ = load_hf_model(model2_name, device_map={"": device})
    print(f"      Loaded in {time.time() - t0:.1f}s")
    
    print(f"      Computing bottom-k vocab for model2 on {len(fingerprints)} fingerprints...")
    t0 = time.time()
    bottomk_model2 = compute_bottomk_vocab_batch(
        model2, tok2,
        prompts=fingerprints,
        k=bottom_k_vocab,
        device=device,
    )
    print(f"      Completed in {time.time() - t0:.1f}s")
    
    # Unload model2
    unload_hf_model(model2, tok2)
    print("      Model2 unloaded")
    
    # ========== Step 4: Compute overlap ratios ==========
    print(f"\n{'='*70}")
    print("COMPUTING OVERLAP RATIOS")
    print(f"{'='*70}")
    
    overlap_ratios = []
    for i in range(len(fingerprints)):
        ratio = overlap_ratio(bottomk_model1[i], bottomk_model2[i])
        overlap_ratios.append(ratio)
        print(f"Fingerprint {i+1}: overlap = {ratio:.4f}")
    
    avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
    min_overlap = min(overlap_ratios)
    max_overlap = max(overlap_ratios)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Average overlap: {avg_overlap:.4f}")
    print(f"Min overlap:     {min_overlap:.4f}")
    print(f"Max overlap:     {max_overlap:.4f}")
    print(f"{'='*70}\n")
    
    return {
        "model1": model1_name,
        "model2": model2_name,
        "overlap_ratios": overlap_ratios,
        "avg_overlap": avg_overlap,
        "min_overlap": min_overlap,
        "max_overlap": max_overlap,
        "num_fingerprints": num_fingerprints,
        "bottom_k_vocab": bottom_k_vocab,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute overlap ratio between two models"
    )
    parser.add_argument("--model1", type=str, required=True,
                        help="First model (HuggingFace ID)")
    parser.add_argument("--model2", type=str, required=True,
                        help="Second model (HuggingFace ID)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--num_fingerprints", type=int, default=5,
                        help="Number of fingerprints to generate")
    parser.add_argument("--k_bottom_sampling", type=int, default=50,
                        help="k for bottom-k sampling during generation")
    parser.add_argument("--fingerprint_length", type=int, default=64,
                        help="Token length of each fingerprint")
    parser.add_argument("--bottom_k_vocab", type=int, default=2000,
                        help="Bottom-k vocabulary size for overlap")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu_id}"
    
    result = compute_overlap_between_models(
        model1_name=args.model1,
        model2_name=args.model2,
        device=device,
        num_fingerprints=args.num_fingerprints,
        k_bottom_sampling=args.k_bottom_sampling,
        fingerprint_length=args.fingerprint_length,
        bottom_k_vocab=args.bottom_k_vocab,
        seed=args.seed,
    )
    
    print(f"\nFinal result: overlap_ratio = {result['avg_overlap']:.4f}")


if __name__ == "__main__":
    main()
