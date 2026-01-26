#!/usr/bin/env python3
"""
Debug script to check model logits distribution.

Usage:
    python debug_logits.py --model_path ./quick_debug/final_model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def check_logits(model_path: str, prompt: str = "The"):
    """Check logits distribution for a model."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Using prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits
    
    logits_np = logits.cpu().numpy()
    
    print("\n" + "="*80)
    print("LOGITS STATISTICS")
    print("="*80)
    print(f"Vocab size: {len(logits_np)}")
    print(f"Mean: {np.mean(logits_np):.4f}")
    print(f"Std: {np.std(logits_np):.4f}")
    print(f"Min: {np.min(logits_np):.4f}")
    print(f"Max: {np.max(logits_np):.4f}")
    print(f"Contains NaN: {np.isnan(logits_np).any()}")
    print(f"Contains Inf: {np.isinf(logits_np).any()}")
    
    # Check if logits are monotonic (sign of collapse)
    sorted_indices = np.argsort(logits_np)
    is_monotonic = np.all(sorted_indices == np.arange(len(logits_np)))
    print(f"Logits are monotonic: {is_monotonic}")
    
    # Top-k and Bottom-k
    print("\n" + "="*80)
    print("TOP-20 TOKENS (Highest logits)")
    print("="*80)
    top_indices = np.argsort(logits_np)[-20:][::-1]
    for i, idx in enumerate(top_indices):
        token_text = tokenizer.decode([idx])
        print(f"{i+1:2d}. ID={idx:6d} Logit={logits_np[idx]:8.4f} Text='{token_text}'")
    
    print("\n" + "="*80)
    print("BOTTOM-20 TOKENS (Lowest logits)")
    print("="*80)
    bottom_indices = np.argsort(logits_np)[:20]
    for i, idx in enumerate(bottom_indices):
        token_text = tokenizer.decode([idx])
        print(f"{i+1:2d}. ID={idx:6d} Logit={logits_np[idx]:8.4f} Text='{token_text}'")
    
    # Check if bottom-k is just 0-1999
    bottom_2000 = np.argsort(logits_np)[:2000]
    is_sequential = np.all(bottom_2000 == np.arange(2000))
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    if is_sequential:
        print("❌ CRITICAL: Bottom-2000 tokens are exactly IDs 0-1999!")
        print("   This indicates the model's logits have collapsed.")
        print("   The model is broken and needs to be retrained.")
    elif is_monotonic:
        print("❌ WARNING: Logits are monotonic (perfectly sorted by ID)")
        print("   This is highly unusual and suggests numerical issues.")
    elif np.isnan(logits_np).any() or np.isinf(logits_np).any():
        print("❌ ERROR: Logits contain NaN or Inf values!")
        print("   Training diverged. Check loss and grad_norm in logs.")
    else:
        print("✓ Logits distribution looks normal.")
        print("  Mean and std are reasonable, no NaN/Inf detected.")
    
    # Check logit range
    logit_range = np.max(logits_np) - np.min(logits_np)
    print(f"\nLogit range: {logit_range:.4f}")
    if logit_range < 1.0:
        print("❌ WARNING: Logit range is very small (< 1.0)")
        print("   Model may have collapsed or is undertrained.")
    elif logit_range > 100.0:
        print("❌ WARNING: Logit range is very large (> 100.0)")
        print("   Model may have numerical instability.")
    else:
        print("✓ Logit range is reasonable.")


def main():
    parser = argparse.ArgumentParser(description="Debug model logits distribution")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (e.g., ./quick_debug/final_model)")
    parser.add_argument("--prompt", type=str, default="The",
                       help="Prompt to use for logits computation")
    
    args = parser.parse_args()
    check_logits(args.model_path, args.prompt)


if __name__ == "__main__":
    main()
