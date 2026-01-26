#!/usr/bin/env python3
"""
Analyze wordlist changes during training to debug overlap decrease.

Usage:
    python analyze_wordlist_changes.py --output_dir ./exp_output
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_wordlist_debug(output_dir: str, step: int) -> Dict[str, Any]:
    """Load wordlist debug file for a specific step."""
    debug_file = Path(output_dir) / f"wordlist_debug_step_{step}.json"
    if not debug_file.exists():
        return None
    
    with open(debug_file, 'r') as f:
        return json.load(f)


def analyze_token_changes(base_tokens: List[int], ft_tokens: List[int], tokenizer) -> Dict[str, Any]:
    """Analyze what changed between base and fine-tuned wordlists."""
    base_set = set(base_tokens)
    ft_set = set(ft_tokens)
    
    # Tokens that stayed
    intersection = base_set & ft_set
    
    # Tokens that disappeared from base
    removed = base_set - ft_set
    
    # New tokens in fine-tuned
    added = ft_set - base_set
    
    return {
        "overlap_count": len(intersection),
        "overlap_ratio": len(intersection) / len(base_set) if base_set else 0,
        "removed_count": len(removed),
        "added_count": len(added),
        "removed_tokens": list(removed)[:20],  # First 20
        "added_tokens": list(added)[:20],  # First 20
        "removed_text": [tokenizer.decode([t]) for t in list(removed)[:20]],
        "added_text": [tokenizer.decode([t]) for t in list(added)[:20]],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze wordlist changes during training")
    parser.add_argument("--output_dir", type=str, required=True, help="Training output directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 10, 20, 50, 100],
                       help="Steps to analyze (default: 1 10 20 50 100)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Tokenizer to use for decoding")
    
    args = parser.parse_args()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    
    # Find all available debug files
    output_path = Path(args.output_dir)
    debug_files = sorted(output_path.glob("wordlist_debug_step_*.json"))
    available_steps = [int(f.stem.split("_")[-1]) for f in debug_files]
    
    print(f"\nAvailable steps: {available_steps}")
    
    # Analyze specified steps
    steps_to_analyze = [s for s in args.steps if s in available_steps]
    if not steps_to_analyze:
        steps_to_analyze = available_steps[:5]  # First 5 if none specified
    
    print(f"Analyzing steps: {steps_to_analyze}\n")
    
    # Load step 1 as baseline
    step1_data = load_wordlist_debug(args.output_dir, 1)
    if not step1_data:
        print("Error: Could not load step 1 data")
        return
    
    print("=" * 80)
    print("WORDLIST CHANGES ANALYSIS")
    print("=" * 80)
    
    for step in steps_to_analyze:
        data = load_wordlist_debug(args.output_dir, step)
        if not data:
            print(f"\nStep {step}: No data available")
            continue
        
        print(f"\n{'='*80}")
        print(f"STEP {step} (Avg Overlap: {data['avg_overlap']:.4f})")
        print(f"{'='*80}")
        
        # Analyze each fingerprint
        for fp_data in data['fingerprints'][:3]:  # First 3 fingerprints
            idx = fp_data['fingerprint_idx']
            # Use full wordlists if available, otherwise use preview
            base_tokens = fp_data.get('base_bottomk_full', fp_data.get('base_bottomk', []))
            ft_tokens = fp_data.get('ft_bottomk_full', fp_data.get('ft_bottomk', []))
            
            analysis = analyze_token_changes(base_tokens, ft_tokens, tokenizer)
            
            print(f"\nFingerprint {idx}:")
            print(f"  Prompt: {fp_data['prompt'][:80]}...")
            print(f"  Overlap: {analysis['overlap_ratio']:.4f} ({analysis['overlap_count']}/{len(base_tokens)})")
            print(f"  Removed: {analysis['removed_count']} tokens")
            print(f"  Added: {analysis['added_count']} tokens")
            
            # Show some examples of removed/added tokens
            if analysis['removed_text']:
                print(f"  Removed examples: {analysis['removed_text'][:10]}")
            if analysis['added_text']:
                print(f"  Added examples: {analysis['added_text'][:10]}")
    
    # Compare step 1 vs latest step
    print(f"\n{'='*80}")
    print("COMPARISON: Step 1 vs Latest Step")
    print(f"{'='*80}")
    
    latest_step = max(available_steps)
    latest_data = load_wordlist_debug(args.output_dir, latest_step)
    
    if latest_data:
        print(f"\nStep 1 → Step {latest_step}")
        print(f"Overlap change: 1.0000 → {latest_data['avg_overlap']:.4f}")
        
        # Detailed comparison for first fingerprint
        fp1_step1 = step1_data['fingerprints'][0]
        fp1_latest = latest_data['fingerprints'][0]
        
        base_tokens = fp1_step1.get('base_bottomk_full', fp1_step1.get('base_bottomk', []))
        ft_tokens = fp1_latest.get('ft_bottomk_full', fp1_latest.get('ft_bottomk', []))
        
        analysis = analyze_token_changes(base_tokens, ft_tokens, tokenizer)
        
        print(f"\nFirst Fingerprint Detailed Analysis:")
        print(f"  Overlap: {analysis['overlap_ratio']:.4f}")
        print(f"  Tokens removed: {analysis['removed_count']}")
        print(f"  Tokens added: {analysis['added_count']}")
        
        print(f"\n  Top 20 removed tokens (from base bottom-k):")
        for i, (token_id, text) in enumerate(zip(analysis['removed_tokens'][:20], 
                                                   analysis['removed_text'][:20])):
            print(f"    {i+1}. ID={token_id:6d} Text='{text}'")
        
        print(f"\n  Top 20 added tokens (to fine-tuned bottom-k):")
        for i, (token_id, text) in enumerate(zip(analysis['added_tokens'][:20],
                                                   analysis['added_text'][:20])):
            print(f"    {i+1}. ID={token_id:6d} Text='{text}'")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
