"""
run_bottomk_lineage_pipeline_v2.py

New Algorithm:
    fingerprint_prompt = generate_prompts(base_model)
    bottomk_tokens_base = get_tokens(base_model, base_tokenizer)

    base_response = constrained_generate(base_model, prompt, bottomk_tokens_base)
    derived_response = constrained_generate(derived_model, prompt, bottomk_tokens_base)

    score = similarity(base_response, derived_response)

Key difference from v1:
    - Both base and derived models use the SAME constraint (base's bottom-k vocab)
    - This provides a fairer comparison under identical generation constraints
"""

from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)

from bottomk_logits_processor import (
    BottomKLogitsProcessor,
    compute_bottomk_vocab_for_model,
)

from test_batch import (
    sample_fingerprint_prompt,
    format_full_prompt,
    metric_prefix_match,
    metric_lcs_ratio,
    metric_signature_overlap,
)


# ----------------- helpers -----------------


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    torch_dtype=torch.float16,
):
    """Load a HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer) -> None:
    """Free GPU memory."""
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------- core functions -----------------


def generate_fingerprint_prompts(
    model,
    tokenizer,
    num_prompts: int,
    prompt_style: str = "raw",
    l_random_prefix: int = 8,
    total_len: int = 64,
    k_bottom: int = 50,
    device: Optional[str] = None,
) -> List[str]:
    """
    Generate fingerprint prompts using the base model.
    
    Returns a list of formatted prompts ready for generation.
    """
    if device is None:
        device = next(model.parameters()).device

    prompts = []
    for i in range(num_prompts):
        # Generate x' using bottom-k sampling
        x_prime = sample_fingerprint_prompt(
            model=model,
            tokenizer=tokenizer,
            device=device,
            l_random_prefix=l_random_prefix,
            total_len=total_len,
            k_bottom=k_bottom,
        )
        # Format the prompt
        full_prompt = format_full_prompt(x_prime, prompt_style=prompt_style)
        prompts.append(full_prompt)
        print(f"[prompt {i+1}/{num_prompts}] x': {x_prime[:80]}...")

    return prompts


@torch.no_grad()
def constrained_generate(
    model,
    tokenizer,
    prompt: str,
    allowed_token_ids: List[int],
    max_new_tokens: int = 64,
    device: Optional[str] = None,
) -> str:
    """
    Generate text constrained to only use tokens in allowed_token_ids.
    
    Args:
        model: HuggingFace model
        tokenizer: corresponding tokenizer
        prompt: input prompt text
        allowed_token_ids: list of token ids that are allowed to be generated
        max_new_tokens: maximum number of new tokens to generate
        device: device to run on
    
    Returns:
        Generated continuation text (excluding the prompt)
    """
    if device is None:
        device = next(model.parameters()).device

    # Create the constrained logits processor
    processors = LogitsProcessorList([
        BottomKLogitsProcessor(allowed_token_ids=allowed_token_ids)
    ])

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy decoding
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
    )[0]

    # Extract only the generated part
    gen_ids = output_ids[input_len:]
    if gen_ids.numel() > 0:
        return tokenizer.decode(gen_ids, skip_special_tokens=True)
    return ""


def compute_similarity(
    base_response: str,
    derived_response: str,
    min_prefix_len: int = 30,
    sig_min_tok_len: int = 6,
) -> Dict[str, float]:
    """
    Compute similarity metrics between base and derived responses.
    
    Returns dict with:
        - prefix_match: 1 if first min_prefix_len chars match, else 0
        - lcs_ratio: longest common subsequence ratio
        - sig_overlap: signature token overlap rate
        - avg_score: average of all metrics
    """
    prefix_match = metric_prefix_match(base_response, derived_response, min_len=min_prefix_len)
    lcs_ratio = metric_lcs_ratio(base_response, derived_response)
    sig_hits, sig_total = metric_signature_overlap(base_response, derived_response, min_tok_len=sig_min_tok_len)
    sig_overlap = sig_hits / sig_total if sig_total > 0 else 0.0

    avg_score = (prefix_match + lcs_ratio + sig_overlap) / 3.0

    return {
        "prefix_match": float(prefix_match),
        "lcs_ratio": float(lcs_ratio),
        "sig_overlap": float(sig_overlap),
        "avg_score": float(avg_score),
    }


# ----------------- main pipeline -----------------


def eval_base_derived_pair(
    base_model_name: str,
    derived_model_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Evaluate similarity between a base model and a derived (suspect) model.
    
    Algorithm:
        1. Load base model, generate fingerprint prompts
        2. Compute base model's bottom-k vocab
        3. For each prompt:
            - base_response = constrained_generate(base_model, prompt, base_bottomk)
            - derived_response = constrained_generate(derived_model, prompt, base_bottomk)
            - score = similarity(base_response, derived_response)
        4. Aggregate scores
    """
    device = args.device

    print(f"\n{'='*60}")
    print(f"[eval] Base: {base_model_name}")
    print(f"[eval] Derived: {derived_model_name}")
    print(f"{'='*60}")

    # 1. Load base model
    print(f"\n[1/4] Loading base model: {base_model_name}")
    base_model, base_tok = load_model_and_tokenizer(base_model_name, device=device)

    try:
        # 2. Compute base model's bottom-k vocab
        print(f"[2/4] Computing base model bottom-k vocab (k={args.bottom_k_vocab})...")
        base_bottomk_ids = compute_bottomk_vocab_for_model(
            base_model,
            base_tok,
            k=args.bottom_k_vocab,
            device=device,
        )
        base_bottomk_ids = list(base_bottomk_ids)
        print(f"       Bottom-k vocab size: {len(base_bottomk_ids)}")

        # 3. Generate fingerprint prompts
        print(f"[3/4] Generating {args.num_prompts} fingerprint prompts...")
        prompts = generate_fingerprint_prompts(
            model=base_model,
            tokenizer=base_tok,
            num_prompts=args.num_prompts,
            prompt_style=args.prompt_style,
            l_random_prefix=args.l_random_prefix,
            total_len=args.total_len,
            k_bottom=args.k_bottom_prompt,
            device=device,
        )

        # 4. Generate base responses (constrained to base's bottom-k)
        print(f"[4/4] Generating constrained responses...")
        base_responses = []
        for i, prompt in enumerate(prompts):
            resp = constrained_generate(
                model=base_model,
                tokenizer=base_tok,
                prompt=prompt,
                allowed_token_ids=base_bottomk_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            base_responses.append(resp)
            print(f"  [base {i+1}/{len(prompts)}] {resp[:60]}...")

    finally:
        # Unload base model to free memory before loading derived model
        unload_model(base_model, base_tok)

    # 5. Load derived model
    print(f"\n[5/6] Loading derived model: {derived_model_name}")
    derived_model, derived_tok = load_model_and_tokenizer(derived_model_name, device=device)

    try:
        # 6. Generate derived responses using BASE's bottom-k constraint
        print(f"[6/6] Generating derived model responses (using base's bottom-k constraint)...")
        
        # Check tokenizer compatibility
        base_vocab_size = len(base_tok) if hasattr(base_tok, '__len__') else base_tok.vocab_size
        derived_vocab_size = len(derived_tok) if hasattr(derived_tok, '__len__') else derived_tok.vocab_size
        
        if base_vocab_size != derived_vocab_size:
            print(f"  [warn] Vocab size mismatch: base={base_vocab_size}, derived={derived_vocab_size}")
            # Filter out invalid token ids
            safe_bottomk_ids = [tid for tid in base_bottomk_ids if tid < derived_vocab_size]
            print(f"  [warn] Using {len(safe_bottomk_ids)}/{len(base_bottomk_ids)} valid token ids")
        else:
            safe_bottomk_ids = base_bottomk_ids

        derived_responses = []
        for i, prompt in enumerate(prompts):
            resp = constrained_generate(
                model=derived_model,
                tokenizer=derived_tok,
                prompt=prompt,
                allowed_token_ids=safe_bottomk_ids,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            derived_responses.append(resp)
            print(f"  [derived {i+1}/{len(prompts)}] {resp[:60]}...")

    finally:
        unload_model(derived_model, derived_tok)

    # 7. Compute similarities
    print(f"\n[7/7] Computing similarity scores...")
    all_scores = []
    for i, (base_resp, derived_resp) in enumerate(zip(base_responses, derived_responses)):
        scores = compute_similarity(
            base_resp, derived_resp,
            min_prefix_len=args.min_prefix_len,
            sig_min_tok_len=args.sig_min_tok_len,
        )
        all_scores.append(scores)
        print(f"  [pair {i+1}] prefix={scores['prefix_match']:.0f}, "
              f"lcs={scores['lcs_ratio']:.3f}, sig={scores['sig_overlap']:.3f}, "
              f"avg={scores['avg_score']:.3f}")

    # 8. Aggregate results
    avg_prefix = sum(s["prefix_match"] for s in all_scores) / len(all_scores)
    avg_lcs = sum(s["lcs_ratio"] for s in all_scores) / len(all_scores)
    avg_sig = sum(s["sig_overlap"] for s in all_scores) / len(all_scores)
    avg_overall = sum(s["avg_score"] for s in all_scores) / len(all_scores)

    result = {
        "base_model": base_model_name,
        "derived_model": derived_model_name,
        "num_prompts": len(prompts),
        "bottom_k_vocab_size": len(base_bottomk_ids),
        "avg_prefix_match": avg_prefix,
        "avg_lcs_ratio": avg_lcs,
        "avg_sig_overlap": avg_sig,
        "avg_overall_score": avg_overall,
        "per_prompt_scores": all_scores,
    }

    print(f"\n[result] Summary:")
    print(f"  avg_prefix_match:  {avg_prefix:.4f}")
    print(f"  avg_lcs_ratio:     {avg_lcs:.4f}")
    print(f"  avg_sig_overlap:   {avg_sig:.4f}")
    print(f"  avg_overall_score: {avg_overall:.4f}")

    return result


def append_result_to_csv(result: Dict[str, Any], csv_path: Path) -> None:
    """Append a single result row to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model",
        "derived_model",
        "num_prompts",
        "bottom_k_vocab_size",
        "avg_prefix_match",
        "avg_lcs_ratio",
        "avg_sig_overlap",
        "avg_overall_score",
        "per_prompt_scores_json",
    ]

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "base_model": result["base_model"],
            "derived_model": result["derived_model"],
            "num_prompts": result["num_prompts"],
            "bottom_k_vocab_size": result["bottom_k_vocab_size"],
            "avg_prefix_match": result["avg_prefix_match"],
            "avg_lcs_ratio": result["avg_lcs_ratio"],
            "avg_sig_overlap": result["avg_sig_overlap"],
            "avg_overall_score": result["avg_overall_score"],
            "per_prompt_scores_json": json.dumps(result["per_prompt_scores"]),
        })

    print(f"[csv] Appended result to {csv_path}")


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bottom-k fingerprint lineage pipeline v2: "
                    "Both models generate under base's bottom-k constraint."
    )

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HF model name for the base model.",
    )
    parser.add_argument(
        "--derived_model",
        type=str,
        required=True,
        help="HF model name for the derived (suspect) model.",
    )

    # Or use lists for batch processing
    parser.add_argument(
        "--base_model_list",
        type=str,
        default=None,
        help="Path to txt file with base model names (one per line). "
             "If provided, --base_model is ignored.",
    )
    parser.add_argument(
        "--derived_model_list",
        type=str,
        default=None,
        help="Path to txt file with derived model names (one per line). "
             "If provided, --derived_model is ignored.",
    )

    # Device and seed
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Fingerprint generation parameters
    parser.add_argument("--num_prompts", type=int, default=10,
                        help="Number of fingerprint prompts per base model.")
    parser.add_argument("--prompt_style", type=str, default="raw",
                        choices=["raw", "oneshot", "chatml"])
    parser.add_argument("--l_random_prefix", type=int, default=8,
                        help="Length of random prefix in fingerprint prompt.")
    parser.add_argument("--total_len", type=int, default=64,
                        help="Total length of fingerprint prompt.")
    parser.add_argument("--k_bottom_prompt", type=int, default=50,
                        help="k for bottom-k sampling during prompt generation.")

    # Bottom-k vocab parameters
    parser.add_argument("--bottom_k_vocab", type=int, default=2000,
                        help="Size of bottom-k vocab for constrained generation.")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max new tokens for constrained generation.")

    # Metric parameters
    parser.add_argument("--min_prefix_len", type=int, default=30)
    parser.add_argument("--sig_min_tok_len", type=int, default=6)

    # Output
    parser.add_argument("--csv_path", type=str, default="lineage_scores_v2.csv")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv_path)

    # Determine base and derived model lists
    if args.base_model_list:
        base_models = [
            line.strip()
            for line in Path(args.base_model_list).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        base_models = [args.base_model]

    if args.derived_model_list:
        derived_models = [
            line.strip()
            for line in Path(args.derived_model_list).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        derived_models = [args.derived_model]

    print(f"[main] Base models: {len(base_models)}")
    print(f"[main] Derived models: {len(derived_models)}")
    print(f"[main] Total pairs to evaluate: {len(base_models) * len(derived_models)}")

    total_pairs = len(base_models) * len(derived_models)
    pair_idx = 0

    for base_model in base_models:
        for derived_model in derived_models:
            pair_idx += 1
            print(f"\n{'#'*60}")
            print(f"# Pair {pair_idx}/{total_pairs}")
            print(f"{'#'*60}")

            try:
                result = eval_base_derived_pair(
                    base_model_name=base_model,
                    derived_model_name=derived_model,
                    args=args,
                )
                append_result_to_csv(result, csv_path)

            except Exception as e:
                print(f"[error] Failed on pair ({base_model}, {derived_model}): {e}")
                import traceback
                traceback.print_exc()

    print(f"\n[main] Done! Results saved to {csv_path}")


if __name__ == "__main__":
    main()

