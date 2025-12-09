"""
run_bottomk_lineage_pipeline_text.py

Pipeline for comparing bottom-k generated text between base and suspect models:
  1) For each base (candidate) model:
       - Generate fingerprint prompts x' using RoFL-style bottom-k sampling
       - For each x', generate text y_base using bottom-k constrained decoding
  2) For the suspect model:
       - For each fingerprint prompt x', generate text y_suspect using its own bottom-k constrained decoding
  3) Compare y_base vs y_suspect using text similarity metrics:
       - PAL_k (Prefix Agreement Length)
       - Levenshtein similarity
       - LCS ratio
  4) Write CSV with per-base-model average scores and per-pair details.

This allows detecting if two models share similar "behavior" in the low-probability token space.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)

from test_batch import (
    set_seed,
    generate_fingerprints_batch,
    lineage_score_simple,
)

from bottomk_logits_processor import (
    BottomKLogitsProcessor,
    compute_bottomk_vocab_for_model,
)


# ----------------- helpers -----------------


def default_set_seed(seed: int) -> None:
    """Fallback seed setter if you don't have your own."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_set_seed(seed: int) -> None:
    if set_seed is not None:
        set_seed(seed)
    else:
        default_set_seed(seed)


def load_hf_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    if device:
        model.to(device)
    model.eval()
    return model, tokenizer


def unload_hf_model(model, tokenizer) -> None:
    """Small helper to free CUDA/MPS memory."""
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------- fingerprint generation wrapper -----------------


def generate_fingerprints_for_base(
    model,
    model_name: str,
    tokenizer,
    num_pairs: int,
    prompt_style: str,
    k_bottom_random_prefix: int,
    total_len: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Thin wrapper around your existing `generate_fingerprints_batch`.
    Returns a list of dicts with x_prime, y_response, full_prompt_used, etc.
    """
    pairs, _ = generate_fingerprints_batch(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        num_pairs=num_pairs,
        prompt_style=prompt_style,
        l_random_prefix=8,
        total_len=total_len,
        k_bottom=k_bottom_random_prefix,
        max_new_tokens=max_new_tokens,
        save_json_path=None,
    )
    return pairs


def extract_prompt_from_fingerprint(fp: Dict[str, Any]) -> str:
    """
    Try to extract the textual prompt x' from a fingerprint dict.
    """
    for key in ("x_prime", "prompt", "x", "input_text"):
        if key in fp:
            return fp[key]
    raise KeyError(
        f"Cannot find prompt text in fingerprint dict keys={list(fp.keys())}. "
        "Please adapt `extract_prompt_from_fingerprint`."
    )


def extract_full_prompt_from_fingerprint(fp: Dict[str, Any]) -> str:
    """
    Extract the full prompt (with role markers) from fingerprint dict.
    """
    if "full_prompt_used" in fp:
        return fp["full_prompt_used"]
    x_prime = extract_prompt_from_fingerprint(fp)
    return f"user: {x_prime}\nassistant:"


# ----------------- bottom-k text generation -----------------


@torch.no_grad()
def generate_text_with_bottomk(
    model,
    tokenizer,
    prompt_text: str,
    bottomk_ids: Sequence[int],
    max_new_tokens: int,
    max_input_length: int,
    device: str,
) -> str:
    """
    Generate text using bottom-k constrained decoding.
    Only tokens in bottomk_ids are allowed during generation.
    
    Returns the generated text (excluding prompt).
    """
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    
    input_len = inputs["input_ids"].shape[1]
    
    processors = LogitsProcessorList(
        [BottomKLogitsProcessor(allowed_token_ids=bottomk_ids)]
    )
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy
        logits_processor=processors,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    
    # Decode only the newly generated tokens
    gen_only_ids = output_ids[input_len:]
    gen_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return gen_text


# ----------------- main pipeline logic -----------------


def eval_one_base_model(
    base_model_name: str,
    suspect_model,
    suspect_tokenizer,
    suspect_bottomk_ids: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    For a single base model:
        1) load base model
        2) generate fingerprint prompts x'
        3) compute base bottom-k vocab
        4) for each x':
           - base model generates y_base using its bottom-k constraint
           - suspect model generates y_suspect using its bottom-k constraint
           - compare y_base vs y_suspect using text similarity metrics
    """
    device = args.device

    print(f"\n[base] Loading base model: {base_model_name}")
    base_model, base_tok = load_hf_model_and_tokenizer(base_model_name, device=device)

    try:
        print("[base] Computing base model bottom-k vocab ...")
        base_bottomk_ids = compute_bottomk_vocab_for_model(
            base_model,
            base_tok,
            k=args.bottom_k_vocab,
            device=device,
        )
        print(f"[base] bottom-k size = {len(base_bottomk_ids)}")

        print(f"[base] Generating {args.num_pairs} fingerprint prompts ...")
        fps = generate_fingerprints_for_base(
            model=base_model,
            model_name=base_model_name,
            tokenizer=base_tok,
            num_pairs=args.num_pairs,
            prompt_style=args.prompt_style,
            k_bottom_random_prefix=args.k_bottom_random_prefix,
            total_len=args.total_len,
            max_new_tokens=args.base_max_new_tokens,
        )
        print(f"[base] got {len(fps)} fingerprint pairs")

        # Collect per-pair metrics
        pair_metrics: List[Dict[str, Any]] = []
        all_pal_k = []
        all_lev_sim = []
        all_lcs_ratio = []
        all_scores = []

        for idx, fp in enumerate(fps):
            full_prompt = extract_full_prompt_from_fingerprint(fp)
            
            # Base model generates text using its bottom-k
            y_base = generate_text_with_bottomk(
                model=base_model,
                tokenizer=base_tok,
                prompt_text=full_prompt,
                bottomk_ids=base_bottomk_ids,
                max_new_tokens=args.base_max_new_tokens,
                max_input_length=args.max_input_length,
                device=device,
            )
            
            # Suspect model generates text using its bottom-k
            y_suspect = generate_text_with_bottomk(
                model=suspect_model,
                tokenizer=suspect_tokenizer,
                prompt_text=full_prompt,
                bottomk_ids=suspect_bottomk_ids,
                max_new_tokens=args.suspect_max_new_tokens,
                max_input_length=args.max_input_length,
                device=device,
            )
            
            # Compute text similarity metrics
            metrics = lineage_score_simple(y_base, y_suspect, k_prefix=args.k_prefix)
            
            pair_metrics.append({
                "fingerprint": extract_prompt_from_fingerprint(fp),
                "y_base": y_base,
                "y_suspect": y_suspect,
                "PAL_chars": metrics["PAL_chars"],
                "PAL_k": metrics["PAL_k"],
                "Lev_sim": metrics["Lev_sim"],
                "LCS_ratio": metrics["LCS_ratio"],
                "LineageScoreSimple": metrics["LineageScoreSimple"],
            })
            
            all_pal_k.append(metrics["PAL_k"])
            all_lev_sim.append(metrics["Lev_sim"])
            all_lcs_ratio.append(metrics["LCS_ratio"])
            all_scores.append(metrics["LineageScoreSimple"])
            
            print(
                f"[pair {idx+1}/{len(fps)}] PAL_k={metrics['PAL_k']:.3f}, "
                f"Lev_sim={metrics['Lev_sim']:.3f}, LCS={metrics['LCS_ratio']:.3f}, "
                f"Score={metrics['LineageScoreSimple']:.3f}"
            )

        # Compute averages
        n = len(pair_metrics) if pair_metrics else 1
        avg_pal_k = sum(all_pal_k) / n if all_pal_k else 0.0
        avg_lev_sim = sum(all_lev_sim) / n if all_lev_sim else 0.0
        avg_lcs_ratio = sum(all_lcs_ratio) / n if all_lcs_ratio else 0.0
        avg_score = sum(all_scores) / n if all_scores else 0.0

        result: Dict[str, Any] = {
            "base_model_name": base_model_name,
            "suspect_model_name": args.suspect_model_name,
            "num_pairs": len(pair_metrics),
            "k_prefix": args.k_prefix,
            "bottom_k_vocab_size": args.bottom_k_vocab,
            "avg_pal_k": avg_pal_k,
            "avg_lev_sim": avg_lev_sim,
            "avg_lcs_ratio": avg_lcs_ratio,
            "avg_score": avg_score,
            "pair_metrics": pair_metrics,
        }
        
        print(f"\n[base] Summary for {base_model_name}:")
        print(f"  avg_PAL_k:     {avg_pal_k:.4f}")
        print(f"  avg_Lev_sim:   {avg_lev_sim:.4f}")
        print(f"  avg_LCS_ratio: {avg_lcs_ratio:.4f}")
        print(f"  avg_Score:     {avg_score:.4f}")
        
        return result

    finally:
        unload_hf_model(base_model, base_tok)


def append_result_csv(
    result: Dict[str, Any],
    csv_path: Path,
) -> None:
    """
    Append one base_model result row to the CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model_name",
        "suspect_model_name",
        "num_pairs",
        "k_prefix",
        "bottom_k_vocab_size",
        "avg_pal_k",
        "avg_lev_sim",
        "avg_lcs_ratio",
        "avg_score",
        "pair_metrics_json",
    ]

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "base_model_name": result["base_model_name"],
                "suspect_model_name": result["suspect_model_name"],
                "num_pairs": result["num_pairs"],
                "k_prefix": result["k_prefix"],
                "bottom_k_vocab_size": result["bottom_k_vocab_size"],
                "avg_pal_k": round(result["avg_pal_k"], 6),
                "avg_lev_sim": round(result["avg_lev_sim"], 6),
                "avg_lcs_ratio": round(result["avg_lcs_ratio"], 6),
                "avg_score": round(result["avg_score"], 6),
                "pair_metrics_json": json.dumps(result["pair_metrics"]),
            }
        )

    print(f"[csv] appended result for base={result['base_model_name']} to {csv_path}")


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bottom-k text comparison lineage pipeline."
    )

    parser.add_argument(
        "--suspect_model_name",
        type=str,
        required=True,
        help="HF model name for the fixed suspect model.",
    )
    parser.add_argument(
        "--candidate_list",
        type=str,
        required=True,
        help="Path to a txt file containing candidate base model names, one per line.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on: cuda / mps / cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=10,
        help="Number of fingerprint pairs per base model.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="raw",
        choices=["raw", "oneshot", "chatml"],
        help="Prompt format used by your fingerprint generator.",
    )
    parser.add_argument(
        "--k_bottom_random_prefix",
        type=int,
        default=50,
        help="k_bottom parameter used when sampling random bottom-k prefix in fingerprint generation.",
    )
    parser.add_argument(
        "--total_len",
        type=int,
        default=64,
        help="Total length used when generating fingerprint prompts.",
    )
    parser.add_argument(
        "--base_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens when generating text on the base model.",
    )
    parser.add_argument(
        "--suspect_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate on suspect model.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Max input length when tokenizing fingerprint prompts.",
    )
    parser.add_argument(
        "--bottom_k_vocab",
        type=int,
        default=2000,
        help="Size of bottom-k vocab for both base + suspect models.",
    )
    parser.add_argument(
        "--k_prefix",
        type=int,
        default=30,
        help="Prefix length for PAL_k metric (chars).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="lineage_bottomk_text_scores.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_set_seed(args.seed)

    device = args.device

    # 1) load suspect model
    print(f"[suspect] Loading suspect model: {args.suspect_model_name}")
    suspect_model, suspect_tok = load_hf_model_and_tokenizer(
        args.suspect_model_name,
        device=device,
    )

    try:
        # 2) compute suspect's own bottom-k vocab
        print("[suspect] Computing suspect bottom-k vocab ...")
        suspect_bottomk_ids = compute_bottomk_vocab_for_model(
            suspect_model,
            suspect_tok,
            k=args.bottom_k_vocab,
            device=device,
        )
        print(f"[suspect] bottom-k size = {len(suspect_bottomk_ids)}")

        # 3) load candidate base model list
        candidate_path = Path(args.candidate_list)
        base_model_names = [
            line.strip()
            for line in candidate_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"[main] Loaded {len(base_model_names)} candidate base models.")

        csv_path = Path(args.csv_path)

        for idx, base_model_name in enumerate(base_model_names):
            print(f"\n[main] ====== ({idx+1}/{len(base_model_names)}) {base_model_name} ======")
            try:
                res = eval_one_base_model(
                    base_model_name=base_model_name,
                    suspect_model=suspect_model,
                    suspect_tokenizer=suspect_tok,
                    suspect_bottomk_ids=suspect_bottomk_ids,
                    args=args,
                )
                append_result_csv(res, csv_path)
            except Exception as e:
                print(f"[error] Failed on base model {base_model_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[main] Done. All processed bases have been appended to {csv_path}.")

    finally:
        unload_hf_model(suspect_model, suspect_tok)


if __name__ == "__main__":
    main()

