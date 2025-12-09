"""
run_bottomk_subspace_overlap.py

Pipeline: compare bottom-k subspace overlap between base and suspect models.

Steps:
  1) For each base (candidate) model, generate fingerprint prompts x'
     using your existing random + bottom-k fingerprint generator.
  2) For each fingerprint prompt:
       - compute base bottom-k vocab conditioned on that prompt
       - compute suspect bottom-k vocab conditioned on the same prompt
       - compute overlap ratio between the two bottom-k vocab sets
  3) Write CSV summarizing per-base-model average overlap, plus per-pair scores.

Differences vs run_bottomk_lineage_pipeline.py:
  - We do NOT compare generated text.
  - The score is purely subspace overlap: |intersection| / k.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from test_batch import (
    set_seed,
    sample_fingerprint_prompt,
)

from bottomk_logits_processor import (
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
    Generate ONLY fingerprint prompts (x') using bottom-k sampling.
    Does NOT generate y; avoids extra generation cost.
    """
    device = next(model.parameters()).device
    pairs: List[Dict[str, Any]] = []
    for i in range(num_pairs):
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


# ----------------- main pipeline logic -----------------


def overlap_ratio(set_a: Sequence[int], set_b: Sequence[int]) -> float:
    """Compute |intersection| / |set_a| (both sets are size k, so symmetric)."""
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    inter = len(sa.intersection(sb))
    return inter / float(len(sa))


def eval_one_base_model(
    base_model_name: str,
    suspect_model,
    suspect_tokenizer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    For a single base model:
        1) load base model
        2) generate fingerprint prompts x'
        3) for each prompt, compute bottom-k vocab for base and suspect (conditioned on prompt)
        4) measure overlap ratio between the two bottom-k sets
    """
    device = args.device

    print(f"\n[base] Loading base model: {base_model_name}")
    base_model, base_tok = load_hf_model_and_tokenizer(base_model_name, device=device)

    try:
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
        print(f"[base] got {len(fps)} fingerprint prompts")

        pair_scores: List[float] = []
        pair_records: List[Dict[str, Any]] = []

        for idx, fp in enumerate(fps):
            prompt_text = extract_prompt_from_fingerprint(fp)

            # Compute bottom-k vocab for base on this prompt
            base_bottomk_ids = compute_bottomk_vocab_for_model(
                base_model,
                base_tok,
                k=args.bottom_k_vocab,
                device=device,
                prompt=prompt_text,
            )

            # Compute bottom-k vocab for suspect on the same prompt
            suspect_bottomk_ids = compute_bottomk_vocab_for_model(
                suspect_model,
                suspect_tokenizer,
                k=args.bottom_k_vocab,
                device=device,
                prompt=prompt_text,
            )

            score = overlap_ratio(base_bottomk_ids, suspect_bottomk_ids)
            pair_scores.append(score)

            pair_records.append(
                {
                    "fingerprint": prompt_text,
                    "overlap_ratio": score,
                    "base_bottomk_ids": base_bottomk_ids,
                    "suspect_bottomk_ids": suspect_bottomk_ids,
                }
            )

            print(
                f"[pair {idx+1}/{len(fps)}] overlap ratio between bottom-k sets = {score:.4f}"
            )

        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

        result: Dict[str, Any] = {
            "base_model_name": base_model_name,
            "suspect_model_name": args.suspect_model_name,
            "num_pairs": len(pair_scores),
            "avg_overlap_ratio": avg_score,
            "bottom_k_vocab_size": args.bottom_k_vocab,
            "pair_scores": pair_scores,
            "pair_records": pair_records,
        }
        return result

    finally:
        unload_hf_model(base_model, base_tok)


def append_result_csv(
    result: Dict[str, Any],
    csv_path: Path,
) -> None:
    """
    Append one base_model result row-by-row to the CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model_name",
        "suspect_model_name",
        "num_pairs",
        "avg_overlap_ratio",
        "bottom_k_vocab_size",
        "pair_scores_json",
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
                "avg_overlap_ratio": result["avg_overlap_ratio"],
                "bottom_k_vocab_size": result["bottom_k_vocab_size"],
                "pair_scores_json": json.dumps(result["pair_scores"]),
            }
        )

    print(f"[csv] appended result for base={result['base_model_name']} to {csv_path}")


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bottom-k subspace overlap lineage pipeline."
    )

    parser.add_argument(
        "--suspect_model_name",
        type=str,
        required=True,
        help="HF model name for the fixed suspect model (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0).",
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
        help="Number of fingerprint prompts per base model.",
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
        help="Max new tokens when generating fingerprints (unused for scoring).",
    )
    parser.add_argument(
        "--bottom_k_vocab",
        type=int,
        default=2000,
        help="Size of bottom-k vocab for both base + suspect models.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="lineage_bottomk_subspace_overlap.csv",
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
        # 2) load candidate base model list
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
                    args=args,
                )
                # After finishing one base, write the result row immediately
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

