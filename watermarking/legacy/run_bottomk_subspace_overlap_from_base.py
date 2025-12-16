"""
run_bottomk_subspace_overlap_from_base.py

Pipeline: Fixed base model, test subspace overlap on multiple derived models.

Inverted logic from run_bottomk_subspace_overlap.py:
  - Original: candidate base models -> generate fingerprints -> test on fixed suspect
  - This version: fixed base model -> generate fingerprints once -> test on all derived models from CSV

Steps:
  1) Load the fixed base model (e.g., Qwen2.5-0.5B)
  2) Generate fingerprint prompts x' once
  3) Compute bottom-k vocab for base model conditioned on each fingerprint
  4) For each derived model from CSV:
       - compute derived model's bottom-k vocab conditioned on each fingerprint
       - compute overlap ratio between base and derived bottom-k vocab sets
  5) Write CSV summarizing per-derived-model average overlap, plus per-pair scores.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional

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


def load_derived_model_ids_from_csv(csv_path: str, model_id_column: str = "model_id") -> List[str]:
    """Load model IDs from a CSV file."""
    ids: List[str] = []
    if not Path(csv_path).exists():
        print(f"[warn] CSV not found: {csv_path}")
        return ids
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = (row.get(model_id_column) or "").strip()
            if mid:
                ids.append(mid)
    return ids


# ----------------- main pipeline logic -----------------


def overlap_ratio(set_a: Sequence[int], set_b: Sequence[int]) -> float:
    """Compute |intersection| / |set_a| (both sets are size k, so symmetric)."""
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    inter = len(sa.intersection(sb))
    return inter / float(len(sa))


def eval_one_derived_model(
    derived_model_name: str,
    base_model,
    base_tokenizer,
    fingerprints: List[Dict[str, Any]],
    base_bottomk_cache: Dict[str, List[int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    For a single derived model:
        1) load derived model
        2) for each fingerprint prompt, compute derived model's bottom-k vocab
        3) compare with base model's bottom-k (already cached)
        4) measure overlap ratio
    """
    device = args.device

    print(f"\n[derived] Loading derived model: {derived_model_name}")
    try:
        derived_model, derived_tok = load_hf_model_and_tokenizer(derived_model_name, device=device)
    except Exception as e:
        print(f"[error] Failed to load derived model {derived_model_name}: {e}")
        return {
            "base_model_name": args.base_model_name,
            "derived_model_name": derived_model_name,
            "num_pairs": 0,
            "avg_overlap_ratio": -1.0,  # error marker
            "bottom_k_vocab_size": args.bottom_k_vocab,
            "pair_scores": [],
            "error": str(e),
        }

    try:
        pair_scores: List[float] = []
        pair_records: List[Dict[str, Any]] = []

        for idx, fp in enumerate(fingerprints):
            prompt_text = extract_prompt_from_fingerprint(fp)

            # Get base bottom-k vocab from cache
            base_bottomk_ids = base_bottomk_cache[prompt_text]

            # Compute bottom-k vocab for derived model on the same prompt
            derived_bottomk_ids = compute_bottomk_vocab_for_model(
                derived_model,
                derived_tok,
                k=args.bottom_k_vocab,
                device=device,
                prompt=prompt_text,
            )

            score = overlap_ratio(base_bottomk_ids, derived_bottomk_ids)
            pair_scores.append(score)

            pair_records.append(
                {
                    "fingerprint": prompt_text,
                    "overlap_ratio": score,
                    "base_bottomk_ids": base_bottomk_ids,
                    "derived_bottomk_ids": derived_bottomk_ids,
                }
            )

            print(
                f"[pair {idx+1}/{len(fingerprints)}] overlap ratio between bottom-k sets = {score:.4f}"
            )

        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

        result: Dict[str, Any] = {
            "base_model_name": args.base_model_name,
            "derived_model_name": derived_model_name,
            "num_pairs": len(pair_scores),
            "avg_overlap_ratio": avg_score,
            "bottom_k_vocab_size": args.bottom_k_vocab,
            "pair_scores": pair_scores,
            "pair_records": pair_records,
        }
        return result

    finally:
        unload_hf_model(derived_model, derived_tok)


def append_result_csv(
    result: Dict[str, Any],
    csv_path: Path,
) -> None:
    """
    Append one derived_model result row-by-row to the CSV.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model_name",
        "derived_model_name",
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
                "derived_model_name": result["derived_model_name"],
                "num_pairs": result["num_pairs"],
                "avg_overlap_ratio": result["avg_overlap_ratio"],
                "bottom_k_vocab_size": result["bottom_k_vocab_size"],
                "pair_scores_json": json.dumps(result["pair_scores"]),
            }
        )

    print(f"[csv] appended result for derived={result['derived_model_name']} to {csv_path}")


# ----------------- CLI -----------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bottom-k subspace overlap pipeline: fixed base model, test on derived models."
    )

    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="HF model name for the fixed base model (e.g. Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--derived_model_csv",
        type=str,
        required=True,
        help="Path to a CSV file containing derived model names (must have a 'model_id' column).",
    )
    parser.add_argument(
        "--model_id_column",
        type=str,
        default="model_id",
        help="Column name in CSV for model IDs. Default: 'model_id'.",
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
        help="Number of fingerprint prompts to generate from base model.",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="raw",
        choices=["raw", "oneshot", "chatml"],
        help="Prompt format used by fingerprint generator.",
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
        help="Size of bottom-k vocab for both base + derived models.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="lineage_bottomk_subspace_overlap_from_base.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--save_fingerprints",
        type=str,
        default=None,
        help="Optional path to save generated fingerprints as JSON (for reproducibility).",
    )
    parser.add_argument(
        "--load_fingerprints",
        type=str,
        default=None,
        help="Optional path to load existing fingerprints from JSON (skip generation).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_set_seed(args.seed)

    device = args.device

    # 1) Load base model
    print(f"[base] Loading base model: {args.base_model_name}")
    base_model, base_tok = load_hf_model_and_tokenizer(
        args.base_model_name,
        device=device,
    )

    try:
        # 2) Generate (or load) fingerprints ONCE from base model
        if args.load_fingerprints and Path(args.load_fingerprints).exists():
            print(f"[base] Loading fingerprints from: {args.load_fingerprints}")
            with open(args.load_fingerprints, "r", encoding="utf-8") as f:
                fingerprints = json.load(f)
            print(f"[base] Loaded {len(fingerprints)} fingerprints")
        else:
            print(f"[base] Generating {args.num_pairs} fingerprint prompts ...")
            fingerprints = generate_fingerprints_for_base(
                model=base_model,
                model_name=args.base_model_name,
                tokenizer=base_tok,
                num_pairs=args.num_pairs,
                prompt_style=args.prompt_style,
                k_bottom_random_prefix=args.k_bottom_random_prefix,
                total_len=args.total_len,
                max_new_tokens=args.base_max_new_tokens,
            )
            print(f"[base] Generated {len(fingerprints)} fingerprint prompts")

            # Optionally save fingerprints for later reuse
            if args.save_fingerprints:
                with open(args.save_fingerprints, "w", encoding="utf-8") as f:
                    json.dump(fingerprints, f, ensure_ascii=False, indent=2)
                print(f"[base] Saved fingerprints to: {args.save_fingerprints}")

        # 3) Precompute base model's bottom-k vocab for each fingerprint (cache)
        print(f"[base] Precomputing base model bottom-k vocab for {len(fingerprints)} fingerprints ...")
        base_bottomk_cache: Dict[str, List[int]] = {}
        for idx, fp in enumerate(fingerprints):
            prompt_text = extract_prompt_from_fingerprint(fp)
            base_bottomk_ids = compute_bottomk_vocab_for_model(
                base_model,
                base_tok,
                k=args.bottom_k_vocab,
                device=device,
                prompt=prompt_text,
            )
            base_bottomk_cache[prompt_text] = base_bottomk_ids
            print(f"[base] Cached bottom-k for fingerprint {idx+1}/{len(fingerprints)}")

        # 4) Load derived model list from CSV
        derived_model_names = load_derived_model_ids_from_csv(
            args.derived_model_csv,
            model_id_column=args.model_id_column,
        )
        print(f"[main] Loaded {len(derived_model_names)} derived models from CSV.")

        if not derived_model_names:
            print("[main] No derived models to test. Exiting.")
            return

        csv_path = Path(args.csv_path)

        # 5) Iterate through derived models and evaluate
        for idx, derived_model_name in enumerate(derived_model_names):
            print(f"\n[main] ====== ({idx+1}/{len(derived_model_names)}) {derived_model_name} ======")
            try:
                res = eval_one_derived_model(
                    derived_model_name=derived_model_name,
                    base_model=base_model,
                    base_tokenizer=base_tok,
                    fingerprints=fingerprints,
                    base_bottomk_cache=base_bottomk_cache,
                    args=args,
                )
                # After finishing one derived model, write the result row immediately
                append_result_csv(res, csv_path)
            except Exception as e:
                print(f"[error] Failed on derived model {derived_model_name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[main] Done. All processed derived models have been appended to {csv_path}.")

    finally:
        unload_hf_model(base_model, base_tok)


if __name__ == "__main__":
    main()
