"""
run_bottomk_lineage_pipeline.py

Pipeline:
  1) For each base (candidate) model, generate fingerprint prompts x' using your
     existing random + bottom-k fingerprint generator.
  2) For each base model, compute a fixed bottom-k vocab (its "fingerprint space").
  3) For a fixed suspect model:
       - compute its own fixed bottom-k vocab (used as decode constraint)
       - for each fingerprint prompt x', let the suspect model generate y_suspect
         using greedy decoding constrained to its bottom-k vocab.
       - measure how many tokens in y_suspect fall inside *base model's* bottom-k vocab.
  4) Write a CSV summarizing per-base-model average overlap, plus per-pair scores.

This script is meant to be a *skeleton* that you can plug into your existing
fingerprint_tools / generate_fingerprints_batch implementation.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
)

from test_batch import (
        set_seed,
        generate_fingerprints_batch,
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure pad_token_id is set for some models
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)
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

    Expected behavior:
        - returns a list of dicts, where each dict has at least one of:
            - "x_prime" (string prompt)
            - "prompt"  (string prompt)
            - "x"       (string prompt)
        - you can modify this function to match your actual structure.

    If `generate_fingerprints_batch` is not available, this function will raise
    an error and you should replace its body with your own implementation.
    """

    # Call your existing batch generator; keep parameter names aligned with your implementation
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

    Modify this if your fingerprint dict uses different keys.
    """
    for key in ("x_prime", "prompt", "x", "input_text"):
        if key in fp:
            return fp[key]
    raise KeyError(
        f"Cannot find prompt text in fingerprint dict keys={list(fp.keys())}. "
        "Please adapt `extract_prompt_from_fingerprint`."
    )


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
        3) compute base bottom-k vocab (fingerprint space)
        4) for each x', let suspect model generate y under its own bottom-k constraint
        5) measure overlap ratio with base bottom-k vocab
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
        base_bottomk_set = set(base_bottomk_ids)
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

        # Prepare suspect logits processor: greedy decoding in its own bottom-k
        processors = LogitsProcessorList(
            [BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids)]
        )

        pair_scores: List[float] = []

        for idx, fp in enumerate(fps):
            prompt_text = extract_prompt_from_fingerprint(fp)
            # Tokenize with suspect tokenizer
            inputs = suspect_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)

            with torch.no_grad():
                output_ids = suspect_model.generate(
                    **inputs,
                    max_new_tokens=args.suspect_max_new_tokens,
                    do_sample=False,  # greedy
                    logits_processor=processors,
                )[0]

            # Only count newly generated tokens (exclude prompt part)
            gen_only_ids = output_ids[inputs["input_ids"].shape[1] :]

            if gen_only_ids.numel() == 0:
                score = 0.0
            else:
                overlap = sum(
                    int(t.item() in base_bottomk_set) for t in gen_only_ids
                )
                score = overlap / float(gen_only_ids.numel())

            pair_scores.append(score)
            print(
                f"[pair {idx+1}/{len(fps)}] overlap ratio with base bottom-k = {score:.4f}"
            )

        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

        result: Dict[str, Any] = {
            "base_model_name": base_model_name,
            "suspect_model_name": args.suspect_model_name,
            "num_pairs": len(pair_scores),
            "avg_overlap_ratio": avg_score,
            "pair_scores": pair_scores,
            "bottom_k_vocab_size": args.bottom_k_vocab,
        }
        return result

    finally:
        unload_hf_model(base_model, base_tok)


from pathlib import Path
import csv
import json

def append_result_csv(
    result: Dict[str, Any],
    csv_path: Path,
) -> None:
    """
    Append one base_model result row-by-row to the CSV.

    If the file does not exist, write the header first;
    Otherwise, just append one row.
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

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Write header if this is a new file
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
        description="Bottom-k fingerprint lineage pipeline."
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
        help="Max new tokens when generating fingerprint y on the base model (if used).",
    )
    parser.add_argument(
        "--suspect_max_new_tokens",
        type=int,
        default=64,
        help="Max new tokens to generate on suspect model under bottom-k constraint.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Max input length when tokenizing fingerprint prompts for suspect model.",
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
        default="lineage_bottomk_scores.csv",
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
        # 2) compute suspect's own bottom-k vocab (decode constraint)
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
                # After finishing one base, write the result row immediately
                append_result_csv(res, csv_path)
            except Exception as e:
                print(f"[error] Failed on base model {base_model_name}: {e}")

        print(f"[main] Done. All processed bases have been appended to {csv_path}.")

    finally:
        unload_hf_model(suspect_model, suspect_tok)

if __name__ == "__main__":
    main()
