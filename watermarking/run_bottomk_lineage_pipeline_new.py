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
    metric_prefix_match,
    metric_lcs_ratio,
    metric_signature_overlap,
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

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen–Shannon divergence between two 1D probability vectors p, q on same support.
    返回一个标量 tensor。
    """
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = (p * (p.log() - m.log())).sum()
    kl_qm = (q * (q.log() - m.log())).sum()
    return 0.5 * (kl_pm + kl_qm)


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

    # 这里调用你原来的 batch 生成函数；注意参数名要和你的实现对应
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
        2) generate fingerprint pairs (x', y_base)
        3) compute base fixed bottom-k vocab (fingerprint space)
        4) for each x':
             - compute last-step JS divergence between base vs suspect on base bottom-k vocab
             - let suspect generate greedy continuation under its *own* bottom-k constraint
               and compare with y_base using the 3 text metrics
        5) aggregate metrics and return.
    """
    device = args.device

    print(f"\n[base] Loading base model: {base_model_name}")
    base_model, base_tok = load_hf_model_and_tokenizer(base_model_name, device=device)

    try:
        # 1) base fixed bottom-k vocab
        print("[base] Computing base model bottom-k vocab ...")
        base_bottomk_ids = compute_bottomk_vocab_for_model(
            base_model,
            base_tok,
            k=args.bottom_k_vocab,
            device=device,
        )
        base_bottomk_ids = list(base_bottomk_ids)
        base_bottomk_set = set(base_bottomk_ids)
        print(f"[base] bottom-k size = {len(base_bottomk_ids)}")

        # 2) generate fingerprint pairs (x', y_base)
        print(f"[base] Generating {args.num_pairs} fingerprint pairs ...")
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

        # suspect 用自己的 fixed bottom-k vocab 做 greedy，用于 text metrics
        processors = LogitsProcessorList(
            [BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids)]
        )

        js_scores: List[float] = []
        prefix_hits: List[float] = []
        lcs_scores: List[float] = []
        sig_hits: List[int] = []
        sig_totals: List[int] = []

        # ---- main loop over fingerprint pairs ----
        for idx, fp in enumerate(fps):
            prompt_text = extract_prompt_from_fingerprint(fp)

            # 2.1  JS divergence：只看最后一个位置的下一 token 分布
            # base logits
            base_inputs = base_tok(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            with torch.no_grad():
                base_out = base_model(**base_inputs)
            base_logits_last = base_out.logits[0, -1, :]  # [V]
            base_probs = torch.softmax(base_logits_last, dim=-1)

            # suspect logits（只 forward 一次，不生成）
            sus_inputs = suspect_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            with torch.no_grad():
                sus_out = suspect_model(**sus_inputs)
            sus_logits_last = sus_out.logits[0, -1, :]
            sus_probs = torch.softmax(sus_logits_last, dim=-1)

            # 限制到 base 的 fixed bottom-k vocab
            vocab_size_sus = sus_probs.size(0)

            # 过滤掉在 suspect vocab 里越界的 id
            safe_base_bottomk_ids = [tid for tid in base_bottomk_ids if 0 <= tid < vocab_size_sus]

            if len(safe_base_bottomk_ids) == 0:
                # 极端情况：两边 vocab 完全对不上，直接跳过这个 pair
                print(f"[warn] no safe bottom-k ids for pair {idx}, skip JS")
                continue

            idx_tensor = torch.tensor(
                safe_base_bottomk_ids, device=device, dtype=torch.long
            )

            p_k = base_probs[idx_tensor]
            q_k = sus_probs[idx_tensor]
            p_k = p_k / (p_k.sum() + 1e-12)
            q_k = q_k / (q_k.sum() + 1e-12)
            js = js_divergence(p_k, q_k).item()
            js_scores.append(js)
            # 重新归一化
            p_k = p_k / (p_k.sum() + 1e-12)
            q_k = q_k / (q_k.sum() + 1e-12)

            js = js_divergence(p_k, q_k).item()
            js_scores.append(js)

            # 2.2  suspect 在自己的 bottom-k vocab 里 greedy 生成 continuation
            sus_gen_inputs = suspect_tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            with torch.no_grad():
                output_ids = suspect_model.generate(
                    **sus_gen_inputs,
                    max_new_tokens=args.suspect_max_new_tokens,
                    do_sample=False,  # greedy
                    logits_processor=processors,
                )[0]

            # 只保留 continuation 部分（去掉 prompt）
            gen_only_ids = output_ids[sus_gen_inputs["input_ids"].shape[1] :]
            if gen_only_ids.numel() > 0:
                suspect_y = suspect_tokenizer.decode(
                    gen_only_ids, skip_special_tokens=True
                )
            else:
                suspect_y = ""

            # base 的 y 从 fingerprint 里拿
            base_y = fp.get("y_response", "")

            # 2.3  文本相似度：沿用你之前那三个 metric
            min_prefix_len = getattr(args, "min_prefix_len", 30)
            sig_min_tok_len = getattr(args, "sig_min_tok_len", 6)

            pm = metric_prefix_match(base_y, suspect_y, min_len=min_prefix_len)
            lcs = metric_lcs_ratio(base_y, suspect_y)
            h, tot = metric_signature_overlap(
                base_y, suspect_y, min_tok_len=sig_min_tok_len
            )

            prefix_hits.append(float(pm))
            lcs_scores.append(float(lcs))
            sig_hits.append(int(h))
            sig_totals.append(int(tot))

            print(
                f"[pair {idx+1}/{len(fps)}] "
                f"JS={js:.4f}, prefix={pm}, lcs={lcs:.3f}, sig=({h}/{tot})"
            )

        # ---- aggregate over pairs ----
        avg_js = float(sum(js_scores) / len(js_scores)) if js_scores else 0.0
        prefix_match_rate = (
            float(sum(prefix_hits) / len(prefix_hits)) if prefix_hits else 0.0
        )
        avg_lcs = float(sum(lcs_scores) / len(lcs_scores)) if lcs_scores else 0.0
        if sum(sig_totals) > 0:
            sig_overlap_rate = float(sum(sig_hits)) / float(sum(sig_totals))
        else:
            sig_overlap_rate = 0.0

        # 简单平均一个 text overall 分数
        avg_text_score = (prefix_match_rate + avg_lcs + sig_overlap_rate) / 3.0

        result: Dict[str, Any] = {
            "base_model_name": base_model_name,
            "suspect_model_name": args.suspect_model_name,
            "num_pairs": len(js_scores),
            "avg_js_divergence": avg_js,
            "avg_prefix_match_rate": prefix_match_rate,
            "avg_lcs_ratio": avg_lcs,
            "avg_sig_overlap_rate": sig_overlap_rate,
            "avg_text_score": avg_text_score,
            # 方便以后想画分布：
            "per_pair_js": js_scores,
            "bottom_k_vocab_size": len(base_bottomk_ids),
        }

        print("\n[base] Summary for", base_model_name)
        print("  avg_js_divergence      :", avg_js)
        print("  avg_prefix_match_rate  :", prefix_match_rate)
        print("  avg_lcs_ratio          :", avg_lcs)
        print("  avg_sig_overlap_rate   :", sig_overlap_rate)
        print("  avg_text_score (mean)  :", avg_text_score)

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
    逐条把一个 base_model 的结果 append 到 CSV 里。

    如果文件不存在，就先写 header；
    如果已经存在，就只 append 一行。
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "base_model_name",
        "suspect_model_name",
        "num_pairs",
        "avg_js_divergence",
        "avg_prefix_match_rate",
        "avg_lcs_ratio",
        "avg_sig_overlap_rate",
        "avg_text_score",
        "bottom_k_vocab_size",
        "per_pair_js_json",
    ]

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "base_model_name": result["base_model_name"],
                "suspect_model_name": result["suspect_model_name"],
                "num_pairs": result["num_pairs"],
                "avg_js_divergence": result["avg_js_divergence"],
                "avg_prefix_match_rate": result["avg_prefix_match_rate"],
                "avg_lcs_ratio": result["avg_lcs_ratio"],
                "avg_sig_overlap_rate": result["avg_sig_overlap_rate"],
                "avg_text_score": result["avg_text_score"],
                "bottom_k_vocab_size": result["bottom_k_vocab_size"],
                "per_pair_js_json": json.dumps(result["per_pair_js"]),
            }
        )

    print(f"[csv] appended result for base={result['base_model_name']}")


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
                # 跑完一个 base 立刻写入一行
                append_result_csv(res, csv_path)
            except Exception as e:
                print(f"[error] Failed on base model {base_model_name}: {e}")

        print(f"[main] Done. All processed bases have been appended to {csv_path}.")

    finally:
        unload_hf_model(suspect_model, suspect_tok)

if __name__ == "__main__":
    main()