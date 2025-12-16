"""Core RoFL fingerprint generation and evaluation utilities."""
from __future__ import annotations

import difflib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    """Provide deterministic behavior across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ts_path(path: str | Path, model_name: str) -> str:
    """Append a timestamp and model identifier to the provided filename."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(str(path))
    return f"{base}_{model_name}{ext or '.json'}"


def format_full_prompt(x_prime_text: str, prompt_style: str = "oneshot") -> str:
    """Format a fingerprint prefix as a standalone prompt without a system block."""
    if prompt_style == "chatml":
        return (
            "<|im_start|>user\n" + x_prime_text + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style == "raw":
        return x_prime_text
    return f"user: {x_prime_text}\nassistant:"


def _build_allowed_token_set(tokenizer) -> List[int]:
    disallow = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            disallow.add(tid)
    if hasattr(tokenizer, "all_special_ids"):
        disallow.update(tokenizer.all_special_ids)
    return [tid for tid in range(tokenizer.vocab_size) if tid not in disallow]


@torch.no_grad()
def sample_fingerprint_prompt(
    model,
    tokenizer,
    device: Optional[str] = None,
    l_random_prefix: int = 8,
    total_len: int = 64,
    k_bottom: int = 50,
) -> str:
    """Simplified RoFL step 1 prompt sampler."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    allowed = _build_allowed_token_set(tokenizer)
    allowed_t = torch.tensor(allowed, device=device)

    prefix_ids = random.choices(allowed, k=l_random_prefix)
    prompt_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

    while prompt_ids.shape[1] < total_len:
        logits = model(prompt_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        masked = probs.clone()
        if len(allowed) < tokenizer.vocab_size:
            mask = torch.ones_like(masked, dtype=torch.bool)
            mask[:, allowed_t] = False
            masked[mask] = 1e9

        _, sorted_idx = torch.sort(masked, dim=-1, descending=False)
        k_eff = min(k_bottom, sorted_idx.shape[1])
        bottomk_idx = sorted_idx[:, :k_eff]
        next_id = bottomk_idx[0, random.randrange(k_eff)].view(1, 1)
        prompt_ids = torch.cat([prompt_ids, next_id.to(device)], dim=1)

    return tokenizer.decode(prompt_ids.squeeze(0).cpu(), skip_special_tokens=True)


@torch.no_grad()
def greedy_response_hf(
    model,
    tokenizer,
    full_prompt_text: str,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
) -> str:
    """Deterministic continuation using HF generation (temperature 0)."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(full_prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    return tokenizer.decode(
        out_ids[input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_fingerprints_batch(
    model,
    model_name: str,
    tokenizer,
    num_pairs: int = 3,
    prompt_style: str = "oneshot",
    l_random_prefix: int = 8,
    total_len: int = 64,
    k_bottom: int = 50,
    max_new_tokens: int = 64,
    save_json_path: Optional[str] = "fingerprints_init.json",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Generate (x', y) fingerprints and optionally save them."""
    device = next(model.parameters()).device
    pairs: List[Dict[str, Any]] = []

    for i in range(num_pairs):
        print(f"[gen] [{i + 1}/{num_pairs}]")
        x_prime = sample_fingerprint_prompt(
            model,
            tokenizer,
            device=device,
            l_random_prefix=l_random_prefix,
            total_len=total_len,
            k_bottom=k_bottom,
        )
        full_prompt = format_full_prompt(x_prime, prompt_style=prompt_style)
        y_resp = greedy_response_hf(
            model,
            tokenizer,
            full_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        print("x':", x_prime[:160].replace("\n", "\\n"))
        print("y :", y_resp[:160].replace("\n", "\\n"))

        pairs.append(
            {
                "prompt_style": prompt_style,
                "x_prime": x_prime,
                "y_response": y_resp,
                "full_prompt_used": full_prompt,
            }
        )

    out_path = None
    if save_json_path:
        out_path = ts_path(save_json_path, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[save] fingerprints -> {out_path}")

    return pairs, out_path


def _normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def metric_prefix_match(a: str, b: str, min_len: int = 30) -> int:
    return int(_normalize_text(a)[:min_len] == _normalize_text(b)[:min_len])


def metric_lcs_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def metric_signature_overlap(a: str, b: str, min_tok_len: int = 6) -> Tuple[int, int]:
    a_norm, b_norm = _normalize_text(a), _normalize_text(b)
    toks = [t for t in a_norm.split() if len(t) >= min_tok_len]
    hits = sum(1 for t in toks if t in b_norm)
    return hits, len(toks)


def _default_stops(prompt_style: str) -> List[str]:
    if prompt_style == "raw":
        return ["</s>", "<|im_end|>"]
    if prompt_style == "chatml":
        return ["</s>", "<|im_end|>", "<|im_start|>user"]
    return ["</s>", "user:", "assistant:"]


def evaluate_fingerprints(
    pairs_json_path: str | Path,
    model_name: str,
    suspect_base_name: str,
    suspect_model: Any,
    suspect_label: str = "suspect",
    save_report_path: str | None = None,
    min_prefix_len: int = 30,
    sig_min_tok_len: int = 6,
    use_timestamp_in_name: bool = False,
):
    """Verify a suspect model by replaying stored fingerprints."""
    with open(pairs_json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"[info] Loaded {len(pairs)} fingerprint pairs from {pairs_json_path}")

    prompt_style = pairs[0].get("prompt_style", "raw")
    stops = _default_stops(prompt_style)

    prefix_hits: List[int] = []
    sim_scores: List[float] = []
    sig_hits: List[int] = []
    sig_totals: List[int] = []
    minimal_records: List[Dict[str, str]] = []

    for idx, pair in enumerate(tqdm(pairs, desc=f"Evaluating on {suspect_label}")):
        x_prime = pair["x_prime"]
        base_y = pair["y_response"]
        full_prompt = pair.get("full_prompt_used") or format_full_prompt(x_prime, prompt_style)

        suspect_y = suspect_model.generate_answer(
            full_prompt,
            max_new_tokens=128,
            stop_tokens=stops,
        )

        pm = metric_prefix_match(base_y, suspect_y, min_len=min_prefix_len)
        sim = metric_lcs_ratio(base_y, suspect_y)
        hit, total = metric_signature_overlap(base_y, suspect_y, min_tok_len=sig_min_tok_len)

        prefix_hits.append(pm)
        sim_scores.append(sim)
        sig_hits.append(hit)
        sig_totals.append(total)

        minimal_records.append(
            {
                "fingerprint": x_prime,
                "base_y": base_y,
                "suspect_y": suspect_y,
            }
        )

    prefix_match_rate = float(np.mean(prefix_hits)) if prefix_hits else 0.0
    avg_edit_sim = float(np.mean(sim_scores)) if sim_scores else 0.0
    sig_overlap_rate = (
        float(np.sum(sig_hits)) / max(1, float(np.sum(sig_totals)))
        if np.sum(sig_totals) > 0
        else 0.0
    )

    print("========== FINGERPRINT VERIFICATION REPORT ==========")
    print(f"Suspect model label: {suspect_label}")
    print(f"#pairs evaluated: {len(pairs)}")
    print(f"Prefix match rate (first {min_prefix_len} chars): {prefix_match_rate:.3f}")
    print(f"Avg edit/sequence similarity (0~1):           {avg_edit_sim:.3f}")
    print(f"Signature phrase overlap (>= {sig_min_tok_len} chars): {sig_overlap_rate:.3f}")
    for rec in minimal_records[:2]:
        print("----")
        print("fingerprint:", rec["fingerprint"][:160].replace("\n", "\\n"))
        print("base_y     :", rec["base_y"][:160].replace("\n", "\\n"))
        print("suspect_y  :", rec["suspect_y"][:160].replace("\n", "\\n"))

    out_obj = {
        "base_model_name": model_name,
        "suspect_model_name": suspect_base_name,
        "prompt_style": prompt_style,
        "num_pairs": len(pairs),
        "records": minimal_records,
    }

    out_path = None
    if save_report_path:
        filename = save_report_path
        if use_timestamp_in_name:
            filename = ts_path(save_report_path, model_name)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        out_path = filename
        print(f"[save] minimal summary -> {out_path}")

    return minimal_records, out_path
