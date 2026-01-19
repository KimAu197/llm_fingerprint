"""
fingerprint_gen.py - Fingerprint prompt generation (RoFL-style)
"""
from __future__ import annotations
import os
import time
import json
import random
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F

from .prompt_format import format_full_prompt
from .text_gen import greedy_response_hf


def _ts_path(path: str, model_name) -> str:
    """Generate timestamped path for saving files."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(path)
    return f"{base}_{model_name}{ext or '.json'}"


def _build_allowed_token_set(tokenizer) -> List[int]:
    """Build list of allowed token IDs (excluding special tokens)."""
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
    """
    RoFL Step 1 (simplified implementation):
        (1) The first l tokens are uniformly sampled from the vocab (excluding special tokens)
        (2) Each subsequent step samples uniformly from the k lowest-probability tokens until total_len
    
    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        device: device to run on
        l_random_prefix: number of random prefix tokens
        total_len: total length of fingerprint prompt
        k_bottom: sample from k lowest-probability tokens
    
    Returns:
        fingerprint prompt string
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    allowed = _build_allowed_token_set(tokenizer)
    allowed_t = torch.tensor(allowed, device=device)

    # (1) Uniformly random prefix
    prefix_ids = random.choices(allowed, k=l_random_prefix)
    prompt_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

    # (2) bottom-k extension
    while prompt_ids.shape[1] < total_len:
        logits = model(prompt_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        masked = probs.clone()
        if len(allowed) < tokenizer.vocab_size:
            mask = torch.ones_like(masked, dtype=torch.bool)
            mask[:, allowed_t] = False
            masked[mask] = 1e9  # Prevent disallowed tokens from being in the bottom-k

        _, sorted_idx = torch.sort(masked, dim=-1, descending=False)  # ascending by prob
        k_eff = min(k_bottom, sorted_idx.shape[1])
        bottomk_idx = sorted_idx[:, :k_eff]
        next_id = bottomk_idx[0, random.randrange(k_eff)].view(1, 1)
        prompt_ids = torch.cat([prompt_ids, next_id.to(device)], dim=1)

    return tokenizer.decode(prompt_ids.squeeze(0).cpu(), skip_special_tokens=True)


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
    """
    Generate num_pairs (x', y) fingerprint pairs.
    
    Args:
        model: HuggingFace causal LM
        model_name: name for saving files
        tokenizer: corresponding tokenizer
        num_pairs: number of fingerprint pairs to generate
        prompt_style: 'oneshot' | 'chatml' | 'raw'
        l_random_prefix: number of random prefix tokens
        total_len: total length of fingerprint prompt
        k_bottom: sample from k lowest-probability tokens
        max_new_tokens: max tokens for response generation
        save_json_path: path to save fingerprints (None to skip)
    
    Returns:
        (pairs, output_path) tuple
    """
    device = next(model.parameters()).device
    pairs: List[Dict[str, Any]] = []

    for i in range(num_pairs):
        print(f"[gen] [{i+1}/{num_pairs}]")
        x_prime = sample_fingerprint_prompt(
            model, tokenizer, device=device,
            l_random_prefix=l_random_prefix, total_len=total_len, k_bottom=k_bottom
        )
        full_prompt = format_full_prompt(x_prime, prompt_style=prompt_style)
        y_resp = greedy_response_hf(model, tokenizer, full_prompt, device=device, max_new_tokens=max_new_tokens)

        print("x':", x_prime[:160].replace("\n", "\\n"))
        print("y :", y_resp[:160].replace("\n", "\\n"))

        pairs.append({
            "prompt_style": prompt_style,
            "x_prime": x_prime,
            "y_response": y_resp,
            "full_prompt_used": full_prompt,
        })

    out_path = None
    if save_json_path:
        out_path = _ts_path(save_json_path, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[save] fingerprints -> {out_path}")

    return pairs, out_path






