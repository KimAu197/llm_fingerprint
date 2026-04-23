"""
GGUF (llama.cpp) helpers for the same bottom-k / fingerprint logic as the HF path.

- Token / decode uses HuggingFace ``AutoTokenizer`` from a model id the GGUF
  was converted from (``tokenizer_from``) so **token id space** matches the
  source checkpoint when the conversion is consistent.
- Logits are taken from ``llama_cpp.Llama`` with ``logits_all=True`` after
  a full ``eval`` on the current token list (re-forward each step, matching the
  HF `sample_fingerprint_prompt` call pattern; causal LM is equivalent to
  incremental forward with cache, but we re-eval the full sequence for clarity).

Requires: ``llama_cpp`` (llama-cpp-python) with a backend suitable for the GGUF
(e.g. Metal/CUDA/CPU as built into the wheel).
"""
from __future__ import annotations

import random
import warnings
from typing import List, Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase

from .fingerprint_gen import _build_allowed_token_set


def _llm_n_vocab(llm) -> int:
    n = getattr(llm, "n_vocab", None)
    if n is None:
        raise RuntimeError("llama_cpp.Llama has no n_vocab")
    return int(n() if callable(n) else n)


def load_hf_tokenizer_only(
    tokenizer_from: str, *, trust_remote_code: bool = True
) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_from, trust_remote_code=trust_remote_code)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = float(np.max(x))
    e = np.exp(x - m)
    s = float(e.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(x, 1.0 / max(x.size, 1), dtype=np.float32)
    return (e / s).astype(np.float32)


def eval_full_sequence_last_logits(llm, token_ids: Sequence[int]) -> np.ndarray:
    """
    ``llm.reset()`` then ``llm.eval(list(token_ids))``; return last-position logits
    of shape (n_vocab,). Requires the Llama instance to have been created with
    ``logits_all=True`` and a sufficient ``n_ctx`` for the sequence length.
    """
    llm.reset()
    tlist = [int(t) for t in token_ids]
    if not tlist:
        raise ValueError("token_ids must be non-empty for last-position logits")
    llm.eval(tlist)
    n = int(llm.n_tokens)
    if n < 1:
        raise RuntimeError("llama eval left n_tokens < 1")
    row = np.array(llm.scores[n - 1, :], copy=True, dtype=np.float32)
    nvc = _llm_n_vocab(llm)
    if row.size != nvc:
        raise RuntimeError(
            f"logits size {row.size} != n_vocab {nvc}"
        )
    return row


def _mask_disallowed_bottomk(probs: np.ndarray, allowed: List[int]) -> np.ndarray:
    m = np.ones(probs.shape[0], dtype=bool)
    m[allowed] = False
    out = probs.copy()
    out[m] = 1.0e9
    return out


def sample_fingerprint_prompt_gguf(
    llm,
    tokenizer: PreTrainedTokenizerBase,
    l_random_prefix: int = 8,
    total_len: int = 64,
    k_bottom: int = 50,
) -> str:
    """
    RoFL-style fingerprint; same control flow as ``sample_fingerprint_prompt``
    (HF) but logits from ``llama_cpp.Llama``.
    """
    n_vm = _llm_n_vocab(llm)
    allowed = _build_allowed_token_set(tokenizer, model_vocab_size=n_vm)
    for tid in allowed:
        if tid < 0 or tid >= n_vm:
            raise ValueError(
                f"allowed id {tid} out of [0, {n_vm}) — tokenizer does not match GGUF; "
                "check --tokenizer_from / converted-from model id"
            )

    prefix_ids = random.choices(allowed, k=l_random_prefix)
    prompt_ids: List[int] = list(prefix_ids)

    while len(prompt_ids) < total_len:
        logits = eval_full_sequence_last_logits(llm, prompt_ids)
        probs = _softmax_np(logits)
        if len(allowed) < int(tokenizer.vocab_size):
            masked = _mask_disallowed_bottomk(probs, allowed)
        else:
            masked = probs
        k_eff = min(k_bottom, masked.size)
        order = np.argsort(masked)  # ascending: lowest prob (bottom) first
        bottomk = order[:k_eff]
        next_id = int(
            bottomk[random.randrange(k_eff)]
        )  # uniform among bottom-k
        if next_id < 0 or next_id >= n_vm:
            raise ValueError(f"next token id {next_id} not in [0, {n_vm})")
        prompt_ids.append(next_id)

    return tokenizer.decode(prompt_ids, skip_special_tokens=True)


def _encode_prompt(tokenizer: PreTrainedTokenizerBase, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=True)


def compute_bottomk_vocab_for_prompts_gguf(
    llm,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    k: int = 2000,
) -> List[List[int]]:
    """
    For each string prompt, tokenize with the HF tokenizer, run eval on the
    model, take bottom-k ids from last-position logits (same definition as
    ``compute_bottomk_vocab_batch`` in the HF path).
    """
    out: List[List[int]] = []
    n_vm = _llm_n_vocab(llm)
    for p in prompts:
        ids = _encode_prompt(tokenizer, p)
        if not ids:
            raise ValueError("empty tokenization for fingerprint string")
        logits = eval_full_sequence_last_logits(llm, ids)
        if logits.size != n_vm:
            raise RuntimeError(f"shape mismatch logits {logits.size} n_vocab {n_vm}")
        k_eff = min(k, n_vm)
        if k_eff < 1:
            raise ValueError("k is 0 or n_vocab is 0")
        order = np.argsort(logits)  # ascending: lowest logits first
        out.append([int(x) for x in order[:k_eff]])
    return out


def compute_bottomk_vocab_gguf_chunked(
    llm,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    k: int = 2000,
    batch_size: int = 1,
) -> List[List[int]]:
    """
    Chunked interface matching ``_compute_all_bottomk_for_model``; for GGUF
    the inner loop is always per-prompt (one LLM in process). ``batch_size`` is
    only a progress/stride tag (default 1).
    """
    results: List[List[int]] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        chunk = prompts[start : start + max(1, batch_size)]
        results.extend(compute_bottomk_vocab_for_prompts_gguf(llm, tokenizer, chunk, k=k))
    return results


def check_tokenizer_vocab_vs_gguf(
    tokenizer: PreTrainedTokenizerBase, llm, extra_check_text: str = "hello"
) -> None:
    """
    Warn (not error) on common size mismatches; can error if a tokenized id
    is out of range.
    """
    n_llm = _llm_n_vocab(llm)
    tvs = int(getattr(tokenizer, "vocab_size", n_llm))
    if tvs != n_llm:
        warnings.warn(
            f"Tokenizer vocab_size ({tvs}) != llama n_vocab ({n_llm}). "
            "For correct overlap, `tokenizer_from` should match the HF source of this GGUF.",
            stacklevel=2,
        )
    toks = _encode_prompt(tokenizer, extra_check_text)
    for t in toks:
        if t < 0 or t >= n_llm:
            raise ValueError(
                f"Token id {t} from ``tokenizer_from`` is outside GGUF n_vocab {n_llm}. "
                "Use the HuggingFace model name this file was actually converted from."
            )
