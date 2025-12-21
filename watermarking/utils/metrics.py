"""
metrics.py - Text similarity metrics for lineage detection
"""
from __future__ import annotations
import re
import unicodedata
import difflib
from typing import Tuple


# =============== Basic text normalization ===============

def _normalize_text(s: str) -> str:
    """Normalize text: lowercase, collapse whitespace."""
    return " ".join(s.strip().lower().split())


# =============== Standard metrics (with normalization) ===============

def metric_prefix_match(a: str, b: str, min_len: int = 30) -> int:
    """Check if first min_len chars match (after normalization)."""
    return int(_normalize_text(a)[:min_len] == _normalize_text(b)[:min_len])


def metric_lcs_ratio(a: str, b: str) -> float:
    """Longest Common Subsequence ratio (0-1)."""
    return difflib.SequenceMatcher(None, _normalize_text(a), _normalize_text(b)).ratio()


def metric_signature_overlap(a: str, b: str, min_tok_len: int = 6) -> Tuple[int, int]:
    """Count how many long tokens from a appear in b."""
    a_norm, b_norm = _normalize_text(a), _normalize_text(b)
    toks = [t for t in a_norm.split() if len(t) >= min_tok_len]
    hits = sum(1 for t in toks if t in b_norm)
    return hits, len(toks)


# =============== Raw metrics (no normalization) ===============

def metric_prefix_match_raw(a: str, b: str, min_len: int = 30) -> int:
    """Check if first min_len chars match (raw, no normalization)."""
    return int(a[:min_len] == b[:min_len])


def metric_lcs_ratio_raw(a: str, b: str) -> float:
    """LCS ratio without normalization."""
    return difflib.SequenceMatcher(None, a, b).ratio()


def metric_signature_overlap_raw(a: str, b: str, min_tok_len: int = 6) -> Tuple[int, int]:
    """Signature overlap without normalization."""
    toks = [t for t in a.split() if len(t) >= min_tok_len]
    hits = sum(1 for t in toks if t in b)
    return hits, len(toks)


# =============== Light cleaning for metrics ===============

_SPECIAL_TAG_PATTERNS = [
    r"<\|[^>]*\|>",                 # ChatML-like: <|system|> <|user|> <|assistant|> <|im_start|> ...
    r"</?s>",                       # <s> </s>
    r"\[/?[A-Z_]+\]",               # [INST], [/INST], [AVAILABLE_TOOLS], [/TOOL_RESULTS], etc.
    r"\[control_\d+\]",             # [control_12] etc.
]


def _clean_for_metrics(s: str) -> str:
    """Clean text by removing special tags and garbage strings."""
    # Normalize encoding & remove control characters/replacement chars
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")                    # '�'
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)         # Control characters

    # Remove common "special tags"
    for pat in _SPECIAL_TAG_PATTERNS:
        s = re.sub(pat, " ", s)

    # Collapse/remove pure separator garbage strings
    s = re.sub(r"(?:<\|){2,}", " ", s)             # Consecutive <|<|<|...
    s = re.sub(r"[<>\|\[\]/]{4,}", " ", s)         # 4+ consecutive < > | [ ] /
    s = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1", s)   # Same punctuation 5+ times → 3 times

    # Normalize whitespace and lowercase
    s = " ".join(s.strip().split()).lower()
    return s


def _normalize_for_metrics(s: str, clean: bool = False) -> str:
    """Normalize text, optionally with cleaning."""
    return _clean_for_metrics(s) if clean else _normalize_text(s)


# =============== Prefix Agreement Length ===============

def _prefix_agreement_len(a: str, b: str, clean: bool = False) -> int:
    """Count how many characters match at the start."""
    a, b = _normalize_for_metrics(a, clean), _normalize_for_metrics(b, clean)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# =============== Levenshtein Similarity ===============

def _norm_lev_sim(a: str, b: str, clean: bool = False) -> float:
    """Normalized Levenshtein similarity (1 - distance/max_len)."""
    a, b = _normalize_for_metrics(a, clean), _normalize_for_metrics(b, clean)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = prev[j-1] if ca == cb else 1 + min(prev[j-1], prev[j], cur[j-1])
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(1, max(la, lb))


# =============== Combined Lineage Score ===============

def lineage_score_simple(base_y: str, suspect_y: str, k_prefix: int = 30, clean: bool = False) -> dict:
    """
    Compute three metrics (equal weight): PAL_k, Levenshtein similarity, LCS ratio.
    
    Args:
        base_y: base model's response
        suspect_y: suspect model's response
        k_prefix: prefix length for PAL_k
        clean: whether to clean special tags before comparison
    
    Returns:
        dict with PAL_chars, PAL_k, Lev_sim, LCS_ratio, LineageScoreSimple
    """
    pal = _prefix_agreement_len(base_y, suspect_y, clean=clean)
    pal_k = 1.0 if pal >= k_prefix else 0.0
    lev_sim = _norm_lev_sim(base_y, suspect_y, clean=clean)
    lcs = metric_lcs_ratio(_normalize_for_metrics(base_y, clean),
                           _normalize_for_metrics(suspect_y, clean))
    score = (pal_k + lev_sim + lcs) / 3.0
    return {
        "PAL_chars": pal,
        "PAL_k": pal_k,
        "Lev_sim": lev_sim,
        "LCS_ratio": lcs,
        "LineageScoreSimple": score,
        "clean_used": clean,
    }



