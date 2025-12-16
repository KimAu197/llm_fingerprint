# fingerprint_tools.py
# ------------------------------------------------------------
# Minimal, tidy utilities for RoFL-style x' init + greedy y,
# and black-box verification with simple metrics.
# NO system prompt anywhere in this file.
# You provide:
#   - (base) model, tokenizer  —— 用于生成指纹 (x', y)
#   - suspect_generate()        —— 被测模型的生成回调（HF 或 GGUF）
# ------------------------------------------------------------
from __future__ import annotations
import os, time, json, random, difflib
from typing import Callable, List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# =============== basics ===============

def set_seed(seed: int = 42):
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ts_path(path: str, model_name) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    base, ext = os.path.splitext(path)
    return f"{base}_{model_name}{ext or '.json'}"

# =============== formatting (NO system) ===============

def format_full_prompt(
    x_prime_text: str,
    prompt_style: str = "oneshot",   # 'oneshot' | 'chatml' | 'raw'
) -> str:
    """
    - oneshot:
        "user: {x'}\nassistant:"
    - chatml (Qwen-like, NO system block):
        "<|im_start|>user\n{x'}\n<|im_end|>\n<|im_start|>assistant\n"
    - raw:
        "{x'}"
    """
    if prompt_style == "chatml":
        return (
            "<|im_start|>user\n" + x_prime_text + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if prompt_style == "raw":
        return x_prime_text
    # default: oneshot
    return f"user: {x_prime_text}\nassistant:"

# =============== x' initialization (RoFL Step 1 spirit) ===============

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
    """
    RoFL Step 1 (简化实现):
      (1) 前 l 个 token 从 vocab(去掉 special) 均匀随机
      (2) 之后每步从“概率最低的 k 个 token”里均匀取一个，直到 total_len
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    allowed = _build_allowed_token_set(tokenizer)
    allowed_t = torch.tensor(allowed, device=device)

    # (1) 均匀随机前缀
    prefix_ids = random.choices(allowed, k=l_random_prefix)
    prompt_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)

    # (2) bottom-k 扩展
    while prompt_ids.shape[1] < total_len:
        logits = model(prompt_ids).logits[:, -1, :]
        probs  = F.softmax(logits, dim=-1)

        masked = probs.clone()
        if len(allowed) < tokenizer.vocab_size:
            mask = torch.ones_like(masked, dtype=torch.bool)
            mask[:, allowed_t] = False
            masked[mask] = 1e9  # 不让不允许 token 落入 bottom-k

        _, sorted_idx = torch.sort(masked, dim=-1, descending=False)  # 概率升序
        k_eff = min(k_bottom, sorted_idx.shape[1])
        bottomk_idx = sorted_idx[:, :k_eff]
        next_id = bottomk_idx[0, random.randrange(k_eff)].view(1, 1)
        prompt_ids = torch.cat([prompt_ids, next_id.to(device)], dim=1)

    return tokenizer.decode(prompt_ids.squeeze(0).cpu(), skip_special_tokens=True)

# =============== greedy y (RoFL Step 2) ===============

@torch.no_grad()
def greedy_response_hf(
    model,
    tokenizer,
    full_prompt_text: str,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
) -> str:
    """
    确定性生成（do_sample=False）。返回 continuation（不含输入）。
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(full_prompt_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    out_ids = model.generate(
        **inputs,
        do_sample=False,                 # 温度=0
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    return tokenizer.decode(out_ids[input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)

# =============== batch: (x', y) generation ===============

def generate_fingerprints_batch(
    model,
    model_name,# 已加载好的 base HF model
    tokenizer,                   # 对应 tokenizer
    num_pairs: int = 3,
    prompt_style: str = "oneshot",      # 'oneshot' | 'chatml' | 'raw'
    l_random_prefix: int = 8,
    total_len: int = 64,
    k_bottom: int = 50,
    max_new_tokens: int = 64,
    save_json_path: Optional[str] = "fingerprints_init.json",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    生成 num_pairs 个 (x', y) 并（可选）保存。
    不涉及任何 system 文本。
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
        out_path = ts_path(save_json_path, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"[save] fingerprints -> {out_path}")

    return pairs, out_path

# =============== metrics ===============

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

def metric_prefix_match_raw(a: str, b: str, min_len: int = 30) -> int:
    return int(a[:min_len] == b[:min_len])

def metric_lcs_ratio_raw(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def metric_signature_overlap_raw(a: str, b: str, min_tok_len: int = 6) -> Tuple[int, int]:
    toks = [t for t in a.split() if len(t) >= min_tok_len]
    hits = sum(1 for t in toks if t in b)
    return hits, len(toks)



# ================== suspect wrappers (HF / GGUF) ==================

class SuspectModelHF:
    """
    Wrapper for a HuggingFace causal LM (transformers).
    We'll greedy-generate continuation from full_prompt.
    """
    def __init__(self, model_name, device=None, torch_dtype=torch.float16):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device},   # put whole model on single device
        )
        self.model.eval()

    @torch.no_grad()
    def generate_answer(self, full_prompt, max_new_tokens=128, stop_tokens=None):
        """
        full_prompt: already includes role markers if any (NO system prompt anywhere).
        returns: continuation only (greedy, do_sample=False).
        """
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        out_ids = self.model.generate(
            **inputs,
            do_sample=False,  # greedy == temperature 0
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )[0]

        new_tokens = out_ids[input_len:]
        text = self.tok.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # optional manual stopping on phrases
        if stop_tokens:
            for st in stop_tokens:
                cut_idx = text.find(st)
                if cut_idx != -1:
                    text = text[:cut_idx]
        return text


class SuspectModelLlamaCpp:
    """
    Wrapper for llama_cpp.Llama GGUF model.
    Assumes you already created: llm = Llama(...)
    """
    def __init__(self, llm):
        self.llm = llm

    def generate_answer(self, full_prompt, max_new_tokens=128, stop_tokens=None):
        if stop_tokens is None:
            # defaults that work for oneshot/chatml/raw
            stop_tokens = ["</s>", "user:", "assistant:", "<|im_end|>", "<|im_start|>user"]
        out = self.llm(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,  # greedy-ish
            stop=stop_tokens,
        )
        return out["choices"][0]["text"]


# ================== evaluation (no system prompt anywhere) ==================
import numpy as np

def evaluate_fingerprints(
    pairs_json_path: str,
    model_name: str,
    suspect_base_name, # 用于文件名
    suspect_model,                  # 有 .generate_answer(full_prompt, max_new_tokens, stop_tokens)
    suspect_label: str = "suspect",
    save_report_path: str | None = None,   # 如 "eval_report.json"
    min_prefix_len: int = 30,
    sig_min_tok_len: int = 6,
    use_timestamp_in_name: bool = False,   # 文件名是否带时间戳
):
    """
    保存的 summary 只包含: fingerprint, base_y, suspect_y （逐样本列表）。
    其他指标只打印，不写入文件。
    """
    # 1) load pairs
    with open(pairs_json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"[info] Loaded {len(pairs)} fingerprint pairs from {pairs_json_path}")

    # 打印用指标（不保存）
    prefix_hits, sim_scores, sig_hits, sig_totals = [], [], [], []

    # 读入 pairs 后（evaluate_fingerprints 里）：
    style = pairs[0].get("prompt_style", "raw")

    if style == "raw":
        stops = ["</s>", "<|im_end|>"]           # 只保留真正的特殊终止
    elif style == "chatml":
        stops = ["</s>", "<|im_end|>", "<|im_start|>user"]
    else:  # oneshot
        stops = ["</s>", "user:", "assistant:"]
    # 最终要写入文件的极简列表
    minimal_records = []

    # 2) eval loop
    for idx, pair in enumerate(tqdm(pairs, desc=f"Evaluating on {suspect_label}")):
        x_prime = pair["x_prime"]              # fingerprint
        base_y  = pair["y_response"]           # base model 的 y
        # 优先使用生成 y 时精确的 full prompt（无 system/有 role，都按你生成时的格式）
        if "full_prompt_used" in pair:
            full_prompt_for_suspect = pair["full_prompt_used"]
        else:
            # 兜底（无 system、最小 role）
            full_prompt_for_suspect = f"user: {x_prime}\nassistant:"

        suspect_y = suspect_model.generate_answer(
            full_prompt_for_suspect,
            max_new_tokens=128,
            stop_tokens=stops,
        )

        # —— 只打印，不保存 ——
        pm  = metric_prefix_match(base_y, suspect_y, min_len=min_prefix_len)
        sim = metric_lcs_ratio(base_y, suspect_y)
        h, tot = metric_signature_overlap(base_y, suspect_y, min_tok_len=sig_min_tok_len)
        prefix_hits.append(pm); sim_scores.append(sim); sig_hits.append(h); sig_totals.append(tot)

        # —— 保存的极简条目 ——
        minimal_records.append({
            "fingerprint": x_prime,
            "base_y": base_y,
            "suspect_y": suspect_y,
        })

    # 3) 打印整体指标（不写入文件）
    prefix_match_rate = float(np.mean(prefix_hits)) if prefix_hits else 0.0
    avg_edit_sim      = float(np.mean(sim_scores)) if sim_scores else 0.0
    sig_overlap_rate  = (
        float(np.sum(sig_hits)) / max(1, float(np.sum(sig_totals)))
        if np.sum(sig_totals) > 0 else 0.0
    )

    print("========== FINGERPRINT VERIFICATION REPORT ==========")
    print(f"Suspect model label: {suspect_label}")
    print(f"#pairs evaluated: {len(pairs)}")
    print(f"Prefix match rate (first {min_prefix_len} chars): {prefix_match_rate:.3f}")
    print(f"Avg edit/sequence similarity (0~1):           {avg_edit_sim:.3f}")
    print(f"Signature phrase overlap (>= {sig_min_tok_len} chars): {sig_overlap_rate:.3f}")
    print("Preview (first 2 minimal records):")
    for rec in minimal_records[:2]:
        print("----")
        print("fingerprint:", rec["fingerprint"][:160].replace("\n","\\n"))
        print("base_y     :", rec["base_y"][:160].replace("\n","\\n"))
        print("suspect_y  :", rec["suspect_y"][:160].replace("\n","\\n"))

    out_obj = {
        "base_model_name": model_name,
        "suspect_model_name": suspect_base_name,
        "prompt_style": style,
        "num_pairs": len(pairs),
        "records": minimal_records,
    }

    # 4) 保存：仅极简列表
    out_path = None
    if save_report_path:
        out_path = ts_path(save_report_path, model_name=model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[save] minimal summary -> {out_path}")

    return minimal_records, out_path


def evaluate_fingerprint_unclean(
    pairs_json_path: str,
    model_name: str,
    suspect_base_name,   
    suspect_model,        
    suspect_label: str = "suspect",
    save_report_path: str | None = None,   
    min_prefix_len: int = 30,
    sig_min_tok_len: int = 6,
    use_timestamp_in_name: bool = False,  
):
    # 1) load pairs
    with open(pairs_json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"[info] Loaded {len(pairs)} fingerprint pairs from {pairs_json_path}")

    # 打印用指标（不保存）
    prefix_hits, sim_scores, sig_hits, sig_totals = [], [], [], []

    # 读入 pairs 后选择 stop tokens（与原版一致）
    style = pairs[0].get("prompt_style", "raw")
    if style == "raw":
        stops = ["</s>", "<|im_end|>"]
    elif style == "chatml":
        stops = ["</s>", "<|im_end|>", "<|im_start|>user"]
    else:  # oneshot
        stops = ["</s>", "user:", "assistant:"]

    # 最终要写入文件的极简列表
    minimal_records = []

    for idx, pair in enumerate(tqdm(pairs, desc=f"[UNCLEAN] Evaluating on {suspect_label}")):
        x_prime = pair["x_prime"]            # fingerprint
        base_y  = pair["y_response"]         # base model 的 y

        if "full_prompt_used" in pair:
            full_prompt_for_suspect = pair["full_prompt_used"]
        else:
            full_prompt_for_suspect = f"user: {x_prime}\nassistant:"

        suspect_y = suspect_model.generate_answer(
            full_prompt_for_suspect,
            max_new_tokens=128,
            stop_tokens=stops,
        )

        # —— 只打印，不保存（全部 raw 版本）——
        pm  = metric_prefix_match_raw(base_y, suspect_y, min_len=min_prefix_len)
        sim = metric_lcs_ratio_raw(base_y, suspect_y)
        h, tot = metric_signature_overlap_raw(base_y, suspect_y, min_tok_len=sig_min_tok_len)
        prefix_hits.append(pm); sim_scores.append(sim); sig_hits.append(h); sig_totals.append(tot)

        # —— 保存的极简条目（与原版一致）——
        minimal_records.append({
            "fingerprint": x_prime,
            "base_y": base_y,
            "suspect_y": suspect_y,
        })

    # 3) 打印整体指标（不写入文件）
    prefix_match_rate = float(np.mean(prefix_hits)) if prefix_hits else 0.0
    avg_edit_sim      = float(np.mean(sim_scores)) if sim_scores else 0.0
    sig_overlap_rate  = (
        float(np.sum(sig_hits)) / max(1, float(np.sum(sig_totals)))
        if np.sum(sig_totals) > 0 else 0.0
    )

    print("========== FINGERPRINT VERIFICATION REPORT (UNCLEAN) ==========")
    print(f"Suspect model label: {suspect_label}")
    print(f"#pairs evaluated: {len(pairs)}")
    print(f"Prefix match rate (first {min_prefix_len} RAW chars): {prefix_match_rate:.3f}")
    print(f"Avg RAW LCS-like ratio (0~1):                    {avg_edit_sim:.3f}")
    print(f"Signature phrase overlap RAW (>= {sig_min_tok_len} chars): {sig_overlap_rate:.3f}")
    print("Preview (first 2 minimal records):")
    for rec in minimal_records[:2]:
        print("----")
        print("fingerprint:", rec["fingerprint"][:160].replace("\n","\\n"))
        print("base_y     :", rec["base_y"][:160].replace("\n","\\n"))
        print("suspect_y  :", rec["suspect_y"][:160].replace("\n","\\n"))

    out_obj = {
        "base_model_name": model_name,
        "suspect_model_name": suspect_base_name,
        "prompt_style": style,
        "num_pairs": len(pairs),
        "records": minimal_records,
    }

    out_path = None
    if save_report_path:
        out_path = ts_path(save_report_path, model_name=model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[save] minimal summary -> {out_path}")

    return minimal_records, out_path

# ======== Light cleaning before metrics ========
import re, unicodedata

_SPECIAL_TAG_PATTERNS = [
    r"<\|[^>]*\|>",                 # ChatML-like: <|system|> <|user|> <|assistant|> <|im_start|> ...
    r"</?s>",                       # <s> </s>
    r"\[/?[A-Z_]+\]",               # [INST], [/INST], [AVAILABLE_TOOLS], [/TOOL_RESULTS], etc.
    r"\[control_\d+\]",             # [control_12] 之类
]

def _clean_for_metrics(s: str) -> str:
    # 统一编码 & 去控制字符/替换符
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")                    # '�'
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)         # 控制字符

    # 去掉常见“特殊标记”
    for pat in _SPECIAL_TAG_PATTERNS:
        s = re.sub(pat, " ", s)

    # 折叠/清除纯分隔符型垃圾串（如 <|<|<|<|...）
    s = re.sub(r"(?:<\|){2,}", " ", s)             # 连续 <|<|<|...
    s = re.sub(r"[<>\|\[\]/]{4,}", " ", s)         # 4+ 个由 < > | [ ] / 组成的连串
    s = re.sub(r"([^\w\s])\1{4,}", r"\1\1\1", s)   # 同一标点 5+ 次 → 压到 3 次

    # 标准化空白并小写（与 _normalize_text 一致）
    s = " ".join(s.strip().split()).lower()
    return s

# —— 如果你更希望保留“原有的 _normalize_text 行为”，也可以在清洗后再走一遍 —— 
def _normalize_for_metrics(s: str, clean: bool = False) -> str:
    return _clean_for_metrics(s) if clean else _normalize_text(s)

# ======== 更新简单 3 指标打分（加了 clean 开关） ========

def _prefix_agreement_len(a: str, b: str, clean: bool = False) -> int:
    a, b = _normalize_for_metrics(a, clean), _normalize_for_metrics(b, clean)
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _norm_lev_sim(a: str, b: str, clean: bool = False) -> float:
    a, b = _normalize_for_metrics(a, clean), _normalize_for_metrics(b, clean)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0: return 1.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = prev[j-1] if ca == cb else 1 + min(prev[j-1], prev[j], cur[j-1])
        prev = cur
    dist = prev[lb]
    return 1.0 - dist / max(1, max(la, lb))

def lineage_score_simple(base_y: str, suspect_y: str, k_prefix: int = 30, clean: bool = False) -> dict:
    """
    三指标（等权）：PAL_k, Levenshtein 相似度, LCS 比率
    clean=True 时先剥离 <|system|> / <|<|<|... 之类标记/垃圾串。
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

def evaluate_lineage_simple(source, k_prefix: int = 30, verbose: bool = True):
    """
    source: 
      - str: 路径（evaluate_fingerprints 保存的 minimal JSON）
      - list[dict]: 形如 [{"fingerprint":..., "base_y":..., "suspect_y":...}, ...]
    打印整体均值并返回包含逐样本与汇总的 dict。
    """
    import json

    # 1) 载入 records
    meta = {}
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "records" in obj:
            meta = {k: obj[k] for k in obj.keys() if k != "records"}
            records = obj["records"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError("Unrecognized JSON format for simple evaluation.")
    else:
        records = source

    # 2) 逐样本计算
    per_pair = []
    for r in records:
        by, sy = r["base_y"], r["suspect_y"]
        m = lineage_score_simple(by, sy, k_prefix=k_prefix)
        per_pair.append({
            "PAL_chars": m["PAL_chars"],
            "PAL_k": m["PAL_k"],
            "Lev_sim": m["Lev_sim"],
            "LCS_ratio": m["LCS_ratio"],
            "LineageScoreSimple": m["LineageScoreSimple"],
        })

    # 3) 汇总平均
    n = max(1, len(per_pair))
    mean_pal_chars = sum(x["PAL_chars"] for x in per_pair) / n
    mean_pal_k     = sum(x["PAL_k"] for x in per_pair) / n
    mean_lev       = sum(x["Lev_sim"] for x in per_pair) / n
    mean_lcs       = sum(x["LCS_ratio"] for x in per_pair) / n
    mean_score     = sum(x["LineageScoreSimple"] for x in per_pair) / n

    summary = {
        "k_prefix": k_prefix,
        "num_pairs": len(per_pair),
        "mean_PAL_chars": mean_pal_chars,
        "mean_PAL_k": mean_pal_k,
        "mean_Lev_sim": mean_lev,
        "mean_LCS_ratio": mean_lcs,
        "mean_LineageScoreSimple": mean_score,
    }

    if verbose:
        header = ""
        if meta:
            header = f"[{meta.get('base_model_name','?')} vs {meta.get('suspect_model_name','?')}] "
        print(f"{header}Simple 3-metric summary (k={k_prefix})")
        print(f"  #pairs:          {summary['num_pairs']}")
        print(f"  mean PAL(chars): {summary['mean_PAL_chars']:.1f}")
        print(f"  mean PAL_k:      {summary['mean_PAL_k']:.3f}")
        print(f"  mean Lev_sim:    {summary['mean_Lev_sim']:.3f}")
        print(f"  mean LCS_ratio:  {summary['mean_LCS_ratio']:.3f}")
        print(f"  ==> mean Score:  {summary['mean_LineageScoreSimple']:.3f}")

    return {"summary": summary, "per_pair": per_pair, "meta": meta}


# ======== Append simple lineage scores to CSV ========
import os, csv

def append_lineage_score_csv(res: dict,
                             csv_path: str = "lineage_scores.csv",
                             base_model_name: str | None = None,
                             suspect_model_name: str | None = None,
                             relation: str | None = None):
    """
    将 evaluate_lineage_simple(...) 的结果写入 CSV（追加一行汇总）。
    列：base_model_name, suspect_model_name, relation, num_pairs, k_prefix,
        pal_chars_mean, pal_k_mean, lev_sim_mean, lcs_ratio_mean, score_mean
    """
    meta = res.get("meta", {}) if isinstance(res, dict) else {}
    summary = res.get("summary", {}) if isinstance(res, dict) else {}

    base = (base_model_name or meta.get("base_model_name", "")).strip()
    suspect = (suspect_model_name or meta.get("suspect_model_name", "")).strip()

    n_pairs   = int(summary.get("num_pairs", 0))
    k_prefix  = int(summary.get("k_prefix", 30))
    pal_chars = float(summary.get("mean_PAL_chars", 0.0))
    pal_k     = float(summary.get("mean_PAL_k", 0.0))
    lev_sim   = float(summary.get("mean_Lev_sim", 0.0))
    lcs_ratio = float(summary.get("mean_LCS_ratio", 0.0))
    score     = float(summary.get("mean_LineageScoreSimple", 0.0))

    header = [
        "base_model_name", "suspect_model_name", "relation",
        "num_pairs", "k_prefix",
        "pal_chars_mean", "pal_k_mean", "lev_sim_mean", "lcs_ratio_mean",
        "score_mean",
    ]
    need_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if need_header:
            w.writeheader()
        w.writerow({
            "base_model_name": base,
            "suspect_model_name": suspect,
            "relation": relation,
            "num_pairs": n_pairs,
            "k_prefix": k_prefix,
            "pal_chars_mean": round(pal_chars, 3),
            "pal_k_mean": round(pal_k, 6),
            "lev_sim_mean": round(lev_sim, 6),
            "lcs_ratio_mean": round(lcs_ratio, 6),
            "score_mean": round(score, 6),
        })
    print(f"[csv] appended -> {csv_path} | base={base} suspect={suspect} relation={relation or '-'} score={score:.6f}")

def evaluate_and_append_simple(source,
                               csv_path: str = "lineage_scores.csv",
                               k_prefix: int = 30,
                               base_model_name: str | None = None,
                               suspect_model_name: str | None = None,
                               relation: str | None = None,
                               verbose: bool = True):
    """
    一步到位：评估 + 追加到CSV。返回 res。
    新增参数 relation：传 'same' 或 'diff'（也接受 'pos'/'neg' 等别名）。
    source 可为：
      - str: evaluate_fingerprints() 保存的 minimal JSON 路径
      - list[dict]: 直接传 records 列表
    """
    res = evaluate_lineage_simple(source, k_prefix=k_prefix, verbose=verbose)
    append_lineage_score_csv(
        res,
        csv_path=csv_path,
        base_model_name=base_model_name,
        suspect_model_name=suspect_model_name,
        relation=relation,
    )
    return res

import torch

class SuspectFromLoadedHF:
    def __init__(self, model, tok):
        self.model = model.eval()
        self.tok = tok
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate_answer(self, full_prompt, max_new_tokens=128, stop_tokens=None):
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        inp_len = inputs["input_ids"].shape[1]
        out_ids = self.model.generate(
            **inputs,
            do_sample=False,  # greedy (= temp 0)
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )[0]
        text = self.tok.decode(out_ids[inp_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if stop_tokens:
            for st in stop_tokens:
                i = text.find(st)
                if i != -1:
                    text = text[:i]
        return text

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_hf_model(model_id, fourbit=False, torch_dtype=torch.float16, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(trust_remote_code=True, device_map=device_map)
    if fourbit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok, next(model.parameters()).device

# === memory_cleanup.py（或直接放到一个cell里）===
import gc, torch

def unload_hf_model(model=None, tokenizer=None):
    """释放 HF 模型与显存/内存。兼容 device_map('auto') / 单卡。"""
    try:
        if model is not None:
            try:
                model.to('cpu')  # 先转回 CPU，避免有残留显存句柄
            except Exception:
                pass
            del model
    except Exception:
        pass
    try:
        if tokenizer is not None:
            del tokenizer
    except Exception:
        pass

    # Python 对象回收
    gc.collect()

    # CUDA 显存清理（含跨进程共享块）
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

def unload_llama_cpp(llm=None):
    """释放 llama.cpp GGUF 实例（如用到的话）。"""
    try:
        if llm is not None and hasattr(llm, "close"):
            llm.close()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass



MODEL_LIST_CSV = "qwen2_0_5b_textgen_derivatives.csv"
DEFAULT_BASE_MODEL = "Qwen/Qwen2-0.5B"
RELATION = "same"


def _safe_model_label(model_name: str) -> str:
    """Use the last path component as a filesystem-safe label."""
    if not model_name:
        return "unknown-model"
    parts = model_name.split("/")
    return parts[-1] or model_name.replace("/", "_")


def load_model_ids(csv_path: str = MODEL_LIST_CSV) -> List[str]:
    """Load model IDs from a simple CSV with a `model_id` column."""
    ids: List[str] = []
    if not os.path.exists(csv_path):
        print(f"[warn] model list CSV not found: {csv_path}")
        return ids
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = (row.get("model_id") or "").strip()
            if mid:
                ids.append(mid)
    return ids


def run_single_candidate(model_name: str,
                         suspect_wrapper: SuspectFromLoadedHF,
                         suspect_model_name: str,
                         *,
                         num_pairs: int = 5,
                         prompt_style: str = "raw") -> None:
    """Generate fingerprints for one candidate and evaluate against the suspect."""
    model_orgin = _safe_model_label(model_name)
    print(f"\n===== Testing candidate: {model_name} (label: {model_orgin}) =====")
    try:
        model1, tok1, dev1 = load_hf_model(model_name, fourbit=False)
    except Exception as exc:
        print(f"[error] Failed to load candidate model {model_name}: {exc}")
        return

    try:
        print("Loaded candidate on", dev1)
        set_seed(42)
        pairs, pairs_path = generate_fingerprints_batch(
            model=model1,
            model_name=model_orgin,
            tokenizer=tok1,
            num_pairs=num_pairs,
            prompt_style=prompt_style,
            l_random_prefix=8,
            total_len=64,
            k_bottom=50,
            max_new_tokens=64,
            save_json_path="fingerprints_init.json",
        )
        if not pairs_path:
            print("[warn] No fingerprint file produced, skipping evaluation.")
            return

        suspect_label = f"{suspect_model_name}-on-{model_orgin}"
        _, report_path = evaluate_fingerprints(
            pairs_json_path=pairs_path,
            model_name=model_orgin,
            suspect_base_name=suspect_model_name,
            suspect_model=suspect_wrapper,
            suspect_label=suspect_label,
            save_report_path=f"eval_report_{model_orgin}.json",
        )
        if not report_path:
            print("[warn] No evaluation report path returned, skipping lineage score logging.")
            return

        res = evaluate_and_append_simple(
            report_path,
            csv_path="lineage_scores.csv",
            k_prefix=30,
            base_model_name=model_name,
            suspect_model_name=suspect_model_name,
            relation=RELATION,
        )
        print(res["summary"])
    finally:
        unload_hf_model(model1, tok1)


def main():
    candidate_models = load_model_ids(MODEL_LIST_CSV)
    if not candidate_models:
        print("[info] No candidate models found to evaluate.")
        return

    model_base_name = DEFAULT_BASE_MODEL
    print(f"[setup] Loading suspect/base model: {model_base_name}")
    model2, tok2, dev2 = load_hf_model(model_base_name, fourbit=False)
    print("Loaded suspect/base on", dev2)
    suspect = SuspectFromLoadedHF(model2, tok2)

    try:
        for idx, candidate in enumerate(candidate_models, 1):
            print(f"\n### ({idx}/{len(candidate_models)}) Processing {candidate}")
            try:
                run_single_candidate(
                    candidate,
                    suspect,
                    model_base_name,
                    num_pairs=20,
                    prompt_style="raw",
                )
            except Exception as exc:
                print(f"[error] Unexpected failure while processing {candidate}: {exc}")
    finally:
        unload_hf_model(model2, tok2)


if __name__ == "__main__":
    main()
