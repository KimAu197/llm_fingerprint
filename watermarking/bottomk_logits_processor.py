"""
bottomk_logits_processor.py

Hard-constrained logits processor for a *fixed* bottom-k vocabulary.

和之前“每一步重新算 bottom-k”不同，这个版本的逻辑是：

    - 你事先为某个模型算出一个固定的 bottom-k 词表 S (比如大小=2000)
    - 在生成时，每一步 logits 计算好之后：
        - 只保留 S 中 token 的原始 logits
        - 所有不在 S 里的 token logits 设为 -inf
    - 然后用 greedy / sampling 生成，就等价于“只在固定 bottom-k 里生成”

这非常适合你现在的 fingerprint setting：
    1) base model 上先定义一个固定的 bottom-k 词表（fingerprint 空间）
    2) suspend model 在自己的固定 bottom-k 里 greedy 生成 y
    3) 最后看 y 里的 token 有多少 ∈ base model 的 bottom-k 词表
"""

from typing import Iterable, List, Optional

import torch
from transformers import LogitsProcessor


class BottomKLogitsProcessor(LogitsProcessor):
    """
    Hard-constrained logits processor for a *fixed* allowed vocab set.

    在初始化时传入 allowed_token_ids（比如一个模型的 bottom-2000 token id 列表），
    在每个生成 step：

        new_scores = -inf
        new_scores[..., allowed_token_ids] = 原 scores[..., allowed_token_ids]

    这样后续 softmax / greedy / sampling 都只能在 allowed_token_ids 子集里进行。

    参数:
        allowed_token_ids:  允许输出的 token id 集合（如 bottom-k vocab）。
    """

    def __init__(self, allowed_token_ids: Iterable[int]):
        allowed_token_ids = list(allowed_token_ids)
        if len(allowed_token_ids) == 0:
            raise ValueError("`allowed_token_ids` must be a non-empty list of token ids.")

        # 保存为长整型 tensor，便于后续 scatter / 索引
        self.allowed_token_ids_tensor = torch.tensor(allowed_token_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        参数:
            input_ids: (batch_size, seq_len)，当前已生成的 token 序列（这里其实不用）。
            scores:    (batch_size, vocab_size)，当前 step 的 logits。

        返回:
            new_scores: (batch_size, vocab_size)，除了 allowed_token_ids 之外全部为 -inf。
        """
        if scores.ndim != 2:
            raise ValueError(
                f"`scores` is expected to be 2D (batch_size, vocab_size), got shape {scores.shape}"
            )

        batch_size, vocab_size = scores.shape

        # 把 allowed ids 移到和 scores 同一个设备上
        allowed_ids = self.allowed_token_ids_tensor.to(scores.device)

        # 防御：如果 vocab_size 比我们预期的小，截断一下（几乎不会发生）
        allowed_ids = allowed_ids[allowed_ids < vocab_size]
        if allowed_ids.numel() == 0:
            raise ValueError(
                "After filtering with vocab_size, `allowed_token_ids` became empty. "
                f"vocab_size={vocab_size}."
            )

        # 先全部设为 -inf
        new_scores = scores.new_full(scores.shape, float("-inf"))
        # 再把 allowed_ids 上的 logit 从原 scores 拷回来
        # gather scores[..., allowed_ids] 的 shape 是 (batch_size, len(allowed_ids))
        selected_scores = scores.index_select(dim=-1, index=allowed_ids)
        new_scores.scatter_(-1, allowed_ids.unsqueeze(0).expand(batch_size, -1), selected_scores)

        return new_scores


# ============================
# 辅助函数：为某个模型计算“全局 bottom-k vocab”（非常粗糙版）
# 你可以根据需要换成更复杂的统计方式。
# ============================

def compute_bottomk_vocab_for_model(
    model,
    tokenizer,
    k: int = 2000,
    device: Optional[str] = None,
    prompt: Optional[str] = None,
) -> List[int]:
    """
    简单地为一个 causal LM 计算一次 bottom-k vocab 作为 fingerprint 空间。

    思路（简化版）：
        - 用一个固定 prompt（或 BOS）跑一次 forward
        - 取最后一个位置的 logits: (vocab_size,)
        - 按 logits 升序排序，取前 k 个 token id 作为 bottom-k

    这是一个粗糙但可行的近似：真正的“全局 bottom-k”可以用更多 prompt 做平均，
    但作为实验起点已经够用。

    参数:
        model:      HF AutoModelForCausalLM
        tokenizer:  对应 tokenizer
        k:          bottom-k 大小，例如 2000
        device:     "cuda" / "mps" / "cpu"；若为 None 则自动取 model.device
        prompt:     可选的上下文；若为 None，则用 tokenizer.bos_token 或简单占位文本。

    返回:
        bottomk_ids: 长度为 k 的 token id 列表
    """
    model.eval()

    if device is None:
        # 尝试从模型参数推断设备
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if isinstance(device, torch.device):
        device = device.type

    # 准备一个简单 prompt
    if prompt is None:
        if tokenizer.bos_token is not None:
            prompt = tokenizer.bos_token
        else:
            prompt = "Fingerprint base prompt."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 取最后一个位置的 logits: (batch_size=1, seq_len, vocab_size) → (vocab_size,)
        logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    vocab_size = logits.shape[0]
    k = min(k, vocab_size)

    # 取 bottom-k：logits 最小的 k 个 token
    _, bottomk_indices = torch.topk(logits, k=k, largest=False)

    bottomk_ids = bottomk_indices.tolist()
    return bottomk_ids


# ============================
# Demo: 如何使用（你可以根据项目改掉）
# ============================
if __name__ == "__main__":
    """
    Demo 流程（示意）：

    1. 选一个 base model，算它的 bottom-k 词表：base_bottomk_ids
    2. 选一个 suspect model，算它自己的 bottom-k 词表：suspect_bottomk_ids
       （如果你想让它“在自己的 fixed bottom-k 里 greedy”，就用它自己的这个集合）
    3. 对 suspect model 的 generate:
        - 使用 BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids)
        - do_sample=False（greedy）或 True（sampling in its bottom-k）
    4. 生成完 y_suspect 后，对 token 做统计：
        - 有多少 token id ∈ base_bottomk_ids
    """

    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

    # 只是 demo：实际你会用 Qwen / TinyLlama 等
    base_name = "gpt2"
    suspect_name = "gpt2"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device)

    print("Computing base model bottom-k ids ...")
    base_bottomk_ids = compute_bottomk_vocab_for_model(
        base_model,
        base_tokenizer,
        k=2000,
        device=device,
    )
    print(f"Base model bottom-k size = {len(base_bottomk_ids)}")

    print(f"\nLoading suspect model: {suspect_name}")
    suspect_tokenizer = AutoTokenizer.from_pretrained(suspect_name)
    suspect_model = AutoModelForCausalLM.from_pretrained(suspect_name).to(device)

    # 这里给一个简单 fingerprint prompt，实际会用你生成好的 x'
    fingerprint_prompt = "This is a fingerprint prompt: "

    inputs = suspect_tokenizer(fingerprint_prompt, return_tensors="pt").to(device)

    # 如果你想让 suspect 在“自己的 bottom-k 里生成”，可以先算一遍：
    suspect_bottomk_ids = compute_bottomk_vocab_for_model(
        suspect_model,
        suspect_tokenizer,
        k=2000,
        device=device,
    )

    # 然后用 suspect_bottomk_ids 当作 allowed set：
    logits_processors = LogitsProcessorList(
        [
            BottomKLogitsProcessor(allowed_token_ids=suspect_bottomk_ids),
        ]
    )

    with torch.no_grad():
        outputs = suspect_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # greedy in its fixed bottom-k
            logits_processor=logits_processors,
        )

    generated_ids = outputs[0]
    generated_text = suspect_tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n=== Suspect model generated text (greedy in its bottom-k) ===\n")
    print(generated_text)

    # 统计：生成序列里，有多少 token id ∈ base_bottomk_ids
    base_bottomk_set = set(base_bottomk_ids)
    overlap_count = sum(int(t.item() in base_bottomk_set) for t in generated_ids)
    overlap_ratio = overlap_count / len(generated_ids)
    print(f"\nOverlap with base bottom-k vocab: {overlap_count}/{len(generated_ids)} "
          f"({overlap_ratio:.4f})")