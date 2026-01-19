"""
bottomk_processor.py - Bottom-k logits processor and vocab computation

Hard-constrained logits processor for a *fixed* bottom-k vocabulary.

    - Precompute a fixed bottom-k vocab S for a model (e.g., size=2000).
    - During generation, after each logits computation:
        - Keep the original logits only for tokens in S.
        - Set logits for tokens not in S to -inf.
    - Greedy or sampling then becomes "generate only within the fixed bottom-k."
"""
from typing import Iterable, List, Optional

import torch
from transformers import LogitsProcessor


class BottomKLogitsProcessor(LogitsProcessor):
    """
    Hard-constrained logits processor for a *fixed* allowed vocab set.

    Pass allowed_token_ids at init (e.g., a model's bottom-2000 token id list).
    At each generation step:

        new_scores = -inf
        new_scores[..., allowed_token_ids] = original scores[..., allowed_token_ids]

    This forces softmax/greedy/sampling to operate only on allowed_token_ids.

    Args:
        allowed_token_ids: token ids allowed to be produced (e.g., bottom-k vocab).
    """

    def __init__(self, allowed_token_ids: Iterable[int]):
        allowed_token_ids = list(allowed_token_ids)
        if len(allowed_token_ids) == 0:
            raise ValueError("`allowed_token_ids` must be a non-empty list of token ids.")

        self.allowed_token_ids_tensor = torch.tensor(allowed_token_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, seq_len) current generated token sequence.
            scores:    (batch_size, vocab_size) logits at the current step.

        Returns:
            new_scores: (batch_size, vocab_size) with everything outside allowed_token_ids set to -inf.
        """
        if scores.ndim != 2:
            raise ValueError(
                f"`scores` is expected to be 2D (batch_size, vocab_size), got shape {scores.shape}"
            )

        batch_size, vocab_size = scores.shape

        # Move allowed ids to the same device as scores
        allowed_ids = self.allowed_token_ids_tensor.to(scores.device)

        # Guard: if vocab_size is smaller than expected, truncate
        allowed_ids = allowed_ids[allowed_ids < vocab_size]
        if allowed_ids.numel() == 0:
            raise ValueError(
                "After filtering with vocab_size, `allowed_token_ids` became empty. "
                f"vocab_size={vocab_size}."
            )

        # Set everything to -inf first
        new_scores = scores.new_full(scores.shape, float("-inf"))
        # Then copy logits at allowed_ids from the original scores
        selected_scores = scores.index_select(dim=-1, index=allowed_ids)
        new_scores.scatter_(-1, allowed_ids.unsqueeze(0).expand(batch_size, -1), selected_scores)

        return new_scores


def compute_bottomk_vocab_for_model(
    model,
    tokenizer,
    k: int = 2000,
    device: Optional[str] = None,
    prompt: Optional[str] = None,
) -> List[int]:
    """
    Compute a bottom-k vocab for a causal LM as a fingerprint space.

    Simplified idea:
        - Run one forward pass with a fixed prompt (or BOS).
        - Take the logits at the last position: (vocab_size,).
        - Sort logits ascending and take the first k token ids as bottom-k.

    Args:
        model:      HF AutoModelForCausalLM
        tokenizer:  corresponding tokenizer
        k:          bottom-k size, e.g., 2000
        device:     "cuda" / "mps" / "cpu"; if None, auto-detect
        prompt:     optional context; if None, use tokenizer.bos_token

    Returns:
        bottomk_ids: token id list of length k
    """
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if isinstance(device, torch.device):
        device = device.type

    # Prepare a simple prompt
    if prompt is None:
        if tokenizer.bos_token is not None:
            prompt = tokenizer.bos_token
        else:
            prompt = "Fingerprint base prompt."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)

    vocab_size = logits.shape[0]
    k = min(k, vocab_size)

    # Take bottom-k: the k tokens with the lowest logits
    _, bottomk_indices = torch.topk(logits, k=k, largest=False)

    bottomk_ids = bottomk_indices.tolist()
    return bottomk_ids






