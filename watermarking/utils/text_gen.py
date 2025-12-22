"""
text_gen.py - Text generation utilities
"""
from __future__ import annotations
from typing import Optional

import torch


@torch.no_grad()
def greedy_response_hf(
    model,
    tokenizer,
    full_prompt_text: str,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
) -> str:
    """
    Deterministic generation (do_sample=False). Returns the continuation (excluding input).
    
    Args:
        model: HuggingFace causal LM
        tokenizer: corresponding tokenizer
        full_prompt_text: input prompt
        device: device to run on
        max_new_tokens: max tokens to generate
    
    Returns:
        generated text (continuation only)
    """
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
    return tokenizer.decode(out_ids[input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)




