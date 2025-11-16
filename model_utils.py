"""Model loading helpers used by the RoFL experiments."""
from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

__all__ = ["load_hf_model", "greedy_once"]


def load_hf_model(
    model_id: str,
    fourbit: bool = False,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
) -> Tuple[torch.nn.Module, "PreTrainedTokenizer", str]:
    """Load a causal LM with optional 4-bit quantization."""
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

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    device = next(model.parameters()).device
    return model, tok, device


@torch.no_grad()
def greedy_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Quick deterministic sample for smoke-testing a loaded model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )[0]
    return tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
