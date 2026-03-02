"""
model_loader.py - Model loading, unloading, and seed utilities
"""
from __future__ import annotations
import gc
import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# =============== Seed ===============

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============== Model Loading ===============

def load_hf_model(model_id: str, fourbit: bool = False, torch_dtype=torch.float16, device_map: str = "auto"):
    """
    Load a HuggingFace model and tokenizer.
    
    Args:
        model_id: HuggingFace model ID
        fourbit: whether to use 4-bit quantization
        torch_dtype: torch dtype for model weights
        device_map: device map strategy
    
    Returns:
        (model, tokenizer, device) tuple
    """
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


# =============== Model Unloading ===============

def unload_hf_model(model=None, tokenizer=None):
    """
    Release HF model and GPU/CPU memory.
    Compatible with device_map('auto') / single GPU.
    """
    try:
        if model is not None:
            try:
                # Skip .to('cpu') for models dispatched with accelerate hooks
                # (device_map="auto"), as this triggers a warning and is a no-op.
                if not getattr(model, 'hf_device_map', None):
                    model.to('cpu')
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

    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def unload_llama_cpp(llm=None):
    """Release llama.cpp GGUF instance."""
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


# =============== Model Wrappers ===============

class SuspectModelHF:
    """
    Wrapper for a HuggingFace causal LM.
    Greedy-generates continuation from full_prompt.
    """
    def __init__(self, model_name: str, device: Optional[str] = None, torch_dtype=torch.float16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device},
        )
        self.model.eval()

    @torch.no_grad()
    def generate_answer(self, full_prompt: str, max_new_tokens: int = 128, stop_tokens=None) -> str:
        """Generate continuation (greedy, do_sample=False)."""
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )[0]

        new_tokens = out_ids[input_len:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

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

    def generate_answer(self, full_prompt: str, max_new_tokens: int = 128, stop_tokens=None) -> str:
        if stop_tokens is None:
            stop_tokens = ["</s>", "user:", "assistant:", "<|im_end|>", "<|im_start|>user"]
        out = self.llm(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=stop_tokens,
        )
        return out["choices"][0]["text"]


class SuspectFromLoadedHF:
    """
    Wrapper for an already-loaded HuggingFace model.
    """
    def __init__(self, model, tok):
        self.model = model.eval()
        self.tok = tok
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate_answer(self, full_prompt: str, max_new_tokens: int = 128, stop_tokens=None) -> str:
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        inp_len = inputs["input_ids"].shape[1]
        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
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

