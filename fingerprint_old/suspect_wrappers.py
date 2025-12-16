"""Wrappers that provide a uniform ``generate_answer`` API for suspects."""
from __future__ import annotations

from typing import Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

__all__ = [
    "SuspectModelHF",
    "SuspectModelLlamaCpp",
    "SuspectFromLoadedHF",
]


class SuspectModelHF:
    """Convenience wrapper that lazily loads a Hugging Face causal LM."""

    def __init__(self, model_name: str, device: str | None = None, torch_dtype=torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
        ).eval()

    @torch.no_grad()
    def generate_answer(
        self,
        full_prompt: str,
        max_new_tokens: int = 128,
        stop_tokens: Sequence[str] | None = None,
    ) -> str:
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]
        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )[0]

        text = self.tok.decode(
            out_ids[input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if stop_tokens:
            for st in stop_tokens:
                idx = text.find(st)
                if idx != -1:
                    text = text[:idx]
                    break
        return text


class SuspectModelLlamaCpp:
    """Wrap an existing ``llama_cpp.Llama`` instance."""

    def __init__(self, llm):
        self.llm = llm

    def generate_answer(
        self,
        full_prompt: str,
        max_new_tokens: int = 128,
        stop_tokens: Sequence[str] | None = None,
    ) -> str:
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
    """Adapter for already-loaded HF models (saves memory + load time)."""

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tok = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate_answer(
        self,
        full_prompt: str,
        max_new_tokens: int = 128,
        stop_tokens: Sequence[str] | None = None,
    ) -> str:
        inputs = self.tok(full_prompt, return_tensors="pt").to(self.device)
        inp_len = inputs["input_ids"].shape[1]
        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
        )[0]

        text = self.tok.decode(
            out_ids[inp_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if stop_tokens:
            for st in stop_tokens:
                idx = text.find(st)
                if idx != -1:
                    text = text[:idx]
                    break
        return text
