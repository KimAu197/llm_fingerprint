"""Memory / resource cleanup helpers."""
from __future__ import annotations

import gc

import torch

__all__ = ["unload_hf_model", "unload_llama_cpp"]


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_hf_model(model=None, tokenizer=None) -> None:
    """Release GPU/CPU memory for HF models loaded with transformers."""
    try:
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
    except Exception:
        pass

    if tokenizer is not None:
        try:
            del tokenizer
        except Exception:
            pass

    gc.collect()
    _clear_cuda()


def unload_llama_cpp(llm=None) -> None:
    """Release llama.cpp backed models."""
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
    _clear_cuda()
