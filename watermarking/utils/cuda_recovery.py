"""Best-effort CUDA context recovery after hard failures (OOM, device assert, etc.)."""

from __future__ import annotations

import gc
from typing import Optional

import torch


def try_cuda_device_reset(device_index: int = 0) -> bool:
    """
    Reset the current process's CUDA state on one device (cudaDeviceReset).

    empty_cache/synchronize do not clear a poisoned context after device-side
    asserts; this is stronger and allows the next from_pretrained to succeed.

    Returns True if reset appears to have been invoked without raising.
    """
    if not torch.cuda.is_available():
        return False

    gc.collect()
    try:
        torch.cuda.set_device(device_index)
    except Exception:
        pass

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    err: Optional[int] = None
    try:
        cudart = torch.cuda.cudart()
        if cudart is not None:
            err = int(cudart.cudaDeviceReset())
    except Exception:
        err = None

    if err is None:
        err = _cuda_device_reset_ctypes()

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return err == 0 if err is not None else False


def _cuda_device_reset_ctypes() -> Optional[int]:
    """Fallback: libcudart cudaDeviceReset. Returns error code or None if unavailable."""
    try:
        import ctypes

        cdll = None
        for lib in (
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ):
            try:
                cdll = ctypes.CDLL(lib)
                break
            except OSError:
                continue
        if cdll is None:
            return None
        cdll.cudaDeviceReset.argtypes = []
        cdll.cudaDeviceReset.restype = ctypes.c_int
        return int(cdll.cudaDeviceReset())
    except Exception:
        return None


def is_out_of_memory_error(exc: BaseException) -> bool:
    """
    True for PyTorch / CUDA out-of-memory errors only.

    Used to avoid cudaDeviceReset after OOM: reset in-process often breaks the
    next torch.cuda / from_pretrained in the same process.
    """
    oom_types: list[type] = []
    for mod in (torch, torch.cuda):
        t = getattr(mod, "OutOfMemoryError", None)
        if t is not None and t not in oom_types:
            oom_types.append(t)
    if oom_types and isinstance(exc, tuple(oom_types)):
        return True
    if "outofmemory" in type(exc).__name__.lower():
        return True
    return False


def is_likely_cuda_hard_failure(exc: BaseException) -> bool:
    """Heuristic: failure may have left the CUDA context unusable."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "outofmemory" in name or "out of memory" in msg:
        return True
    if "cuda" in name or "cuda" in msg:
        return True
    if "accelerator" in name:
        return True
    if "cudnn" in msg or "nccl" in msg:
        return True
    if "device-side assert" in msg or "device side assert" in msg:
        return True
    return False
