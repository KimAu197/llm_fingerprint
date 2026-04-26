"""Parallel helpers for retrying Phase 2 overlap rows across multiple GPUs."""

from __future__ import annotations

import gc
import traceback
from types import SimpleNamespace
from typing import Dict, List, Sequence, Set

from .bottomk_processor import compute_bottomk_vocab_batch
from .cuda_recovery import try_cuda_device_reset
from .metrics import overlap_ratio
from .model_loader import load_hf_model, unload_hf_model


def gpu_ids_from_cli(args) -> List[int]:
    raw = getattr(args, "gpu_ids", None)
    if not raw or not str(raw).strip():
        return [0]
    ids = [s.strip() for s in str(raw).split(",") if s.strip()]
    parsed: List[int] = []
    seen: Set[int] = set()
    for gid in ids:
        value = int(gid)
        if value not in seen:
            seen.add(value)
            parsed.append(value)
    return parsed


def partition_models_by_gpu(
    models: Sequence[str], gpu_ids: Sequence[int]
) -> List[tuple[int, List[str]]]:
    if not gpu_ids:
        gpu_ids = [0]
    chunks: Dict[int, List[str]] = {gid: [] for gid in gpu_ids}
    for i, model_id in enumerate(models):
        chunks[gpu_ids[i % len(gpu_ids)]].append(model_id)
    return [(gid, chunk) for gid, chunk in chunks.items() if chunk]


def phase2_args_for_worker(args) -> Dict[str, object]:
    return {
        "bottom_k_vocab": args.bottom_k_vocab,
        "batch_size_bottomk": args.batch_size_bottomk,
        "cuda_device_reset_each_model": args.cuda_device_reset_each_model,
        "no_cuda_device_reset_on_error": args.no_cuda_device_reset_on_error,
        "cuda_device_reset_after_oom": args.cuda_device_reset_after_oom,
    }


def parallel_cross_worker(
    worker_id: int,
    gpu_id: int,
    chunk_models: List[str],
    targets: List[str],
    all_prompts: List[str],
    all_fps: Dict[str, List[str]],
    args_dict: Dict[str, object],
    queue,
) -> None:
    args = SimpleNamespace(**args_dict)
    try:
        pinned: Dict[str, Dict[str, List[int]]] = {}
        queue.put(
            {
                "type": "status",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "message": f"pinning {len(targets)} target cache(s)",
            }
        )
        for target in targets:
            pinned[target] = run_one_model_bottomk(target, all_prompts, args, gpu_id)

        for model_id in chunk_models:
            try:
                cache_j = run_one_model_bottomk(model_id, all_prompts, args, gpu_id)
                cells = []
                for target in targets:
                    cells.append(
                        (
                            target,
                            model_id,
                            matrix_cell_overlap(
                                target, all_fps, pinned[target], cache_j
                            ),
                            matrix_cell_overlap(
                                model_id, all_fps, cache_j, pinned[target]
                            ),
                        )
                    )
                queue.put(
                    {
                        "type": "cells",
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "model_id": model_id,
                        "cells": cells,
                    }
                )
            except Exception:
                queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "gpu_id": gpu_id,
                        "model_id": model_id,
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        queue.put({"type": "done", "worker_id": worker_id, "gpu_id": gpu_id})


def matrix_cell_overlap(
    model_id: str,
    all_fps: Dict[str, List[str]],
    cache_i: Dict[str, List[int]],
    cache_j: Dict[str, List[int]],
) -> float:
    if model_id not in all_fps:
        return -1.0
    scores: List[float] = []
    for fp in all_fps[model_id]:
        if fp in cache_i and fp in cache_j:
            scores.append(overlap_ratio(cache_i[fp], cache_j[fp]))
    if scores:
        return float(sum(scores) / len(scores))
    return -1.0


def run_one_model_bottomk(
    name: str,
    all_prompts: List[str],
    args,
    gpu_id: int,
) -> Dict[str, List[int]]:
    device = f"cuda:{gpu_id}"
    model = None
    tok = None
    try:
        device_map: Dict[str, str] = {"": device}
        model, tok, _ = load_hf_model(name, device_map=device_map)
        bottomk_lists = compute_all_bottomk_for_model(
            model,
            tok,
            all_prompts,
            int(args.bottom_k_vocab),
            device,
            int(args.batch_size_bottomk),
        )
        return dict(zip(all_prompts, bottomk_lists))
    finally:
        try:
            if model is not None or tok is not None:
                unload_hf_model(model, tok)
        finally:
            gc.collect()
            _empty_cuda_cache()
            if getattr(args, "cuda_device_reset_each_model", False):
                try_cuda_device_reset(gpu_id)


def compute_all_bottomk_for_model(
    model,
    tok,
    all_prompts: List[str],
    k: int,
    device: str,
    batch_size: int = 64,
) -> List[List[int]]:
    results: List[List[int]] = []
    for start in range(0, len(all_prompts), batch_size):
        chunk = all_prompts[start : start + batch_size]
        results.extend(compute_bottomk_vocab_batch(model, tok, prompts=chunk, k=k, device=device))
    return results


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
