"""
Append GGUF models to an existing labeled overlap matrix.

This is the GGUF-target counterpart of ``run_pairwise_overlap_phase2_retry.py``:

- ``--csv_path`` is a GGUF CSV with columns:
  ``model_key,tokenizer_from,hf_repo,hf_filename`` or ``gguf_path``.
- Existing matrix rows/columns are assumed to be HuggingFace model ids.
- New GGUF rows/columns are keyed by ``model_key``.
- Missing GGUF fingerprints are generated and merged into ``--fingerprints_file``.

Use this first with a small CSV such as ``csv_unsloth_requested_gguf_q4km.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pairwise_overlap_matrix import (
    _compute_all_bottomk_for_model,
    _maybe_reset_cuda_after_model,
    format_time,
    print_progress,
    save_matrix_artifacts,
    setup_run_logging,
)
from run_pairwise_overlap_matrix_gguf import (
    GgufModelRow,
    _gguf_post_unload_gc,
    _load_llama,
    resolve_gguf_csv,
)
from run_pairwise_overlap_phase2_retry import (
    PHASE2_PROGRESS_NAME,
    PHASE2_STATUS_NAME,
    _dedup_all_prompts,
    _dedupe_preserve_order,
    _matrix_cell_mi_mj,
    _models_with_existing_labeled_matrix_seed,
    _overlap_npy_row_order_from_cli,
    load_overlap_matrix_from_path,
)
from utils import load_hf_model, set_seed, unload_hf_model, unload_llama_cpp
from utils.gguf_fingerprint import (
    check_tokenizer_vocab_vs_gguf,
    compute_bottomk_vocab_gguf_chunked,
    load_hf_tokenizer_only,
    sample_fingerprint_prompt_gguf,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Append GGUF target models to an existing overlap matrix"
    )
    p.add_argument("--csv_path", type=str, required=True, help="GGUF CSV")
    p.add_argument("--fingerprints_file", type=str, required=True)
    p.add_argument("--existing_overlap_matrix", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./pairwise_retry_gguf_out")
    p.add_argument("--gpu_ids", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_fingerprints", type=int, default=5)
    p.add_argument("--fingerprint_length", type=int, default=64)
    p.add_argument("--k_bottom_sampling", type=int, default=50)
    p.add_argument("--bottom_k_vocab", type=int, default=2000)
    p.add_argument("--batch_size_bottomk", type=int, default=24)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_batch", type=int, default=512)
    p.add_argument("--n_gpu_layers", type=int, default=-1)
    p.add_argument("--llama_verbose", action="store_true")
    p.add_argument("--no_live_overlap_matrix", action="store_true")
    p.add_argument("--strict_overlap_matrix_shape", action="store_true")
    p.add_argument("--existing_overlap_npy_row_order", type=str, default=None)
    p.add_argument("--cuda_device_reset_each_model", action="store_true")
    p.add_argument("--no_cuda_device_reset_on_error", action="store_true")
    p.add_argument("--cuda_device_reset_after_oom", action="store_true")
    p.add_argument("--hf_token", type=str, default=None)
    return p.parse_args()


def _gpu_id_from_args(args: argparse.Namespace) -> int:
    return int(args.gpu_ids.split(",")[0]) if args.gpu_ids else 0


def _load_fingerprints(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        print(f"[INFO] No existing file at {path}; starting empty fingerprints dict.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_fingerprints(path: Path, all_fps: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_fps, f, indent=2)


def ensure_gguf_fingerprints(
    targets: List[GgufModelRow],
    all_fps: Dict[str, List[str]],
    args: argparse.Namespace,
    gpu_id: int,
    fp_path: Path,
    output_dir: Path,
) -> Dict[str, List[str]]:
    missing = [row for row in targets if row.name not in all_fps or not all_fps.get(row.name)]
    if not missing:
        return all_fps

    print("\n" + "=" * 70)
    print("PHASE 1 (GENERATE MISSING GGUF FINGERPRINTS)")
    print("=" * 70)
    merged = dict(all_fps)
    errors: Dict[str, str] = {}
    times: List[float] = []
    t0 = time.time()

    for i, row in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] fingerprints: {row.name}")
        print(f"  gguf_path: {row.gguf_path}")
        print(f"  tokenizer_from: {row.tokenizer_from}")
        mt = time.time()
        llm = None
        tok = None
        model_failed = False
        cleanup_failed = False
        last_exc: Optional[BaseException] = None
        try:
            tok = load_hf_tokenizer_only(row.tokenizer_from)
            llm = _load_llama(row.gguf_path, args)
            check_tokenizer_vocab_vs_gguf(tok, llm)
            fps = []
            for fi in range(args.num_fingerprints):
                fp = sample_fingerprint_prompt_gguf(
                    llm,
                    tok,
                    l_random_prefix=8,
                    total_len=args.fingerprint_length,
                    k_bottom=args.k_bottom_sampling,
                )
                fps.append(fp)
                print(f"  fp {fi+1}/{args.num_fingerprints} generated")
            merged[row.name] = fps
            _save_fingerprints(fp_path, merged)
            print(f"  [SAVED] {fp_path}")
        except Exception as e:
            model_failed = True
            last_exc = e
            print(f"  [ERROR] {e}")
            errors[row.name] = traceback.format_exc()
        finally:
            try:
                if llm is not None:
                    unload_llama_cpp(llm)
                if tok is not None:
                    del tok
            except Exception:
                cleanup_failed = True
            _gguf_post_unload_gc()
            _maybe_reset_cuda_after_model(
                gpu_id,
                args,
                model_failed=model_failed,
                cleanup_failed=cleanup_failed,
                last_exception=last_exc,
            )
        times.append(time.time() - mt)
        print_progress(i, len(missing), times, t0)

    if errors:
        err_path = output_dir / "fingerprint_regenerate_errors_gguf.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        raise SystemExit(
            f"Phase 1 could not create GGUF fingerprints for: {list(errors)}\n"
            f"Details: {err_path}"
        )
    return merged


def _run_gguf_bottomk(
    row: GgufModelRow,
    all_prompts: List[str],
    args: argparse.Namespace,
    gpu_id: int,
) -> Dict[str, List[int]]:
    llm = None
    tok = None
    try:
        tok = load_hf_tokenizer_only(row.tokenizer_from)
        llm = _load_llama(row.gguf_path, args)
        check_tokenizer_vocab_vs_gguf(tok, llm)
        bottomk_lists = compute_bottomk_vocab_gguf_chunked(
            llm,
            tok,
            all_prompts,
            k=args.bottom_k_vocab,
            batch_size=max(1, args.batch_size_bottomk),
        )
        return dict(zip(all_prompts, bottomk_lists))
    finally:
        try:
            if llm is not None:
                unload_llama_cpp(llm)
            if tok is not None:
                del tok
        finally:
            _gguf_post_unload_gc()
            _maybe_reset_cuda_after_model(
                gpu_id,
                args,
                model_failed=False,
                cleanup_failed=False,
                last_exception=None,
            )


def _run_hf_bottomk(
    model_id: str,
    all_prompts: List[str],
    args: argparse.Namespace,
    gpu_id: int,
) -> Dict[str, List[int]]:
    device = f"cuda:{gpu_id}"
    model = None
    tok = None
    try:
        model, tok, _ = load_hf_model(model_id, device_map={"": device})
        bottomk_lists = _compute_all_bottomk_for_model(
            model,
            tok,
            all_prompts,
            args.bottom_k_vocab,
            device,
            args.batch_size_bottomk,
        )
        return dict(zip(all_prompts, bottomk_lists))
    finally:
        if model is not None or tok is not None:
            unload_hf_model(model, tok)


def run_append_gguf(
    models: List[str],
    target_rows: List[GgufModelRow],
    all_fps: Dict[str, List[str]],
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
) -> None:
    targets = [row.name for row in target_rows]
    target_set = set(targets)
    row_by_name = {row.name: row for row in target_rows}
    all_prompts = _dedup_all_prompts(models, all_fps)
    target_idx = {target: models.index(target) for target in targets}

    print("\n" + "=" * 70)
    print("PHASE 2 (APPEND GGUF TARGETS)")
    print("=" * 70)
    print(f"[INFO] Unique prompts: {len(all_prompts)}")
    print(f"[INFO] GGUF targets:   {len(targets)}")

    matrix = load_overlap_matrix_from_path(
        Path(args.existing_overlap_matrix),
        models,
        strict_shape=args.strict_overlap_matrix_shape,
        npy_row_order=_overlap_npy_row_order_from_cli(args),
    )
    print(f"[INFO] Seeded matrix from {args.existing_overlap_matrix}")

    pinned: Dict[str, Dict[str, List[int]]] = {}
    print("\n--- Bottom-k for GGUF target(s) ---")
    for i, row in enumerate(target_rows):
        print(f"\n[target {i+1}/{len(target_rows)}] {row.name}")
        pinned[row.name] = _run_gguf_bottomk(row, all_prompts, args, gpu_id)
        matrix[target_idx[row.name], target_idx[row.name]] = 1.0
        save_matrix_artifacts(matrix, models, output_dir, sync=True)

    for t1, t2 in zip(targets, targets):
        matrix[target_idx[t1], target_idx[t2]] = 1.0

    for i, t1 in enumerate(targets):
        for t2 in targets[i + 1 :]:
            i1, i2 = target_idx[t1], target_idx[t2]
            matrix[i1, i2] = _matrix_cell_mi_mj(t1, t2, all_fps, pinned[t1], pinned[t2])
            matrix[i2, i1] = _matrix_cell_mi_mj(t2, t1, all_fps, pinned[t2], pinned[t1])
    save_matrix_artifacts(matrix, models, output_dir, sync=True)

    others = [model_id for model_id in models if model_id not in target_set]
    times: List[float] = []
    t0 = time.time()
    errors: Dict[str, str] = {}
    print(f"\n--- Cross {len(others)} non-target model(s) vs GGUF target(s) ---")
    for step, model_id in enumerate(others):
        print(f"\n[{step+1}/{len(others)}] vs {model_id}")
        mt = time.time()
        try:
            if model_id in row_by_name:
                cache_j = pinned[model_id]
            else:
                cache_j = _run_hf_bottomk(model_id, all_prompts, args, gpu_id)
            ij = models.index(model_id)
            for target in targets:
                it = target_idx[target]
                matrix[it, ij] = _matrix_cell_mi_mj(
                    target, model_id, all_fps, pinned[target], cache_j
                )
                matrix[ij, it] = _matrix_cell_mi_mj(
                    model_id, target, all_fps, cache_j, pinned[target]
                )
            save_matrix_artifacts(matrix, models, output_dir, sync=True)
            with open(output_dir / PHASE2_PROGRESS_NAME, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": "append_gguf_targets",
                        "targets": targets,
                        "last_cross_with": model_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            errors[model_id] = traceback.format_exc()
            print(f"  [ERROR] {model_id}")
            print(errors[model_id])
        times.append(time.time() - mt)
        print_progress(step, len(others), times, t0)

    save_matrix_artifacts(matrix, models, output_dir, sync=True)
    with open(output_dir / PHASE2_STATUS_NAME, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "append_gguf_targets",
                "targets": targets,
                "num_models": len(models),
                "num_prompts": len(all_prompts),
                "errors": errors,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Final matrix -> {output_dir / 'overlap_matrix.csv'}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    gpu_id = _gpu_id_from_args(args)
    fp_path = Path(args.fingerprints_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(output_dir)

    gguf_rows = resolve_gguf_csv(args.csv_path, hf_token=args.hf_token)
    target_names = [row.name for row in gguf_rows]
    models = _models_with_existing_labeled_matrix_seed(
        csv_models=target_names,
        target_models=target_names,
        existing_overlap_matrix=args.existing_overlap_matrix,
        use_phase2_models_csv=True,
    )
    models = _dedupe_preserve_order(models)

    all_fps = _load_fingerprints(fp_path)
    all_fps = ensure_gguf_fingerprints(
        gguf_rows,
        all_fps,
        args,
        gpu_id,
        fp_path,
        output_dir,
    )
    missing_old = [m for m in models if m not in all_fps and m not in target_names]
    if missing_old:
        print(
            f"[WARNING] {len(missing_old)} existing matrix model(s) lack fingerprints; "
            "their row -> GGUF target cells will remain -1."
        )

    t0 = time.time()
    run_append_gguf(models, gguf_rows, all_fps, args, gpu_id, output_dir)
    metadata = {
        "script": "run_pairwise_overlap_phase2_retry_gguf.py",
        "targets": target_names,
        "num_models": len(models),
        "seed": args.seed,
        "time_seconds": time.time() - t0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "phase2_retry_gguf_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone in {format_time(time.time() - t0)}. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
