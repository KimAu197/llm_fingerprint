"""
run_pairwise_overlap_matrix_gguf.py

Same 3-phase workflow as run_pairwise_overlap_matrix.py, but each row is a GGUF
file + a HuggingFace *tokenizer* id (``tokenizer_from``) for the source
checkpoint, so token ids line up with the file.

The Hub repo
https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF
ships multiple quants (Q2_K, Q3_K_M, Q4_K_M, ... F16). Each .gguf can be
one row in the CSV, so you get one fingerprint column per quant and a full
pairwise overlap matrix across all listed files.

CSV columns (header required):
  - tokenizer_from: HuggingFace id for ``AutoTokenizer.from_pretrained``
    (must match the conversion source for that GGUF), e.g.
    Qwen/Qwen2.5-Coder-7B-Instruct
  - Either:
    - gguf_path (or path): existing local .gguf file, OR
    - hf_repo (or repo_id) + hf_filename: file is downloaded with
      huggingface_hub (cache under HF cache; no manual download needed)

Optional:
  - model_key (or name): short id used in fingerprints.json keys and the overlap
    matrix; default is repo::filename or the basename of gguf_path

Requires: llama-cpp-python, huggingface_hub (``pip install llama-cpp-python huggingface_hub``)

Usage (Hub download, no local file):
  python run_pairwise_overlap_matrix_gguf.py \\
    --csv_path data/csv_unsloth_qwen2.5_coder7b_instruct_gguf_sample.csv \\
    --gpu_ids 0 --output_dir ./out_gguf
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_WMD = Path(__file__).resolve().parent.parent
if str(_WMD) not in sys.path:
    sys.path.insert(0, str(_WMD))
_TEST = Path(__file__).resolve().parent
if str(_TEST) not in sys.path:
    sys.path.insert(0, str(_TEST))

from utils import (
    set_seed,
    unload_llama_cpp,
)
from utils.gguf_fingerprint import (
    check_tokenizer_vocab_vs_gguf,
    compute_bottomk_vocab_gguf_chunked,
    load_hf_tokenizer_only,
    sample_fingerprint_prompt_gguf,
)
from run_pairwise_overlap_matrix import (
    _FlushFileHandler,
    _maybe_reset_cuda_after_model,
    format_time,
    print_progress,
    phase3_build_matrix,
    save_matrix_artifacts,
    update_overlap_matrix_for_new_cache,
)

_LOG = logging.getLogger("pairwise_overlap_gguf")


@dataclasses.dataclass(frozen=True)
class GgufModelRow:
    """One GGUF in the run: stable ``name`` (dict/matrix key), local path, tokenizer id."""

    name: str
    tokenizer_from: str
    gguf_path: str


def _get_csv_field(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _resolve_gguf_one(
    row: dict,
    *,
    hf_token: Optional[str] = None,
) -> GgufModelRow:
    tokenizer_from = _get_csv_field(row, "tokenizer_from", "based_on", "source_tokenizer")
    if not tokenizer_from:
        raise ValueError("CSV row missing tokenizer_from (HuggingFace tokenizer id)")

    model_key = _get_csv_field(row, "model_key", "name", "id")
    gguf_path = _get_csv_field(row, "gguf_path", "path", "local_gguf")
    hf_repo = _get_csv_field(
        row, "hf_repo", "huggingface_repo", "repo_id", "hub_repo"
    )
    hf_fn = _get_csv_field(
        row,
        "hf_filename",
        "gguf_filename",
        "filename",
        "hf_file",
        "huggingface_file",
    )
    local: Optional[str] = None
    if gguf_path and os.path.isfile(gguf_path):
        local = os.path.abspath(gguf_path)
    elif hf_repo and hf_fn:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to download gguf. pip install huggingface_hub"
            ) from e
        tok = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        local = str(
            hf_hub_download(
                repo_id=hf_repo,
                filename=hf_fn,
                token=tok,
            )
        )
        _LOG.info("Resolved hub file %s/%s -> %s", hf_repo, hf_fn, local)
    elif gguf_path and not os.path.isfile(gguf_path):
        raise FileNotFoundError(
            f"gguf_path not found: {gguf_path!r}. Use an absolute path, or set "
            f"hf_repo and hf_filename for auto-download."
        )
    else:
        raise ValueError(
            "Need either (1) existing file in gguf_path, or (2) hf_repo + hf_filename for Hub download"
        )

    if not local or not os.path.isfile(local):
        raise RuntimeError(f"invalid local gguf: {local!r}")

    name = model_key or (f"{hf_repo}::{hf_fn}" if (hf_repo and hf_fn) else os.path.basename(local))
    if not name:
        name = local

    return GgufModelRow(name=name, tokenizer_from=tokenizer_from, gguf_path=local)


def _assert_unique_model_names(models: List[GgufModelRow]) -> None:
    seen: set = set()
    for m in models:
        if m.name in seen:
            raise ValueError(
                f"Duplicate model_key/name: {m.name!r} — use unique model_key per row"
            )
        seen.add(m.name)


def _gguf_post_unload_gc() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def setup_gguf_run_logging(output_dir: Path) -> None:
    _LOG.setLevel(logging.DEBUG)
    _LOG.handlers.clear()
    log_path = output_dir / "pairwise_overlap_gguf.log"
    fh = _FlushFileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _LOG.addHandler(fh)
    _LOG.info("Logging to %s", log_path.resolve())


def _load_csv_dict_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def resolve_gguf_csv(
    csv_path: str,
    *,
    hf_token: Optional[str] = None,
) -> List[GgufModelRow]:
    """
    One CSV row -> one GgufModelRow; Hub files are downloaded via ``hf_hub_download``.
    """
    out: List[GgufModelRow] = []
    for raw in _load_csv_dict_rows(csv_path):
        if not raw or not any((v and str(v).strip()) for v in raw.values() if v is not None):
            continue
        out.append(_resolve_gguf_one(raw, hf_token=hf_token))
    if not out:
        raise SystemExit(
            "No valid rows in CSV: need tokenizer_from, and (existing gguf_path) "
            "or (hf_repo + hf_filename)"
        )
    _assert_unique_model_names(out)
    return out


def _load_llama(gguf_path: str, args: argparse.Namespace):
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "llama_cpp is required for this script. Install e.g. "
            "`pip install llama-cpp-python` (or a CUDA/Metal build)."
        ) from e

    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    return Llama(
        model_path=gguf_path,
        n_ctx=int(args.n_ctx),
        n_batch=int(args.n_batch),
        logits_all=True,
        n_gpu_layers=int(args.n_gpu_layers),
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1_generate_fingerprints(
    model_rows: List[GgufModelRow],
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
) -> Dict[str, List[str]]:
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATE FINGERPRINTS (GGUF + HF tokenizer)")
    print("=" * 70)

    all_fps: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    times: List[float] = []
    t0 = time.time()

    for i, mrow in enumerate(model_rows):
        name = mrow.name
        print(f"\n[{i+1}/{len(model_rows)}] {name}")
        print(f"  gguf_path: {mrow.gguf_path}")
        print(f"  tokenizer_from: {mrow.tokenizer_from}")
        mt = time.time()

        llm = None
        tok = None
        model_failed = False
        cleanup_failed = False
        last_exc: Optional[BaseException] = None

        try:
            tok = load_hf_tokenizer_only(mrow.tokenizer_from)
            llm = _load_llama(mrow.gguf_path, args)
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

            all_fps[name] = fps

        except Exception as e:
            model_failed = True
            last_exc = e
            print(f"  [ERROR] {e}")
            errors[name] = traceback.format_exc()
            _LOG.exception("Phase 1 failed for %s", name)
        finally:
            try:
                if llm is not None:
                    unload_llama_cpp(llm)
                if tok is not None:
                    del tok
            except Exception as cleanup_err:
                cleanup_failed = True
                print(f"  [WARNING] Cleanup failed: {cleanup_err}")
            _gguf_post_unload_gc()

            _maybe_reset_cuda_after_model(
                gpu_id, args, model_failed=model_failed, cleanup_failed=cleanup_failed,
                last_exception=last_exc,
            )

        times.append(time.time() - mt)
        print_progress(i, len(model_rows), times, t0)

        if (i + 1) % 5 == 0 or (i + 1) == len(model_rows):
            fp_path = output_dir / "fingerprints.json"
            with open(fp_path, "w", encoding="utf-8") as f:
                json.dump(all_fps, f, indent=2)
            print(f"  [AUTOSAVE] {fp_path}")

    if errors:
        err_path = output_dir / "fingerprint_errors.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"\n[WARNING] {len(errors)} models failed fingerprint generation")

    print(
        f"\nPhase 1 done: {len(all_fps)}/{len(model_rows)} models OK  "
        f"({format_time(time.time() - t0)})"
    )
    return all_fps


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def phase2_compute_caches(
    model_rows: List[GgufModelRow],
    all_fps: Dict[str, List[str]],
    args: argparse.Namespace,
    gpu_id: int,
    output_dir: Path,
) -> Dict[str, Dict[str, List[int]]]:
    print("\n" + "=" * 70)
    print("PHASE 2: COMPUTE BOTTOM-K CACHES (GGUF)")
    print("=" * 70)

    models = [m.name for m in model_rows]
    tok_map = {m.name: m.tokenizer_from for m in model_rows}
    path_map = {m.name: m.gguf_path for m in model_rows}

    all_prompts: List[str] = []
    for m in models:
        if m in all_fps:
            all_prompts.extend(all_fps[m])
    seen: set = set()
    unique_prompts: List[str] = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    all_prompts = unique_prompts
    n_prompts = len(all_prompts)
    print(f"Total unique fingerprints to evaluate: {n_prompts}")
    print()

    caches: Dict[str, Dict[str, List[int]]] = {}
    failures: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    times: List[float] = []
    t0 = time.time()

    live_matrix = not getattr(args, "no_live_overlap_matrix", False)
    overlap_matrix: Optional[np.ndarray] = None
    if live_matrix:
        overlap_matrix = np.full((len(models), len(models)), -1.0)
        print(
            "[INFO] Live overlap matrix: overlap_matrix.csv / .npy updated after each success."
        )

    for i, name in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] {name}")
        tokenizer_from = tok_map[name]

        if name not in all_fps:
            reason = "No fingerprints (Phase 1 failed or missing from JSON)"
            print(f"  [SKIP] {reason}")
            skipped[name] = reason
            _LOG.warning("Phase 2 skip %s: %s", name, reason)
            continue

        mt = time.time()
        llm = None
        tok = None
        model_failed = False
        cleanup_failed = False
        last_exc: Optional[BaseException] = None

        try:
            tok = load_hf_tokenizer_only(tokenizer_from)
            llm = _load_llama(path_map[name], args)
            check_tokenizer_vocab_vs_gguf(tok, llm)

            bottomk_lists = compute_bottomk_vocab_gguf_chunked(
                llm,
                tok,
                all_prompts,
                k=args.bottom_k_vocab,
                batch_size=max(1, getattr(args, "batch_size_bottomk", 1)),
            )
            cache = {fp: bk for fp, bk in zip(all_prompts, bottomk_lists)}
            caches[name] = cache
            print(f"  Computed bottom-k for {n_prompts} fingerprints")

            if live_matrix and overlap_matrix is not None:
                update_overlap_matrix_for_new_cache(
                    overlap_matrix, models, all_fps, caches, name
                )
                save_matrix_artifacts(overlap_matrix, models, output_dir, sync=True)
                prog = output_dir / "phase2_progress.json"
                with open(prog, "w", encoding="utf-8") as pf:
                    json.dump(
                        {
                            "caches_ok_models": list(caches.keys()),
                            "num_caches_ok": len(caches),
                            "last_completed": name,
                            "timestamp": datetime.now().isoformat(),
                        },
                        pf,
                        indent=2,
                    )
                print(
                    f"  [CHECKPOINT] overlap matrix + {prog.name} "
                    f"({len(caches)} models in caches)"
                )
        except Exception as e:
            model_failed = True
            last_exc = e
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failures[name] = traceback.format_exc()
            _LOG.exception("Phase 2 failed for %s", name)
        finally:
            try:
                if llm is not None:
                    unload_llama_cpp(llm)
                if tok is not None:
                    del tok
            except Exception as cleanup_err:
                cleanup_failed = True
                print(f"  [WARNING] Cleanup failed: {cleanup_err}")
            _gguf_post_unload_gc()

            _maybe_reset_cuda_after_model(
                gpu_id, args, model_failed=model_failed, cleanup_failed=cleanup_failed,
                last_exception=last_exc,
            )

        times.append(time.time() - mt)
        print_progress(i, len(models), times, t0)

    print(
        f"\nPhase 2 done: {len(caches)}/{len(models)} models OK  "
        f"({format_time(time.time() - t0)})"
    )
    status_path = output_dir / "phase2_cache_status.json"
    status = {
        "failures": failures,
        "skipped": skipped,
        "num_caches_ok": len(caches),
        "num_models": len(models),
        "num_unique_prompts": n_prompts,
        "batch_size_bottomk": getattr(args, "batch_size_bottomk", 1),
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    if failures or skipped:
        print(
            f"\n[WARNING] Phase 2: {len(failures)} failure(s), {len(skipped)} skip(s). "
            f"See {status_path.name} and pairwise_overlap_gguf.log"
        )
    return caches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            str(args.gpu_ids).split(",")[0].strip()
        )

    set_seed(args.seed)
    t_start = time.time()
    gpu_id = int(str(args.gpu_ids or "0").split(",")[0])

    hf_tok = getattr(args, "hf_token", None)
    model_rows = resolve_gguf_csv(args.csv_path, hf_token=hf_tok)

    models = [m.name for m in model_rows]
    n = len(models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_gguf_run_logging(output_dir)

    print("=" * 70)
    print("PAIRWISE OVERLAP MATRIX (GGUF + HF tokenizer)")
    print("=" * 70)
    print(f"Models:          {n}")
    print(f"Fingerprints:    {args.num_fingerprints} per model")
    print(f"Bottom-k vocab:  {args.bottom_k_vocab}")
    print(f"n_ctx / n_batch: {args.n_ctx} / {args.n_batch}  n_gpu_layers: {args.n_gpu_layers}")
    print(f"Output:          {output_dir}")
    if args.fingerprints_file:
        print(f"Fingerprints:    {args.fingerprints_file} (skip Phase 1)")
    print()

    if args.fingerprints_file:
        with open(args.fingerprints_file, "r", encoding="utf-8") as f:
            all_fps = json.load(f)
    else:
        all_fps = phase1_generate_fingerprints(model_rows, args, gpu_id, output_dir)

    caches = phase2_compute_caches(model_rows, all_fps, args, gpu_id, output_dir)
    phase3_build_matrix(models, all_fps, caches, output_dir)

    metadata = {
        "models": models,
        "tokenizer_by_model": {m.name: m.tokenizer_from for m in model_rows},
        "gguf_path_by_model": {m.name: m.gguf_path for m in model_rows},
        "num_models": n,
        "num_fingerprints": args.num_fingerprints,
        "bottom_k_vocab": args.bottom_k_vocab,
        "k_bottom_sampling": args.k_bottom_sampling,
        "fingerprint_length": args.fingerprint_length,
        "seed": args.seed,
        "gpu_id": gpu_id,
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_gpu_layers": args.n_gpu_layers,
        "backend": "llama.cpp_gguf",
        "total_time_seconds": time.time() - t_start,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "experiment_metadata_gguf.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTotal time:  {format_time(time.time() - t_start)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise overlap (GGUF via llama.cpp, tokenizer from HuggingFace id)"
    )
    p.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help=(
            "CSV: tokenizer_from, and either local gguf_path, or hf_repo+hf_filename "
            "(downloaded to HF cache; see data/csv_unsloth_*.csv)"
        ),
    )
    p.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional HF token; defaults to env HF_TOKEN / HUGGINGFACE_HUB_TOKEN (gated/gated LFS).",
    )
    p.add_argument("--output_dir", type=str, default="./pairwise_gguf_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_fingerprints", type=int, default=5)
    p.add_argument("--k_bottom_sampling", type=int, default=50)
    p.add_argument("--fingerprint_length", type=int, default=64)
    p.add_argument("--bottom_k_vocab", type=int, default=2000)
    p.add_argument("--gpu_ids", type=str, default="0", help="Sets CUDA_VISIBLE_DEVICES to first id")
    p.add_argument(
        "--fingerprints_file", type=str, default=None, help="Skip Phase 1; load this JSON"
    )
    p.add_argument(
        "--batch_size_bottomk", type=int, default=1,
        help="Stride for progress (GGUF is per-prompt; keep 1)",
    )
    p.add_argument(
        "--n_ctx", type=int, default=8192, help="llama.cpp context size (>= longest tokenized prompt)"
    )
    p.add_argument(
        "--n_batch",
        type=int,
        default=512,
        help="llama.cpp n_batch (>= number of tokens per eval chunk)",
    )
    p.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="llama.cpp: GPU layers (-1 = all, 0 = CPU only)",
    )
    p.add_argument(
        "--cuda_device_reset_each_model", action="store_true",
        help="Call cuda device reset after each model (opt-in, same as HF script)",
    )
    p.add_argument(
        "--no_cuda_device_reset_on_error", action="store_true", help="Deprecated (no-op)"
    )
    p.add_argument(
        "--cuda_device_reset_after_oom", action="store_true", help="Deprecated (no-op)"
    )
    p.add_argument(
        "--no_live_overlap_matrix",
        action="store_true",
        help="Do not update overlap matrix after each Phase-2 model",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
