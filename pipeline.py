"""High-level orchestration for RoFL fingerprint generation + evaluation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .cleanup import unload_hf_model
from .fingerprint_tools import evaluate_fingerprints, generate_fingerprints_batch, set_seed
from .model_utils import load_hf_model
from .suspect_wrappers import SuspectFromLoadedHF


@dataclass
class FingerprintPipelineConfig:
    base_model_id: str
    suspect_model_id: str
    base_model_label: Optional[str] = None
    suspect_model_label: Optional[str] = None
    num_pairs: int = 3
    prompt_style: str = "raw"
    l_random_prefix: int = 8
    total_len: int = 64
    k_bottom: int = 50
    max_new_tokens: int = 64
    seed: int = 42
    fingerprint_save_path: str = "fingerprints_init.json"
    reuse_fingerprints_path: Optional[str] = None
    report_save_path: str = "eval_report.json"
    base_fourbit: bool = False
    suspect_fourbit: bool = False
    timestamp_outputs: bool = True


def run_fingerprint_pipeline(config: FingerprintPipelineConfig):
    """Execute the end-to-end RoFL fingerprint test."""
    set_seed(config.seed)

    base_label = config.base_model_label or config.base_model_id
    suspect_label = config.suspect_model_label or config.suspect_model_id

    base_model, base_tok, _ = load_hf_model(config.base_model_id, fourbit=config.base_fourbit)
    suspect_model, suspect_tok, _ = load_hf_model(config.suspect_model_id, fourbit=config.suspect_fourbit)

    try:
        if config.reuse_fingerprints_path:
            pairs_path = config.reuse_fingerprints_path
            if not Path(pairs_path).exists():
                raise FileNotFoundError(pairs_path)
        else:
            _, pairs_path = generate_fingerprints_batch(
                model=base_model,
                model_name=base_label,
                tokenizer=base_tok,
                num_pairs=config.num_pairs,
                prompt_style=config.prompt_style,
                l_random_prefix=config.l_random_prefix,
                total_len=config.total_len,
                k_bottom=config.k_bottom,
                max_new_tokens=config.max_new_tokens,
                save_json_path=config.fingerprint_save_path,
            )
        if not pairs_path:
            raise RuntimeError("Failed to create fingerprint file")

        suspect_wrapper = SuspectFromLoadedHF(suspect_model, suspect_tok)
        _, report_path = evaluate_fingerprints(
            pairs_json_path=pairs_path,
            model_name=base_label,
            suspect_base_name=suspect_label,
            suspect_model=suspect_wrapper,
            suspect_label=f"{suspect_label}-on-{base_label}",
            save_report_path=config.report_save_path,
            use_timestamp_in_name=config.timestamp_outputs,
        )
        return {
            "fingerprints_path": pairs_path,
            "report_path": report_path,
        }
    finally:
        unload_hf_model(base_model, base_tok)
        unload_hf_model(suspect_model, suspect_tok)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoFL fingerprint pipeline")
    parser.add_argument("--base-model", required=True, help="Model used to create fingerprints")
    parser.add_argument("--suspect-model", required=True, help="Model being evaluated")
    parser.add_argument("--num-pairs", type=int, default=3)
    parser.add_argument("--prompt-style", default="raw", choices=["raw", "oneshot", "chatml"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-path", default="fingerprints_init.json")
    parser.add_argument("--reuse-fingerprints", help="Skip generation and reuse an existing pair file")
    parser.add_argument("--report-path", default="eval_report.json")
    parser.add_argument("--base-fourbit", action="store_true")
    parser.add_argument("--suspect-fourbit", action="store_true")
    parser.add_argument("--no-timestamp", dest="timestamp_outputs", action="store_false")
    parser.set_defaults(timestamp_outputs=True)
    return parser


def main(argv: list[str] | None = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    config = FingerprintPipelineConfig(
        base_model_id=args.base_model,
        suspect_model_id=args.suspect_model,
        num_pairs=args.num_pairs,
        prompt_style=args.prompt_style,
        seed=args.seed,
        fingerprint_save_path=args.fingerprint_path,
        reuse_fingerprints_path=args.reuse_fingerprints,
        report_save_path=args.report_path,
        base_fourbit=args.base_fourbit,
        suspect_fourbit=args.suspect_fourbit,
        timestamp_outputs=args.timestamp_outputs,
    )

    results = run_fingerprint_pipeline(config)
    print("Pipeline finished:", results)


if __name__ == "__main__":
    main()
