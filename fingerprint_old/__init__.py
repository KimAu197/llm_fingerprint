"""RoFL fingerprint testing helpers."""
from .fingerprint_tools import (
    evaluate_fingerprints,
    format_full_prompt,
    generate_fingerprints_batch,
    greedy_response_hf,
    sample_fingerprint_prompt,
    set_seed,
    ts_path,
)
from .model_utils import greedy_once, load_hf_model
from .suspect_wrappers import (
    SuspectFromLoadedHF,
    SuspectModelHF,
    SuspectModelLlamaCpp,
)

__all__ = [
    "evaluate_fingerprints",
    "format_full_prompt",
    "generate_fingerprints_batch",
    "greedy_once",
    "greedy_response_hf",
    "load_hf_model",
    "sample_fingerprint_prompt",
    "set_seed",
    "ts_path",
    "SuspectFromLoadedHF",
    "SuspectModelHF",
    "SuspectModelLlamaCpp",
]
