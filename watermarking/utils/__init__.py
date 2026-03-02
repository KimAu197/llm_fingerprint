# Utils for bottom-k fingerprinting

from .prompt_format import format_full_prompt
from .fingerprint_gen import sample_fingerprint_prompt, generate_fingerprints_batch
from .text_gen import greedy_response_hf
from .metrics import (
    metric_prefix_match,
    metric_lcs_ratio,
    metric_signature_overlap,
    metric_prefix_match_raw,
    metric_lcs_ratio_raw,
    metric_signature_overlap_raw,
    lineage_score_simple,
    overlap_ratio,
)
from .model_loader import (
    set_seed,
    load_hf_model,
    unload_hf_model,
    unload_llama_cpp,
    SuspectModelHF,
    SuspectModelLlamaCpp,
    SuspectFromLoadedHF,
)
from .bottomk_processor import (
    BottomKLogitsProcessor,
    compute_bottomk_vocab_for_model,
    compute_bottomk_vocab_batch,
)

