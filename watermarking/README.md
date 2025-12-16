# Bottom-k Fingerprinting for LLM Lineage Detection

This module implements bottom-k fingerprinting methods to detect if a model is derived (finetuned) from a base model.

## Directory Structure

```
watermarking/
├── README.md                 # This file
├── utils/                    # Utility modules
│   ├── __init__.py          # Main exports
│   ├── prompt_format.py     # Prompt formatting
│   ├── fingerprint_gen.py   # Fingerprint generation (RoFL-style)
│   ├── text_gen.py          # Text generation utilities
│   ├── metrics.py           # Text similarity metrics
│   ├── model_loader.py      # Model loading/unloading + set_seed
│   └── bottomk_processor.py # Bottom-k logits processor
├── new/                      # Main pipeline scripts
│   ├── run_bottomk_subspace_overlap_from_base.py
│   ├── run_bottomk_lineage_pipeline_text_from_base.py
│   └── visualization/       # Plotting scripts
│       ├── plot_overlap_distribution.py
│       └── plot_text_metrics_distribution.py
└── legacy/                   # Older versions (for reference)
```

## Core Concepts

### Bottom-k Vocabulary
The "bottom-k" vocabulary refers to the k tokens with the **lowest** logits (probability) for a given model when conditioned on a prompt. This is used as a fingerprint space because:
- Derived models tend to share similar low-probability token distributions with their base model
- Unrelated models have different bottom-k vocabularies

### Fingerprint Prompts (x')
Special prompts generated using bottom-k sampling to maximize the fingerprint signal:
1. Start with random tokens
2. Extend by sampling from bottom-k tokens at each step

## Pipelines

### 1. Subspace Overlap (`run_bottomk_subspace_overlap_from_base.py`)

Measures the **vocabulary overlap** between base and derived models' bottom-k sets.

**Usage:**
```bash
python new/run_bottomk_subspace_overlap_from_base.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --derived_model_csv "derived_models.csv" \
    --num_pairs 10 \
    --bottom_k_vocab 2000 \
    --csv_path "results_overlap.csv"
```

**Key Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--base_model_name` | HuggingFace model ID for base model | Required |
| `--derived_model_csv` | CSV file with derived model IDs | Required |
| `--model_id_column` | Column name for model IDs in CSV | `model_id` |
| `--num_pairs` | Number of fingerprint prompts | 10 |
| `--bottom_k_vocab` | Size of bottom-k vocabulary | 2000 |
| `--device` | Device (cuda/mps/cpu) | `cuda` |
| `--csv_path` | Output CSV path | `lineage_bottomk_overlap.csv` |
| `--save_fingerprints` | Save fingerprints to JSON | None |
| `--load_fingerprints` | Load fingerprints from JSON | None |

**Output CSV columns:**
- `base_model_name`: Base model identifier
- `derived_model_name`: Derived model identifier
- `num_pairs`: Number of fingerprint pairs used
- `avg_overlap_ratio`: Average overlap ratio (0-1)
- `bottom_k_vocab_size`: Size of bottom-k vocabulary
- `pair_scores_json`: Per-pair scores as JSON

### 2. Text Similarity (`run_bottomk_lineage_pipeline_text_from_base.py`)

Compares **generated text** when both models generate using their bottom-k vocabularies.

**Usage:**
```bash
python new/run_bottomk_lineage_pipeline_text_from_base.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --derived_model_csv "derived_models.csv" \
    --num_pairs 10 \
    --bottom_k_vocab 2000 \
    --csv_path "results_text.csv"
```

**Additional Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--k_prefix` | Prefix length for PAL_k metric | 30 |
| `--base_max_new_tokens` | Max tokens for base model generation | 64 |
| `--derived_max_new_tokens` | Max tokens for derived model generation | 64 |
| `--max_input_length` | Max input length (truncation) | 256 |

**Output CSV columns:**
- `base_model_name`, `derived_model_name`, `num_pairs`
- `k_prefix`: Prefix length used for PAL_k
- `bottom_k_vocab_size`: Size of bottom-k vocabulary
- `avg_pal_k`: Average prefix agreement (0 or 1)
- `avg_lev_sim`: Average Levenshtein similarity (0-1)
- `avg_lcs_ratio`: Average LCS ratio (0-1)
- `avg_score`: Average combined score (0-1)
- `pair_metrics_json`: Per-pair metrics as JSON

## Input CSV Format

The derived model CSV should have a column with model IDs:

```csv
model_id
user/model-finetune-v1
user/model-finetune-v2
another-user/model-variant
```

## Metrics Explanation

### Overlap Ratio (Subspace)
```
overlap_ratio = |base_bottomk ∩ derived_bottomk| / |base_bottomk|
```
Higher overlap suggests the derived model shares the base model's low-probability token distribution.

### Text Similarity Metrics
- **PAL_k (Prefix Agreement Length)**: 1 if first k characters match, else 0
- **Lev_sim (Levenshtein Similarity)**: 1 - (edit_distance / max_length)
- **LCS_ratio (Longest Common Subsequence)**: Length of LCS / total length
- **Combined Score**: (PAL_k + Lev_sim + LCS_ratio) / 3

## Error Handling

If a model fails to load, the output will show:
- `avg_overlap_ratio = -1.0` or `avg_score = -1.0`
- `num_pairs = 0`
- `error` field with the error message

Filter these out when analyzing results:
```python
df = df[df['avg_overlap_ratio'] != -1.0]
```

## Python API Usage

```python
import sys
sys.path.insert(0, "path/to/watermarking")

from utils import (
    set_seed,
    load_hf_model,
    unload_hf_model,
    sample_fingerprint_prompt,
    compute_bottomk_vocab_for_model,
    lineage_score_simple,
)

# Set seed for reproducibility
set_seed(42)

# Load model
model, tok, device = load_hf_model("Qwen/Qwen2.5-0.5B")

# Generate fingerprint prompt
x_prime = sample_fingerprint_prompt(model, tok, device=device)

# Compute bottom-k vocab
bottomk_ids = compute_bottomk_vocab_for_model(model, tok, k=2000)

# Compare two texts
metrics = lineage_score_simple(text_a, text_b, k_prefix=30)
print(f"Score: {metrics['LineageScoreSimple']}")

# Clean up
unload_hf_model(model, tok)
```

## Visualization

### Plot Overlap Distribution

```bash
python new/visualization/plot_overlap_distribution.py \
    --same_csv results/overlap_same.csv \
    --diff_csv results/overlap_diff.csv \
    --output_dir results/figures/ \
    --model_name qwen
```

### Plot Text Metrics Distribution

```bash
python new/visualization/plot_text_metrics_distribution.py \
    --same_csv results/text_same.csv \
    --diff_csv results/text_diff.csv \
    --output_dir results/figures/ \
    --model_name qwen
```

**Visualization Arguments:**
| Argument | Description |
|----------|-------------|
| `--same_csv` | CSV for same-lineage models |
| `--diff_csv` | CSV for diff-lineage models |
| `--output_dir` | Output directory for figures (auto-created) |
| `--model_name` | Model name for figure title |
| `--same_label` | Custom legend label for "same" |
| `--diff_label` | Custom legend label for "diff" |
| `--metrics` | (text only) Specific metrics to plot |

## Tips

1. **Memory Management**: The pipelines automatically unload models after use. For large-scale experiments, monitor GPU memory.

2. **Reproducibility**: Use `--save_fingerprints` to save generated fingerprints for later reuse with `--load_fingerprints`.

3. **Batch Processing**: Results are appended to CSV after each model, so you can resume interrupted runs.

4. **Filtering Results**: Always filter out `-1.0` values when analyzing results - these indicate failed model loads.

5. **Output Organization**: Use `--output_dir` in visualization scripts to automatically save figures to a specific folder.
