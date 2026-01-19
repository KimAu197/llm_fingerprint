# Model Lineage Testing

Tools for detecting whether a language model is derived (finetuned) from a base model.

## Table of Contents

- [RoFL Fingerprinting (Random + Bottom-k Sampling)](#rofl-fingerprinting-random--bottom-k-sampling)
- [Watermarking (Bottom-k Fingerprinting)](#watermarking-bottom-k-fingerprinting)
- [Other Methods](#other-methods)

---

## RoFL Fingerprinting (Random + Bottom-k Sampling)

Located in `fingerprint_old/`

### Overview

This method implements the RoFL (Robust Fingerprinting of Large Language Models) approach to detect model lineage. The core idea is to generate unique fingerprints using a two-stage sampling strategy and compare the deterministic responses between models.

### Method

**Step 1: Generate Fingerprint (x', y) on Candidate Model (Fine-tuned)**

1. **Prompt Generation (x')**:
   - First `n` tokens: Random sampling from vocabulary (excluding special tokens)
   - Remaining `l` tokens: Bottom-k sampling (select from k tokens with lowest probability)
   - Default: `n=8`, `total_len=64`, `k_bottom=50`

2. **Response Generation (y)**:
   - Use greedy decoding (temperature=0) to generate deterministic response
   - This creates the fingerprint pair `(x', y)`

**Step 2: Verify on Suspect Model (Base Model)**

1. Use the same fingerprint prompt `x'`
2. Generate response `suspect_y` with greedy decoding
3. Compare text similarity between `base_y` and `suspect_y`

**Step 3: Similarity Metrics**

Three metrics are used to measure similarity:
- **Prefix Match (PAL_k)**: Whether first k characters match exactly
- **LCS Ratio**: Longest Common Subsequence similarity (edit distance)
- **Signature Overlap**: Overlap rate of long words (≥6 chars)

Final lineage score = average of the three metrics (0-1, higher = more likely derived)

### Quick Start

**Generate and Evaluate Fingerprints**

```bash
cd fingerprint_old

# Run batch evaluation on multiple fine-tuned models
python test_batch.py \
    --model_list_csv "../data/qwen2_5_0.5B_ft.csv" \
    --suspect_model "Qwen/Qwen2-0.5B" \
    --num_pairs 20 \
    --k_bottom 50 \
    --prompt_style raw \
    --csv_path "lineage_scores.csv" \
    --save_report_dir "eval_reports/" \
    --k_prefix 30 \
    --relation same
```

**Or use the pipeline script**

```bash
python fingerprint_pipeline.py \
    --model_list_csv "../data/qwen2_5_0.5B_ft.csv" \
    --suspect_model "Qwen/Qwen2-0.5B" \
    --num_pairs 20 \
    --k_bottom 50 \
    --csv_path "lineage_scores.csv"
```

### Parameters

- `--model_list_csv`: CSV file containing candidate models (must have `model_id` column)
- `--suspect_model`: Base model to test against (HuggingFace model ID)
- `--num_pairs`: Number of fingerprint pairs to generate per model (default: 20)
- `--k_bottom`: Size of bottom-k sampling pool (default: 50)
- `--prompt_style`: Prompt format - `raw`, `oneshot`, or `chatml` (default: `raw`)
- `--k_prefix`: Prefix length for PAL_k metric (default: 30)
- `--relation`: Relationship label - `same` or `diff` (for organizing results)
- `--csv_path`: Output CSV file for lineage scores
- `--save_report_dir`: Directory to save detailed evaluation reports (JSON)

### Input Format

The model list CSV should have a `model_id` column:

```csv
model_id
user/qwen2-0.5b-finetune-v1
user/qwen2-0.5b-finetune-v2
another-user/qwen2-variant
```

### Output

**1. Lineage Scores CSV** (`lineage_scores.csv`)

Contains aggregated metrics for each model pair:

```csv
base_model_name,suspect_model_name,relation,num_pairs,k_prefix,pal_chars_mean,pal_k_mean,lev_sim_mean,lcs_ratio_mean,score_mean
user/qwen2-ft,Qwen/Qwen2-0.5B,same,20,30,45.2,0.85,0.72,0.68,0.75
```

**2. Evaluation Reports** (`eval_report_{model}.json`)

Detailed per-fingerprint results:

```json
{
  "base_model_name": "user/qwen2-ft",
  "suspect_model_name": "Qwen/Qwen2-0.5B",
  "prompt_style": "raw",
  "num_pairs": 20,
  "records": [
    {
      "fingerprint": "x' prompt text...",
      "base_y": "response from candidate model...",
      "suspect_y": "response from suspect model..."
    }
  ]
}
```

### Key Files

- `test_batch.py`: Main batch evaluation script with all utilities
- `fingerprint_pipeline.py`: Simplified pipeline wrapper
- `fingerprint_tools.py`: Core fingerprint generation and evaluation functions
- `suspect_wrappers.py`: Model wrapper classes for uniform API
- `model_utils.py`: Model loading utilities
- `cleanup.py`: Memory cleanup utilities

### Notes

- All generation uses **greedy decoding** (do_sample=False) for deterministic results
- Special tokens (BOS, EOS, PAD, UNK) are excluded from random/bottom-k sampling
- Set `seed=42` for reproducibility
- Models are loaded with `device_map="auto"` for automatic GPU allocation
- Memory is cleaned up between model loads to prevent OOM errors

---

## Watermarking (Bottom-k Fingerprinting)

Located in `watermarking/new/`

### Quick Start

**1. Subspace Overlap Detection**

Test if derived models share the same bottom-k vocabulary as a base model:

```bash
cd watermarking

python new/run_bottomk_subspace_overlap_from_base.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --derived_model_csv "../data/qwen_derived_models.csv" \
    --num_pairs 10 \
    --bottom_k_vocab 2000 \
    --csv_path "results/overlap_results.csv"
```

**2. Text Similarity Detection**

Compare bottom-k constrained text generation between base and derived models:

```bash
python new/run_bottomk_lineage_pipeline_text_from_base.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --derived_model_csv "../data/qwen_derived_models.csv" \
    --num_pairs 10 \
    --bottom_k_vocab 2000 \
    --csv_path "results/text_results.csv"
```

**3. Visualize Results**

Plot distribution histograms comparing same-lineage vs different-lineage models:

```bash
# Overlap ratio distribution
python new/visualization/plot_overlap_distribution.py \
    --same_csv results/overlap_same.csv \
    --diff_csv results/overlap_diff.csv \
    --output_dir results/figures/ \
    --model_name qwen

# Text similarity metrics distribution
python new/visualization/plot_text_metrics_distribution.py \
    --same_csv results/text_same.csv \
    --diff_csv results/text_diff.csv \
    --output_dir results/figures/ \
    --model_name qwen
```

### Input Format

The derived model CSV should have a `model_id` column:

```csv
model_id
user/model-finetune-v1
user/model-finetune-v2
another-user/model-variant
```

### Output

- **Overlap pipeline**: `avg_overlap_ratio` (0-1), higher = more likely derived
- **Text pipeline**: `avg_pal_k`, `avg_lev_sim`, `avg_lcs_ratio`, `avg_score` (0-1)
- **Error marker**: `-1.0` indicates failed model load (filter out in analysis)

For detailed documentation, see `watermarking/README.md`.

---

## Other Methods

*Coming soon...*

---

## Results

Experiment results are stored in the following directories:

- `result_10.10/`: Early experimental results
- `result_11.17/`: November experiments with various settings
- `result_12.1/`: December results comparing same vs different lineage
- `result_12.15/`: Latest results with text and wordlist approaches
  - `text/`: Text similarity based detection
  - `wordlist/`: Vocabulary overlap based detection
- `report_11.24/`: Analysis reports with distribution plots
- `report_12.8/`: Comparative analysis for Llama and Qwen models

### Typical Workflow

1. **Generate fingerprints** on fine-tuned models
2. **Evaluate** against base model to get similarity scores
3. **Compare distributions** between same-lineage and different-lineage pairs
4. **Visualize** results to determine detection threshold

### Expected Results

For models with true lineage relationship (fine-tuned from base):
- **PAL_k mean**: 0.7 - 0.9 (high prefix agreement)
- **LCS ratio mean**: 0.6 - 0.8 (high sequence similarity)
- **Lev sim mean**: 0.6 - 0.8 (high edit similarity)
- **Overall score**: 0.7 - 0.85

For unrelated models:
- **PAL_k mean**: 0.0 - 0.2 (low prefix agreement)
- **LCS ratio mean**: 0.1 - 0.3 (low sequence similarity)
- **Lev sim mean**: 0.1 - 0.3 (low edit similarity)
- **Overall score**: 0.1 - 0.25

---

## Citation

This implementation is based on the RoFL paper:

```
Robust Fingerprinting of Large Language Models
```

For watermarking methods, see `watermarking/README.md` for detailed documentation.

