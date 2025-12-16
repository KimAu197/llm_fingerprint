# Model Lineage Testing

Tools for detecting whether a language model is derived (finetuned) from a base model.

## Table of Contents

- [Watermarking (Bottom-k Fingerprinting)](#watermarking-bottom-k-fingerprinting)
- [Other Methods](#other-methods)

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

Experiment results are stored in `result_*` and `report_*` directories.

