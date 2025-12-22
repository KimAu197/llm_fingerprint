# LLM Fingerprinting for Model Lineage Detection

Tools for detecting whether a language model is derived (fine-tuned) from a specific base model using only forward-pass access to model logits.

## Overview

This repository implements two complementary fingerprinting approaches:

1. **Text-Based Fingerprinting (RoFL-style)**: Generate unique prompts using random prefix + bottom-k sampling, then compare generated text responses
2. **Vocabulary Overlap (Bottom-k Subspace)**: Directly compare the low-probability token sets between models

**Key Finding**: Vocabulary overlap significantly outperforms text-based methods, achieving **AUC 0.988** (Qwen2.5-0.5B) and **0.977** (TinyLlama-1.1B-Chat) for distinguishing same-lineage from different-lineage models.

## Table of Contents

- [Quick Start](#quick-start)
- [Methods](#methods)
  - [Text-Based Fingerprinting](#text-based-fingerprinting-rofl-style)
  - [Vocabulary Overlap](#vocabulary-overlap-bottom-k-subspace)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KimAu197/llm_fingerprint.git
cd llm_fingerprint

# Install dependencies
pip install torch transformers pandas numpy scikit-learn matplotlib
```

### Run Vocabulary Overlap Detection (Recommended)

```bash
cd watermarking

# Test if models are derived from Qwen2.5-0.5B
python new/run_bottomk_subspace_overlap_from_base.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --derived_model_csv "../data/qwen2_5_0.5B_ft.csv" \
    --num_pairs 5 \
    --bottom_k_vocab 2000 \
    --csv_path "results/overlap_results.csv"
```

### Run Text-Based Detection

```bash
cd fingerprint_old

python test_batch.py \
    --model_list_csv "../data/qwen2_5_0.5B_ft.csv" \
    --suspect_model "Qwen/Qwen2.5-0.5B" \
    --num_pairs 20 \
    --k_bottom 50 \
    --csv_path "lineage_scores.csv"
```

---

## Methods

### Text-Based Fingerprinting (RoFL-style)

Located in `fingerprint_old/`

**Concept**: Generate unique fingerprint prompts using bottom-k sampling, then compare the deterministic responses between models.

**Pipeline**:
1. **Generate Fingerprint Prompt (x')**: 
   - First n tokens: Random sampling from vocabulary
   - Remaining tokens: Bottom-k sampling (lowest probability tokens)
2. **Generate Response (y)**: Greedy decoding for deterministic output
3. **Compare Responses**: Measure similarity using PAL-k, LCS ratio, and Levenshtein distance

**Metrics**:
- **$\mathrm{PAL}_k$**: Prefix Agreement Length (first k characters match)
- **LCS Ratio**: Longest Common Subsequence / max length
- **Lev Similarity**: 1 - (edit distance / max length)

**Typical AUC**: 0.6 - 0.83 (varies with settings)

### Vocabulary Overlap (Bottom-k Subspace)

Located in `watermarking/`

**Concept**: The bottom-k vocabulary (tokens with lowest logits) is remarkably stable under fine-tuning. Derived models share similar low-probability token distributions with their base model.

**Pipeline**:
1. Generate fingerprint prompts from the base model
2. For each prompt, compute bottom-k token sets for both base and candidate models
3. Calculate overlap ratio: $|\text{bottomk}_{\text{base}} \cap \text{bottomk}_{\text{cand}}| / k$

**Key Parameters**:
- `bottom_k_vocab`: 2000 (recommended)
- `num_pairs`: 5 fingerprint prompts for averaging

**Performance**:
| Model | AUC | Threshold | Accuracy |
|-------|-----|-----------|----------|
| Qwen2.5-0.5B | 0.988 | 0.0082 | 98.6% |
| TinyLlama-1.1B-Chat | 0.977 | 0.0254 | 96.7% |

---

## Results

### Experimental Summary

We evaluated both methods on 100+ fine-tuned models per base model family:

| Method | Qwen AUC | TinyLlama AUC | Speed |
|--------|----------|---------------|-------|
| Text-Based | 0.601 | 0.763 | Slower |
| Vocabulary Overlap | **0.988** | **0.977** | Faster |

### Three-Group Classification

We also tested distinguishing between:
- **Same**: Models derived from Qwen2.5-0.5B
- **Diff**: Models derived from TinyLlama (different architecture)
- **Diff2**: Models derived from Qwen2-0.5B (same family, different version)

With threshold adjustment (0.0719), we achieve **94.5% overall accuracy** across all three groups.

### Result Directories

Experimental results are organized by date:

```
result/
├── result_10.10/   # Initial text-based experiments
├── result_11.10/   # Per-metric analysis (PAL-k, LCS, Lev)
├── result_11.17/   # Hyperparameter tuning
├── result_11.24/   # Initial vocabulary overlap experiments
├── result_12.1/    # JS divergence experiments
├── result_12.8/    # Text vs. word comparison (derived generates fingerprint)
├── result_12.15/   # Fixed base fingerprint (base generates fingerprint)
│   ├── text/       # Text similarity results
│   └── wordlist/   # Vocabulary overlap results
└── result_12.22/   # Three-group classification (Qwen2.5 vs Qwen2 vs TinyLlama)
```

---

## Project Structure

```
llm_fingerprint/
├── README.md                    # This file
├── data/                        # Model lists for experiments
│   ├── qwen2_5_0.5B_ft.csv     # Qwen2.5-0.5B fine-tuned models
│   ├── qwen2_0.5B_ft.csv       # Qwen2-0.5B fine-tuned models
│   └── tinyllama_ft.csv        # TinyLlama fine-tuned models
├── fingerprint_old/             # Text-based fingerprinting (RoFL-style)
│   ├── test_batch.py           # Main batch evaluation script
│   ├── fingerprint_tools.py    # Core fingerprint generation
│   ├── suspect_wrappers.py     # Model wrapper classes
│   └── model_utils.py          # Model loading utilities
├── watermarking/                # Vocabulary overlap methods
│   ├── README.md               # Detailed documentation
│   ├── new/                    # Main pipeline scripts
│   │   ├── run_bottomk_subspace_overlap_from_base.py
│   │   ├── run_bottomk_lineage_pipeline_text_from_base.py
│   │   └── visualization/      # Plotting scripts
│   ├── utils/                  # Shared utilities
│   │   ├── bottomk_processor.py
│   │   ├── fingerprint_gen.py
│   │   ├── metrics.py
│   │   └── model_loader.py
│   └── legacy/                 # Older implementations
└── result/                      # Experimental results
```

---

## Input/Output Format

### Input CSV

Model list CSV should have a `model_id` column:

```csv
model_id
user/qwen2-finetune-v1
user/qwen2-finetune-v2
another-user/model-variant
```

### Output CSV

**Vocabulary Overlap**:
```csv
base_model_name,derived_model_name,avg_overlap_ratio,bottom_k_vocab_size
Qwen/Qwen2.5-0.5B,user/model-ft,0.85,2000
```

**Text-Based**:
```csv
base_model_name,suspect_model_name,pal_k_mean,lev_sim_mean,lcs_ratio_mean,score_mean
user/model-ft,Qwen/Qwen2.5-0.5B,0.85,0.72,0.68,0.75
```

### Error Handling

Failed model loads are marked with `-1.0` in the output. Filter these when analyzing:

```python
df = df[df['avg_overlap_ratio'] != -1.0]
```

---

## Key Findings

1. **Vocabulary overlap captures a fundamental model property**: The structure of the logit distribution is remarkably stable under fine-tuning.

2. **Text-based methods have limitations**: High variance due to generation stochasticity and prompt sensitivity.

3. **Fixed base fingerprint works better**: Using consistent prompts from the base model reduces variance.

4. **Sibling model detection is possible**: With threshold adjustment, we can distinguish between different versions of the same model family (e.g., Qwen2.5 vs Qwen2).

---

## Citation

This implementation draws inspiration from:

```
RoFL: Robust Fingerprinting of Large Language Models
Kirchenbauer et al., A Watermark for Large Language Models
```

---

## License

MIT License
