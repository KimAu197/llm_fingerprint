# Weekly Results Summary - Week of January 26, 2026

## Overview

This week's experiments focused on two main research directions:
1. **Download Count Impact Analysis**: Investigating whether model download counts affect overlap patterns
2. **FT Training Experiments**: Studying how fine-tuning affects bottom-k wordlist overlap

---

## 1. Download Count Impact Analysis

### Motivation
We hypothesized that filtering models by download count could help identify more reliable patterns, as low-download models might represent personal/experimental fine-tunes with noisy characteristics.

### Approach
- Filtered models with **downloads > 10** to exclude low-quality/personal models
- Excluded models that cannot directly load logits and CV task models
- Compared overlap patterns between Qwen and LLaMA families

### Key Findings

#### For Previously Tested Models:
Download count does show **some impact** on overlap patterns:
- Low-download models (≤10) exhibit more noise and variance
- High-download models show more stable and reliable overlap patterns

#### For New Filtered Experiment (downloads > 10):
After filtering out models with downloads ≤10 and removing non-compatible models:

**Qwen Family:**
- Still shows relatively **low overlap** with derivative models
- Overlap patterns remain consistent but with reduced noise

**LLaMA Family:**
- Shows **better performance** compared to Qwen
- Higher and more stable overlap with same-family derivatives
- More reliable fingerprinting characteristics

### Analysis Location
- **Detailed Analysis**: `result_1.26/download/ANALYSIS_REPORT.md`
- **Experiment Summary**: `result_1.26/download/experiment/SUMMARY.md` (if exists)
- **Data Files**: 
  - Filtered results: `result_1.26/download/filtered/`
  - Original data: `result_1.26/download/original/data/`
  - Popular models: `result_1.26/download/experiment/data/`
- **Visualizations**: 
  - `result_1.26/download/experiment/overlap_vs_downloads_analysis.png`
  - `result_1.26/download/experiment/qwen_vs_llama_comparison.png`

---

## 2. SFT Training Experiments

### Motivation
Understanding how supervised fine-tuning affects model fingerprints and whether overlap changes are predictable.

### Experimental Challenges & Parameter Sensitivity

#### Initial Issues:
- **Max token length = 512**: Gradient explosion occurred frequently, overlap decreased extremely rapidly
- Training was unstable and results were unreliable

#### Adjustments Made:
1. **Increased max_length to 2048**: Improved stability but still experienced gradient explosion leading to NaN values
2. **Added BF16 mixed precision**: Training finally stabilized

#### Critical Concern:
**If the method is so sensitive to parameter settings, is it still a robust approach?**

This raises important questions about:
- The reliability of overlap-based fingerprinting under different training configurations
- Whether the method can generalize across various fine-tuning scenarios

### Current Results (500 steps)

#### Training Setup:
- **Base Model**: Qwen2.5-0.5B
- **Training Steps**: 500
- **Datasets**: English Wikipedia (`20231101.en`) and Japanese Wikipedia (`20231101.ja`)
- **Evaluation Frequency**: Every 2 steps
- **Key Parameters**:
  - Batch size: 4 per device
  - Gradient accumulation: 8 steps (effective batch size = 32)
  - Learning rate: 2e-6
  - Max length: 2048
  - BF16: Enabled

#### Overlap Changes:
- **Initial overlap**: ~97.8%
- **Final overlap (step 500)**: 
  - English: ~14.5%
  - Japanese: ~14.9%
- **Lowest overlap observed**: 
  - English: ~9.7% (step 402)
  - Japanese: ~7.3% (step 300)

**Key Observation**: Overlap decreases **dramatically** during training, but the lowest values remain **above 0.1** (10%).

#### Critical Finding: Language Independence

**Surprising discovery**: Training data language has **no impact** on overlap change patterns!

- English and Japanese Wikipedia training show nearly identical overlap trajectories
- Step 50: English 0.8437 vs Japanese 0.8435 (almost identical)
- Step 500: English 0.1452 vs Japanese 0.1490 (difference < 0.4%)
- Total decrease: ~83% for both languages

**Implication**: The overlap change mechanism is an inherent property of the training process itself, independent of the language characteristics of the training data.

### Analysis Location
- **Experiment Summary**: `result_1.26/trainning_sft/EXPERIMENT_SUMMARY.md`
- **Data Files**:
  - English results: `result_1.26/trainning_sft/overlap_summary-en.csv`
  - Japanese results: `result_1.26/trainning_sft/overlap_summary-jp.csv`
- **Visualization**: `result_1.26/trainning_sft/overlap_training.png`
- **Training Scripts**:
  - Main code: `llm_fingerprint/watermarking/train/train_and_eval_overlap.py`
  - English run: `llm_fingerprint/watermarking/train/run_wikipedia_en.sh`
  - Japanese run: `llm_fingerprint/watermarking/train/run_wikipedia_ja.sh`

---

## Summary & Next Steps

### Critical Questions to Address

1. **Convergence behavior**: Run longer training experiments to see if overlap stabilizes or continues decreasing
2. **Robustness**: Investigate why the method is so sensitive to training parameters
3. **Generalization**: Test on more model families and training scenarios
