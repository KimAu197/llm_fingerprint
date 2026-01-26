# SFT Training Experiment Summary

## Experiment Overview

This experiment investigates the changes in bottom-k wordlist overlap when fine-tuning the Qwen2.5-0.5B model on Wikipedia data in different languages using Supervised Fine-Tuning (SFT).

**Key Finding: The training language does not affect the overlap change trend. As long as training occurs, the overlap decreases significantly.**

---

## Experimental Setup

### Base Model
- **Model**: `Qwen/Qwen2.5-0.5B`
- **Model Type**: Causal Language Model
- **Parameters**: 0.5B

### Training Data
The experiment was conducted on Wikipedia data in two different languages:

1. **English Wikipedia**: `wikimedia/wikipedia` (config: `20231101.en`)
2. **Japanese Wikipedia**: `wikimedia/wikipedia` (config: `20231101.ja`)

### Training Parameters

Based on the run scripts `run_wikipedia_en.sh` and `run_wikipedia_ja.sh`, the specific parameters are:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_model_name` | `Qwen/Qwen2.5-0.5B` | Base model |
| `dataset_name` | `wikimedia/wikipedia` | Dataset name |
| `dataset_config` | `20231101.en` / `20231101.ja` | English/Japanese config |
| `max_steps` | 500 | Maximum training steps |
| `eval_steps` | 2 | Evaluate overlap every 2 steps |
| `num_train_samples` | 20,000 | Number of training samples |
| `per_device_train_batch_size` | 4 | Batch size per device |
| `gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `learning_rate` | 2e-6 | Learning rate |
| `warmup_steps` | 300 | Warmup steps |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `use_bf16` | True | Use BF16 mixed precision |
| `logging_steps` |  5 | Logging frequency |

**Effective Batch Size**: `per_device_train_batch_size × gradient_accumulation_steps = 4 × 8 = 32`

### Overlap Evaluation Parameters
- **Bottom-k Vocabulary Size**: 2000
- **Number of Fingerprints**: 5
- **Fingerprint Length**: 64 tokens
- **Evaluation Frequency**: Every 2 training steps

---

## Experimental Results

### 1. English Wikipedia Training Results

| Training Step | Overlap Ratio | Trend |
|--------------|---------------|-------|
| 1 | 0.9776 | Initial |
| 50 | 0.8437 | ↓ 13.7% |
| 100 | 0.6731 | ↓ 31.1% |
| 200 | 0.2591 | ↓ 73.5% |
| 300 | 0.1167 | ↓ 88.1% |
| 400 | 0.1042 | ↓ 89.3% |
| 500 | 0.1452 | ↓ 85.1% |

**Key Observations**:
- Initial overlap: **0.9776** (97.76%)
- Final overlap (step 500): **0.1452** (14.52%)
- Lowest overlap (step 402): **0.0969** (9.69%)
- **Total decrease**: 83.24 percentage points
- Overlap reaches its lowest between steps 200-400, followed by slight fluctuations

### 2. Japanese Wikipedia Training Results

| Training Step | Overlap Ratio | Trend |
|--------------|---------------|-------|
| 1 | 0.9776 | Initial |
| 50 | 0.8435 | ↓ 13.7% |
| 100 | 0.5927 | ↓ 39.4% |
| 200 | 0.2938 | ↓ 70.0% |
| 300 | 0.0734 | ↓ 92.5% |
| 400 | 0.1352 | ↓ 86.2% |
| 500 | 0.1490 | ↓ 84.8% |

**Key Observations**:
- Initial overlap: **0.9776** (97.76%)
- Final overlap (step 500): **0.1490** (14.90%)
- Lowest overlap (step 300): **0.0734** (7.34%)
- **Total decrease**: 82.86 percentage points
- Japanese training reaches its lowest point at step 300, earlier than English



---

<!-- ## Core Conclusions

###  Main Findings

**Language Independence**: The training language (English vs Japanese) has negligible impact on overlap changes. This finding suggests:

1. **Bottom-k wordlist changes are an inherent characteristic of the training process itself**, not influenced by specific language data
2. **The mechanism of model fingerprint changes is independent of training data language features**
3. **Any fine-tuning training, regardless of the language of the data used, leads to significant overlap decrease**

###  Quantitative Results

- **Overlap decrease rate**: Approximately 30-40 percentage points per 100 steps (early phase)
- **Stable state**: After 400-500 steps, overlap stabilizes around 10-15%
- **Language difference**: Final overlap difference between the two languages is less than 0.4%

###  Experimental Significance

This experiment validates:
1. **Robustness of model fingerprints**: Bottom-k wordlist as a model fingerprint exhibits language-independent change patterns
2. **Impact of fine-tuning**: Even relatively few training steps (500 steps) lead to significant changes in model fingerprints
3. **Potential applications**: Overlap ratio can be used to quantify the degree of model fine-tuning without considering the language of training data

--- -->

