# Wikipedia Overlap Experiments

Quick guide for running English vs Japanese Wikipedia fine-tuning experiments.

## Overview

This experiment compares how fine-tuning on different languages (English vs Japanese) affects the overlap ratio with the base model.

**Key Points:**
- Uses Wikipedia datasets (same schema, different languages)
- No special text formatting - raw Wikipedia articles are used directly
- Matches the evaluation approach (raw prompts only)

## Understanding Training Steps vs Samples

**Important:** Training steps ≠ number of samples!

- **1 step** = 1 batch update
- **Effective batch size** = `per_device_batch_size × gradient_accumulation_steps × num_gpus`
- **Samples per step** = effective batch size

**Example:**
- `per_device_batch_size = 4`
- `gradient_accumulation_steps = 4`
- `num_gpus = 1`
- **→ 16 samples per step**
- **→ 1000 steps = 16,000 samples**

**Recommendation:** Set `--num_train_samples` to avoid downloading entire Wikipedia:
```bash
--num_train_samples 20000  # For 1000 steps with default settings
```

## Quick Start

### Option 1: Use the Provided Script

```bash
bash run_wikipedia_experiments.sh
```

This will:
1. Fine-tune on English Wikipedia (10k samples, ~625 steps)
2. Fine-tune on Japanese Wikipedia (10k samples, ~625 steps)
3. Compare the results

**Note:** The script uses streaming mode to avoid downloading all 41 Wikipedia files!

### Option 2: Run Manually

**Step 1: English Wikipedia**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --save_fingerprints "./fingerprints.json"
```

**Why 20000 samples for 1000 steps?**
- Default: `batch_size=4`, `gradient_accumulation=4` → 16 samples/step
- 1000 steps × 16 = 16,000 samples
- 20,000 gives some buffer

**Step 2: Japanese Wikipedia**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.ja" \
    --output_dir "./wiki_ja" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --load_fingerprints "./fingerprints.json"
```

**Step 3: Compare**
```bash
python compare_overlap_experiments.py \
    --exp_dirs ./wiki_en ./wiki_ja \
    --labels "English" "Japanese"
```

## Wikipedia Dataset Details

### English Wikipedia
- **Dataset:** `wikimedia/wikipedia`
- **Config:** `20231101.en`
- **Fields:** `id`, `url`, `title`, `text`
- **Text:** Clean Wikipedia articles (non-content structure removed)

### Japanese Wikipedia
- **Dataset:** `wikimedia/wikipedia`
- **Config:** `20231101.ja`
- **Fields:** Same as English
- **Text:** Clean Wikipedia articles in Japanese

### Why Wikipedia?

1. **Same schema:** Both datasets have identical structure
2. **Clean text:** Pre-processed and cleaned
3. **Language comparison:** Perfect for English vs Japanese experiments
4. **Large scale:** Millions of articles available

## Text Formatting

**Important:** No special formatting is applied!

❌ **Not used:**
```
### Instruction:
{instruction}

### Response:
{output}
```

✅ **Used:**
```
{raw_wikipedia_text}
```

This matches the evaluation approach where only raw prompts are used to compute bottom-k vocabularies.

## Expected Results

You can compare:

1. **Overlap decrease rate:** Does Japanese cause faster/slower overlap decrease than English?
2. **Final overlap:** Which language results in lower final overlap?
3. **Training dynamics:** Are the overlap curves similar or different?

## Output Files

Each experiment generates:

```
wiki_en/
├── overlap_results.json              # Detailed results
├── overlap_summary.csv               # CSV summary
├── overlap_vs_steps.png              # Visualization
├── overlap_analysis_combined.png     # Combined analysis
└── final_model/                      # Fine-tuned model

wiki_ja/
├── (same structure)
```

Comparison:
```
comparison.png                        # Side-by-side comparison
```

## Customization

### Adjust Training Steps

```bash
--max_steps 2000 \
--eval_steps 200
```

### Limit Training Samples

```bash
--num_train_samples 5000  # Use only 5k articles
```

### Use Different Model

```bash
--base_model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Adjust Overlap Parameters

```bash
--bottom_k_vocab 3000 \      # Larger wordlist
--num_fingerprints 50        # More fingerprints
```

## Troubleshooting

### Out of Memory

Reduce batch size and samples:
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--num_train_samples 5000
```

### Dataset Loading Slow / Too Much Data

**Solution 1:** Always set `--num_train_samples`:
```bash
--num_train_samples 20000  # Only load what you need
```

**Solution 2:** The script now uses streaming mode automatically, so it won't download all 41 files.

**How many samples do I need?**
```
samples_needed = max_steps × batch_size × gradient_accumulation_steps × 1.25
```

Examples:
- 1000 steps: ~20,000 samples
- 500 steps: ~10,000 samples
- 100 steps: ~2,000 samples

### Different Tokenizers

If using a model with a different tokenizer, the overlap patterns may vary significantly.

## Research Questions

This experiment can help answer:

1. **Language influence:** Does the training language affect overlap differently?
2. **Cross-lingual effects:** How does Japanese training affect a model's English capabilities?
3. **Generalization:** Are overlap patterns consistent across languages?

## Notes

- Both experiments use the **same fingerprints** (via `--save_fingerprints` and `--load_fingerprints`)
- This ensures fair comparison between English and Japanese
- The base model is Qwen2.5-0.5B which supports both English and Japanese
