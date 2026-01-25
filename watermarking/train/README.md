# Fine-tuning Overlap Experiment

Track how wordlist overlap changes during model fine-tuning.

## Overview

This tool fine-tunes a base model and evaluates the overlap ratio between the fine-tuned model's bottom-k wordlist and the original base model's wordlist at different training steps.

**Key Features:**
- Uses the same overlap testing functions from `../new/` directory
- Supports HuggingFace datasets and custom CSV files
- Tracks overlap changes during training
- Generates visualization plots

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_overlap_experiment.txt
```

### 2. Run Experiment

**Using Wikipedia English:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./experiment_wiki_en" \
    --max_steps 1000 \
    --eval_steps 100
```

**Using Wikipedia Japanese:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.ja" \
    --output_dir "./experiment_wiki_ja" \
    --max_steps 1000 \
    --eval_steps 100
```

**Using custom CSV:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./example_data_simple.csv" \
    --output_dir "./experiment_output" \
    --max_steps 1000 \
    --eval_steps 100
```

### 3. Visualize Results

```bash
python plot_overlap_vs_steps.py --result_dir "./experiment_output"
```

## Files

### Core Scripts
- **`train_and_eval_overlap.py`** - Main training script with overlap evaluation
- **`plot_overlap_vs_steps.py`** - Visualization tool
- **`compare_overlap_experiments.py`** - Compare multiple experiments

### Data Examples
- **`example_data_simple.csv`** - Simple text format example
- **`example_data_instruction.csv`** - Instruction format example

### Documentation
- **`README.md`** - This file
- **`CSV_FORMAT_README.md`** - Detailed CSV format guide
- **`requirements_overlap_experiment.txt`** - Python dependencies

## Data Format

### HuggingFace Datasets

The script works with any HuggingFace dataset that has a `text` column. No special formatting is applied - text is used as-is.

**Recommended: Wikipedia datasets**
- English: `wikimedia/wikipedia` with config `20231101.en`
- Japanese: `wikimedia/wikipedia` with config `20231101.ja`

### CSV Format

Simple CSV with a `text` column:

```csv
text
"Your training text here..."
"Another training example..."
```

**Usage:**
```bash
python train_and_eval_overlap.py \
    --csv_path "./my_data.csv" \
    --text_column "text"
```

**Note:** No special formatting (like "### Instruction:") is applied. Text is used directly for training, matching the evaluation approach where only raw prompts are used.

## Key Parameters

### Model & Device
- `--base_model_name`: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
- `--device`: cuda/mps/cpu (auto-detected by default)

### Data Source (choose one)
- `--dataset_name`: HuggingFace dataset name (e.g., "wikimedia/wikipedia")
- `--dataset_config`: Dataset config/subset (e.g., "20231101.en" for English, "20231101.ja" for Japanese)
- `--csv_path`: Path to custom CSV file

### Training
- `--max_steps`: Maximum training steps (default: 1000)
- `--eval_steps`: Evaluate overlap every N steps (default: 100)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--learning_rate`: Learning rate (default: 2e-5)

### Overlap Evaluation
- `--bottom_k_vocab`: Size of bottom-k vocabulary (default: 2000)
- `--num_fingerprints`: Number of fingerprint prompts (default: 20)

### Other
- `--num_train_samples`: Limit training samples (None = use all)
- `--seed`: Random seed (default: 42)
- `--save_fingerprints`: Save fingerprints for reuse
- `--load_fingerprints`: Load existing fingerprints

## Output Files

Each experiment generates:

```
output_dir/
├── args.json                          # Experiment parameters
├── base_bottomk_cache.json            # Base model wordlists
├── overlap_results.json               # Detailed results (JSON)
├── overlap_summary.csv                # Summary (CSV)
├── overlap_vs_steps.png               # Visualization
├── overlap_decrease_rate.png          # Rate analysis
├── overlap_analysis_combined.png      # Combined analysis
├── checkpoint-*/                      # Training checkpoints
└── final_model/                       # Final fine-tuned model
```

## Examples

### Example 1: Quick Test (5 minutes)

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./test" \
    --max_steps 50 \
    --eval_steps 10 \
    --num_fingerprints 5 \
    --num_train_samples 500
```

### Example 2: Compare English vs Japanese Wikipedia

```bash
# English Wikipedia
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./exp_wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --save_fingerprints "./fingerprints_shared.json"

# Japanese Wikipedia (reuse same fingerprints)
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.ja" \
    --output_dir "./exp_wiki_ja" \
    --max_steps 1000 \
    --eval_steps 100 \
    --load_fingerprints "./fingerprints_shared.json"

# Compare results
python compare_overlap_experiments.py \
    --exp_dirs ./exp_wiki_en ./exp_wiki_ja \
    --labels "English Wikipedia" "Japanese Wikipedia"
```

### Example 3: Using Custom CSV Data

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./example_data_simple.csv" \
    --output_dir "./custom_data_exp" \
    --max_steps 1000 \
    --eval_steps 100
```

## Relationship to `../new/` Directory

This tool uses the same overlap testing methodology from the `../new/` directory:

| Function | Source | Purpose |
|----------|--------|---------|
| `compute_bottomk_vocab_for_model()` | `utils/bottomk_processor.py` | Compute bottom-k vocabulary |
| `sample_fingerprint_prompt()` | `utils/fingerprint_gen.py` | Generate fingerprint prompts |
| `overlap_ratio()` | Same logic as `new/run_bottomk_subspace_overlap_from_base.py` | Calculate overlap |

**Key Difference:**
- `new/`: Compares a fixed base model with multiple already-fine-tuned models
- `train/`: Tracks overlap changes **during** the fine-tuning process

## Troubleshooting

### Out of Memory

```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--num_fingerprints 10
```

### Training Too Slow

```bash
--num_train_samples 5000 \
--eval_steps 200
```

### CSV Format Error

Check your CSV:
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.columns)
print(df.head())
```
