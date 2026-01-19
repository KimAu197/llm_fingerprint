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

**Using HuggingFace dataset:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "yahma/alpaca-cleaned" \
    --output_dir "./experiment_output" \
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

## CSV Data Format

### Format 1: Simple Text

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

### Format 2: Instruction Format

```csv
instruction,input,output
"Task description","Input context","Expected output"
```

**Usage:**
```bash
python train_and_eval_overlap.py \
    --csv_path "./my_data.csv" \
    --instruction_column "instruction" \
    --input_column "input" \
    --output_column "output"
```

See `CSV_FORMAT_README.md` for detailed format specifications.

## Key Parameters

### Model & Device
- `--base_model_name`: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
- `--device`: cuda/mps/cpu (auto-detected by default)

### Data Source (choose one)
- `--dataset_name`: HuggingFace dataset name
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
    --dataset_name "yahma/alpaca-cleaned" \
    --output_dir "./test" \
    --max_steps 50 \
    --eval_steps 10 \
    --num_fingerprints 5 \
    --num_train_samples 500
```

### Example 2: Compare Different Training Steps

```bash
# Train for different numbers of steps
for steps in 100 500 1000; do
    python train_and_eval_overlap.py \
        --base_model_name "Qwen/Qwen2.5-0.5B" \
        --dataset_name "yahma/alpaca-cleaned" \
        --output_dir "./exp_${steps}steps" \
        --max_steps $steps \
        --eval_steps $(($steps / 10))
done

# Compare results
python compare_overlap_experiments.py \
    --exp_dirs ./exp_100steps ./exp_500steps ./exp_1000steps \
    --labels "100 steps" "500 steps" "1000 steps"
```

### Example 3: Using Custom CSV Data

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./example_data_instruction.csv" \
    --instruction_column "instruction" \
    --input_column "input" \
    --output_column "output" \
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

## Research Applications

This tool helps answer:

1. **Training dynamics**: How does overlap change as training progresses?
2. **Linear relationship**: Is overlap decrease proportional to training steps?
3. **Dataset influence**: Do different datasets cause different overlap patterns?
4. **Early prediction**: Can we predict final overlap from early training?

## Citation

If you use this tool in your research, please cite the RoFL paper:

```bibtex
@article{rofl2023,
  title={RoFL: Robust Fingerprinting of Large Language Models},
  author={...},
  year={2023}
}
```
