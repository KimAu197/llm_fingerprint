# Weights & Biases (wandb) Integration Guide

Track your overlap experiments with Weights & Biases for better visualization and comparison.

## Quick Start

### 1. Install wandb

```bash
pip install wandb
```

### 2. Get Your API Key

1. Sign up at [wandb.ai](https://wandb.ai)
2. Go to [Settings → API Keys](https://wandb.ai/settings)
3. Copy your API key (format: `wandb_v1_...`)

### 3. Set API Key

**Option A: Environment variable (recommended)**
```bash
export WANDB_API_KEY="wandb_v1_YOUR_API_KEY_HERE"
```

**Option B: Pass directly in command**
```bash
python train_and_eval_overlap.py \
    --use_wandb \
    --wandb_api_key "wandb_v1_YOUR_API_KEY_HERE" \
    ...
```

### 4. Run Experiment

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./exp_wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --use_wandb \
    --wandb_project "model-overlap" \
    --wandb_run_name "wiki_en_1000steps"
```

## What Gets Logged

### Training Metrics (every `--logging_steps`)
- `train/loss`: Training loss
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm
- `train/epoch`: Current epoch

### Overlap Metrics (every `--eval_steps`)
- `overlap/avg_overlap_ratio`: Average overlap across all fingerprints
- `overlap/min_overlap`: Minimum overlap (best case)
- `overlap/max_overlap`: Maximum overlap (worst case)

### Summary Statistics (at end)
- `initial_overlap`: Overlap at step 1
- `final_overlap`: Overlap at final step
- `total_decrease`: How much overlap decreased
- `decrease_rate_per_step`: Rate of decrease per training step

### Configuration
- All command-line arguments
- Model name, dataset, hyperparameters, etc.

## Example Commands

### Basic Usage

```bash
export WANDB_API_KEY="wandb_v1_YOUR_API_KEY"

python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./exp1" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --use_wandb
```

### Compare Multiple Experiments

```bash
# Experiment 1: English Wikipedia
python train_and_eval_overlap.py \
    --dataset_config "20231101.en" \
    --output_dir "./exp_en" \
    --use_wandb \
    --wandb_run_name "wiki_en" \
    --num_train_samples 20000

# Experiment 2: Japanese Wikipedia
python train_and_eval_overlap.py \
    --dataset_config "20231101.ja" \
    --output_dir "./exp_ja" \
    --use_wandb \
    --wandb_run_name "wiki_ja" \
    --num_train_samples 20000

# Experiment 3: Different learning rate
python train_and_eval_overlap.py \
    --dataset_config "20231101.en" \
    --output_dir "./exp_lr_low" \
    --use_wandb \
    --wandb_run_name "wiki_en_lr1e6" \
    --learning_rate 1e-6 \
    --num_train_samples 20000
```

Then compare all three runs in the wandb dashboard!

### Using the Convenience Script

```bash
# Edit run_with_wandb.sh to set your parameters
bash run_with_wandb.sh
```

## Wandb Dashboard Features

### Charts You'll See

1. **Training Loss Over Time**
   - See if training is stable
   - Detect NaN or explosion

2. **Overlap Ratio Over Steps**
   - Main metric: how overlap decreases
   - Min/max bounds show variance

3. **Learning Rate Schedule**
   - Verify warmup is working
   - Check learning rate decay

4. **Gradient Norm**
   - Detect gradient explosion
   - Verify gradient clipping is working

### Comparing Runs

1. Go to your project page
2. Select multiple runs (checkbox on left)
3. Click "Compare" button
4. View side-by-side charts

### Filtering and Grouping

- **Group by dataset**: Compare en vs ja
- **Group by learning rate**: Compare different LRs
- **Filter by tags**: Auto-tagged with model name

## Tips

### 1. Meaningful Run Names

**Bad:**
```bash
--wandb_run_name "experiment1"
```

**Good:**
```bash
--wandb_run_name "wiki_en_lr5e6_1000steps"
```

### 2. Use Same Project for Related Experiments

```bash
--wandb_project "model-overlap"  # All overlap experiments
```

### 3. Add Custom Tags

Edit `train_and_eval_overlap.py` line ~650:
```python
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config=vars(args),
    tags=["overlap-experiment", args.base_model_name.split("/")[-1], "my-custom-tag"],
)
```

### 4. Offline Mode (No Internet)

```bash
export WANDB_MODE=offline

# Run experiments...

# Later, sync when online:
wandb sync ./wandb/offline-run-*
```

### 5. Disable Wandb Temporarily

Just remove the `--use_wandb` flag:
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    # ... other args ...
    # --use_wandb  # commented out
```

## Troubleshooting

### "wandb not installed"

```bash
pip install wandb
```

### "wandb login required"

```bash
wandb login
# Paste your API key when prompted
```

Or set environment variable:
```bash
export WANDB_API_KEY="wandb_v1_YOUR_KEY"
```

### "Permission denied" or "Invalid API key"

1. Check your API key is correct
2. Make sure there are no extra spaces
3. Try logging in again: `wandb login`

### Runs Not Showing Up

1. Check you're logged into the correct account
2. Verify project name matches
3. Wait a few seconds for sync

### Too Much Data / Slow Logging

Increase logging intervals:
```bash
--logging_steps 200  # Log less frequently
--eval_steps 200     # Evaluate less frequently
```

## Advanced: Custom Metrics

To log additional metrics, edit the `OverlapEvaluationCallback` in `train_and_eval_overlap.py`:

```python
# Around line 470
if self.use_wandb and WANDB_AVAILABLE:
    wandb.log({
        "overlap/avg_overlap_ratio": avg_overlap,
        "overlap/min_overlap": min(overlap_scores),
        "overlap/max_overlap": max(overlap_scores),
        # Add your custom metrics here:
        "overlap/std": np.std(overlap_scores),
        "overlap/median": np.median(overlap_scores),
        "step": current_step,
    })
```

## Example Workflow

```bash
# 1. Set API key once
export WANDB_API_KEY="wandb_v1_YOUR_KEY"

# 2. Run multiple experiments
for lr in 1e-6 5e-6 1e-5; do
    python train_and_eval_overlap.py \
        --base_model_name "Qwen/Qwen2.5-0.5B" \
        --dataset_name "wikimedia/wikipedia" \
        --dataset_config "20231101.en" \
        --output_dir "./exp_lr_${lr}" \
        --max_steps 1000 \
        --eval_steps 100 \
        --num_train_samples 20000 \
        --learning_rate $lr \
        --use_wandb \
        --wandb_run_name "wiki_en_lr_${lr}"
done

# 3. Compare all runs in wandb dashboard
# Go to: https://wandb.ai/YOUR_USERNAME/model-overlap
```

## Resources

- [Wandb Documentation](https://docs.wandb.ai/)
- [Wandb with HuggingFace](https://docs.wandb.ai/guides/integrations/huggingface)
- [Wandb Python API](https://docs.wandb.ai/ref/python)

## Summary

**Minimum setup:**
```bash
pip install wandb
export WANDB_API_KEY="wandb_v1_YOUR_KEY"
python train_and_eval_overlap.py --use_wandb ...
```

**What you get:**
- Real-time training monitoring
- Automatic metric logging
- Easy experiment comparison
- Shareable results
- No extra code needed!
