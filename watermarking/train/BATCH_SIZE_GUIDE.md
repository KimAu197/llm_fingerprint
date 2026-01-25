# Batch Size and Gradient Accumulation Guide

## Quick Answer

### Change via Command Line (Recommended)

```bash
python train_and_eval_overlap.py \
    --per_device_train_batch_size 8 \      # Change this
    --gradient_accumulation_steps 2        # Change this
    # ... other parameters
```

### Change in Shell Script

Edit `run_wikipedia_experiments.sh`:

```bash
BATCH_SIZE=8              # Line 4: Change from 4 to 8
GRAD_ACCUMULATION=2       # Line 5: Change from 4 to 2
```

## Understanding the Parameters

### Effective Batch Size

```
Effective Batch Size = per_device_batch_size × gradient_accumulation × num_gpus
```

**Example:**
- `per_device_batch_size = 4`
- `gradient_accumulation = 4`
- `num_gpus = 1`
- **→ Effective batch size = 16**

### Samples per Step

```
Samples per Step = Effective Batch Size
```

So with default settings:
- **1 step = 16 samples**
- **1000 steps = 16,000 samples**

## Common Configurations

### Default (Balanced)
```bash
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4
# Effective batch size: 16
# Memory: Moderate
# Speed: Moderate
```

### High Memory (Faster Training)
```bash
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2
# Effective batch size: 16 (same)
# Memory: High
# Speed: Faster (fewer gradient accumulation steps)
```

### Low Memory (Slower Training)
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8
# Effective batch size: 16 (same)
# Memory: Low
# Speed: Slower (more gradient accumulation steps)
```

### Very Low Memory
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
# Effective batch size: 16 (same)
# Memory: Very Low
# Speed: Slowest
```

### Larger Effective Batch Size
```bash
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4
# Effective batch size: 32
# Memory: High
# Speed: Fast
# Note: Need to adjust num_train_samples!
```

## Adjusting Training Samples

**Important:** If you change the effective batch size, adjust `num_train_samples`:

```
num_samples = max_steps × effective_batch_size × 1.25
```

### Examples

**Default (batch_size=4, grad_accum=4):**
```bash
--max_steps 1000 \
--num_train_samples 20000  # 1000 × 16 × 1.25
```

**High Memory (batch_size=8, grad_accum=2):**
```bash
--max_steps 1000 \
--num_train_samples 20000  # 1000 × 16 × 1.25 (same)
```

**Larger Batch (batch_size=8, grad_accum=4):**
```bash
--max_steps 1000 \
--num_train_samples 40000  # 1000 × 32 × 1.25
```

**Low Memory (batch_size=2, grad_accum=8):**
```bash
--max_steps 1000 \
--num_train_samples 20000  # 1000 × 16 × 1.25 (same)
```

## Memory vs Speed Trade-off

| Config | Memory | Speed | Use When |
|--------|--------|-------|----------|
| batch=8, accum=2 | High | Fast | You have GPU memory |
| batch=4, accum=4 | Medium | Medium | Default/balanced |
| batch=2, accum=8 | Low | Slow | Limited GPU memory |
| batch=1, accum=16 | Very Low | Very Slow | Very limited memory |

## How to Choose

### Step 1: Check Your GPU Memory

```bash
nvidia-smi
```

Look at "Memory-Usage" - how much free memory do you have?

### Step 2: Choose Configuration

| Free GPU Memory | Recommended Config |
|-----------------|-------------------|
| > 16 GB | `batch=8, accum=2` or `batch=8, accum=4` |
| 8-16 GB | `batch=4, accum=4` (default) |
| 4-8 GB | `batch=2, accum=8` |
| < 4 GB | `batch=1, accum=16` |

### Step 3: Test and Adjust

Start with recommended config. If you get OOM (Out of Memory):
1. Reduce `per_device_train_batch_size` by half
2. Double `gradient_accumulation_steps`
3. Keep effective batch size the same

## Complete Examples

### Example 1: Default Settings

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### Example 2: High Memory GPU

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2
```

### Example 3: Low Memory GPU

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 20000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

### Example 4: Larger Effective Batch Size

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./wiki_en" \
    --max_steps 1000 \
    --eval_steps 100 \
    --num_train_samples 40000 \              # Doubled!
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4
```

## Modifying Shell Script

Edit `run_wikipedia_experiments.sh`:

```bash
#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-0.5B"
MAX_STEPS=1000
EVAL_STEPS=100
NUM_FINGERPRINTS=20
BATCH_SIZE=8              # ← Change this
GRAD_ACCUMULATION=2       # ← Change this
NUM_SAMPLES=20000         # ← Adjust if needed
```

Then run:
```bash
bash run_wikipedia_experiments.sh
```

## FAQ

### Q: What's the difference between batch_size and gradient_accumulation?

**A:** 
- `batch_size`: How many samples are processed at once (affects memory)
- `gradient_accumulation`: How many batches to accumulate before updating weights (doesn't affect memory much)

### Q: Should I keep effective batch size constant?

**A:** Usually yes! Changing effective batch size affects:
- Training dynamics
- Learning rate effectiveness
- Convergence behavior

If you must change it, also adjust learning rate proportionally.

### Q: I got OOM error, what should I do?

**A:** Reduce batch_size and increase gradient_accumulation:
```bash
# If this fails:
--per_device_train_batch_size 4 --gradient_accumulation_steps 4

# Try this:
--per_device_train_batch_size 2 --gradient_accumulation_steps 8

# Still failing? Try this:
--per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

### Q: Can I use different settings for English and Japanese?

**A:** Yes, but for fair comparison, keep them the same!

## Summary

**To change batch size and gradient accumulation:**

1. **Command line:** Add `--per_device_train_batch_size X --gradient_accumulation_steps Y`
2. **Shell script:** Edit `BATCH_SIZE` and `GRAD_ACCUMULATION` variables
3. **Keep effective batch size constant:** If you halve batch_size, double gradient_accumulation
4. **Adjust num_train_samples:** If you change effective batch size, recalculate samples needed
