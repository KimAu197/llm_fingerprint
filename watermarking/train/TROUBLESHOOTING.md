# Troubleshooting Guide

## Common Errors and Solutions

### 1. FP16 Gradient Scaling Error ⚠️

**Error Message:**
```
ValueError: Attempting to unscale FP16 gradients.
```

**Cause:** FP16 mixed precision training can cause gradient scaling issues on some systems.

**Solution:** FP16 is now **disabled by default**. The script uses FP32 (full precision) which is more stable.

**If you want to try FP16 (not recommended):**
```bash
python train_and_eval_overlap.py \
    --use_fp16 \  # Add this flag
    # ... other parameters
```

**Why this happens:**
- Some PyTorch/CUDA versions have issues with FP16 gradient unscaling
- FP16 can be unstable during training
- FP32 is slower but more reliable

---

### 2. Downloading Too Much Data

**Problem:** Wikipedia dataset is downloading all 41 files (~10GB total)

**Solution:** Always specify `--num_train_samples`:

```bash
python train_and_eval_overlap.py \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --num_train_samples 20000  # ← Add this!
```

**How to calculate samples needed:**
```
samples = max_steps × batch_size × gradient_accumulation × 1.25
```

**Examples:**
- 100 steps → 2,000 samples
- 500 steps → 10,000 samples
- 1000 steps → 20,000 samples
- 2000 steps → 40,000 samples

---

### 3. Out of Memory (OOM)

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Reduce batch size**
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--num_fingerprints 5
```

**Solution 2: Even smaller batch size**
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--num_fingerprints 5
```

**Solution 3: Use smaller model**
```bash
--base_model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

**Check GPU memory:**
```bash
nvidia-smi
```

---

### 4. Training Stuck at 0%

**Problem:** Training starts but doesn't progress

**Possible Causes:**

1. **Dataset loading is slow** - Wait a bit, first epoch can be slow
2. **Too many workers** - Reduce dataloader workers:
   ```bash
   # Edit train_and_eval_overlap.py line ~710
   dataloader_num_workers=0,  # Change from 4 to 0
   ```
3. **Evaluation taking too long** - Reduce fingerprints:
   ```bash
   --num_fingerprints 5
   ```

---

### 5. Dataset Not Found

**Error Message:**
```
FileNotFoundError: Dataset not found
```

**Solution:** Check dataset name and config:

```bash
# Correct
--dataset_name "wikimedia/wikipedia" \
--dataset_config "20231101.en"

# Wrong
--dataset_name "wikipedia"  # Missing "wikimedia/"
```

**Available configs:**
- English: `20231101.en`
- Japanese: `20231101.ja`
- Chinese: `20231101.zh`
- French: `20231101.fr`
- German: `20231101.de`

---

### 6. Tokenizer Errors

**Error Message:**
```
KeyError: 'text'
```

**Cause:** Dataset doesn't have a 'text' column

**Solution:** Specify correct column name:
```bash
--text_column "content"  # If your column is named "content"
```

**For CSV files:**
```bash
--csv_path "./my_data.csv" \
--text_column "text"  # Make sure this column exists
```

---

### 7. Fingerprint Generation Slow

**Problem:** Generating fingerprints takes too long

**Solution:** Reduce number of fingerprints:
```bash
--num_fingerprints 5  # Default is 20
```

**Or load pre-generated fingerprints:**
```bash
--load_fingerprints "./fingerprints.json"
```

---

### 8. Checkpoint Saving Failed

**Error Message:**
```
OSError: [Errno 28] No space left on device
```

**Solution 1: Reduce checkpoint frequency**
```bash
--save_steps 1000  # Save less often
--save_total_limit 2  # Keep fewer checkpoints
```

**Solution 2: Disable checkpoint saving**
```bash
--save_steps 999999  # Effectively disable
```

---

### 9. Training Too Slow

**Problem:** Training is taking too long

**Solutions:**

1. **Reduce training samples:**
   ```bash
   --num_train_samples 10000
   ```

2. **Reduce max steps:**
   ```bash
   --max_steps 500
   ```

3. **Reduce evaluation frequency:**
   ```bash
   --eval_steps 200
   ```

4. **Increase batch size (if memory allows):**
   ```bash
   --per_device_train_batch_size 8 \
   --gradient_accumulation_steps 2
   ```

5. **Reduce fingerprints:**
   ```bash
   --num_fingerprints 5
   ```

---

### 10. Import Errors

**Error Message:**
```
ImportError: No module named 'transformers'
```

**Solution:** Install dependencies:
```bash
pip install -r requirements_overlap_experiment.txt
```

**Or manually:**
```bash
pip install torch transformers datasets accelerate matplotlib numpy pandas
```

---

### 11. CUDA Not Available

**Error Message:**
```
RuntimeError: CUDA not available
```

**Solution:** Use CPU (slower):
```bash
--device cpu
```

**Or check CUDA installation:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

### 12. Streaming Dataset Error

**Error Message:**
```
TypeError: 'IterableDataset' object is not subscriptable
```

**Cause:** Trying to access streaming dataset incorrectly

**Solution:** This should be fixed in the latest version. Make sure you're using the updated `train_and_eval_overlap.py`.

---

## Performance Tips

### Speed Up Training

1. **Use larger batch size** (if memory allows):
   ```bash
   --per_device_train_batch_size 8
   ```

2. **Reduce evaluation frequency**:
   ```bash
   --eval_steps 200
   ```

3. **Use fewer fingerprints**:
   ```bash
   --num_fingerprints 10
   ```

4. **Limit training samples**:
   ```bash
   --num_train_samples 10000
   ```

### Reduce Memory Usage

1. **Smaller batch size**:
   ```bash
   --per_device_train_batch_size 2
   ```

2. **Fewer fingerprints**:
   ```bash
   --num_fingerprints 5
   ```

3. **Smaller model**:
   ```bash
   --base_model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

4. **Disable FP16** (default now):
   ```bash
   # Don't add --use_fp16
   ```

---

## Getting Help

### Check Logs

Look at the error message carefully. Most errors tell you exactly what's wrong.

### Test with Minimal Config

```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --output_dir "./test" \
    --max_steps 10 \
    --eval_steps 5 \
    --num_fingerprints 2 \
    --num_train_samples 100
```

If this works, gradually increase parameters.

### Check System Resources

```bash
# GPU memory
nvidia-smi

# Disk space
df -h

# RAM
free -h
```

---

## Quick Fixes Summary

| Problem | Quick Fix |
|---------|-----------|
| FP16 error | Don't use `--use_fp16` (disabled by default) |
| OOM | `--per_device_train_batch_size 2` |
| Too slow | `--num_train_samples 10000` |
| Downloading too much | `--num_train_samples 20000` |
| Fingerprints slow | `--num_fingerprints 5` |
| No CUDA | `--device cpu` |
