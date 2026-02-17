# Running Experiments Family by Family

## 📁 Individual Family CSV Files

Created in: `llm_fingerprint/result/result_2.10/data/`

```
experiment_family_3.csv  - Family 3: meta-llama/Llama-3.1-8B-Instruct (5 models)
experiment_family_4.csv  - Family 4: Qwen/Qwen2.5-7B (5 models)
experiment_family_5.csv  - Family 5: Qwen/Qwen3-0.6B-Base (5 models)
experiment_family_6.csv  - Family 6: meta-llama/Llama-3.2-3B-Instruct (5 models)
```

## 🚀 How to Run Each Family Separately

### Family 3: meta-llama/Llama-3.1-8B-Instruct

```bash
cd /content/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_3.csv \
    --output_dir test_results_family_3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda
```

**Expected:**
- 5 positive samples
- 25 negative samples (5 derivatives × 5 negatives each)
- Total: 30 comparisons

### Family 4: Qwen/Qwen2.5-7B

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_4.csv \
    --output_dir test_results_family_4 \
    --num_pairs 5 \
    --num_negative_samples 5
```

### Family 5: Qwen/Qwen3-0.6B-Base

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_5.csv \
    --output_dir test_results_family_5 \
    --num_pairs 5 \
    --num_negative_samples 5
```

### Family 6: meta-llama/Llama-3.2-3B-Instruct

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_6.csv \
    --output_dir test_results_family_6 \
    --num_pairs 5 \
    --num_negative_samples 5
```

## 📊 Progress Tracking

### Completed So Far
- ✅ Family 1: meta-llama/Llama-3.1-8B (done, ~36 tests)
- ✅ Family 2: Qwen/Qwen2.5-1.5B (done, ~36 tests) 
- ⏸️ Family 3: Qwen/Qwen2.5-7B (partially done, need to finish)

### Remaining
- ❌ Family 3: meta-llama/Llama-3.1-8B-Instruct
- ❌ Family 4: Qwen/Qwen2.5-7B (or finish remaining tests)
- ❌ Family 5: Qwen/Qwen3-0.6B-Base
- ❌ Family 6: meta-llama/Llama-3.2-3B-Instruct

## 💾 Memory Management Tips for Colab

### 1. Run One Family at a Time
This prevents memory accumulation.

### 2. Clear Memory Between Runs
```python
import gc
import torch

# After each family
gc.collect()
torch.cuda.empty_cache()

# Check memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### 3. Use Smaller Models First
Run in this order to test memory requirements:
1. Family 5: Qwen3-0.6B-Base (smallest - 0.6B)
2. Family 4: Qwen2.5-7B (medium - 7B)
3. Family 3: Llama-3.1-8B-Instruct (larger - 8B)
4. Family 6: Llama-3.2-3B-Instruct (small - 3B)

### 4. Reduce Negative Samples If Needed
```bash
# Test with fewer negative samples first
--num_negative_samples 2  # Instead of 5
```

This reduces tests per family from 30 to 15.

## 🔄 Merge Results Later

After running each family separately, merge all CSVs:

```python
import pandas as pd

# Read all family results
dfs = []
for i in range(3, 7):
    df = pd.read_csv(f'test_results_family_{i}/base_family_overlap_results.csv')
    dfs.append(df)

# Merge
combined = pd.concat(dfs, ignore_index=True)

# Save
combined.to_csv('all_families_combined.csv', index=False)
print(f"Combined: {len(combined)} total tests")
```

## 📝 Recommended Running Order

For Colab (from smallest to largest memory requirements):

```bash
# 1. Smallest model - test memory
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_5.csv \
    --output_dir test_results_family_5 \
    --num_pairs 5 --num_negative_samples 5

# 2. Small model
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_6.csv \
    --output_dir test_results_family_6 \
    --num_pairs 5 --num_negative_samples 5

# 3. Medium model
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_4.csv \
    --output_dir test_results_family_4 \
    --num_pairs 5 --num_negative_samples 5

# 4. Larger model
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_3.csv \
    --output_dir test_results_family_3 \
    --num_pairs 5 --num_negative_samples 5
```

## 🎯 Start with Family 3 Now

```bash
cd /content/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_family_3.csv \
    --output_dir test_results_family_3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda
```

This will:
- Test 5 models from Llama-3.1-8B-Instruct family
- Generate 5 positive + 25 negative = 30 comparisons
- Save to separate directory: `test_results_family_3/`
