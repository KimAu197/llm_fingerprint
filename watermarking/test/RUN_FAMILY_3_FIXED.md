# 🚀 Quick Commands for Family 3 (Fixed!)

## ⚠️ Issues Found

1. **Tee error**: Directory doesn't exist yet
2. **Gated model**: meta-llama/Llama-3.1-8B-Instruct needs HF auth

## ✅ Solution

### Option 1: Run Family 4 First (No Auth Needed)

```bash
cd /content/llm_fingerprint/watermarking

# Create directory first
mkdir -p test_results_family_4

# Run Family 4
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_4/experiment.log
```

### Option 2: Setup HF Auth Then Run Family 3

```python
# In Colab cell 1: Login
from huggingface_hub import notebook_login
notebook_login()
```

```bash
# In Colab cell 2: Run Family 3
!cd /content/llm_fingerprint/watermarking && \
mkdir -p test_results_family_3 && \
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_3 \
    --family_index 3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_3/experiment.log
```

## 📋 All Families Status

| Family | Base Model | Size | Gated? | Status |
|--------|-----------|------|--------|--------|
| 1 | meta-llama/Llama-3.1-8B | 8B | ⚠️ Yes | ✅ Done |
| 2 | Qwen/Qwen2.5-1.5B | 1.5B | ✅ No | ✅ Done |
| 3 | meta-llama/Llama-3.1-8B-Instruct | 8B | ⚠️ Yes | ⏭️ Next (needs auth) |
| 4 | Qwen/Qwen2.5-7B | 7B | ✅ No | ❌ Not started |
| 5 | Qwen/Qwen3-0.6B-Base | 0.6B | ✅ No | ❌ Not started |
| 6 | meta-llama/Llama-3.2-3B-Instruct | 3B | ⚠️ Yes | ❌ Not started |

## 🎯 Recommended Running Order

### Without HF Auth (Run These First)

```bash
# 1. Family 5 (smallest, no auth)
mkdir -p test_results_family_5
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_5 \
    --family_index 5 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_5/experiment.log

# 2. Family 4 (medium, no auth)
mkdir -p test_results_family_4
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_4/experiment.log
```

### With HF Auth (Run After Login)

```python
# Login first
from huggingface_hub import notebook_login
notebook_login()
```

```bash
# 3. Family 3
mkdir -p test_results_family_3
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_3 \
    --family_index 3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_3/experiment.log

# 4. Family 6
mkdir -p test_results_family_6
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_6 \
    --family_index 6 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_6/experiment.log
```

## 🎯 Start Now: Family 4

**This will work without authentication:**

```bash
cd /content/llm_fingerprint/watermarking

mkdir -p test_results_family_4

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_4/experiment.log
```

Family 4 is **Qwen/Qwen2.5-7B** - no gating, should work!
