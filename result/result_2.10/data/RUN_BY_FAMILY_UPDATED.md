# Running Experiments One Family at a Time

## 🎯 Solution

Use the **SAME CSV** (all 30 models) but specify `--family_index` to run only one family.

This way:
- ✅ All 30 models are loaded (needed for negative samples)
- ✅ Only ONE family is actually tested
- ✅ Negative samples can still be drawn from other families
- ✅ Saves memory by only processing one family at a time

## 🚀 Run Individual Families

### Family 3: meta-llama/Llama-3.1-8B-Instruct (5 models)

```bash
cd /content/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_3 \
    --family_index 3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda
```

**Or use the convenience script:**
```bash
./test/run_family_3.sh
```

### Family 4: Qwen/Qwen2.5-7B (5 models)

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5
```

**Or:**
```bash
./test/run_family_4.sh
```

### Family 5: Qwen/Qwen3-0.6B-Base (5 models)

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_5 \
    --family_index 5 \
    --num_pairs 5 \
    --num_negative_samples 5
```

**Or:**
```bash
./test/run_family_5.sh
```

### Family 6: meta-llama/Llama-3.2-3B-Instruct (5 models)

```bash
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_6 \
    --family_index 6 \
    --num_pairs 5 \
    --num_negative_samples 5
```

**Or:**
```bash
./test/run_family_6.sh
```

## 📊 Family Index Reference

| Index | Base Model | Models | Size |
|-------|-----------|--------|------|
| 1 | meta-llama/Llama-3.1-8B | 5 | 8B |
| 2 | Qwen/Qwen2.5-1.5B | 5 | 1.5B |
| **3** | **meta-llama/Llama-3.1-8B-Instruct** | **5** | **8B** |
| 4 | Qwen/Qwen2.5-7B | 5 | 7B |
| 5 | Qwen/Qwen3-0.6B-Base | 5 | 0.6B |
| 6 | meta-llama/Llama-3.2-3B-Instruct | 5 | 3B |

## 📈 What Each Family Run Does

For each family (e.g., Family 3):
1. Loads **full 30-model CSV** (all families)
2. Identifies Family 3's base model and derivatives
3. Loads base model and generates fingerprints
4. Tests **5 positive samples** (derivatives vs base)
5. Tests **25 negative samples** (5 derivatives × 5 random from other families)
6. Saves results to `test_results_family_3/`

## 🔄 Merge Results Later

After running all families (3-6), merge them:

```bash
cd /content/llm_fingerprint/watermarking

python3 test/merge_family_results.py \
    --family_dirs \
        test_results_family_3 \
        test_results_family_4 \
        test_results_family_5 \
        test_results_family_6 \
    --output test_results_combined/all_families.csv
```

Or include families 1 and 2 (already done):

```python
import pandas as pd

# Load all results
dfs = []
dfs.append(pd.read_csv('/content/Downloads/base_family_overlap_results.csv'))  # Families 1-2
dfs.append(pd.read_csv('test_results_family_3/base_family_overlap_results.csv'))
dfs.append(pd.read_csv('test_results_family_4/base_family_overlap_results.csv'))
dfs.append(pd.read_csv('test_results_family_5/base_family_overlap_results.csv'))
dfs.append(pd.read_csv('test_results_family_6/base_family_overlap_results.csv'))

combined = pd.concat(dfs, ignore_index=True)

# Remove duplicates
combined = combined.drop_duplicates(
    subset=['base_model', 'test_model', 'sample_type'], 
    keep='first'
)

combined.to_csv('all_families_complete.csv', index=False)
print(f"✅ Saved {len(combined)} results")
```

## 💾 Memory-Efficient Running Order

Run from smallest to largest:

```bash
# 1. Smallest (0.6B) - quick test
./test/run_family_5.sh

# 2. Small (3B)
./test/run_family_6.sh

# 3. Medium (7B) - if family 2 didn't complete this
./test/run_family_4.sh

# 4. Large (8B)
./test/run_family_3.sh
```

## 🎯 Start with Family 3

**In Colab:**

```python
# Family 3: Llama-3.1-8B-Instruct
!cd /content/llm_fingerprint/watermarking && \
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_3 \
    --family_index 3 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    --device cuda \
    2>&1 | tee test_results_family_3/experiment.log
```

This will:
- Load all 30 models from the full CSV
- Run ONLY Family 3
- Use other 25 models as negative samples
- Save logs to file
- Save results to separate directory

## ✅ Why This Works

```
Input: experiment_models_base_family.csv (30 models, 6 families)
                    ↓
        --family_index 3 (filter)
                    ↓
Process: Only Family 3's 5 models
                    ↓
Negative samples: Randomly select from other 25 models
                    ↓
Output: test_results_family_3/base_family_overlap_results.csv (30 tests)
```

Each family is self-contained but has access to all other families for negatives!
