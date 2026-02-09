# 🚀 Quick Start Guide

## Fastest Start

```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking
./test/run_experiment.sh
```

Follow the prompts to select experiment type!

---

## Experiment Overview

### 📊 Experiment Design

This experiment tests whether fingerprinting can distinguish:
- ✅ **Positive samples**: Models from same family (base model vs derived model)
- ❌ **Negative samples**: Models from different families (cross-family comparison)

### 📈 Dataset

- **30 models** from 6 base model families
- **5 derivatives** per family

**6 Families:**
1. `meta-llama/Llama-3.1-8B` (5 derivatives)
2. `Qwen/Qwen2.5-1.5B` (5 derivatives)
3. `meta-llama/Llama-3.1-8B-Instruct` (5 derivatives)
4. `Qwen/Qwen2.5-7B` (5 derivatives)
5. `Qwen/Qwen3-0.6B-Base` (5 derivatives)
6. `Qwen/Qwen3-4B-Base` (5 derivatives)

### 🎯 Test Pairs

- **Positive samples**: 30 pairs (each derivative vs its base model)
- **Negative samples**: 150 pairs (each model vs 5 random models from other families)
- **Total**: 180 comparisons

---

## Three Ways to Run

### 1️⃣ Interactive Mode (Recommended)

```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking
./test/run_experiment.sh
```

You'll be prompted to choose:
- Quick test
- Standard ⭐️ Recommended
- Full
- Custom parameters

### 2️⃣ Direct Run (Standard Config)

```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results \
    --num_pairs 10 \
    --num_negative_samples 5
```

### 3️⃣ Quick Test (Code Verification)

```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_quick \
    --num_pairs 3 \
    --num_negative_samples 2
```

---

## View Results

### 1. View Raw CSV

```bash
cat test_results/base_family_overlap_results.csv
```

Or open with Excel/Numbers:
```
test_results/base_family_overlap_results.csv
```

### 2. Statistical Analysis + Visualization

```bash
python3 test/analyze_results.py \
    --input test_results/base_family_overlap_results.csv
```

Generates:
- 📊 `overlap_distribution.png` - Positive vs negative distribution
- 📊 `per_family_comparison.png` - Per-family overlap comparison
- 📊 `positive_vs_negative_scatter.png` - Scatter plot

### 3. Custom Python Analysis

```python
import pandas as pd
import json

# Load results
df = pd.read_csv('test_results/base_family_overlap_results.csv')

# Separate positive and negative samples
positive = df[df['sample_type'] == 'positive']
negative = df[df['sample_type'] == 'negative']

# Statistics
print(f"Positive mean: {positive['avg_overlap_ratio'].mean():.4f}")
print(f"Negative mean: {negative['avg_overlap_ratio'].mean():.4f}")
print(f"Separation: {positive['avg_overlap_ratio'].mean() - negative['avg_overlap_ratio'].mean():.4f}")

# View specific model scores
model_results = df[df['test_model'] == 'meta-llama/Llama-3.1-8B-Instruct']
for _, row in model_results.iterrows():
    scores = json.loads(row['pair_scores_json'])
    print(f"{row['sample_type']}: {row['base_model']} -> avg={row['avg_overlap_ratio']:.4f}")
```

---

## Parameters

### Required

- `--csv_path`: Path to experiment_models_base_family.csv

### Optional

| Parameter | Default | Description | Recommended |
|-----------|---------|-------------|-------------|
| `--output_dir` | `./test_results` | Output directory | - |
| `--device` | `cuda` | Compute device | `cuda` / `mps` / `cpu` |
| `--num_pairs` | `10` | Number of fingerprints per base model | 3-20 |
| `--num_negative_samples` | `5` | Number of negative samples per model | 3-10 |
| `--bottom_k_vocab` | `2000` | Bottom-k vocabulary size | 1000-3000 |
| `--seed` | `42` | Random seed | - |

### Parameter Impact

- **Larger `num_pairs`**:
  - ✅ More accurate results
  - ❌ Longer runtime
  - ❌ More memory usage

- **Larger `num_negative_samples`**:
  - ✅ Better negative sample coverage
  - ❌ Longer runtime (linear)

### Recommended Configurations

**Quick verification** (10-20 minutes):
```bash
--num_pairs 3 --num_negative_samples 2
```

**Standard experiment** (30-60 minutes):
```bash
--num_pairs 10 --num_negative_samples 5
```

**Full experiment** (1-2 hours):
```bash
--num_pairs 20 --num_negative_samples 5
```

---

## Expected Output

### Runtime Process

```
================================================================================
LOADING EXPERIMENT DATA
================================================================================

Loaded 6 base model families
Total models: 30
  meta-llama/Llama-3.1-8B: 5 derivatives
  Qwen/Qwen2.5-1.5B: 5 derivatives
  ...

================================================================================
FAMILY 1/6: meta-llama/Llama-3.1-8B
================================================================================
Derivatives: 5

[1] Loading base model...
  Loading: meta-llama/Llama-3.1-8B

[2] Generating 10 fingerprints...
    Generating fingerprint 1/10
    Generating fingerprint 2/10
    ...

[3] Computing base model bottom-k vocab...
    Cached bottom-k for fingerprint 1/10
    ...

[4] Testing 5 derived models (POSITIVE SAMPLES)...

  [1/5] Testing: meta-llama/Llama-3.1-8B-Instruct
    Computing derived model bottom-k vocab...
    Avg overlap: 0.8523

[5] Testing NEGATIVE SAMPLES (cross-family)...
  Testing meta-llama/Llama-3.1-8B-Instruct against 5 other-family models...
    [1/5] vs Qwen/Qwen2.5-7B-Instruct
      Avg overlap: 0.1234
    ...
```

### Example Results CSV

```csv
sample_type,base_model,test_model,num_pairs,avg_overlap_ratio,min_overlap,max_overlap,bottom_k_vocab_size,pair_scores_json
positive,meta-llama/Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct,10,0.8523,0.8201,0.8765,2000,"[0.8523, 0.8432, ...]"
negative,meta-llama/Llama-3.1-8B,Qwen/Qwen2.5-7B-Instruct,10,0.1234,0.1105,0.1389,2000,"[0.1234, 0.1156, ...]"
```

### Success Metrics

✅ **Good results should show:**
- Positive samples avg overlap > 0.7
- Negative samples avg overlap < 0.3
- Clear separation between positive and negative (gap > 0.4)

⚠️ **Signs of improvement needed:**
- Positive overlap < 0.6
- Negative overlap > 0.4
- Significant overlap between positive and negative distributions

---

## FAQ

### Q: Runtime too long?

A: Reduce parameters:
```bash
--num_pairs 3 --num_negative_samples 2
```

### Q: Out of memory (CUDA OOM)?

A: 
1. Run on fewer models at once
2. Reduce `--bottom_k_vocab` size
3. Use CPU (will be slow):
   ```bash
   --device cpu
   ```

### Q: Some models fail to load?

A: Normal! Script will skip failed models and continue. Check `[ERROR]` messages in output.

### Q: CSV file path not found?

A: Ensure experiment_models_base_family.csv is in the correct location:
```bash
# Check if file exists
ls ../result/result_2.10/data/experiment_models_base_family.csv

# Or use absolute path
--csv_path /absolute/path/to/experiment_models_base_family.csv
```

---

## Next Steps

Based on experiment results:

### ✅ If results are good (clear positive/negative separation)
→ Scale up to more models (80-100)

### ⚠️ If results are moderate (some separation but not ideal)
→ Tune parameters:
- Increase `--num_pairs` (more fingerprints)
- Adjust `--bottom_k_vocab` (different k values)

### ❌ If results are poor (no clear separation)
→ Method improvement needed:
- Analyze which families work well/poorly
- Consider different fingerprint generation strategies
- Check if special model types (VL, Coder, Math) affect results

---

## Need Help?

See detailed documentation:
- `test/README.md` - Complete experiment guide
- `test/analyze_results.py --help` - Analysis tool help
- `test/run_base_family_experiment.py --help` - Experiment script help
