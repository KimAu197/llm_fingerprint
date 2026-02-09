# Base Model Family Fingerprinting Experiment

## Experiment Design

### Objective
Test the fingerprinting technique's ability to distinguish between models within the same family (positive samples) and across different families (negative samples).

### Dataset
`experiment_models_base_family.csv` - 30 models from 6 base model families, with 5 derivatives per family.

### Experiment Design

#### 1️⃣ Positive Samples (Same Family - Should be Similar)
- **Test**: Each derivative model compared with its base model
- **Count**: 30 pairs (6 families × 5 derivatives)
- **Expected**: High overlap ratio (high similarity)

#### 2️⃣ Negative Samples (Different Family - Should be Dissimilar)
- **Test**: Each derivative model compared with models from other families
- **Count**: 30 × 5 = 150 pairs (each derivative randomly paired with 5 models from other families)
- **Expected**: Low overlap ratio (low similarity)

## How to Run

### Basic Usage

```bash
cd /Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/watermarking

python test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results \
    --device cuda \
    --num_pairs 10 \
    --bottom_k_vocab 2000 \
    --num_negative_samples 5
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv_path` | (required) | Path to experiment_models_base_family.csv |
| `--output_dir` | `./test_results` | Output directory for results |
| `--device` | `cuda` | Device (cuda, mps, cpu) |
| `--num_pairs` | `10` | Number of fingerprints per base model |
| `--bottom_k_vocab` | `2000` | Bottom-k vocabulary size |
| `--num_negative_samples` | `5` | Number of negative samples per model |

### Quick Test (Small Scale)

```bash
# Fewer fingerprints for quick verification
python test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_quick \
    --num_pairs 3 \
    --num_negative_samples 2
```

### Full Experiment

```bash
# More fingerprints for more accurate results
python test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_full \
    --num_pairs 20 \
    --num_negative_samples 5 \
    --bottom_k_vocab 2000
```

## Output Results

### Output File
`test_results/base_family_overlap_results.csv`

### CSV Format

| Column | Description |
|--------|-------------|
| `sample_type` | "positive" (same family) or "negative" (cross-family) |
| `base_model` | Base model name |
| `test_model` | Test model name |
| `num_pairs` | Number of fingerprints used |
| `avg_overlap_ratio` | Average overlap ratio |
| `min_overlap` | Minimum overlap |
| `max_overlap` | Maximum overlap |
| `bottom_k_vocab_size` | Bottom-k vocabulary size |
| `pair_scores_json` | Detailed scores for each fingerprint (JSON) |

### Example Results

```csv
sample_type,base_model,test_model,num_pairs,avg_overlap_ratio,min_overlap,max_overlap,bottom_k_vocab_size,pair_scores_json
positive,meta-llama/Llama-3.1-8B,meta-llama/Llama-3.1-8B-Instruct,10,0.8523,0.8201,0.8765,2000,"[0.8523, 0.8432, ...]"
negative,meta-llama/Llama-3.1-8B,Qwen/Qwen2.5-7B-Instruct,10,0.1234,0.1105,0.1389,2000,"[0.1234, 0.1156, ...]"
```

## Experiment Workflow

### For each base model family:

1. **Load Base Model**
   ```
   Loading: meta-llama/Llama-3.1-8B
   ```

2. **Generate Fingerprints**
   ```
   Generating 10 fingerprints...
   Generating fingerprint 1/10
   Generating fingerprint 2/10
   ...
   ```

3. **Compute Base Model's Bottom-K Vocab**
   ```
   Computing base model bottom-k vocab...
   Cached bottom-k for fingerprint 1/10
   ...
   ```

4. **Test Derivative Models (Positive Samples)**
   ```
   Testing derived models (POSITIVE SAMPLES)...
   [1/5] Testing: meta-llama/Llama-3.1-8B-Instruct
     Computing derived model bottom-k vocab...
     Avg overlap: 0.8523
   ```

5. **Test Cross-Family Models (Negative Samples)**
   ```
   Testing NEGATIVE SAMPLES (cross-family)...
   Testing meta-llama/Llama-3.1-8B-Instruct against 5 other-family models...
     [1/5] vs Qwen/Qwen2.5-7B-Instruct
       Avg overlap: 0.1234
   ```

## Analyze Results

### Using Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read results
df = pd.read_csv('test_results/base_family_overlap_results.csv')

# Group statistics
positive = df[df['sample_type'] == 'positive']
negative = df[df['sample_type'] == 'negative']

print(f"Positive samples - Mean: {positive['avg_overlap_ratio'].mean():.4f}")
print(f"Negative samples - Mean: {negative['avg_overlap_ratio'].mean():.4f}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(positive['avg_overlap_ratio'], bins=20, alpha=0.5, label='Positive (same family)')
plt.hist(negative['avg_overlap_ratio'], bins=20, alpha=0.5, label='Negative (different family)')
plt.xlabel('Overlap Ratio')
plt.ylabel('Count')
plt.legend()
plt.title('Overlap Ratio Distribution: Positive vs Negative Samples')
plt.savefig('test_results/overlap_distribution.png')
plt.show()
```

## Expected Results

### Successful fingerprinting should show:

1. **High Discrimination**
   - Positive samples' overlap ratio significantly higher than negative samples
   - Positive samples: average > 0.7
   - Negative samples: average < 0.3

2. **Consistency**
   - All derivatives within same family should have high overlap with base model
   - Cross-family overlaps should be consistently low

3. **Separation**
   - Clear separation between positive and negative distributions
   - Minimal or no overlap region between the two distributions

## Challenge Recording

While running experiments, record the following challenges:

### 1. Model Loading Issues
- [ ] Which models fail to load?
- [ ] Out of memory?
- [ ] Corrupted or unavailable model files?

### 2. Computational Resources
- [ ] How long does each model pair take?
- [ ] Memory usage?
- [ ] Need for batch processing?

### 3. Fingerprinting Effectiveness
- [ ] Clear distinction between positive and negative samples?
- [ ] Which families work well/poorly?
- [ ] False positives/negatives?

### 4. Other Issues
- [ ] API limits?
- [ ] Performance on special model types (VL, Coder, Math)?

## Next Steps

Based on experiment results:

1. **If results are good**: Scale up to more models (80-100)
2. **If results are moderate**: Adjust parameters (bottom_k_vocab, num_pairs)
3. **If results are poor**: Analyze specific issues, may need to improve fingerprint generation method
