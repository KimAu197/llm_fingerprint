# Handling Gated Models in Colab

## Problem

Some models require HuggingFace authentication:
- ❌ **meta-llama/Llama-3.1-8B-Instruct** (gated)
- ❌ **meta-llama/Llama-3.1-8B** (gated)
- ❌ **meta-llama/Llama-3.2-3B-Instruct** (gated)

## Solution 1: Login to HuggingFace in Colab

### Option A: Using HF Token

```python
# In Colab, first cell
from huggingface_hub import login

# Get your token from: https://huggingface.co/settings/tokens
login(token="hf_YOUR_TOKEN_HERE")
```

### Option B: Interactive Login

```python
from huggingface_hub import notebook_login
notebook_login()
```

### Option C: Using Environment Variable

```python
import os
os.environ['HF_TOKEN'] = 'hf_YOUR_TOKEN_HERE'
```

Then in the script, add to model loading:

```python
from huggingface_hub import login
import os

# At the start of script
if 'HF_TOKEN' in os.environ:
    login(token=os.environ['HF_TOKEN'])
```

## Solution 2: Run Non-Gated Families First (Recommended) ⭐️

Skip the Llama families, run Qwen families first:

### Run These First (No Authentication Needed):

```bash
# Family 2: Qwen/Qwen2.5-1.5B (already done)
# ✅ Done

# Family 4: Qwen/Qwen2.5-7B
cd /content/llm_fingerprint/watermarking
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5

# Family 5: Qwen/Qwen3-0.6B-Base
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_5 \
    --family_index 5 \
    --num_pairs 5 \
    --num_negative_samples 5
```

### Then Setup HF Auth for Llama Models:

```python
# Login to HuggingFace
from huggingface_hub import notebook_login
notebook_login()

# Then run Llama families
!cd /content/llm_fingerprint/watermarking && \
python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_1 \
    --family_index 1 \
    --num_pairs 5 \
    --num_negative_samples 5
```

## Solution 3: Replace Gated Models

Replace gated Llama models with open-access alternatives:

### Alternative Base Models (No Gating):
- ✅ `mistralai/Mistral-7B-v0.1` (9 derivatives)
- ✅ `microsoft/Phi-3.5-mini-instruct` (9 derivatives)
- ✅ `google/gemma-3-12b-pt` (6 derivatives)

Want me to regenerate the CSV with non-gated models?

## Recommended Order (No Auth Needed)

Run in this order to avoid gated models:

```bash
# 1. Family 5: Qwen3-0.6B-Base (smallest, no auth)
./test/run_family_5.sh

# 2. Family 4: Qwen2.5-7B (no auth)
./test/run_family_4.sh

# 3. Then setup HF auth and run:
# - Family 1: Llama-3.1-8B
# - Family 3: Llama-3.1-8B-Instruct
# - Family 6: Llama-3.2-3B-Instruct
```

## Quick Fix for Current Error

The directory error is now fixed in the code. But you still need to handle gated models.

**Easiest next step:**

```bash
# Run Family 4 (Qwen, no auth needed)
cd /content/llm_fingerprint/watermarking

python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results_family_4 \
    --family_index 4 \
    --num_pairs 5 \
    --num_negative_samples 5 \
    2>&1 | tee test_results_family_4/experiment.log
```

This will work without authentication!
