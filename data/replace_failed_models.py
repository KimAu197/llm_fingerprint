import pandas as pd
import random

SRC = "../result/result_2.10/data/model_ground_truth_finetune_llm_logit_access.csv"
DST = "random_100_base10.csv"

# Failed models to replace
FAILED_MODELS = [
    'Open-Bee/Bee-8B-RL',
    'arcee-ai/SuperNova-Medius',
    'canopylabs/3b-de-pretrain-research_release',
    'microsoft/Phi-3.5-mini-instruct',
    'microsoft/Phi-3-mini-4k-instruct',
    'mistralai/Mistral-Nemo-Instruct-FP8-2407',
    'furiosa-ai/Llama-3.1-8B-Instruct-FP8',
    'furiosa-ai/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-2-7b-hf'
]

# Load current models
current = pd.read_csv(DST)
print(f"Current models: {len(current)}")
print(f"Models to replace: {len(FAILED_MODELS)}\n")

# Load full dataset
full_df = pd.read_csv(SRC)
full_df = full_df.drop_duplicates(subset=['model_id'], keep='first')

# Find small models (<=3B) excluding failed ones
current_set = set(current['model_id'])
exclude_set = current_set.union(set(FAILED_MODELS))

small_models = []
for _, row in full_df.iterrows():
    model_id = row['model_id']
    if model_id in exclude_set:
        continue
    
    # Only include small models (0.5B, 1B, 1.5B, 3B)
    if any(size in model_id for size in ['0.5B', '1B', '1.5B', '3B']):
        if not any(x in model_id for x in ['70B', '72B', '170B', 'FP8', 'FP4']):
            small_models.append(model_id)

print(f"Found {len(small_models)} small models available")

# Randomly select replacements
random.seed(42)
replacements = random.sample(small_models, min(len(FAILED_MODELS), len(small_models)))

print("\nReplacements:")
for i, (old, new) in enumerate(zip(FAILED_MODELS, replacements), 1):
    print(f"{i}. {old}")
    print(f"   -> {new}\n")

# Replace failed models
new_models = []
for model in current['model_id']:
    if model in FAILED_MODELS:
        idx = FAILED_MODELS.index(model)
        new_models.append(replacements[idx])
    else:
        new_models.append(model)

# Save
result = pd.DataFrame({'model_id': sorted(new_models)})
result.to_csv(DST, index=False)

print(f"Saved {len(result)} models to {DST}")
print(f"Unique models: {result['model_id'].nunique()}")
