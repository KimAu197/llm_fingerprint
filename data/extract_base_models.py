import pandas as pd

SRC_MODELS = "random_100_base10.csv"
SRC_FULL = "../result/result_2.10/data/model_ground_truth_finetune_llm_logit_access.csv"
DST = "random_100_base10_with_bases.csv"

# Load the 100 models
models_df = pd.read_csv(SRC_MODELS)
print(f"Loaded {len(models_df)} models")

# Load full dataset
full_df = pd.read_csv(SRC_FULL)
full_df = full_df.drop_duplicates(subset=['model_id'], keep='first')
print(f"Full dataset: {len(full_df)} unique models")

# Get base models for the 100 models
models_with_bases = []
for model_id in models_df['model_id']:
    row = full_df[full_df['model_id'] == model_id]
    if len(row) > 0:
        base = row.iloc[0]['effective_base_model']
        models_with_bases.append({
            'model_id': model_id,
            'base_model': base
        })
    else:
        models_with_bases.append({
            'model_id': model_id,
            'base_model': None
        })
        print(f"Warning: {model_id} not found in full dataset")

result = pd.DataFrame(models_with_bases)
result.to_csv(DST, index=False)
print(f"\nSaved to {DST}")
print(f"Models with base info: {result['base_model'].notna().sum()}/{len(result)}")
