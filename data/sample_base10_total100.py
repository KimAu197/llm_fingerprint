import pandas as pd

SRC = "../result/result_2.10/data/model_ground_truth_finetune_llm_logit_access.csv"
DST = "random_100_base10.csv"
N_TOTAL = 100
N_BASE = 10
SEED = 42

# Base models to use (replaced Llama-3.3-70B with Qwen2.5-14B)
TARGET_BASES = [
    'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-14B',  # Replaced: was meta-llama/Llama-3.3-70B-Instruct
    'Qwen/Qwen2.5-VL-7B-Instruct',
    'meta-llama/Llama-3.1-8B',
    'Qwen/Qwen2.5-32B',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'Qwen/Qwen2.5-VL-3B-Instruct',
    'microsoft/Phi-3.5-mini-instruct'
]

df = pd.read_csv(SRC)
df = df.drop_duplicates(subset=['model_id'], keep='first')

print(f"Total unique models: {len(df)}")
print(f"\nSelected {N_BASE} base models:")
for i, base in enumerate(TARGET_BASES, 1):
    count = len(df[df['effective_base_model'] == base])
    print(f"{i}. {base}: {count} finetuned models")

# Calculate how many derived models to sample from each base
n_derived = N_TOTAL - N_BASE
per_base = n_derived // N_BASE
remainder = n_derived % N_BASE

print(f"\nTarget: {N_BASE} base models + {n_derived} derived models = {N_TOTAL} total")
print(f"Strategy: {per_base} derived models per base, +1 for first {remainder} bases\n")

# Sample derived models from each base
selected_derived = []
for i, base in enumerate(TARGET_BASES):
    # How many derived models to sample from this base
    n_sample = per_base + (1 if i < remainder else 0)
    
    # Get all models with this base (exclude base itself and any already selected)
    derived = df[df['effective_base_model'] == base]['model_id'].tolist()
    derived = [d for d in derived if d != base and d not in TARGET_BASES]
    
    # Sample (or take all if not enough)
    if len(derived) <= n_sample:
        sampled = derived
        print(f"{base}: took all {len(sampled)}/{len(derived)} derived models")
    else:
        sampled = pd.Series(derived).sample(n=n_sample, random_state=SEED+i).tolist()
        print(f"{base}: sampled {len(sampled)}/{len(derived)} derived models")
    
    selected_derived.extend(sampled)

# Combine base + derived
all_models = TARGET_BASES + selected_derived

print(f"\nTotal selected: {len(all_models)} models")
print(f"  - {len(TARGET_BASES)} base models")
print(f"  - {len(selected_derived)} derived models")
print(f"  - Unique: {len(set(all_models))}")

# Save
result = pd.DataFrame({'model_id': sorted(all_models)})
result.to_csv(DST, index=False)
print(f"\nSaved to {DST}")
