import pandas as pd

SRC = "../result/result_2.10/data/model_ground_truth_finetune_llm_logit_access.csv"
DST = "random_100_base10.csv"
N_TOTAL = 100
N_BASE = 10
SEED = 42

df = pd.read_csv(SRC)
df = df.drop_duplicates(subset=['model_id'], keep='first')

print(f"Total unique models: {len(df)}")

# Count finetuned models per base
base_counts = df['effective_base_model'].value_counts()
print(f"\nTop {N_BASE} base models by number of finetuned models:")
top_bases = base_counts.head(N_BASE).index.tolist()
for i, base in enumerate(top_bases, 1):
    print(f"{i}. {base}: {base_counts[base]} finetuned models")

# Calculate how many derived models to sample from each base
n_derived = N_TOTAL - N_BASE
per_base = n_derived // N_BASE
remainder = n_derived % N_BASE

print(f"\nTarget: {N_BASE} base models + {n_derived} derived models = {N_TOTAL} total")
print(f"Strategy: {per_base} derived models per base, +1 for first {remainder} bases")

# Sample derived models from each base
selected_derived = []
for i, base in enumerate(top_bases):
    # How many derived models to sample from this base
    n_sample = per_base + (1 if i < remainder else 0)
    
    # Get all models with this base (exclude base itself and any already selected)
    derived = df[df['effective_base_model'] == base]['model_id'].tolist()
    derived = [d for d in derived if d != base and d not in top_bases]
    
    # Sample (or take all if not enough)
    if len(derived) <= n_sample:
        sampled = derived
        print(f"{base}: took all {len(sampled)}/{len(derived)} derived models")
    else:
        sampled = pd.Series(derived).sample(n=n_sample, random_state=SEED+i).tolist()
        print(f"{base}: sampled {len(sampled)}/{len(derived)} derived models")
    
    selected_derived.extend(sampled)

# Combine base + derived
all_models = top_bases + selected_derived

print(f"\nTotal selected: {len(all_models)} models")
print(f"  - {len(top_bases)} base models")
print(f"  - {len(selected_derived)} derived models")
print(f"  - Unique: {len(set(all_models))}")

# Save
result = pd.DataFrame({'model_id': sorted(all_models)})
result.to_csv(DST, index=False)
print(f"\nSaved to {DST}")

