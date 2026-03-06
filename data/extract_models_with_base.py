import pandas as pd

SRC = "random_100_models.csv"
DST = "random_100_with_base.csv"

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} models")

models = set(df['model_id'].tolist())
print(f"Initial models: {len(models)}")

base_models = set()
for base in df['effective_base_model'].dropna():
    if '|' in str(base):
        base_models.update(str(base).split('|'))
    else:
        base_models.add(str(base))

print(f"Base models found: {len(base_models)}")

all_models = sorted(models.union(base_models))
print(f"Total unique models: {len(all_models)}")

result = pd.DataFrame({'model_id': all_models})
result.to_csv(DST, index=False)
print(f"Saved to {DST}")
