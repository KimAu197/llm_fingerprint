import pandas as pd

SRC = "../result/result_2.10/data/model_ground_truth_finetune_llm_logit_access.csv"
DST = "random_100_models.csv"
N = 100
SEED = 42

df = pd.read_csv(SRC)
print(f"Total rows: {len(df)}")
print(f"Unique model_ids: {df['model_id'].nunique()}")

df_unique = df.drop_duplicates(subset=['model_id'], keep='first')
print(f"After deduplication: {len(df_unique)}")

sampled = df_unique.sample(n=N, random_state=SEED)
sampled.to_csv(DST, index=False)
print(f"Sampled {len(sampled)} unique models -> {DST}")
