import pandas as pd
import json

# Load data
qwen_same = pd.read_csv('lineage_bottomk_overlap_qwen_same.csv')
llama_same = pd.read_csv('lineage_bottomk_overlap_llama_same.csv')

# Filter out valid data (num_pairs > 0)
qwen_same_valid = qwen_same[qwen_same['num_pairs'] > 0].copy()
llama_same_valid = llama_same[llama_same['num_pairs'] > 0].copy()

# Define low overlap ratio threshold
LOW_THRESHOLD = 0.3

print("=" * 80)
print("Analyzing low overlap ratio cases in same lineage")
print("=" * 80)

# Analyze Qwen models
print("\n" + "=" * 80)
print("Qwen2.5-0.5B Series - Low overlap ratio cases (< {})".format(LOW_THRESHOLD))
print("=" * 80)

qwen_low = qwen_same_valid[qwen_same_valid['avg_overlap_ratio'] < LOW_THRESHOLD].sort_values('avg_overlap_ratio')

print(f"\nFound {len(qwen_low)} derived models with low overlap ratio\n")

for idx, row in qwen_low.iterrows():
    print(f"Model: {row['derived_model_name']}")
    print(f"  Average overlap ratio: {row['avg_overlap_ratio']:.4f}")
    print(f"  Number of test pairs: {row['num_pairs']}")
    scores = json.loads(row['pair_scores_json'])
    print(f"  Individual test scores: {scores}")
    print(f"  Min score: {min(scores):.4f}, Max score: {max(scores):.4f}")
    print()

# Analyze TinyLlama models
print("\n" + "=" * 80)
print("TinyLlama-1.1B-Chat-v1.0 Series - Low overlap ratio cases (< {})".format(LOW_THRESHOLD))
print("=" * 80)

llama_low = llama_same_valid[llama_same_valid['avg_overlap_ratio'] < LOW_THRESHOLD].sort_values('avg_overlap_ratio')

print(f"\nFound {len(llama_low)} derived models with low overlap ratio\n")

for idx, row in llama_low.iterrows():
    print(f"Model: {row['derived_model_name']}")
    print(f"  Average overlap ratio: {row['avg_overlap_ratio']:.4f}")
    print(f"  Number of test pairs: {row['num_pairs']}")
    scores = json.loads(row['pair_scores_json'])
    print(f"  Individual test scores: {scores}")
    print(f"  Min score: {min(scores):.4f}, Max score: {max(scores):.4f}")
    print()

# Statistical summary
print("\n" + "=" * 80)
print("Statistical Summary")
print("=" * 80)

print("\nQwen2.5-0.5B:")
print(f"  Total derived models (valid data): {len(qwen_same_valid)}")
print(f"  Low overlap ratio models (< {LOW_THRESHOLD}): {len(qwen_low)}")
print(f"  Percentage: {len(qwen_low)/len(qwen_same_valid)*100:.2f}%")
print(f"  Mean overlap ratio: {qwen_same_valid['avg_overlap_ratio'].mean():.4f}")
print(f"  Median overlap ratio: {qwen_same_valid['avg_overlap_ratio'].median():.4f}")
print(f"  Min overlap ratio: {qwen_same_valid['avg_overlap_ratio'].min():.4f}")

print("\nTinyLlama-1.1B-Chat-v1.0:")
print(f"  Total derived models (valid data): {len(llama_same_valid)}")
print(f"  Low overlap ratio models (< {LOW_THRESHOLD}): {len(llama_low)}")
print(f"  Percentage: {len(llama_low)/len(llama_same_valid)*100:.2f}%")
print(f"  Mean overlap ratio: {llama_same_valid['avg_overlap_ratio'].mean():.4f}")
print(f"  Median overlap ratio: {llama_same_valid['avg_overlap_ratio'].median():.4f}")
print(f"  Min overlap ratio: {llama_same_valid['avg_overlap_ratio'].min():.4f}")

# Save low overlap ratio models to CSV
qwen_low.to_csv('qwen_low_overlap_analysis.csv', index=False)
llama_low.to_csv('llama_low_overlap_analysis.csv', index=False)

print("\nAnalysis results saved to:")
print("  - qwen_low_overlap_analysis.csv")
print("  - llama_low_overlap_analysis.csv")

# Analyze possible reasons
print("\n" + "=" * 80)
print("Possible Reasons for Low Overlap Ratio")
print("=" * 80)

print("\nInferred from model names:")

# Analyze Qwen low overlap models
print("\nQwen models:")
for idx, row in qwen_low.iterrows():
    name = row['derived_model_name'].lower()
    reasons = []
    
    if 'coder' in name or 'code' in name:
        reasons.append("Code-specialized model - may use code-specific vocabulary")
    if 'math' in name or 'prm' in name:
        reasons.append("Math-specialized model - may use mathematical symbols and terms")
    if 'law' in name or 'legal' in name:
        reasons.append("Legal-specialized model - domain-specific vocabulary")
    if 'medical' in name or 'med' in name:
        reasons.append("Medical-specialized model - medical terminology")
    if 'grpo' in name or 'dpo' in name or 'rl' in name:
        reasons.append("RL-trained model - may have changed vocabulary distribution")
    if 'sailor' in name:
        reasons.append("Multilingual model - may include other language vocabularies")
    if 'toxicity' in name:
        reasons.append("Toxicity detection model - task-specific vocabulary")
    
    if reasons:
        print(f"  {row['derived_model_name']}: {row['avg_overlap_ratio']:.4f}")
        for reason in reasons:
            print(f"    - {reason}")

# Analyze TinyLlama low overlap models
print("\nTinyLlama models:")
for idx, row in llama_low.iterrows():
    name = row['derived_model_name'].lower()
    reasons = []
    
    if 'jp' in name or 'japanese' in name:
        reasons.append("Japanese model - uses Japanese vocabulary")
    if 'fictional' in name:
        reasons.append("Fiction generation model - domain-specific vocabulary")
    if 'sql' in name or 'text2sql' in name:
        reasons.append("SQL generation model - SQL syntax vocabulary")
    if 'sec' in name:
        reasons.append("SEC document model - financial terminology")
    if 'bit' in name:
        reasons.append("May be a quantized or special architecture model")
    if 'ibrain' in name and 'q4' in name:
        reasons.append("Quantized model - may affect vocabulary")
    
    if reasons:
        print(f"  {row['derived_model_name']}: {row['avg_overlap_ratio']:.4f}")
        for reason in reasons:
            print(f"    - {reason}")

print("\n" + "=" * 80)
