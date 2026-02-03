"""
Create a detailed table showing TP/FN/TPR/FNR at different thresholds.
Compare before and after download filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
ROC_DIR = SCRIPT_DIR / "roc_analysis"

# Load ROC data
qwen_roc = pd.read_csv(ROC_DIR / "roc_data_qwen.csv")
llama_roc = pd.read_csv(ROC_DIR / "roc_data_llama.csv")

print("=" * 100)
print("THRESHOLD ANALYSIS: TP/FN/TPR/FNR at Different Thresholds")
print("=" * 100)
print("\nComparing Before vs After Download Filtering (downloads > 10)")
print()

def create_threshold_table(roc_df, family_name):
    """Create a table showing metrics at key thresholds."""
    
    # Key thresholds to examine
    key_thresholds = [0.0, 0.005, 0.01, 0.02, 0.025, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    results = []
    
    for threshold in key_thresholds:
        # Get closest threshold in data
        before_row = roc_df[(roc_df["filter"] == "before") & 
                            (np.abs(roc_df["threshold"] - threshold) < 0.001)].iloc[0]
        after_row = roc_df[(roc_df["filter"] == "after") & 
                           (np.abs(roc_df["threshold"] - threshold) < 0.001)].iloc[0]
        
        results.append({
            "Threshold": threshold,
            "Before_N": before_row["n"],
            "Before_TP": before_row["tp"],
            "Before_FN": before_row["fn"],
            "Before_TPR": before_row["tpr"],
            "Before_FNR": before_row["fnr"],
            "After_N": after_row["n"],
            "After_TP": after_row["tp"],
            "After_FN": after_row["fn"],
            "After_TPR": after_row["tpr"],
            "After_FNR": after_row["fnr"],
        })
    
    df = pd.DataFrame(results)
    
    # Add change columns
    df["TPR_Change"] = df["After_TPR"] - df["Before_TPR"]
    df["FNR_Change"] = df["After_FNR"] - df["Before_FNR"]
    
    return df


# Create tables for both families
print("\n" + "=" * 100)
print("QWEN FAMILY")
print("=" * 100)
qwen_table = create_threshold_table(qwen_roc, "Qwen")
print(qwen_table.to_string(index=False))

# Save
qwen_table.to_csv(ROC_DIR / "threshold_table_qwen.csv", index=False)
print(f"\nSaved: {ROC_DIR / 'threshold_table_qwen.csv'}")

print("\n" + "=" * 100)
print("LLAMA FAMILY")
print("=" * 100)
llama_table = create_threshold_table(llama_roc, "LLaMA")
print(llama_table.to_string(index=False))

# Save
llama_table.to_csv(ROC_DIR / "threshold_table_llama.csv", index=False)
print(f"\nSaved: {ROC_DIR / 'threshold_table_llama.csv'}")

# Create a summary table focusing on key findings
print("\n" + "=" * 100)
print("KEY FINDINGS: Optimal Thresholds for 95% TPR")
print("=" * 100)

def find_optimal_threshold(roc_df, target_tpr=0.95):
    """Find the threshold that achieves target TPR."""
    before = roc_df[roc_df["filter"] == "before"]
    after = roc_df[roc_df["filter"] == "after"]
    
    # Find minimum threshold that achieves target TPR
    before_optimal = before[before["tpr"] >= target_tpr].iloc[-1]  # Last one (highest threshold)
    after_optimal = after[after["tpr"] >= target_tpr].iloc[-1]
    
    return {
        "before_threshold": before_optimal["threshold"],
        "before_tpr": before_optimal["tpr"],
        "before_fnr": before_optimal["fnr"],
        "before_tp": before_optimal["tp"],
        "before_fn": before_optimal["fn"],
        "before_n": before_optimal["n"],
        "after_threshold": after_optimal["threshold"],
        "after_tpr": after_optimal["tpr"],
        "after_fnr": after_optimal["fnr"],
        "after_tp": after_optimal["tp"],
        "after_fn": after_optimal["fn"],
        "after_n": after_optimal["n"],
    }

qwen_optimal = find_optimal_threshold(qwen_roc, target_tpr=0.95)
llama_optimal = find_optimal_threshold(llama_roc, target_tpr=0.95)

optimal_df = pd.DataFrame([
    {
        "Family": "Qwen",
        **qwen_optimal,
        "threshold_change": qwen_optimal["after_threshold"] - qwen_optimal["before_threshold"],
    },
    {
        "Family": "LLaMA",
        **llama_optimal,
        "threshold_change": llama_optimal["after_threshold"] - llama_optimal["before_threshold"],
    }
])

print("\nOptimal thresholds to achieve 95% TPR (True Positive Rate):")
print(optimal_df.to_string(index=False))

optimal_df.to_csv(ROC_DIR / "optimal_thresholds_95tpr.csv", index=False)
print(f"\nSaved: {ROC_DIR / 'optimal_thresholds_95tpr.csv'}")

print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)
print("""
Key Metrics:
- TP (True Positive): Same-family models correctly identified (overlap >= threshold)
- FN (False Negative): Same-family models missed (overlap < threshold)
- TPR (True Positive Rate): TP / (TP + FN) = Recall = Detection rate
- FNR (False Negative Rate): FN / (TP + FN) = 1 - TPR = Miss rate

Goal: Find threshold that maximizes TPR (minimizes FNR) while maintaining accuracy.

Filtering Effect (downloads > 10):
- Reduces sample size from ~116-117 to 27 models
- May change optimal threshold if low-download models have different overlap patterns
""")

print("\n" + "=" * 100)
