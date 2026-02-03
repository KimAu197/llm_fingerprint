"""
Create complete threshold table including False Positives (FP) from diff data.

Metrics:
- From SAME data: TP, FN, TPR, FNR
- From DIFF data: FP, TN, FPR, TNR (Specificity)
- Combined: Accuracy, Precision, F1-Score
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
ROC_DIR = SCRIPT_DIR / "roc_analysis"

# Load original diff data (for false positives)
ORIGINAL_DATA_DIR = SCRIPT_DIR.parent.parent.parent / "result_1.26" / "download" / "original" / "data"
diff_qwen = pd.read_csv(ORIGINAL_DATA_DIR / "lineage_bottomk_overlap_qwen_diff_with_downloads.csv")
diff_llama = pd.read_csv(ORIGINAL_DATA_DIR / "lineage_bottomk_overlap_llama_diff_with_downloads.csv")

# Load merged same data
same_qwen = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_qwen_same_merged.csv")
same_llama = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_llama_same_merged.csv")

print("=" * 100)
print("COMPLETE THRESHOLD ANALYSIS: Including False Positives (FP)")
print("=" * 100)
print("\nMetrics:")
print("  From SAME data: TP, FN, TPR, FNR")
print("  From DIFF data: FP, TN, FPR, Specificity")
print("  Combined: Accuracy, Precision, F1-Score")
print()


def prepare_data(df, filter_downloads=None):
    """Prepare valid overlap data."""
    valid = df[(df["num_pairs"] > 0) & (df["avg_overlap_ratio"] >= 0)].copy()
    
    if filter_downloads is not None:
        valid = valid[valid["derived_model_downloads"] > filter_downloads]
    
    return valid


def compute_complete_metrics_at_threshold(same_df, diff_df, threshold):
    """
    Compute complete confusion matrix metrics at a given threshold.
    
    From SAME data (true positives):
    - TP: overlap >= threshold (correctly identified as same)
    - FN: overlap < threshold (incorrectly identified as different)
    
    From DIFF data (true negatives):
    - TN: overlap < threshold (correctly identified as different)
    - FP: overlap >= threshold (incorrectly identified as same)
    """
    # Same data metrics
    if len(same_df) == 0:
        tp, fn = 0, 0
    else:
        tp = (same_df["avg_overlap_ratio"] >= threshold).sum()
        fn = (same_df["avg_overlap_ratio"] < threshold).sum()
    
    # Diff data metrics
    if len(diff_df) == 0:
        tn, fp = 0, 0
    else:
        tn = (diff_df["avg_overlap_ratio"] < threshold).sum()
        fp = (diff_df["avg_overlap_ratio"] >= threshold).sum()
    
    # Calculate rates
    total_same = tp + fn
    total_diff = tn + fp
    total = total_same + total_diff
    
    tpr = tp / total_same if total_same > 0 else 0  # True Positive Rate (Recall/Sensitivity)
    fnr = fn / total_same if total_same > 0 else 0  # False Negative Rate
    tnr = tn / total_diff if total_diff > 0 else 0  # True Negative Rate (Specificity)
    fpr = fp / total_diff if total_diff > 0 else 0  # False Positive Rate
    
    # Combined metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    
    return {
        "threshold": threshold,
        # Counts
        "same_n": total_same,
        "diff_n": total_diff,
        "total_n": total,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        # Rates
        "tpr": tpr,
        "fnr": fnr,
        "tnr": tnr,
        "fpr": fpr,
        # Combined
        "accuracy": accuracy,
        "precision": precision,
        "f1": f1,
    }


def create_complete_threshold_table(same_df, diff_df, family_name, filter_downloads=None):
    """Create complete threshold table with FP included."""
    
    # Prepare data
    same_valid = prepare_data(same_df, filter_downloads)
    diff_valid = prepare_data(diff_df, filter_downloads)
    
    # Key thresholds to examine
    key_thresholds = [0.0, 0.005, 0.01, 0.02, 0.025, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    results = []
    for threshold in key_thresholds:
        metrics = compute_complete_metrics_at_threshold(same_valid, diff_valid, threshold)
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Format percentages for display
    df_display = df.copy()
    for col in ["tpr", "fnr", "tnr", "fpr", "accuracy", "precision", "f1"]:
        df_display[f"{col}_pct"] = (df[col] * 100).round(2)
    
    return df, df_display


def analyze_family_complete(name, same_df, diff_df):
    """Analyze one model family with complete metrics (before and after filtering)."""
    print(f"\n{'=' * 100}")
    print(f"{name} FAMILY - COMPLETE METRICS")
    print(f"{'=' * 100}")
    
    # Before filtering
    before_df, before_display = create_complete_threshold_table(same_df, diff_df, name, filter_downloads=None)
    
    # After filtering
    after_df, after_display = create_complete_threshold_table(same_df, diff_df, name, filter_downloads=10)
    
    print(f"\nSample sizes:")
    print(f"  Before filtering:")
    print(f"    Same (lineage): {before_df['same_n'].iloc[0]} models")
    print(f"    Diff (non-lineage): {before_df['diff_n'].iloc[0]} models")
    print(f"    Total: {before_df['total_n'].iloc[0]} models")
    print(f"  After filtering (downloads > 10):")
    print(f"    Same (lineage): {after_df['same_n'].iloc[0]} models")
    print(f"    Diff (non-lineage): {after_df['diff_n'].iloc[0]} models")
    print(f"    Total: {after_df['total_n'].iloc[0]} models")
    
    # Save detailed tables
    before_csv = ROC_DIR / f"complete_threshold_table_{name.lower()}_before.csv"
    after_csv = ROC_DIR / f"complete_threshold_table_{name.lower()}_after.csv"
    
    before_df.to_csv(before_csv, index=False)
    after_df.to_csv(after_csv, index=False)
    
    print(f"\nSaved:")
    print(f"  Before: {before_csv}")
    print(f"  After:  {after_csv}")
    
    # Print summary table
    print(f"\n{name} - BEFORE FILTERING (all models)")
    print("-" * 100)
    summary_cols = ["threshold", "tp", "fn", "fp", "tn", "tpr_pct", "fnr_pct", "fpr_pct", "tnr_pct", "accuracy_pct", "precision_pct", "f1_pct"]
    print(before_display[summary_cols].to_string(index=False))
    
    print(f"\n{name} - AFTER FILTERING (downloads > 10)")
    print("-" * 100)
    print(after_display[summary_cols].to_string(index=False))
    
    return before_df, after_df


# Analyze both families
print("\n" + "=" * 100)
print("ANALYSIS START")
print("=" * 100)

qwen_before, qwen_after = analyze_family_complete("Qwen", same_qwen, diff_qwen)
llama_before, llama_after = analyze_family_complete("LLaMA", same_llama, diff_llama)

# Create comparison summary
print("\n" + "=" * 100)
print("KEY FINDINGS: Effect of Download Filtering on False Positives")
print("=" * 100)

def summarize_at_threshold(before_df, after_df, threshold, family_name):
    """Summarize metrics at a specific threshold."""
    before_row = before_df[before_df["threshold"] == threshold].iloc[0]
    after_row = after_df[after_df["threshold"] == threshold].iloc[0]
    
    return {
        "Family": family_name,
        "Threshold": threshold,
        "Before_FP": before_row["fp"],
        "Before_FPR": f"{before_row['fpr']*100:.2f}%",
        "Before_Accuracy": f"{before_row['accuracy']*100:.2f}%",
        "Before_F1": f"{before_row['f1']*100:.2f}%",
        "After_FP": after_row["fp"],
        "After_FPR": f"{after_row['fpr']*100:.2f}%",
        "After_Accuracy": f"{after_row['accuracy']*100:.2f}%",
        "After_F1": f"{after_row['f1']*100:.2f}%",
        "FP_Change": int(after_row["fp"] - before_row["fp"]),
    }

# Compare at key thresholds
comparison_thresholds = [0.005, 0.01, 0.025, 0.05]
comparison_results = []

for threshold in comparison_thresholds:
    comparison_results.append(summarize_at_threshold(qwen_before, qwen_after, threshold, "Qwen"))
    comparison_results.append(summarize_at_threshold(llama_before, llama_after, threshold, "LLaMA"))

comparison_df = pd.DataFrame(comparison_results)
print("\nComparison at Key Thresholds:")
print(comparison_df.to_string(index=False))

comparison_csv = ROC_DIR / "fp_comparison_summary.csv"
comparison_df.to_csv(comparison_csv, index=False)
print(f"\nSaved: {comparison_csv}")

print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)
print("""
Complete Confusion Matrix:
- TP (True Positive): Same-family correctly identified as same (overlap >= threshold)
- FN (False Negative): Same-family incorrectly identified as different (overlap < threshold)
- TN (True Negative): Different-family correctly identified as different (overlap < threshold)
- FP (False Positive): Different-family incorrectly identified as same (overlap >= threshold)

Key Metrics:
- TPR (True Positive Rate) = TP/(TP+FN) = Recall = Sensitivity
- FNR (False Negative Rate) = FN/(TP+FN) = 1 - TPR
- TNR (True Negative Rate) = TN/(TN+FP) = Specificity
- FPR (False Positive Rate) = FP/(TN+FP) = 1 - TNR
- Accuracy = (TP+TN)/(TP+FN+TN+FP)
- Precision = TP/(TP+FP)
- F1-Score = 2*(Precision*Recall)/(Precision+Recall)

Goal: Maximize TPR and TNR (minimize FNR and FPR) while maintaining high accuracy.
""")

print("\n" + "=" * 100)
print("Analysis complete!")
print("=" * 100)
