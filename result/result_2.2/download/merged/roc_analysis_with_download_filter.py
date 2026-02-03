"""
ROC Analysis: Compare Before/After Download Filtering

Analyze how filtering by downloads affects the ROC curve and optimal threshold.
Uses only SAME data (true lineage pairs).

Strategy:
- Vary threshold from 0 to 1
- For each threshold, compute TP, FP, FN, TN
- TP: same-family correctly identified (overlap >= threshold)
- FN: same-family missed (overlap < threshold)
- Plot ROC curves for before/after filtering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import auc

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "roc_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load merged same data
same_qwen = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_qwen_same_merged.csv")
same_llama = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_llama_same_merged.csv")

print("=" * 80)
print("ROC Analysis: Effect of Download Filtering on Same-Family Detection")
print("=" * 80)
print("\nData: Merged dataset (original + experiment)")
print("Analysis: Vary threshold to see TP/FN trade-offs")
print("Compare: Before filtering vs After filtering (downloads > 10)")
print()


def prepare_data(df, filter_downloads=None):
    """
    Prepare valid overlap data.
    
    Returns:
        valid_df: DataFrame with valid overlap ratios
    """
    # Valid: num_pairs > 0 and avg_overlap_ratio >= 0
    valid = df[(df["num_pairs"] > 0) & (df["avg_overlap_ratio"] >= 0)].copy()
    
    if filter_downloads is not None:
        valid = valid[valid["derived_model_downloads"] > filter_downloads]
    
    return valid


def compute_metrics_at_threshold(valid_df, threshold):
    """
    Compute TP, FN, TPR, FNR at a given threshold.
    
    For same-family data:
    - TP: overlap >= threshold (correctly identified as same)
    - FN: overlap < threshold (incorrectly identified as different)
    - TPR (Recall) = TP / (TP + FN)
    - FNR = FN / (TP + FN) = 1 - TPR
    """
    if len(valid_df) == 0:
        return {"tp": 0, "fn": 0, "tpr": 0, "fnr": 1.0, "n": 0}
    
    tp = (valid_df["avg_overlap_ratio"] >= threshold).sum()
    fn = (valid_df["avg_overlap_ratio"] < threshold).sum()
    total = len(valid_df)
    
    tpr = tp / total if total > 0 else 0
    fnr = fn / total if total > 0 else 0
    
    return {
        "tp": tp,
        "fn": fn,
        "tpr": tpr,
        "fnr": fnr,
        "n": total
    }


def compute_roc_curve(valid_df, thresholds):
    """
    Compute ROC curve points.
    
    For same-family detection:
    - X-axis: threshold
    - Y-axis: TPR (True Positive Rate = Recall)
    
    Also compute FNR for reference.
    """
    results = []
    
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(valid_df, threshold)
        results.append({
            "threshold": threshold,
            "tpr": metrics["tpr"],
            "fnr": metrics["fnr"],
            "tp": metrics["tp"],
            "fn": metrics["fn"],
            "n": metrics["n"]
        })
    
    return pd.DataFrame(results)


def find_best_threshold(roc_df, target_tpr=0.95):
    """
    Find the minimum threshold that achieves target TPR (e.g., 95% recall).
    
    This minimizes FNR while maintaining high detection rate.
    """
    # Filter to thresholds that meet target TPR
    candidates = roc_df[roc_df["tpr"] >= target_tpr]
    
    if len(candidates) == 0:
        # If no threshold meets target, return threshold with highest TPR
        best_idx = roc_df["tpr"].idxmax()
        return roc_df.iloc[best_idx]
    
    # Among candidates, choose the one with minimum threshold (most conservative)
    best_idx = candidates["threshold"].idxmin()
    return candidates.loc[best_idx]


def analyze_family(name, same_df):
    """Analyze one model family with and without download filtering."""
    print(f"\n{'=' * 80}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 80}")
    
    # Prepare data
    before_df = prepare_data(same_df, filter_downloads=None)
    after_df = prepare_data(same_df, filter_downloads=10)
    
    print(f"\nSample sizes:")
    print(f"  Before filtering: {len(before_df)} models")
    print(f"  After filtering (downloads > 10): {len(after_df)} models")
    
    # Define threshold range
    thresholds = np.linspace(0, 1, 201)  # 0.00, 0.005, 0.01, ..., 1.00
    
    # Compute ROC curves
    roc_before = compute_roc_curve(before_df, thresholds)
    roc_after = compute_roc_curve(after_df, thresholds)
    
    # Find best thresholds (95% TPR)
    best_before = find_best_threshold(roc_before, target_tpr=0.95)
    best_after = find_best_threshold(roc_after, target_tpr=0.95)
    
    print(f"\nBest thresholds (targeting 95% TPR):")
    print(f"  Before: threshold={best_before['threshold']:.4f}, TPR={best_before['tpr']:.4f}, FNR={best_before['fnr']:.4f}")
    print(f"  After:  threshold={best_after['threshold']:.4f}, TPR={best_after['tpr']:.4f}, FNR={best_after['fnr']:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: TPR vs Threshold
    ax1 = axes[0]
    ax1.plot(roc_before["threshold"], roc_before["tpr"], 'b-', linewidth=2, label='Before filtering (all models)')
    ax1.plot(roc_after["threshold"], roc_after["tpr"], 'r-', linewidth=2, label='After filtering (downloads > 10)')
    
    # Mark best thresholds
    ax1.axvline(best_before["threshold"], color='b', linestyle='--', alpha=0.5, label=f'Best before: {best_before["threshold"]:.4f}')
    ax1.axvline(best_after["threshold"], color='r', linestyle='--', alpha=0.5, label=f'Best after: {best_after["threshold"]:.4f}')
    ax1.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='Target TPR: 0.95')
    
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{name}: TPR vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.3])  # Focus on relevant range
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: FNR vs Threshold
    ax2 = axes[1]
    ax2.plot(roc_before["threshold"], roc_before["fnr"], 'b-', linewidth=2, label='Before filtering (all models)')
    ax2.plot(roc_after["threshold"], roc_after["fnr"], 'r-', linewidth=2, label='After filtering (downloads > 10)')
    
    # Mark best thresholds
    ax2.axvline(best_before["threshold"], color='b', linestyle='--', alpha=0.5, label=f'Best before: {best_before["threshold"]:.4f}')
    ax2.axvline(best_after["threshold"], color='r', linestyle='--', alpha=0.5, label=f'Best after: {best_after["threshold"]:.4f}')
    ax2.axhline(0.05, color='gray', linestyle=':', alpha=0.5, label='Target FNR: 0.05')
    
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('False Negative Rate (FNR)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{name}: FNR vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.3])  # Focus on relevant range
    ax2.set_ylim([0, 0.5])
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"roc_analysis_{name.lower()}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")
    plt.close()
    
    # Save detailed results
    roc_before["filter"] = "before"
    roc_after["filter"] = "after"
    roc_combined = pd.concat([roc_before, roc_after], ignore_index=True)
    csv_path = OUTPUT_DIR / f"roc_data_{name.lower()}.csv"
    roc_combined.to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path}")
    
    return {
        "family": name,
        "before_n": len(before_df),
        "after_n": len(after_df),
        "best_threshold_before": best_before["threshold"],
        "best_tpr_before": best_before["tpr"],
        "best_fnr_before": best_before["fnr"],
        "best_threshold_after": best_after["threshold"],
        "best_tpr_after": best_after["tpr"],
        "best_fnr_after": best_after["fnr"],
        "threshold_change": best_after["threshold"] - best_before["threshold"],
    }


# Run analysis
summary_results = []

for name, same_df in [("Qwen", same_qwen), ("LLaMA", same_llama)]:
    result = analyze_family(name, same_df)
    summary_results.append(result)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY: Best Thresholds Before vs After Filtering")
print("=" * 80)
print("\nTarget: 95% TPR (True Positive Rate / Recall)")
print()

summary_df = pd.DataFrame(summary_results)
print(summary_df.to_string(index=False))

# Save summary
summary_path = OUTPUT_DIR / "roc_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")

print("\n" + "=" * 80)
print("Analysis complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("=" * 80)
