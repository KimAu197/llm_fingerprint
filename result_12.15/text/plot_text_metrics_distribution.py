"""
plot_text_metrics_distribution.py

Plot distribution histograms for same vs diff text similarity metrics.
Filters out -1 values (failed generations).
Generates 4 separate figures for: avg_pal_k, avg_lev_sim, avg_lcs_ratio, avg_score
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============== Configuration ==============

# File paths - adjust these as needed
SAME_CSV = "lineage_bottomk_text_same_qwen.csv"
DIFF_CSV = "lineage_bottomk_text_diff_qwen.csv"

# Metrics to plot
METRICS = [
    ("avg_pal_k", "PAL_k (Prefix Agreement)"),
    ("avg_lev_sim", "Levenshtein Similarity"),
    ("avg_lcs_ratio", "LCS Ratio"),
    ("avg_score", "Combined Score"),
]

# Labels for legend
SAME_LABEL = "Same (qwen derived)"
DIFF_LABEL = "Diff (qwen base → llama derived)"

# Model name for title
MODEL_NAME = "qwen"

# ============== Load and filter data ==============

def load_and_filter(csv_path: str, metric: str) -> np.ndarray:
    """Load CSV and filter out -1 (failed) values for a specific metric."""
    df = pd.read_csv(csv_path)
    values = df[metric].values
    # Filter out -1 (error marker)
    valid = values[values >= 0]
    print(f"[{csv_path}] {metric}: Total={len(values)}, Valid={len(valid)}, Filtered={len(values) - len(valid)}")
    return valid


def plot_single_metric(same_values: np.ndarray, diff_values: np.ndarray, 
                       metric_name: str, metric_display: str, output_path: str):
    """Plot a single metric distribution."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine bins
    all_values = np.concatenate([same_values, diff_values])
    min_val = max(0, all_values.min() - 0.05)
    max_val = min(1, all_values.max() + 0.05)
    bins = np.linspace(min_val, max_val, 31)
    
    # Plot histograms with transparency
    ax.hist(same_values, bins=bins, alpha=0.6, label=SAME_LABEL, 
            color='#2ecc71', edgecolor='white', linewidth=0.5)
    ax.hist(diff_values, bins=bins, alpha=0.6, label=DIFF_LABEL, 
            color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    # Add vertical lines for means
    ax.axvline(same_values.mean(), color='#27ae60', linestyle='--', linewidth=2, 
               label=f'Same mean: {same_values.mean():.4f}')
    ax.axvline(diff_values.mean(), color='#c0392b', linestyle='--', linewidth=2, 
               label=f'Diff mean: {diff_values.mean():.4f}')
    
    # Labels and title
    ax.set_xlabel(metric_display, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {metric_display}: Same vs Diff ({MODEL_NAME})', 
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[saved] {output_path}")
    
    plt.close()


def main():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    # Print statistics for all metrics
    print("\n=== Statistics ===")
    
    for metric_name, metric_display in METRICS:
        same_values = load_and_filter(SAME_CSV, metric_name)
        diff_values = load_and_filter(DIFF_CSV, metric_name)
        
        print(f"\n{metric_display}:")
        print(f"  Same: mean={same_values.mean():.4f}, std={same_values.std():.4f}, "
              f"min={same_values.min():.4f}, max={same_values.max():.4f}")
        print(f"  Diff: mean={diff_values.mean():.4f}, std={diff_values.std():.4f}, "
              f"min={diff_values.min():.4f}, max={diff_values.max():.4f}")
        
        # Plot
        output_path = f"dist_{metric_name}_{MODEL_NAME.lower()}.png"
        plot_single_metric(same_values, diff_values, metric_name, metric_display, output_path)
    
    print("\n" + "=" * 60)
    print("All figures saved!")
    print("=" * 60)


if __name__ == "__main__":
    main()

