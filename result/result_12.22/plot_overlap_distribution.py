"""
plot_overlap_distribution.py

Plot distribution histograms for same vs diff vs diff2 overlap ratios.
Filters out -1 values (failed generations).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============== Configuration ==============

# File paths - adjust these as needed
SAME_CSV = "lineage_bottomk_overlap_qwen_same.csv"
DIFF_CSV = "lineage_bottomk_overlap_qwen_diff.csv"
DIFF2_CSV = "lineage_bottomk_overlap_qwen_diff2.csv"  # Qwen2.5 vs Qwen2

# Output figure path
OUTPUT_FIG = "overlap_distribution_qwen_3groups.png"

# Labels for legend
SAME_LABEL = "Same (Qwen2.5 -> Qwen2.5 derived)"
DIFF_LABEL = "Diff (Qwen2.5 -> TinyLlama derived)"
DIFF2_LABEL = "Diff2 (Qwen2.5 -> Qwen2 derived)"

# Figure title
TITLE = "Distribution of Bottom-k Subspace Overlap Ratio"

# ============== Load and filter data ==============

def load_and_filter(csv_path: str) -> np.ndarray:
    """Load CSV and filter out -1 (failed) values."""
    df = pd.read_csv(csv_path)
    values = df["avg_overlap_ratio"].values
    # Filter out -1 (error marker)
    valid = values[values >= 0]
    print(f"[{csv_path}] Total: {len(values)}, Valid (>= 0): {len(valid)}, Filtered: {len(values) - len(valid)}")
    return valid


def main():
    # Load data
    same_values = load_and_filter(SAME_CSV)
    diff_values = load_and_filter(DIFF_CSV)
    diff2_values = load_and_filter(DIFF2_CSV)
    
    # Print statistics
    print(f"\n=== Statistics ===")
    print(f"Same:  mean={same_values.mean():.4f}, std={same_values.std():.4f}, min={same_values.min():.4f}, max={same_values.max():.4f}")
    print(f"Diff:  mean={diff_values.mean():.4f}, std={diff_values.std():.4f}, min={diff_values.min():.4f}, max={diff_values.max():.4f}")
    print(f"Diff2: mean={diff2_values.mean():.4f}, std={diff2_values.std():.4f}, min={diff2_values.min():.4f}, max={diff2_values.max():.4f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Determine bins - use same bins for all distributions
    bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
    
    # Plot histograms with transparency
    ax.hist(same_values, bins=bins, alpha=0.5, label=SAME_LABEL, color='#2ecc71', edgecolor='white', linewidth=0.5)
    ax.hist(diff_values, bins=bins, alpha=0.5, label=DIFF_LABEL, color='#e74c3c', edgecolor='white', linewidth=0.5)
    ax.hist(diff2_values, bins=bins, alpha=0.5, label=DIFF2_LABEL, color='#3498db', edgecolor='white', linewidth=0.5)
    
    # Add vertical lines for means
    ax.axvline(same_values.mean(), color='#27ae60', linestyle='--', linewidth=2, label=f'Same mean: {same_values.mean():.3f}')
    ax.axvline(diff_values.mean(), color='#c0392b', linestyle='--', linewidth=2, label=f'Diff mean: {diff_values.mean():.3f}')
    ax.axvline(diff2_values.mean(), color='#2980b9', linestyle='--', linewidth=2, label=f'Diff2 mean: {diff2_values.mean():.3f}')
    
    # Labels and title
    ax.set_xlabel('Average Overlap Ratio', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(TITLE, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper right', fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    # X-axis range
    ax.set_xlim(0, 1)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches='tight')
    print(f"\n[saved] {OUTPUT_FIG}")
    
    # Show figure
    plt.show()


if __name__ == "__main__":
    main()
