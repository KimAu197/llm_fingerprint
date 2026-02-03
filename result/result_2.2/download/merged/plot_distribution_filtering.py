"""
Plot distribution of avg_overlap_ratio before and after download filtering.
Shows same format as the original distribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "roc_analysis" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load same data (merged)
same_qwen = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_qwen_same_merged.csv")
same_llama = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_llama_same_merged.csv")

# Load diff data (original)
ORIGINAL_DATA_DIR = SCRIPT_DIR.parent.parent.parent / "result_1.26" / "download" / "original" / "data"
diff_qwen = pd.read_csv(ORIGINAL_DATA_DIR / "lineage_bottomk_overlap_qwen_diff_with_downloads.csv")
diff_llama = pd.read_csv(ORIGINAL_DATA_DIR / "lineage_bottomk_overlap_llama_diff_with_downloads.csv")

print("=" * 80)
print("Creating Distribution Plots: Before vs After Download Filtering")
print("=" * 80)


def prepare_data(df):
    """Prepare valid overlap data."""
    valid = df[(df["num_pairs"] > 0) & (df["avg_overlap_ratio"] >= 0)].copy()
    return valid


def plot_distribution_comparison(same_df, diff_df, family_name, output_path):
    """
    Plot distribution comparison: before vs after filtering.
    
    Style matches the original distribution plots:
    - Same (green) vs Diff (red/pink)
    - Mean lines for each distribution
    - Histogram with alpha transparency
    """
    # Prepare data
    same_before = prepare_data(same_df)
    same_after = same_before[same_before["derived_model_downloads"] > 10]
    diff_before = prepare_data(diff_df)
    diff_after = diff_before[diff_before["derived_model_downloads"] > 10]
    
    # Calculate means
    same_before_mean = same_before["avg_overlap_ratio"].mean()
    same_after_mean = same_after["avg_overlap_ratio"].mean()
    diff_before_mean = diff_before["avg_overlap_ratio"].mean()
    diff_after_mean = diff_after["avg_overlap_ratio"].mean()
    
    print(f"\n{family_name}:")
    print(f"  Same - Before: n={len(same_before)}, mean={same_before_mean:.3f}")
    print(f"  Same - After:  n={len(same_after)}, mean={same_after_mean:.3f}")
    print(f"  Diff - Before: n={len(diff_before)}, mean={diff_before_mean:.3f}")
    print(f"  Diff - After:  n={len(diff_after)}, mean={diff_after_mean:.3f}")
    
    # Create figure with 2 subplots (before and after)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Common histogram settings
    bins = np.linspace(0, 1, 50)
    
    # ============= LEFT: Before Filtering =============
    ax = axes[0]
    
    # Plot histograms
    ax.hist(diff_before["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#FF6B6B', label=f'Diff ({family_name} → Other derived)\nμ={diff_before_mean:.3f}', 
            edgecolor='none')
    ax.hist(same_before["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#90EE90', label=f'Same ({family_name} → {family_name} derived)\nμ={same_before_mean:.3f}',
            edgecolor='none')
    
    # Mean lines
    ax.axvline(diff_before_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(same_before_mean, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('avg_overlap_ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'{family_name}: Before Filtering (All Models)\nn_same={len(same_before)}, n_diff={len(diff_before)}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    # ============= RIGHT: After Filtering =============
    ax = axes[1]
    
    # Plot histograms
    ax.hist(diff_after["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#FF6B6B', label=f'Diff ({family_name} → Other derived)\nμ={diff_after_mean:.3f}', 
            edgecolor='none')
    ax.hist(same_after["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#90EE90', label=f'Same ({family_name} → {family_name} derived)\nμ={same_after_mean:.3f}',
            edgecolor='none')
    
    # Mean lines
    ax.axvline(diff_after_mean, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(same_after_mean, color='green', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('avg_overlap_ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'{family_name}: After Filtering (Downloads > 10)\nn_same={len(same_after)}, n_diff={len(diff_after)}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_single_distribution(same_df, diff_df, family_name, filtering, output_path):
    """
    Plot single distribution (matching original style exactly).
    
    Args:
        filtering: "before" or "after"
    """
    # Prepare data
    same_valid = prepare_data(same_df)
    diff_valid = prepare_data(diff_df)
    
    if filtering == "after":
        same_valid = same_valid[same_valid["derived_model_downloads"] > 10]
        diff_valid = diff_valid[diff_valid["derived_model_downloads"] > 10]
        title_suffix = " (After Filtering: downloads > 10)"
        n_info = f"n_same={len(same_valid)}, n_diff={len(diff_valid)}"
    else:
        title_suffix = " (Before Filtering: All Models)"
        n_info = f"n_same={len(same_valid)}, n_diff={len(diff_valid)}"
    
    # Calculate means
    same_mean = same_valid["avg_overlap_ratio"].mean()
    diff_mean = diff_valid["avg_overlap_ratio"].mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram settings (matching original)
    bins = np.linspace(0, 1, 50)
    
    # Plot histograms (Diff first, then Same to match original layering)
    ax.hist(diff_valid["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#FFB6C1', 
            label=f'Diff ({family_name} → Other derived)\nμ={diff_mean:.3f}', 
            edgecolor='white', linewidth=0.5)
    ax.hist(same_valid["avg_overlap_ratio"], bins=bins, alpha=0.6, 
            color='#90EE90', 
            label=f'Same ({family_name} → {family_name} derived)\nμ={same_mean:.3f}',
            edgecolor='white', linewidth=0.5)
    
    # Mean lines (dashed vertical lines)
    ax.axvline(diff_mean, color='red', linestyle='--', linewidth=2.5, alpha=0.9)
    ax.axvline(same_mean, color='green', linestyle='--', linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('avg_overlap_ratio', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_title(f'{family_name}: Distribution of avg_overlap_ratio (Same vs Diff){title_suffix}\n{n_info}', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim([0, 1.05])
    
    # Match original style more closely
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# ============================================================================
# Create all distribution plots
# ============================================================================

print("\n1. Creating side-by-side comparison plots...")
plot_distribution_comparison(same_qwen, diff_qwen, "Qwen", 
                            OUTPUT_DIR / "distribution_qwen_filtering_comparison.png")
plot_distribution_comparison(same_llama, diff_llama, "LLaMA", 
                            OUTPUT_DIR / "distribution_llama_filtering_comparison.png")

print("\n2. Creating individual distribution plots (original style)...")

# Before filtering
plot_single_distribution(same_qwen, diff_qwen, "Qwen", "before",
                        OUTPUT_DIR / "distribution_qwen_before_filtering.png")
plot_single_distribution(same_llama, diff_llama, "LLaMA", "before",
                        OUTPUT_DIR / "distribution_llama_before_filtering.png")

# After filtering
plot_single_distribution(same_qwen, diff_qwen, "Qwen", "after",
                        OUTPUT_DIR / "distribution_qwen_after_filtering.png")
plot_single_distribution(same_llama, diff_llama, "LLaMA", "after",
                        OUTPUT_DIR / "distribution_llama_after_filtering.png")

print("\n" + "=" * 80)
print("Distribution plots complete!")
print("=" * 80)
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("\nGenerated plots:")
print("  1. distribution_qwen_filtering_comparison.png - Side-by-side comparison")
print("  2. distribution_llama_filtering_comparison.png - Side-by-side comparison")
print("  3. distribution_qwen_before_filtering.png - Before filtering (original style)")
print("  4. distribution_qwen_after_filtering.png - After filtering (original style)")
print("  5. distribution_llama_before_filtering.png - Before filtering (original style)")
print("  6. distribution_llama_after_filtering.png - After filtering (original style)")
print("=" * 80)
