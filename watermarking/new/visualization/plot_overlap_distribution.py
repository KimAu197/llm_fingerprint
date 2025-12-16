"""
plot_overlap_distribution.py

Plot distribution histograms for same vs diff overlap ratios.
Filters out -1 values (failed model loads).

Usage:
    python plot_overlap_distribution.py \
        --same_csv path/to/same.csv \
        --diff_csv path/to/diff.csv \
        --output_dir results/ \
        --model_name qwen
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_and_filter(csv_path: str) -> np.ndarray:
    """Load CSV and filter out -1 (failed) values."""
    df = pd.read_csv(csv_path)
    values = df["avg_overlap_ratio"].values
    valid = values[values >= 0]
    print(f"[{csv_path}] Total: {len(values)}, Valid: {len(valid)}, Filtered: {len(values) - len(valid)}")
    return valid


def plot_distribution(
    same_values: np.ndarray,
    diff_values: np.ndarray,
    output_path: Path,
    model_name: str,
    same_label: str,
    diff_label: str,
):
    """Plot overlap distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 31)
    
    ax.hist(same_values, bins=bins, alpha=0.6, label=same_label, 
            color='#2ecc71', edgecolor='white', linewidth=0.5)
    ax.hist(diff_values, bins=bins, alpha=0.6, label=diff_label, 
            color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    ax.axvline(same_values.mean(), color='#27ae60', linestyle='--', linewidth=2, 
               label=f'Same mean: {same_values.mean():.3f}')
    ax.axvline(diff_values.mean(), color='#c0392b', linestyle='--', linewidth=2, 
               label=f'Diff mean: {diff_values.mean():.3f}')
    
    ax.set_xlabel('Average Overlap Ratio', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of Bottom-k Subspace Overlap Ratio ({model_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[saved] {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot overlap ratio distribution")
    parser.add_argument("--same_csv", type=str, required=True, help="CSV for same-lineage models")
    parser.add_argument("--diff_csv", type=str, required=True, help="CSV for diff-lineage models")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for figures")
    parser.add_argument("--model_name", type=str, default="model", help="Model name for title")
    parser.add_argument("--same_label", type=str, default=None, help="Legend label for same")
    parser.add_argument("--diff_label", type=str, default=None, help="Legend label for diff")
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    same_values = load_and_filter(args.same_csv)
    diff_values = load_and_filter(args.diff_csv)
    
    print(f"\n=== Statistics ===")
    print(f"Same: mean={same_values.mean():.4f}, std={same_values.std():.4f}")
    print(f"Diff: mean={diff_values.mean():.4f}, std={diff_values.std():.4f}")
    
    same_label = args.same_label or f"Same ({args.model_name} derived)"
    diff_label = args.diff_label or f"Diff ({args.model_name} base → other derived)"
    
    output_path = output_dir / f"overlap_distribution_{args.model_name.lower()}.png"
    plot_distribution(same_values, diff_values, output_path, args.model_name, same_label, diff_label)


if __name__ == "__main__":
    main()

