"""
plot_text_metrics_distribution.py

Plot distribution histograms for same vs diff text similarity metrics.
Filters out -1 values (failed model loads).
Generates 4 separate figures for: avg_pal_k, avg_lev_sim, avg_lcs_ratio, avg_score

Usage:
    python plot_text_metrics_distribution.py \
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


METRICS = [
    ("avg_pal_k", "PAL_k (Prefix Agreement)"),
    ("avg_lev_sim", "Levenshtein Similarity"),
    ("avg_lcs_ratio", "LCS Ratio"),
    ("avg_score", "Combined Score"),
]


def load_and_filter(csv_path: str, metric: str) -> np.ndarray:
    """Load CSV and filter out -1 (failed) values for a specific metric."""
    df = pd.read_csv(csv_path)
    values = df[metric].values
    valid = values[values >= 0]
    print(f"[{csv_path}] {metric}: Total={len(values)}, Valid={len(valid)}, Filtered={len(values) - len(valid)}")
    return valid


def plot_single_metric(
    same_values: np.ndarray,
    diff_values: np.ndarray,
    metric_name: str,
    metric_display: str,
    output_path: Path,
    model_name: str,
    same_label: str,
    diff_label: str,
):
    """Plot a single metric distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_values = np.concatenate([same_values, diff_values])
    min_val = max(0, all_values.min() - 0.05)
    max_val = min(1, all_values.max() + 0.05)
    bins = np.linspace(min_val, max_val, 31)
    
    ax.hist(same_values, bins=bins, alpha=0.6, label=same_label, 
            color='#2ecc71', edgecolor='white', linewidth=0.5)
    ax.hist(diff_values, bins=bins, alpha=0.6, label=diff_label, 
            color='#e74c3c', edgecolor='white', linewidth=0.5)
    
    ax.axvline(same_values.mean(), color='#27ae60', linestyle='--', linewidth=2, 
               label=f'Same mean: {same_values.mean():.4f}')
    ax.axvline(diff_values.mean(), color='#c0392b', linestyle='--', linewidth=2, 
               label=f'Diff mean: {diff_values.mean():.4f}')
    
    ax.set_xlabel(metric_display, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {metric_display}: Same vs Diff ({model_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[saved] {output_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot text similarity metrics distribution")
    parser.add_argument("--same_csv", type=str, required=True, help="CSV for same-lineage models")
    parser.add_argument("--diff_csv", type=str, required=True, help="CSV for diff-lineage models")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for figures")
    parser.add_argument("--model_name", type=str, default="model", help="Model name for title")
    parser.add_argument("--same_label", type=str, default=None, help="Legend label for same")
    parser.add_argument("--diff_label", type=str, default=None, help="Legend label for diff")
    parser.add_argument("--metrics", type=str, nargs="*", default=None, 
                        help="Specific metrics to plot (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    same_label = args.same_label or f"Same ({args.model_name} derived)"
    diff_label = args.diff_label or f"Diff ({args.model_name} base → other derived)"
    
    metrics_to_plot = METRICS
    if args.metrics:
        metrics_to_plot = [(m, d) for m, d in METRICS if m in args.metrics]
    
    print("=" * 60)
    print("Loading data and generating plots...")
    print("=" * 60)
    
    for metric_name, metric_display in metrics_to_plot:
        same_values = load_and_filter(args.same_csv, metric_name)
        diff_values = load_and_filter(args.diff_csv, metric_name)
        
        print(f"\n{metric_display}:")
        print(f"  Same: mean={same_values.mean():.4f}, std={same_values.std():.4f}")
        print(f"  Diff: mean={diff_values.mean():.4f}, std={diff_values.std():.4f}")
        
        output_path = output_dir / f"dist_{metric_name}_{args.model_name.lower()}.png"
        plot_single_metric(
            same_values, diff_values, metric_name, metric_display,
            output_path, args.model_name, same_label, diff_label
        )
    
    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

