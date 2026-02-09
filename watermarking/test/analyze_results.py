"""
analyze_results.py

Analyze the results from base family fingerprinting experiment.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV."""
    df = pd.read_csv(csv_path)
    
    # Parse JSON scores
    df['pair_scores'] = df['pair_scores_json'].apply(json.loads)
    
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()
    
    positive = df[df['sample_type'] == 'positive']
    negative = df[df['sample_type'] == 'negative']
    
    print("📊 Sample Counts:")
    print(f"  Positive (same family): {len(positive)}")
    print(f"  Negative (different family): {len(negative)}")
    print(f"  Total: {len(df)}")
    print()
    
    print("📈 Overlap Ratio Statistics:")
    print()
    print("Positive Samples (same family):")
    print(f"  Mean:   {positive['avg_overlap_ratio'].mean():.4f}")
    print(f"  Median: {positive['avg_overlap_ratio'].median():.4f}")
    print(f"  Std:    {positive['avg_overlap_ratio'].std():.4f}")
    print(f"  Min:    {positive['avg_overlap_ratio'].min():.4f}")
    print(f"  Max:    {positive['avg_overlap_ratio'].max():.4f}")
    print()
    
    print("Negative Samples (different family):")
    print(f"  Mean:   {negative['avg_overlap_ratio'].mean():.4f}")
    print(f"  Median: {negative['avg_overlap_ratio'].median():.4f}")
    print(f"  Std:    {negative['avg_overlap_ratio'].std():.4f}")
    print(f"  Min:    {negative['avg_overlap_ratio'].min():.4f}")
    print(f"  Max:    {negative['avg_overlap_ratio'].max():.4f}")
    print()
    
    # Separation
    pos_mean = positive['avg_overlap_ratio'].mean()
    neg_mean = negative['avg_overlap_ratio'].mean()
    separation = pos_mean - neg_mean
    
    print("🎯 Discrimination Ability:")
    print(f"  Separation (pos_mean - neg_mean): {separation:.4f}")
    
    # Check for overlap in distributions
    pos_min = positive['avg_overlap_ratio'].min()
    neg_max = negative['avg_overlap_ratio'].max()
    
    if pos_min > neg_max:
        print(f"  ✅ Perfect separation! (pos_min={pos_min:.4f} > neg_max={neg_max:.4f})")
    else:
        overlap_count = sum((negative['avg_overlap_ratio'] > pos_min) & 
                           (negative['avg_overlap_ratio'] < positive['avg_overlap_ratio'].max()))
        print(f"  ⚠️  Some overlap exists (pos_min={pos_min:.4f} <= neg_max={neg_max:.4f})")
        print(f"      Overlapping samples: {overlap_count}")
    print()


def print_per_family_stats(df: pd.DataFrame):
    """Print statistics per base model family."""
    print("=" * 80)
    print("PER-FAMILY STATISTICS")
    print("=" * 80)
    print()
    
    positive = df[df['sample_type'] == 'positive']
    
    for base_model in positive['base_model'].unique():
        family_positive = positive[positive['base_model'] == base_model]
        
        print(f"Family: {base_model}")
        print(f"  Derivatives tested: {len(family_positive)}")
        print(f"  Avg overlap: {family_positive['avg_overlap_ratio'].mean():.4f}")
        print(f"  Range: [{family_positive['avg_overlap_ratio'].min():.4f}, "
              f"{family_positive['avg_overlap_ratio'].max():.4f}]")
        print()


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    positive = df[df['sample_type'] == 'positive']
    negative = df[df['sample_type'] == 'negative']
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Distribution plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(positive['avg_overlap_ratio'], bins=20, alpha=0.7, label='Positive (same family)', 
             color='green', edgecolor='black')
    plt.hist(negative['avg_overlap_ratio'], bins=20, alpha=0.7, label='Negative (different family)', 
             color='red', edgecolor='black')
    plt.xlabel('Average Overlap Ratio', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Overlap Ratio Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [positive['avg_overlap_ratio'], negative['avg_overlap_ratio']]
    bp = plt.boxplot(data_to_plot, labels=['Positive\n(same family)', 'Negative\n(different family)'],
                     patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.ylabel('Average Overlap Ratio', fontsize=12)
    plt.title('Overlap Ratio Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'overlap_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot: {plot_path}")
    
    # 3. Per-family comparison
    plt.figure(figsize=(14, 6))
    
    base_models = positive['base_model'].unique()
    family_data = []
    family_labels = []
    
    for base_model in base_models:
        family_positive = positive[positive['base_model'] == base_model]
        family_data.append(family_positive['avg_overlap_ratio'].values)
        # Shorten label for readability
        short_label = base_model.split('/')[-1][:20]
        family_labels.append(short_label)
    
    bp = plt.boxplot(family_data, labels=family_labels, patch_artist=True, notch=True)
    
    # Color boxes
    colors = plt.cm.Set3(range(len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Average Overlap Ratio', fontsize=12)
    plt.xlabel('Base Model Family', fontsize=12)
    plt.title('Overlap Ratio by Base Model Family', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'per_family_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot: {plot_path}")
    
    # 4. Scatter plot: positive vs negative for each base model
    plt.figure(figsize=(10, 8))
    
    for base_model in base_models:
        family_positive = positive[positive['base_model'] == base_model]
        # Get negative samples from this base model
        family_negative = negative[negative['base_model'] == base_model]
        
        if len(family_negative) > 0:
            pos_mean = family_positive['avg_overlap_ratio'].mean()
            neg_mean = family_negative['avg_overlap_ratio'].mean()
            
            short_label = base_model.split('/')[-1][:15]
            plt.scatter(neg_mean, pos_mean, s=100, alpha=0.6, label=short_label)
    
    # Draw diagonal line
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='y=x')
    
    plt.xlabel('Negative (different family) - Mean Overlap', fontsize=12)
    plt.ylabel('Positive (same family) - Mean Overlap', fontsize=12)
    plt.title('Positive vs Negative Overlap by Family', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'positive_vs_negative_scatter.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot: {plot_path}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze fingerprinting experiment results")
    parser.add_argument('--input', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots (default: same as input)')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    df = load_results(args.input)
    print(f"Loaded {len(df)} results")
    print()
    
    # Print statistics
    print_summary_statistics(df)
    print_per_family_stats(df)
    
    # Create visualizations
    if args.output_dir is None:
        args.output_dir = Path(args.input).parent
    
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    print()
    
    create_visualizations(df, args.output_dir)
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Results analyzed from: {args.input}")
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
