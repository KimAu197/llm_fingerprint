"""
Compare overlap distributions between Qwen and TinyLlama
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_distributions():
    # Read data
    df_qwen = pd.read_csv('lineage_bottomk_text_same_qwen.csv')
    df_llama = pd.read_csv('lineage_bottomk_text_same_llama.csv')
    
    # Filter valid data
    df_qwen = df_qwen[df_qwen['avg_overlap_ratio'] >= 0]
    df_llama = df_llama[df_llama['avg_overlap_ratio'] >= 0]
    
    print("\n" + "="*80)
    print("Qwen vs TinyLlama Overlap Comparison Analysis")
    print("="*80 + "\n")
    
    # Basic statistics
    print("Basic Statistics:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Qwen2.5-0.5B':>20} {'TinyLlama-1.1B':>20} {'Difference':>15}")
    print("-" * 80)
    
    qwen_mean = df_qwen['avg_overlap_ratio'].mean()
    llama_mean = df_llama['avg_overlap_ratio'].mean()
    print(f"{'Mean':<20} {qwen_mean:>20.4f} {llama_mean:>20.4f} {llama_mean/qwen_mean:>14.2f}x")
    
    qwen_median = df_qwen['avg_overlap_ratio'].median()
    llama_median = df_llama['avg_overlap_ratio'].median()
    print(f"{'Median':<20} {qwen_median:>20.4f} {llama_median:>20.4f} {llama_median/qwen_median:>14.2f}x")
    
    qwen_std = df_qwen['avg_overlap_ratio'].std()
    llama_std = df_llama['avg_overlap_ratio'].std()
    print(f"{'Std Dev':<20} {qwen_std:>20.4f} {llama_std:>20.4f}")
    
    qwen_min = df_qwen['avg_overlap_ratio'].min()
    llama_min = df_llama['avg_overlap_ratio'].min()
    print(f"{'Min':<20} {qwen_min:>20.4f} {llama_min:>20.4f}")
    
    qwen_max = df_qwen['avg_overlap_ratio'].max()
    llama_max = df_llama['avg_overlap_ratio'].max()
    print(f"{'Max':<20} {qwen_max:>20.4f} {llama_max:>20.4f}")
    
    print(f"{'Sample Size':<20} {len(df_qwen):>20d} {len(df_llama):>20d}")
    
    # Distribution comparison
    print("\n" + "="*80)
    print("Overlap Distribution Comparison:")
    print("="*80 + "\n")
    
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    df_qwen['group'] = pd.cut(df_qwen['avg_overlap_ratio'], bins=bins, labels=labels)
    df_llama['group'] = pd.cut(df_llama['avg_overlap_ratio'], bins=bins, labels=labels)
    
    print(f"{'Overlap Range':<15} {'Qwen Count':>12} {'Qwen %':>12} {'Llama Count':>12} {'Llama %':>12}")
    print("-" * 80)
    
    for label in labels:
        qwen_count = (df_qwen['group'] == label).sum()
        qwen_pct = qwen_count / len(df_qwen) * 100
        llama_count = (df_llama['group'] == label).sum()
        llama_pct = llama_count / len(df_llama) * 100
        print(f"{label:<15} {qwen_count:>12d} {qwen_pct:>11.1f}% {llama_count:>12d} {llama_pct:>11.1f}%")
    
    # Key findings
    print("\n" + "="*80)
    print("Key Findings:")
    print("="*80 + "\n")
    
    qwen_low = (df_qwen['avg_overlap_ratio'] < 0.3).sum() / len(df_qwen) * 100
    llama_low = (df_llama['avg_overlap_ratio'] < 0.3).sum() / len(df_llama) * 100
    print(f"1. Low overlap (<0.3) percentage:")
    print(f"   - Qwen:  {qwen_low:.1f}% (heavy modification dominant)")
    print(f"   - Llama: {llama_low:.1f}% (few cross-domain)")
    
    qwen_high = (df_qwen['avg_overlap_ratio'] > 0.7).sum() / len(df_qwen) * 100
    llama_high = (df_llama['avg_overlap_ratio'] > 0.7).sum() / len(df_llama) * 100
    print(f"\n2. High overlap (>0.7) percentage:")
    print(f"   - Qwen:  {qwen_high:.1f}% (few light fine-tuning)")
    print(f"   - Llama: {llama_high:.1f}% (many light fine-tuning)")
    
    print(f"\n3. TinyLlama's mean overlap is {llama_mean/qwen_mean:.2f}x that of Qwen")
    print(f"   This indicates TinyLlama community prefers conservative fine-tuning strategies")
    
    print("\n" + "="*80)
    
    # Create visualization
    create_comparison_plot(df_qwen, df_llama)

def create_comparison_plot(df_qwen, df_llama):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Qwen vs TinyLlama Overlap Distribution Comparison', fontsize=16, fontweight='bold')
    
    # 1. Histogram comparison
    ax1 = axes[0, 0]
    ax1.hist(df_qwen['avg_overlap_ratio'], bins=20, alpha=0.6, label='Qwen2.5-0.5B', color='blue', edgecolor='black')
    ax1.hist(df_llama['avg_overlap_ratio'], bins=20, alpha=0.6, label='TinyLlama-1.1B', color='orange', edgecolor='black')
    ax1.set_xlabel('Overlap Ratio', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Histogram Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot comparison
    ax2 = axes[0, 1]
    data = [df_qwen['avg_overlap_ratio'], df_llama['avg_overlap_ratio']]
    bp = ax2.boxplot(data, tick_labels=['Qwen2.5-0.5B', 'TinyLlama-1.1B'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    ax2.set_ylabel('Overlap Ratio', fontsize=11)
    ax2.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. CDF comparison
    ax3 = axes[1, 0]
    qwen_sorted = np.sort(df_qwen['avg_overlap_ratio'])
    llama_sorted = np.sort(df_llama['avg_overlap_ratio'])
    qwen_cdf = np.arange(1, len(qwen_sorted) + 1) / len(qwen_sorted)
    llama_cdf = np.arange(1, len(llama_sorted) + 1) / len(llama_sorted)
    ax3.plot(qwen_sorted, qwen_cdf, label='Qwen2.5-0.5B', linewidth=2, color='blue')
    ax3.plot(llama_sorted, llama_cdf, label='TinyLlama-1.1B', linewidth=2, color='orange')
    ax3.set_xlabel('Overlap Ratio', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Grouped bar chart
    ax4 = axes[1, 1]
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    
    df_qwen['group'] = pd.cut(df_qwen['avg_overlap_ratio'], bins=bins, labels=labels)
    df_llama['group'] = pd.cut(df_llama['avg_overlap_ratio'], bins=bins, labels=labels)
    
    qwen_counts = [((df_qwen['group'] == label).sum() / len(df_qwen) * 100) for label in labels]
    llama_counts = [((df_llama['group'] == label).sum() / len(df_llama) * 100) for label in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax4.bar(x - width/2, qwen_counts, width, label='Qwen2.5-0.5B', color='lightblue', edgecolor='black')
    ax4.bar(x + width/2, llama_counts, width, label='TinyLlama-1.1B', color='lightsalmon', edgecolor='black')
    
    ax4.set_xlabel('Overlap Range', fontsize=11)
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title('Distribution by Range', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qwen_vs_llama_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: qwen_vs_llama_comparison.png")

if __name__ == "__main__":
    compare_distributions()
