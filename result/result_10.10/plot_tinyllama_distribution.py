import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
diff = pd.read_csv('lineage_scores_tinyllama_diff.csv')
same1 = pd.read_csv('lineage_scores_llama.csv')
same2 = pd.read_csv('lineage_scores_tinyllama_same2.csv')

# Merge same data
same_combined = pd.concat([same1, same2], ignore_index=True)
same_combined = same_combined.dropna(subset=['score_mean'])

# Diff data
diff_combined = diff.dropna(subset=['score_mean'])

print(f"Diff count: {len(diff_combined)}")
print(f"Same merged count: {len(same_combined)}")

# Set plot style - white background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['pal_k_mean', 'lev_sim_mean', 'lcs_ratio_mean', 'score_mean']
titles = ['PAL K Mean', 'Levenshtein Similarity', 'LCS Ratio', 'Overall Score']

# Color configuration - matching target style
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

for ax, metric, title in zip(axes.flatten(), metrics, titles):
    # Get data
    diff_data = diff_combined[metric].dropna()
    same_data = same_combined[metric].dropna()
    
    # Calculate bins
    all_data = pd.concat([diff_data, same_data])
    bins = np.linspace(0, all_data.max() * 1.05, 25)
    
    # Plot histogram - Same in green, Diff in red
    ax.hist(same_data, bins=bins, alpha=0.7, label=f'Same (TinyLlama derived)', 
            color=same_color, edgecolor='darkgreen', linewidth=0.5)
    ax.hist(diff_data, bins=bins, alpha=0.7, label=f'Diff (TinyLlama base -> Qwen derived)', 
            color=diff_color, edgecolor='darkred', linewidth=0.5)
    
    ax.set_xlabel(title, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Distribution of {title}: Same vs Diff (TinyLlama)', fontsize=13, fontweight='bold')
    
    # Calculate mean
    same_mean = same_data.mean()
    diff_mean = diff_data.mean()
    
    # Add mean lines
    ax.axvline(same_mean, color='green', linestyle='--', linewidth=2)
    ax.axvline(diff_mean, color='darkred', linestyle='--', linewidth=2)
    
    # Legend
    ax.legend([f'Same (TinyLlama derived)', f'Diff (TinyLlama base -> Qwen derived)', 
               f'Same mean: {same_mean:.4f}', f'Diff mean: {diff_mean:.4f}'],
              loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('tinyllama_distribution_same_vs_diff.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n=== Statistical Summary ===")
print("\nDiff group:")
print(diff_combined[metrics].describe())
print("\nSame group:")
print(same_combined[metrics].describe())
