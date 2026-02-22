#!/usr/bin/env python3
"""
Plot overlap distribution for positive and negative samples
Similar to overlap_distribution_qwen.png
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Read the CSV
with open('all.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract all individual overlap scores
positive_scores = []
negative_scores = []

for row in rows:
    scores = json.loads(row['pair_scores_json'])
    if row['sample_type'] == 'positive':
        positive_scores.extend(scores)
    else:
        negative_scores.extend(scores)

print(f"Positive scores: {len(positive_scores)}")
print(f"Negative scores: {len(negative_scores)}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create histograms
bins = np.linspace(0, 1, 50)

ax.hist(negative_scores, bins=bins, alpha=0.7, label='Negative (Different Family)', 
        color='#FF6B6B', edgecolor='black', linewidth=0.5)
ax.hist(positive_scores, bins=bins, alpha=0.7, label='Positive (Same Family)', 
        color='#4ECDC4', edgecolor='black', linewidth=0.5)

# Add statistics text
pos_mean = np.mean(positive_scores)
pos_median = np.median(positive_scores)
neg_mean = np.mean(negative_scores)
neg_median = np.median(negative_scores)

stats_text = f'Positive (n={len(positive_scores)}):\n'
stats_text += f'  Mean: {pos_mean:.4f}\n'
stats_text += f'  Median: {pos_median:.4f}\n\n'
stats_text += f'Negative (n={len(negative_scores)}):\n'
stats_text += f'  Mean: {neg_mean:.4f}\n'
stats_text += f'  Median: {neg_median:.4f}\n\n'
stats_text += f'Separation: {pos_mean/neg_mean:.1f}x'

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontfamily='monospace', fontsize=9)

# Add vertical lines for means
ax.axvline(pos_mean, color='#4ECDC4', linestyle='--', linewidth=2, 
           label=f'Positive Mean ({pos_mean:.3f})')
ax.axvline(neg_mean, color='#FF6B6B', linestyle='--', linewidth=2, 
           label=f'Negative Mean ({neg_mean:.3f})')

# Labels and title
ax.set_xlabel('Overlap Ratio (Bottom-k Vocab)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Overlap Ratios: Positive vs Negative Samples\n' +
             'Model Fingerprinting Experiment (3 Base Families, k=2000)', 
             fontsize=14, fontweight='bold', pad=20)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Set x-axis limits
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('overlap_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: overlap_distribution.png")

# Create a second plot: by base model family
fig2, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

base_models = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-1.5B']
# Different colors for positive samples in each family
pos_colors = ['#4ECDC4', '#95E1D3', '#FFD93D']

for idx, base_model in enumerate(base_models):
    ax = axes[idx]
    
    # Get scores for this family
    pos_scores_fam = []
    neg_scores_fam = []
    
    for row in rows:
        if row['base_model'] == base_model:
            scores = json.loads(row['pair_scores_json'])
            if row['sample_type'] == 'positive':
                pos_scores_fam.extend(scores)
            else:
                neg_scores_fam.extend(scores)
    
    # Plot negative first (in background)
    ax.hist(neg_scores_fam, bins=bins, alpha=0.6, label='Negative', 
            color='#FF6B6B', edgecolor='black', linewidth=0.5)
    # Then positive (in foreground) with distinct color
    ax.hist(pos_scores_fam, bins=bins, alpha=0.7, label='Positive', 
            color=pos_colors[idx], edgecolor='black', linewidth=0.5)
    
    # Stats
    if pos_scores_fam and neg_scores_fam:
        pos_m = np.mean(pos_scores_fam)
        neg_m = np.mean(neg_scores_fam)
        ax.axvline(pos_m, color=pos_colors[idx], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(neg_m, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.8)
        
        stats = f'Pos: {pos_m:.3f} (n={len(pos_scores_fam)})\n'
        stats += f'Neg: {neg_m:.3f} (n={len(neg_scores_fam)})'
        ax.text(0.98, 0.95, stats, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontfamily='monospace', fontsize=8)
    
    # Format
    model_short = base_model.split('/')[-1]
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_short}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

axes[-1].set_xlabel('Overlap Ratio', fontsize=11, fontweight='bold')
fig2.suptitle('Overlap Distribution by Base Model Family', 
              fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('overlap_distribution_by_family.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: overlap_distribution_by_family.png")

plt.close('all')
print("\n✅ Done!")
