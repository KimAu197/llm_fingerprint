#!/usr/bin/env python3
"""
Plot overlap distribution with count of model pairs on Y-axis
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Read the CSV
with open('all_clean.csv', 'r') as f:
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

# Create the main plot
fig, ax = plt.subplots(figsize=(14, 7))

# Create histograms with bins
bins = np.linspace(0, 1, 40)  # 40 bins for better granularity

# Plot histograms
n_neg, bins_neg, patches_neg = ax.hist(negative_scores, bins=bins, alpha=0.7, 
        label=f'Negative (Different Family, n={len(negative_scores)})', 
        color='#FF6B6B', edgecolor='black', linewidth=0.8)

n_pos, bins_pos, patches_pos = ax.hist(positive_scores, bins=bins, alpha=0.75, 
        label=f'Positive (Same Family, n={len(positive_scores)})', 
        color='#4ECDC4', edgecolor='black', linewidth=0.8)

# Add statistics text
pos_mean = np.mean(positive_scores)
pos_median = np.median(positive_scores)
neg_mean = np.mean(negative_scores)
neg_median = np.median(negative_scores)

stats_text = f'Positive Statistics:\n'
stats_text += f'  Mean: {pos_mean:.4f}\n'
stats_text += f'  Median: {pos_median:.4f}\n'
stats_text += f'  Std: {np.std(positive_scores):.4f}\n\n'
stats_text += f'Negative Statistics:\n'
stats_text += f'  Mean: {neg_mean:.4f}\n'
stats_text += f'  Median: {neg_median:.4f}\n'
stats_text += f'  Std: {np.std(negative_scores):.4f}\n\n'
stats_text += f'Separation: {pos_mean/neg_mean:.1f}x'

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
        fontfamily='monospace', fontsize=10, linespacing=1.5)

# Add vertical lines for means
ax.axvline(pos_mean, color='#2C7A7B', linestyle='--', linewidth=2.5, 
           label=f'Positive Mean: {pos_mean:.3f}', alpha=0.8)
ax.axvline(neg_mean, color='#C53030', linestyle='--', linewidth=2.5, 
           label=f'Negative Mean: {neg_mean:.3f}', alpha=0.8)

# Labels and title
ax.set_xlabel('Overlap Ratio (Bottom-k Vocabulary)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Model Pairs', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Overlap Ratios: Positive vs Negative Samples\n' +
             'Model Fingerprinting Experiment (3 Base Families, k=2000)', 
             fontsize=15, fontweight='bold', pad=20)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Set x-axis limits
ax.set_xlim(0, 1)

# Make y-axis show integers only
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('overlap_distribution_counts.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: overlap_distribution_counts.png")

# Create a second plot: Side-by-side comparison
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Negative samples
ax1.hist(negative_scores, bins=bins, color='#FF6B6B', edgecolor='black', linewidth=0.8, alpha=0.8)
ax1.axvline(neg_mean, color='#C53030', linestyle='--', linewidth=2.5, label=f'Mean: {neg_mean:.3f}')
ax1.set_xlabel('Overlap Ratio', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Model Pairs', fontsize=12, fontweight='bold')
ax1.set_title(f'Negative Samples (Different Family)\nn = {len(negative_scores)}', 
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 1)
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Right: Positive samples
ax2.hist(positive_scores, bins=bins, color='#4ECDC4', edgecolor='black', linewidth=0.8, alpha=0.8)
ax2.axvline(pos_mean, color='#2C7A7B', linestyle='--', linewidth=2.5, label=f'Mean: {pos_mean:.3f}')
ax2.set_xlabel('Overlap Ratio', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Model Pairs', fontsize=12, fontweight='bold')
ax2.set_title(f'Positive Samples (Same Family)\nn = {len(positive_scores)}', 
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, 1)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig('overlap_distribution_side_by_side.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: overlap_distribution_side_by_side.png")

# Create a third plot: Box plot comparison
fig3, ax3 = plt.subplots(figsize=(10, 7))

data = [negative_scores, positive_scores]
labels = [f'Negative\n(n={len(negative_scores)})', f'Positive\n(n={len(positive_scores)})']
colors = ['#FF6B6B', '#4ECDC4']

bp = ax3.boxplot(data, labels=labels, patch_artist=True,
                 widths=0.6,
                 medianprops=dict(color='black', linewidth=2),
                 boxprops=dict(linewidth=1.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add mean markers
means = [neg_mean, pos_mean]
ax3.scatter([1, 2], means, marker='D', s=100, color='darkred', 
           zorder=3, label='Mean', edgecolors='black', linewidths=1.5)

# Labels
ax3.set_ylabel('Overlap Ratio', fontsize=13, fontweight='bold')
ax3.set_title('Box Plot Comparison: Overlap Ratios\nPositive vs Negative Samples', 
              fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.legend(fontsize=11)

# Add text annotations for means
for i, (mean, label) in enumerate(zip(means, ['Negative', 'Positive']), 1):
    ax3.text(i, mean, f'  {mean:.3f}', 
            verticalalignment='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('overlap_boxplot.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: overlap_boxplot.png")

plt.close('all')
print("\n✅ Done! Generated 3 plots:")
print("   1. overlap_distribution_counts.png - Overlapping histograms")
print("   2. overlap_distribution_side_by_side.png - Side-by-side comparison")
print("   3. overlap_boxplot.png - Box plot comparison")
