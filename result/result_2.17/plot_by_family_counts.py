#!/usr/bin/env python3
"""
Plot overlap distribution by family with count of model pairs on Y-axis
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

# Create figure with 3 subplots (one for each family)
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

base_models = ['Qwen/Qwen3-0.6B-Base', 'Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-1.5B']
pos_colors = ['#4ECDC4', '#95E1D3', '#FFD93D']
neg_color = '#FF6B6B'

bins = np.linspace(0, 1, 40)  # 40 bins

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
    n_neg, _, _ = ax.hist(neg_scores_fam, bins=bins, alpha=0.7, 
            label=f'Negative (n={len(neg_scores_fam)})', 
            color=neg_color, edgecolor='black', linewidth=0.7)
    
    # Then positive (in foreground) with distinct color
    n_pos, _, _ = ax.hist(pos_scores_fam, bins=bins, alpha=0.8, 
            label=f'Positive (n={len(pos_scores_fam)})', 
            color=pos_colors[idx], edgecolor='black', linewidth=0.7)
    
    # Stats
    if pos_scores_fam and neg_scores_fam:
        pos_m = np.mean(pos_scores_fam)
        neg_m = np.mean(neg_scores_fam)
        ax.axvline(pos_m, color=pos_colors[idx], linestyle='--', linewidth=2.5, alpha=0.9)
        ax.axvline(neg_m, color='#C53030', linestyle='--', linewidth=2.5, alpha=0.9)
        
        stats = f'Positive: {pos_m:.3f}\n'
        stats += f'Negative: {neg_m:.3f}\n'
        stats += f'Ratio: {pos_m/neg_m:.1f}x'
        ax.text(0.98, 0.95, stats, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                fontfamily='monospace', fontsize=10)
    
    # Format
    model_short = base_model.split('/')[-1]
    ax.set_ylabel('Number of Model Pairs', fontsize=11, fontweight='bold')
    ax.set_title(f'{model_short}', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

axes[-1].set_xlabel('Overlap Ratio', fontsize=12, fontweight='bold')
fig.suptitle('Overlap Distribution by Base Model Family\n(Y-axis: Number of Model Pairs)', 
              fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('overlap_by_family_counts.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: overlap_by_family_counts.png")

# Create individual plots for each family
for idx, base_model in enumerate(base_models):
    fig2, ax = plt.subplots(figsize=(12, 6))
    
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
    
    # Plot
    ax.hist(neg_scores_fam, bins=bins, alpha=0.7, 
            label=f'Negative (n={len(neg_scores_fam)})', 
            color=neg_color, edgecolor='black', linewidth=0.8)
    ax.hist(pos_scores_fam, bins=bins, alpha=0.8, 
            label=f'Positive (n={len(pos_scores_fam)})', 
            color=pos_colors[idx], edgecolor='black', linewidth=0.8)
    
    # Stats
    if pos_scores_fam and neg_scores_fam:
        pos_m = np.mean(pos_scores_fam)
        neg_m = np.mean(neg_scores_fam)
        pos_med = np.median(pos_scores_fam)
        neg_med = np.median(neg_scores_fam)
        
        ax.axvline(pos_m, color=pos_colors[idx], linestyle='--', linewidth=2.5, 
                  label=f'Positive Mean: {pos_m:.3f}', alpha=0.9)
        ax.axvline(neg_m, color='#C53030', linestyle='--', linewidth=2.5, 
                  label=f'Negative Mean: {neg_m:.3f}', alpha=0.9)
        
        stats_text = f'Positive Statistics:\n'
        stats_text += f'  Mean: {pos_m:.4f}\n'
        stats_text += f'  Median: {pos_med:.4f}\n'
        stats_text += f'  Std: {np.std(pos_scores_fam):.4f}\n\n'
        stats_text += f'Negative Statistics:\n'
        stats_text += f'  Mean: {neg_m:.4f}\n'
        stats_text += f'  Median: {neg_med:.4f}\n'
        stats_text += f'  Std: {np.std(neg_scores_fam):.4f}\n\n'
        stats_text += f'Separation: {pos_m/neg_m:.1f}x'
        
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'),
                fontfamily='monospace', fontsize=10)
    
    # Format
    model_short = base_model.split('/')[-1]
    ax.set_xlabel('Overlap Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Model Pairs', fontsize=12, fontweight='bold')
    ax.set_title(f'Overlap Distribution: {model_short}\n' + 
                 f'Positive (n={len(pos_scores_fam)}) vs Negative (n={len(neg_scores_fam)})', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filename = f'overlap_family_{idx+1}_{model_short.replace(".", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

plt.close('all')
print("\n✅ Done! Generated 4 plots:")
print("   1. overlap_by_family_counts.png - All 3 families together")
print("   2-4. Individual plots for each family")
