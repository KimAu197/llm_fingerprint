#!/usr/bin/env python3
"""
Plot overlap matrix heatmap
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read the CSV
csv_path = 'overlap_matrix.csv'

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Extract model names and matrix data
model_names = rows[0][1:]  # Skip first empty cell
matrix_data = []

for i in range(1, len(rows)):
    if not rows[i]:
        continue
    values = []
    for val in rows[i][1:]:  # Skip model name in first column
        try:
            values.append(float(val))
        except:
            values.append(0.0)
    matrix_data.append(values)

matrix = np.array(matrix_data)

print(f"Loaded matrix: {matrix.shape}")
print(f"Models: {len(model_names)}")

# Shorten model names for better display
def shorten_name(name):
    if '/' in name:
        parts = name.split('/')
        org = parts[0]
        model = parts[1]
        # Shorten organization names
        org_map = {
            'meta-llama': 'Meta',
            'Qwen': 'Q',
            'nvidia': 'NV',
            'unsloth': 'US',
            'allenai': 'AI',
            'NousResearch': 'Nous',
            'context-labs': 'Ctx',
            'inference-net': 'Inf',
            'Skywork': 'Sky',
            'Team-ACE': 'ACE',
            'acon96': 'acon',
            'Menlo': 'Menlo',
            'Gensyn': 'Gen',
        }
        org_short = org_map.get(org, org[:4])
        # Shorten model name if too long
        if len(model) > 30:
            model = model[:27] + '...'
        return f"{org_short}/{model}"
    return name[:35]

short_names = [shorten_name(name) for name in model_names]

print("Creating heatmap...")

# ========== Figure 1: Simple Heatmap ==========
fig, ax = plt.subplots(figsize=(24, 22))

# Create heatmap with custom colormap
mask = matrix == -1  # Mask error values
matrix_masked = np.ma.masked_where(mask, matrix)

# Use a diverging colormap (green = high overlap)
cmap = sns.color_palette("RdYlGn", as_cmap=True)
im = ax.imshow(matrix_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(short_names)))
ax.set_yticks(np.arange(len(short_names)))
ax.set_xticklabels(short_names, rotation=90, ha='right', fontsize=10)
ax.set_yticklabels(short_names, fontsize=10)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Overlap Score', rotation=270, labelpad=25, fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Add grid
ax.set_xticks(np.arange(len(short_names)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(short_names)) - 0.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

# Add title
ax.set_title(f'Model Overlap Matrix ({len(model_names)}x{len(model_names)})\nGreen = High Overlap, Red = Low Overlap', 
             fontsize=18, fontweight='bold', pad=20)

ax.set_xlabel('Model (Column)', fontsize=14, fontweight='bold')
ax.set_ylabel('Model (Row) - Used to generate fingerprint', fontsize=14, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'overlap_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Basic heatmap saved to: {output_path}")

# ========== Figure 2: Annotated Heatmap ==========
fig2, ax2 = plt.subplots(figsize=(26, 24))

# Create heatmap
im2 = ax2.imshow(matrix_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax2.set_xticks(np.arange(len(short_names)))
ax2.set_yticks(np.arange(len(short_names)))
ax2.set_xticklabels(short_names, rotation=90, ha='right', fontsize=10)
ax2.set_yticklabels(short_names, fontsize=10)

# Add text annotations for significant overlaps
print("Adding annotations...")
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i, j] >= 0:  # Skip errors
            if i == j:
                # Diagonal (self-comparison)
                text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="darkblue", 
                               fontsize=7, fontweight='bold')
            elif matrix[i, j] > 0.4:  # Very high overlap
                text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="darkgreen", 
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            elif matrix[i, j] > 0.2:  # High overlap
                text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", 
                               fontsize=7, fontweight='bold')
            elif matrix[i, j] > 0.1:  # Medium overlap
                text = ax2.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", 
                               fontsize=6)

# Add colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Overlap Score', rotation=270, labelpad=25, fontsize=14)
cbar2.ax.tick_params(labelsize=12)

# Add grid
ax2.set_xticks(np.arange(len(short_names)) - 0.5, minor=True)
ax2.set_yticks(np.arange(len(short_names)) - 0.5, minor=True)
ax2.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

# Add title with legend
title_text = f'Model Overlap Matrix with Annotations ({len(model_names)}x{len(model_names)})\n'
title_text += 'Values shown for overlap > 0.1 | Bold green box = overlap > 0.4'
ax2.set_title(title_text, fontsize=18, fontweight='bold', pad=20)

ax2.set_xlabel('Model (Column)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Model (Row) - Used to generate fingerprint', fontsize=14, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save annotated version
output_path2 = 'overlap_heatmap_annotated.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Annotated heatmap saved to: {output_path2}")

# ========== Statistics ==========
print(f"\n{'='*60}")
print("Matrix Statistics")
print(f"{'='*60}")
print(f"Size: {matrix.shape}")

# Filter out diagonal and negative values
off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
valid_values = off_diagonal[off_diagonal >= 0]

print(f"Min overlap (non-diagonal): {np.min(valid_values):.4f}")
print(f"Max overlap (non-diagonal): {np.max(valid_values):.4f}")
print(f"Mean overlap (non-diagonal): {np.mean(valid_values):.4f}")
print(f"Median overlap (non-diagonal): {np.median(valid_values):.4f}")

# Find top overlaps
print(f"\n{'='*60}")
print("Top 10 Model Pairs (non-self)")
print(f"{'='*60}")
pairs = []
for i in range(len(matrix)):
    for j in range(i+1, len(matrix[0])):
        if matrix[i, j] >= 0:
            pairs.append((matrix[i, j], model_names[i], model_names[j]))

pairs.sort(reverse=True)
for rank, (score, model1, model2) in enumerate(pairs[:10], 1):
    print(f"{rank:2d}. {score:.4f} | {model1} <-> {model2}")

print(f"\n✓ All done! Check the PNG files in this directory.")
