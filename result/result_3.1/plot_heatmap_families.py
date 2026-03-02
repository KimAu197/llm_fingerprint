#!/usr/bin/env python3
"""
Plot overlap matrix heatmap with base model annotations
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Read the overlap matrix CSV
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

# Load base model information
base_model_csv = '/Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/result/result_2.10/data/experiment_models_base_family_fixed.csv'
model_to_base = {}
base_models_set = set()

try:
    with open(base_model_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_id = row['model_id'].strip()
            base_model = row['effective_base_model'].strip()
            if model_id and base_model:
                model_to_base[model_id] = base_model
                base_models_set.add(base_model)
    print(f"Loaded base model info for {len(model_to_base)} models")
    print(f"Found {len(base_models_set)} unique base models")
except Exception as e:
    print(f"Warning: Could not load base model info: {e}")
    print("Continuing without base model annotations...")

# Assign colors to each base model family
base_model_colors = {}
color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
for i, base in enumerate(sorted(base_models_set)):
    base_model_colors[base] = color_palette[i % len(color_palette)]

# Check which families have their base model in the matrix
families_with_base = set()
for model in model_names:
    if model in base_models_set:
        families_with_base.add(model)

# Determine color for each model (RED if family has base model present)
model_colors = []
for model in model_names:
    if model in model_to_base:
        base = model_to_base[model]
        # Check if this family's base model is in the matrix
        if base in families_with_base:
            model_colors.append('red')
        else:
            model_colors.append(base_model_colors.get(base, '#999999'))
    else:
        # Check if this model IS a base model
        if model in base_models_set:
            model_colors.append('red')  # Base models are red
        else:
            model_colors.append('#999999')  # Gray for unknown

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

# Add base model indicator to names
display_names = []
for i, (full_name, short_name) in enumerate(zip(model_names, short_names)):
    if full_name in model_to_base:
        base = model_to_base[full_name]
        base_short = shorten_name(base)
        display_names.append(f"{short_name} → {base_short}")
    elif full_name in base_models_set:
        display_names.append(f"{short_name} [BASE]")
    else:
        display_names.append(short_name)

print("Creating heatmap with base model annotations...")

# ========== Figure: Heatmap with Base Model Colors ==========
fig, ax = plt.subplots(figsize=(28, 26))

# Create heatmap
mask = matrix == -1
matrix_masked = np.ma.masked_where(mask, matrix)
cmap = sns.color_palette("RdYlGn", as_cmap=True)
im = ax.imshow(matrix_masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(display_names)))
ax.set_yticks(np.arange(len(display_names)))
ax.set_xticklabels(display_names, rotation=90, ha='right', fontsize=10)
ax.set_yticklabels(display_names, fontsize=10)

# Color the tick labels based on base model (RED if family has base in matrix)
for i, (tick, color) in enumerate(zip(ax.get_xticklabels(), model_colors)):
    tick.set_color(color)
    tick.set_fontweight('bold' if color == 'red' else 'normal')
for i, (tick, color) in enumerate(zip(ax.get_yticklabels(), model_colors)):
    tick.set_color(color)
    tick.set_fontweight('bold' if color == 'red' else 'normal')

# Add text annotations for values
print("Adding value annotations...")
for i in range(len(matrix)):
    for j in range(len(matrix[0])):
        if matrix[i, j] >= 0:  # Skip errors
            if i == j:
                # Diagonal (self-comparison)
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="darkblue", 
                               fontsize=6, fontweight='bold')
            elif matrix[i, j] > 0.4:  # Very high overlap
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="white", 
                               fontsize=7, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='darkgreen', 
                                       edgecolor='white', linewidth=1.5, alpha=0.9))
            elif matrix[i, j] > 0.2:  # High overlap
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", 
                               fontsize=6, fontweight='bold')
            elif matrix[i, j] > 0.1:  # Medium overlap
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", 
                               fontsize=5)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Overlap Score', rotation=270, labelpad=25, fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Add grid
ax.set_xticks(np.arange(len(display_names)) - 0.5, minor=True)
ax.set_yticks(np.arange(len(display_names)) - 0.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

# Create legend for base model families
legend_handles = []
for base in sorted(base_models_set):
    color = base_model_colors[base]
    base_short = shorten_name(base)
    patch = mpatches.Patch(color=color, label=f'{base_short}')
    legend_handles.append(patch)

# Add legend
if legend_handles:
    legend = ax.legend(handles=legend_handles, 
                      loc='upper left', 
                      bbox_to_anchor=(1.08, 1.0),
                      title='Base Model Families',
                      fontsize=10,
                      title_fontsize=12,
                      frameon=True)

# Add title
title_text = f'Model Overlap Matrix ({len(model_names)}x{len(model_names)})\n'
title_text += 'RED text = Family with Base Model in matrix | [BASE] = Base Model | → = Derived from'
ax.set_title(title_text, fontsize=17, fontweight='bold', pad=20)

ax.set_xlabel('Model (Column)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model (Row) - Used to generate fingerprint', fontsize=13, fontweight='bold')

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'overlap_heatmap_with_families.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Heatmap with base model families saved to: {output_path}")

# ========== Statistics grouped by family ==========
print(f"\n{'='*60}")
print("Base Model Family Analysis")
print(f"{'='*60}")

# Group models by family
families = {}
for model in model_names:
    if model in model_to_base:
        base = model_to_base[model]
    elif model in base_models_set:
        base = model  # It's a base model itself
    else:
        base = "Unknown"
    
    if base not in families:
        families[base] = []
    families[base].append(model)

for base in sorted(families.keys()):
    models = families[base]
    print(f"\n{base} ({len(models)} models):")
    for model in sorted(models):
        is_base = "[BASE]" if model in base_models_set else ""
        print(f"  - {model} {is_base}")

# ========== Top overlaps within same family ==========
print(f"\n{'='*60}")
print("Top 10 Same-Family Overlaps (excluding self-comparison)")
print(f"{'='*60}")

same_family_pairs = []
for i in range(len(matrix)):
    for j in range(i+1, len(matrix[0])):
        if matrix[i, j] >= 0:
            model_i = model_names[i]
            model_j = model_names[j]
            
            # Get base models
            base_i = model_to_base.get(model_i, model_i if model_i in base_models_set else None)
            base_j = model_to_base.get(model_j, model_j if model_j in base_models_set else None)
            
            # Check if same family
            if base_i and base_j and (base_i == base_j or model_i == base_j or model_j == base_i):
                same_family_pairs.append((matrix[i, j], model_i, model_j, base_i))

same_family_pairs.sort(reverse=True)
for rank, (score, model1, model2, base) in enumerate(same_family_pairs[:10], 1):
    base_short = shorten_name(base)
    print(f"{rank:2d}. {score:.4f} | {model1} <-> {model2}")
    print(f"     Family: {base_short}")

print(f"\n✓ All done! Check the PNG file in this directory.")
