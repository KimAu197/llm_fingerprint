#!/usr/bin/env python3
"""
Extract top-2 highest overlaps for each model (excluding self)
Groups models by score buckets - all models with highest score in top1, all with 2nd highest in top2
"""
import csv
import numpy as np

# Read the overlap matrix CSV
csv_path = 'overlap_matrix.csv'

with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

# Extract model names and matrix data
model_names = rows[0][1:]
matrix_data = []

for i in range(1, len(rows)):
    if not rows[i]:
        continue
    values = []
    for val in rows[i][1:]:
        try:
            values.append(float(val))
        except:
            values.append(0.0)
    matrix_data.append(values)

matrix = np.array(matrix_data)

print(f"Loaded matrix: {matrix.shape}")
print(f"Processing top-2 overlap buckets for each model...\n")

# Extract top-2 buckets for each model
results = []

for i, model in enumerate(model_names):
    # Get this model's row
    row = matrix[i, :].copy()
    
    # Mask out self-comparison (diagonal)
    row[i] = -999
    
    # Get unique scores (sorted descending)
    unique_scores = np.unique(row)[::-1]  # Descending order
    unique_scores = unique_scores[unique_scores >= 0]  # Remove negative values
    
    # Get top-1 score and all models with that score
    if len(unique_scores) > 0:
        top1_score = unique_scores[0]
        top1_indices = np.where(row == top1_score)[0]
        top1_models = [model_names[idx] for idx in top1_indices]
        top1_models_str = " | ".join(top1_models)
        top1_count = len(top1_models)
    else:
        top1_score = 0.0
        top1_models_str = ""
        top1_count = 0
    
    # Get top-2 score and all models with that score
    if len(unique_scores) > 1:
        top2_score = unique_scores[1]
        top2_indices = np.where(row == top2_score)[0]
        top2_models = [model_names[idx] for idx in top2_indices]
        top2_models_str = " | ".join(top2_models)
        top2_count = len(top2_models)
    else:
        top2_score = 0.0
        top2_models_str = ""
        top2_count = 0
    
    results.append({
        'model': model,
        'top1_models': top1_models_str,
        'top1_count': top1_count,
        'top1_overlap': top1_score,
        'top2_models': top2_models_str,
        'top2_count': top2_count,
        'top2_overlap': top2_score,
    })
    
    print(f"{model}")
    print(f"  Top-1 ({top1_count} models, score={top1_score:.4f}):")
    for m in top1_models[:5] if top1_count > 0 else []:
        print(f"    - {m}")
    if top1_count > 5:
        print(f"    ... and {top1_count - 5} more")
    
    print(f"  Top-2 ({top2_count} models, score={top2_score:.4f}):")
    for m in top2_models[:5] if top2_count > 0 else []:
        print(f"    - {m}")
    if top2_count > 5:
        print(f"    ... and {top2_count - 5} more")
    print()

# Save to CSV
output_csv = 'top2_overlaps_grouped.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'top1_models', 'top1_count', 'top1_overlap', 'top2_models', 'top2_count', 'top2_overlap'])
    
    for result in results:
        writer.writerow([
            result['model'],
            result['top1_models'],
            result['top1_count'],
            f"{result['top1_overlap']:.4f}",
            result['top2_models'],
            result['top2_count'],
            f"{result['top2_overlap']:.4f}",
        ])

print(f"✓ Results saved to: {output_csv}")
