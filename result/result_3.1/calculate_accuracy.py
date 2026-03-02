#!/usr/bin/env python3
"""
Calculate accuracy: For each derived model, check if highest overlap is with its base model
"""
import csv
import numpy as np
from pathlib import Path

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

# Load base model information
base_model_csv = '/Users/kenzieluo/Desktop/columbia/course/model_lineage/llm_fingerprint/result/result_2.10/data/experiment_models_base_family_fixed.csv'
model_to_base = {}
base_models_set = set()

with open(base_model_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_id = row['model_id'].strip()
        base_model = row['effective_base_model'].strip()
        if model_id and base_model:
            model_to_base[model_id] = base_model
            base_models_set.add(base_model)

print(f"Loaded base model info for {len(model_to_base)} models")
print(f"Found {len(base_models_set)} unique base models\n")

# Calculate accuracy
print("=" * 80)
print("ACCURACY ANALYSIS: Does highest overlap match base model?")
print("=" * 80)
print()

correct = 0
total = 0
results = []

for i, model in enumerate(model_names):
    # Skip if this is a base model itself
    if model in base_models_set:
        continue
    
    # Skip if we don't have base model info
    if model not in model_to_base:
        continue
    
    base = model_to_base[model]
    
    # Check if base model is in the matrix
    if base not in model_names:
        continue
    
    # Get this model's row (excluding self-comparison)
    row = matrix[i, :].copy()
    row[i] = -999  # Mask out self-comparison
    
    # Find the model with highest overlap
    max_idx = np.argmax(row)
    max_overlap = row[max_idx]
    
    # Skip if all overlaps are negative (errors)
    if max_overlap < 0:
        continue
    
    best_match = model_names[max_idx]
    
    # Check if best match is the base model
    is_correct = (best_match == base)
    
    total += 1
    if is_correct:
        correct += 1
    
    results.append({
        'model': model,
        'base': base,
        'best_match': best_match,
        'overlap': max_overlap,
        'correct': is_correct
    })
    
    status = "✓" if is_correct else "✗"
    print(f"{status} {model}")
    print(f"   Base: {base}")
    print(f"   Best match: {best_match} (overlap: {max_overlap:.4f})")
    if not is_correct:
        # Show base model's overlap
        base_idx = model_names.index(base)
        base_overlap = matrix[i, base_idx]
        print(f"   Base overlap: {base_overlap:.4f} (rank: {np.sum(row > base_overlap) + 1})")
    print()

# Calculate overall accuracy
if total > 0:
    accuracy = correct / total * 100
else:
    accuracy = 0.0

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total derived models tested: {total}")
print(f"Correct matches (best = base): {correct}")
print(f"Incorrect matches: {total - correct}")
print(f"Accuracy: {accuracy:.2f}%")
print()

# Breakdown by family
print("=" * 80)
print("ACCURACY BY FAMILY")
print("=" * 80)

families = {}
for result in results:
    base = result['base']
    if base not in families:
        families[base] = {'correct': 0, 'total': 0}
    families[base]['total'] += 1
    if result['correct']:
        families[base]['correct'] += 1

for base in sorted(families.keys()):
    stats = families[base]
    fam_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"{base}:")
    print(f"  Correct: {stats['correct']}/{stats['total']} ({fam_acc:.1f}%)")

# Show incorrect matches
print()
print("=" * 80)
print("INCORRECT MATCHES (Detailed)")
print("=" * 80)

incorrect_results = [r for r in results if not r['correct']]
if incorrect_results:
    for result in incorrect_results:
        print(f"\n{result['model']}")
        print(f"  Expected: {result['base']}")
        print(f"  Got: {result['best_match']} (overlap: {result['overlap']:.4f})")
        
        # Show base overlap
        model_idx = model_names.index(result['model'])
        base_idx = model_names.index(result['base'])
        base_overlap = matrix[model_idx, base_idx]
        print(f"  Base overlap was: {base_overlap:.4f}")
        print(f"  Difference: {result['overlap'] - base_overlap:.4f}")
else:
    print("\nNo incorrect matches - perfect accuracy!")

# Save results to CSV
output_csv = 'accuracy_results.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'base_model', 'best_match', 'overlap_score', 'correct'])
    for result in results:
        writer.writerow([
            result['model'],
            result['base'],
            result['best_match'],
            f"{result['overlap']:.4f}",
            result['correct']
        ])

print(f"\n✓ Results saved to: {output_csv}")
