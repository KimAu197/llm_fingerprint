import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load previously calculated threshold
roc_results = pd.read_csv('/Users/kenzieluo/Desktop/columbia/course/model_linage/llm_fingerprint/result/result_12.15/wordlist/roc_analysis_results.csv')
qwen_threshold = roc_results[roc_results['Model'] == 'Qwen']['Best_Threshold'].values[0]

print(f"Using Qwen threshold: {qwen_threshold}")

# Load new data
diff = pd.read_csv('lineage_bottomk_overlap_qwen_diff.csv')
diff2 = pd.read_csv('lineage_bottomk_overlap_qwen_diff2.csv')
same = pd.read_csv('lineage_bottomk_overlap_qwen_same.csv')

# Filter invalid data
diff = diff[(diff['num_pairs'] > 0) & (diff['avg_overlap_ratio'] >= 0)]
diff2 = diff2[(diff2['num_pairs'] > 0) & (diff2['avg_overlap_ratio'] >= 0)]
same = same[(same['num_pairs'] > 0) & (same['avg_overlap_ratio'] >= 0)]

print(f"Same: {len(same)} samples")
print(f"Diff (TinyLLaMA derived): {len(diff)} samples")
print(f"Diff2 (Qwen2-0.5B derived): {len(diff2)} samples")

# Extract avg_overlap_ratio
same_scores = same['avg_overlap_ratio'].values
diff_scores = diff['avg_overlap_ratio'].values
diff2_scores = diff2['avg_overlap_ratio'].values

# Classify using threshold
# Above threshold -> Same (1), Below threshold -> Diff (0)
same_pred = (same_scores >= qwen_threshold).astype(int)
diff_pred = (diff_scores >= qwen_threshold).astype(int)
diff2_pred = (diff2_scores >= qwen_threshold).astype(int)

# Calculate classification results for each group
# Same group: true label = 1, predict = 1 is TP, predict = 0 is FN
same_tp = np.sum(same_pred == 1)  # Correctly predicted as Same
same_fn = np.sum(same_pred == 0)  # Incorrectly predicted as Diff

# Diff group: true label = 0, predict = 0 is TN, predict = 1 is FP
diff_tn = np.sum(diff_pred == 0)  # Correctly predicted as Diff
diff_fp = np.sum(diff_pred == 1)  # Incorrectly predicted as Same

# Diff2 group: true label = 0, predict = 0 is TN, predict = 1 is FP
diff2_tn = np.sum(diff2_pred == 0)  # Correctly predicted as Diff
diff2_fp = np.sum(diff2_pred == 1)  # Incorrectly predicted as Same

print(f"\n=== Classification Results (threshold = {qwen_threshold}) ===")
print(f"Same group: TP = {same_tp} ({100*same_tp/len(same):.1f}%), FN = {same_fn} ({100*same_fn/len(same):.1f}%)")
print(f"Diff group: TN = {diff_tn} ({100*diff_tn/len(diff):.1f}%), FP = {diff_fp} ({100*diff_fp/len(diff):.1f}%)")
print(f"Diff2 group: TN = {diff2_tn} ({100*diff2_tn/len(diff2):.1f}%), FP = {diff2_fp} ({100*diff2_fp/len(diff2):.1f}%)")

# ==================== Plot 3x2 Matrix ====================
fig, ax = plt.subplots(figsize=(10, 8))

# Create 3x2 matrix data
# Rows: Same, Diff, Diff2
# Columns: Predicted Diff (0), Predicted Same (1)
matrix_counts = np.array([
    [same_fn, same_tp],      # Same group
    [diff_tn, diff_fp],      # Diff group
    [diff2_tn, diff2_fp]     # Diff2 group
])

# Calculate row percentage
row_totals = matrix_counts.sum(axis=1, keepdims=True)
matrix_percent = matrix_counts / row_totals * 100

# Plot heatmap
im = ax.imshow(matrix_percent, cmap='Blues', vmin=0, vmax=100)

# Set labels
row_labels = [
    f'Same\n(Qwen2.5-0.5B derived)\nn={len(same)}',
    f'Diff\n(TinyLLaMA derived)\nn={len(diff)}',
    f'Diff2\n(Qwen2-0.5B derived)\nn={len(diff2)}'
]
col_labels = ['Predicted: Diff (0)', 'Predicted: Same (1)']

ax.set_xticks([0, 1])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticklabels(row_labels, fontsize=10)

# Add text annotations
cell_labels = [
    [f'FN\n{same_fn} ({matrix_percent[0, 0]:.1f}%)', f'TP\n{same_tp} ({matrix_percent[0, 1]:.1f}%)'],
    [f'TN\n{diff_tn} ({matrix_percent[1, 0]:.1f}%)', f'FP\n{diff_fp} ({matrix_percent[1, 1]:.1f}%)'],
    [f'TN\n{diff2_tn} ({matrix_percent[2, 0]:.1f}%)', f'FP\n{diff2_fp} ({matrix_percent[2, 1]:.1f}%)']
]

for i in range(3):
    for j in range(2):
        text_color = "white" if matrix_percent[i, j] > 50 else "black"
        ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                color=text_color, fontsize=12, fontweight='bold')

ax.set_title(f'Classification Results Using Qwen Threshold = {qwen_threshold}\n(avg_overlap_ratio)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Group', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, format='%.0f%%')
cbar.set_label('Percentage within Group', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrix_3groups.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nGenerated confusion_matrix_3groups.png")

# ==================== Overall Accuracy ====================
# Same group correct = TP, Diff group correct = TN, Diff2 group correct = TN
total_correct = same_tp + diff_tn + diff2_tn
total_samples = len(same) + len(diff) + len(diff2)
overall_accuracy = total_correct / total_samples * 100

print(f"\n=== Overall Results ===")
print(f"Total samples: {total_samples}")
print(f"Correctly classified: {total_correct}")
print(f"Overall accuracy: {overall_accuracy:.2f}%")

# Per-group accuracy
print(f"\nPer-group accuracy:")
print(f"  Same group (TPR): {100*same_tp/len(same):.2f}%")
print(f"  Diff group (TNR): {100*diff_tn/len(diff):.2f}%")
print(f"  Diff2 group (TNR): {100*diff2_tn/len(diff2):.2f}%")
