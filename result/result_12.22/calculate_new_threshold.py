import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
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

# Combine all Diff data (Diff + Diff2)
all_diff_scores = np.concatenate([diff_scores, diff2_scores])

print(f"\nCombined Diff total: {len(all_diff_scores)}")

# Prepare data for ROC analysis
# Same = 1, Diff = 0
y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(all_diff_scores))])
y_scores = np.concatenate([same_scores, all_diff_scores])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find best threshold (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[optimal_idx]

print(f"\n=== ROC Analysis Results (Same vs All Diff) ===")
print(f"AUC: {roc_auc:.4f}")
print(f"Best threshold (Youden's J): {best_threshold:.4f}")

# Calculate classification results using best threshold
y_pred = (y_scores >= best_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / len(y_true) * 100
precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nOverall metrics:")
print(f"  Accuracy: {accuracy:.2f}%")
print(f"  Precision: {precision:.2f}%")
print(f"  Recall (TPR): {recall:.2f}%")
print(f"  Specificity (TNR): {specificity:.2f}%")
print(f"  F1 Score: {f1:.4f}")

# ==================== 1. ROC Curve ====================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, zorder=5, 
            label=f'Best threshold = {best_threshold:.4f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Same vs All Diff\n(Diff = TinyLLaMA derived + Qwen2-0.5B derived)', 
          fontsize=12, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('roc_curve_new.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 2. 3x2 Matrix with New Threshold ====================
# Calculate classification results for each group
same_pred = (same_scores >= best_threshold).astype(int)
diff_pred = (diff_scores >= best_threshold).astype(int)
diff2_pred = (diff2_scores >= best_threshold).astype(int)

same_tp = np.sum(same_pred == 1)
same_fn = np.sum(same_pred == 0)
diff_tn = np.sum(diff_pred == 0)
diff_fp = np.sum(diff_pred == 1)
diff2_tn = np.sum(diff2_pred == 0)
diff2_fp = np.sum(diff2_pred == 1)

print(f"\n=== Per-group Classification Results (new threshold = {best_threshold:.4f}) ===")
print(f"Same group: TP = {same_tp} ({100*same_tp/len(same):.1f}%), FN = {same_fn} ({100*same_fn/len(same):.1f}%)")
print(f"Diff group: TN = {diff_tn} ({100*diff_tn/len(diff):.1f}%), FP = {diff_fp} ({100*diff_fp/len(diff):.1f}%)")
print(f"Diff2 group: TN = {diff2_tn} ({100*diff2_tn/len(diff2):.1f}%), FP = {diff2_fp} ({100*diff2_fp/len(diff2):.1f}%)")

# Plot 3x2 matrix
fig, ax = plt.subplots(figsize=(10, 8))

matrix_counts = np.array([
    [same_fn, same_tp],
    [diff_tn, diff_fp],
    [diff2_tn, diff2_fp]
])

row_totals = matrix_counts.sum(axis=1, keepdims=True)
matrix_percent = matrix_counts / row_totals * 100

im = ax.imshow(matrix_percent, cmap='Blues', vmin=0, vmax=100)

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

ax.set_title(f'Classification Results Using NEW Threshold = {best_threshold:.4f}\n(avg_overlap_ratio)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Group', fontsize=12)

cbar = plt.colorbar(im, ax=ax, format='%.0f%%')
cbar.set_label('Percentage within Group', fontsize=10)

plt.tight_layout()
plt.savefig('confusion_matrix_3groups_new_threshold.png', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 3. Compare Old vs New Threshold ====================
old_threshold = 0.0082

print(f"\n=== Old vs New Threshold Comparison ===")
print(f"Old threshold (Qwen only): {old_threshold}")
print(f"New threshold (All 3 groups): {best_threshold:.4f}")

# Save results
results_df = pd.DataFrame({
    'Threshold_Type': ['Old (Qwen only)', 'New (All 3 groups)'],
    'Threshold': [old_threshold, best_threshold],
    'AUC': [0.9884, roc_auc],  # Old AUC from previous results
    'Same_TPR': [97.1, 100*same_tp/len(same)],
    'Diff_TNR': [100.0, 100*diff_tn/len(diff)],
    'Diff2_TNR': [5.0, 100*diff2_tn/len(diff2)]
})
results_df.to_csv('threshold_comparison.csv', index=False)
print("\nResults saved to threshold_comparison.csv")

print("\nGenerated figures: roc_curve_new.png, confusion_matrix_3groups_new_threshold.png")
