import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
diff_llama = pd.read_csv('lineage_bottomk_overlap_llama_diff.csv')
diff_qwen = pd.read_csv('lineage_bottomk_overlap_qwen_diff.csv')
same_llama = pd.read_csv('lineage_bottomk_overlap_llama_same.csv')
same_qwen = pd.read_csv('lineage_bottomk_overlap_qwen_same.csv')

# Filter out invalid data (num_pairs=0 or avg_overlap_ratio=-1)
diff_llama = diff_llama[(diff_llama['num_pairs'] > 0) & (diff_llama['avg_overlap_ratio'] >= 0)]
diff_qwen = diff_qwen[(diff_qwen['num_pairs'] > 0) & (diff_qwen['avg_overlap_ratio'] >= 0)]
same_llama = same_llama[(same_llama['num_pairs'] > 0) & (same_llama['avg_overlap_ratio'] >= 0)]
same_qwen = same_qwen[(same_qwen['num_pairs'] > 0) & (same_qwen['avg_overlap_ratio'] >= 0)]

print(f"LLaMA: Same={len(same_llama)}, Diff={len(diff_llama)}")
print(f"Qwen: Same={len(same_qwen)}, Diff={len(diff_qwen)}")

# Color configuration
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Use avg_overlap_ratio as the main metric
metric = 'avg_overlap_ratio'

# ==================== Analyze Qwen ====================
same_scores_qwen = same_qwen[metric].dropna().values
diff_scores_qwen = diff_qwen[metric].dropna().values

# Prepare data for ROC analysis
y_true_qwen = np.concatenate([np.ones(len(same_scores_qwen)), np.zeros(len(diff_scores_qwen))])
y_scores_qwen = np.concatenate([same_scores_qwen, diff_scores_qwen])

# ROC curve
fpr_qwen, tpr_qwen, thresholds_qwen = roc_curve(y_true_qwen, y_scores_qwen)
roc_auc_qwen = auc(fpr_qwen, tpr_qwen)

# Find best threshold (Youden's J statistic)
j_scores_qwen = tpr_qwen - fpr_qwen
best_idx_qwen = np.argmax(j_scores_qwen)
best_threshold_qwen = thresholds_qwen[best_idx_qwen]

print(f"\n=== Qwen ===")
print(f"AUC: {roc_auc_qwen:.4f}")
print(f"Best Threshold: {best_threshold_qwen:.4f}")

# ==================== Analyze LLaMA ====================
same_scores_llama = same_llama[metric].dropna().values
diff_scores_llama = diff_llama[metric].dropna().values

y_true_llama = np.concatenate([np.ones(len(same_scores_llama)), np.zeros(len(diff_scores_llama))])
y_scores_llama = np.concatenate([same_scores_llama, diff_scores_llama])

fpr_llama, tpr_llama, thresholds_llama = roc_curve(y_true_llama, y_scores_llama)
roc_auc_llama = auc(fpr_llama, tpr_llama)

j_scores_llama = tpr_llama - fpr_llama
best_idx_llama = np.argmax(j_scores_llama)
best_threshold_llama = thresholds_llama[best_idx_llama]

print(f"\n=== LLaMA ===")
print(f"AUC: {roc_auc_llama:.4f}")
print(f"Best Threshold: {best_threshold_llama:.4f}")

# ==================== 1. ROC Curve Comparison ====================
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(fpr_qwen, tpr_qwen, 'b-', linewidth=2, 
        label=f'Qwen (AUC = {roc_auc_qwen:.4f})')
ax.plot(fpr_llama, tpr_llama, 'r-', linewidth=2, 
        label=f'LLaMA (AUC = {roc_auc_llama:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')

ax.scatter([fpr_qwen[best_idx_qwen]], [tpr_qwen[best_idx_qwen]], 
           c='blue', s=100, zorder=5, edgecolors='white', linewidth=2)
ax.scatter([fpr_llama[best_idx_llama]], [tpr_llama[best_idx_llama]], 
           c='red', s=100, zorder=5, edgecolors='white', linewidth=2)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison: Qwen vs LLaMA\n(Bottom-k Overlap Method)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: roc_comparison.png")

# ==================== 2. Confusion Matrices (Row Percentage) ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Qwen Confusion Matrix
ax = axes[0]
y_pred_qwen = (y_scores_qwen >= best_threshold_qwen).astype(int)
cm_qwen = confusion_matrix(y_true_qwen, y_pred_qwen)
tn, fp, fn, tp = cm_qwen.ravel()

# Row percentage
row0_total = tn + fp
row1_total = fn + tp
tn_p = tn / row0_total * 100 if row0_total > 0 else 0
fp_p = fp / row0_total * 100 if row0_total > 0 else 0
fn_p = fn / row1_total * 100 if row1_total > 0 else 0
tp_p = tp / row1_total * 100 if row1_total > 0 else 0

cm_percent_qwen = np.array([[tn_p, fp_p], [fn_p, tp_p]])

im = ax.imshow(cm_percent_qwen, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
ax.set_title(f'Qwen Confusion Matrix\nThreshold = {best_threshold_qwen:.4f}', 
             fontsize=11, fontweight='bold')

labels = [
    [f'TN\n{tn} ({tn_p:.1f}%)', f'FP\n{fp} ({fp_p:.1f}%)'],
    [f'FN\n{fn} ({fn_p:.1f}%)', f'TP\n{tp} ({tp_p:.1f}%)']
]

for i in range(2):
    for j in range(2):
        color = 'white' if cm_percent_qwen[i, j] > 50 else 'black'
        ax.text(j, i, labels[i][j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)

ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Diff', 'Same'])
ax.set_yticklabels(['Diff', 'Same'])

# Calculate metrics
accuracy_qwen = (tp + tn) / (tp + tn + fp + fn) * 100
precision_qwen = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall_qwen = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1_qwen = 2 * precision_qwen * recall_qwen / (precision_qwen + recall_qwen) if (precision_qwen + recall_qwen) > 0 else 0
specificity_qwen = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

print(f"\n=== Qwen Metrics ===")
print(f"Accuracy: {accuracy_qwen:.2f}%")
print(f"Precision: {precision_qwen:.2f}%")
print(f"Recall (TPR): {recall_qwen:.2f}%")
print(f"Specificity (TNR): {specificity_qwen:.2f}%")
print(f"F1 Score: {f1_qwen/100:.4f}")

# LLaMA Confusion Matrix
ax = axes[1]
y_pred_llama = (y_scores_llama >= best_threshold_llama).astype(int)
cm_llama = confusion_matrix(y_true_llama, y_pred_llama)
tn, fp, fn, tp = cm_llama.ravel()

row0_total = tn + fp
row1_total = fn + tp
tn_p = tn / row0_total * 100 if row0_total > 0 else 0
fp_p = fp / row0_total * 100 if row0_total > 0 else 0
fn_p = fn / row1_total * 100 if row1_total > 0 else 0
tp_p = tp / row1_total * 100 if row1_total > 0 else 0

cm_percent_llama = np.array([[tn_p, fp_p], [fn_p, tp_p]])

im = ax.imshow(cm_percent_llama, interpolation='nearest', cmap='Oranges', vmin=0, vmax=100)
ax.set_title(f'LLaMA Confusion Matrix\nThreshold = {best_threshold_llama:.4f}', 
             fontsize=11, fontweight='bold')

labels = [
    [f'TN\n{tn} ({tn_p:.1f}%)', f'FP\n{fp} ({fp_p:.1f}%)'],
    [f'FN\n{fn} ({fn_p:.1f}%)', f'TP\n{tp} ({tp_p:.1f}%)']
]

for i in range(2):
    for j in range(2):
        color = 'white' if cm_percent_llama[i, j] > 50 else 'black'
        ax.text(j, i, labels[i][j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)

ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Diff', 'Same'])
ax.set_yticklabels(['Diff', 'Same'])

# Calculate metrics
accuracy_llama = (tp + tn) / (tp + tn + fp + fn) * 100
precision_llama = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
recall_llama = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
f1_llama = 2 * precision_llama * recall_llama / (precision_llama + recall_llama) if (precision_llama + recall_llama) > 0 else 0
specificity_llama = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0

print(f"\n=== LLaMA Metrics ===")
print(f"Accuracy: {accuracy_llama:.2f}%")
print(f"Precision: {precision_llama:.2f}%")
print(f"Recall (TPR): {recall_llama:.2f}%")
print(f"Specificity (TNR): {specificity_llama:.2f}%")
print(f"F1 Score: {f1_llama/100:.4f}")

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: confusion_matrix_comparison.png")

# ==================== Save Results to CSV ====================
results = pd.DataFrame({
    'Model': ['Qwen', 'LLaMA'],
    'AUC': [roc_auc_qwen, roc_auc_llama],
    'Best_Threshold': [best_threshold_qwen, best_threshold_llama],
    'Accuracy': [accuracy_qwen, accuracy_llama],
    'Precision': [precision_qwen, precision_llama],
    'Recall': [recall_qwen, recall_llama],
    'Specificity': [specificity_qwen, specificity_llama],
    'F1_Score': [f1_qwen/100, f1_llama/100],
    'Same_Count': [len(same_scores_qwen), len(same_scores_llama)],
    'Diff_Count': [len(diff_scores_qwen), len(diff_scores_llama)]
})

results.to_csv('roc_analysis_results.csv', index=False)
print("\nSaved: roc_analysis_results.csv")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
