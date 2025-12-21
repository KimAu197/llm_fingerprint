import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
diff_llama = pd.read_csv('lineage_bottomk_scores_diff_llama.csv')
diff_qwen = pd.read_csv('lineage_bottomk_scores_diff_qwen.csv')
same_llama = pd.read_csv('lineage_bottomk_scores_same_llama.csv')
same_qwen = pd.read_csv('lineage_bottomk_scores_same_qwen.csv')

print(f"LLaMA: Same={len(same_llama)}, Diff={len(diff_llama)}")
print(f"Qwen: Same={len(same_qwen)}, Diff={len(diff_qwen)}")

# Color configuration
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Use avg_score as the main metric
metric = 'avg_score'

# ==================== Analyze Qwen ====================
same_scores_qwen = same_qwen[metric].dropna().values
diff_scores_qwen = diff_qwen[metric].dropna().values

print(f"\n=== Qwen ===")
print(f"Same mean: {same_scores_qwen.mean():.4f}, std: {same_scores_qwen.std():.4f}")
print(f"Diff mean: {diff_scores_qwen.mean():.4f}, std: {diff_scores_qwen.std():.4f}")

# ROC Analysis - Qwen
y_true_qwen = np.concatenate([np.ones(len(same_scores_qwen)), np.zeros(len(diff_scores_qwen))])
y_scores_qwen = np.concatenate([same_scores_qwen, diff_scores_qwen])

fpr_qwen, tpr_qwen, thresholds_qwen = roc_curve(y_true_qwen, y_scores_qwen)
roc_auc_qwen = auc(fpr_qwen, tpr_qwen)

# Find best threshold (Youden's J)
j_scores_qwen = tpr_qwen - fpr_qwen
best_idx_qwen = np.argmax(j_scores_qwen)
best_threshold_qwen = thresholds_qwen[best_idx_qwen]

print(f"AUC: {roc_auc_qwen:.4f}")
print(f"Best threshold: {best_threshold_qwen:.4f}")

# Confusion Matrix - Qwen
y_pred_qwen = (y_scores_qwen >= best_threshold_qwen).astype(int)
cm_qwen = confusion_matrix(y_true_qwen, y_pred_qwen)
tn_q, fp_q, fn_q, tp_q = cm_qwen.ravel()

accuracy_qwen = (tp_q + tn_q) / (tp_q + tn_q + fp_q + fn_q)
precision_qwen = tp_q / (tp_q + fp_q) if (tp_q + fp_q) > 0 else 0
recall_qwen = tp_q / (tp_q + fn_q) if (tp_q + fn_q) > 0 else 0
specificity_qwen = tn_q / (tn_q + fp_q) if (tn_q + fp_q) > 0 else 0
f1_qwen = 2 * precision_qwen * recall_qwen / (precision_qwen + recall_qwen) if (precision_qwen + recall_qwen) > 0 else 0

print(f"Accuracy: {accuracy_qwen:.2%}")
print(f"Precision: {precision_qwen:.2%}, Recall: {recall_qwen:.2%}")
print(f"Specificity: {specificity_qwen:.2%}, F1: {f1_qwen:.4f}")

# ==================== Analyze LLaMA ====================
same_scores_llama = same_llama[metric].dropna().values
diff_scores_llama = diff_llama[metric].dropna().values

print(f"\n=== LLaMA ===")
print(f"Same mean: {same_scores_llama.mean():.4f}, std: {same_scores_llama.std():.4f}")
print(f"Diff mean: {diff_scores_llama.mean():.4f}, std: {diff_scores_llama.std():.4f}")

# ROC Analysis - LLaMA
y_true_llama = np.concatenate([np.ones(len(same_scores_llama)), np.zeros(len(diff_scores_llama))])
y_scores_llama = np.concatenate([same_scores_llama, diff_scores_llama])

fpr_llama, tpr_llama, thresholds_llama = roc_curve(y_true_llama, y_scores_llama)
roc_auc_llama = auc(fpr_llama, tpr_llama)

# Find best threshold
j_scores_llama = tpr_llama - fpr_llama
best_idx_llama = np.argmax(j_scores_llama)
best_threshold_llama = thresholds_llama[best_idx_llama]

print(f"AUC: {roc_auc_llama:.4f}")
print(f"Best threshold: {best_threshold_llama:.4f}")

# Confusion Matrix - LLaMA
y_pred_llama = (y_scores_llama >= best_threshold_llama).astype(int)
cm_llama = confusion_matrix(y_true_llama, y_pred_llama)
tn_l, fp_l, fn_l, tp_l = cm_llama.ravel()

accuracy_llama = (tp_l + tn_l) / (tp_l + tn_l + fp_l + fn_l)
precision_llama = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0
recall_llama = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0
specificity_llama = tn_l / (tn_l + fp_l) if (tn_l + fp_l) > 0 else 0
f1_llama = 2 * precision_llama * recall_llama / (precision_llama + recall_llama) if (precision_llama + recall_llama) > 0 else 0

print(f"Accuracy: {accuracy_llama:.2%}")
print(f"Precision: {precision_llama:.2%}, Recall: {recall_llama:.2%}")
print(f"Specificity: {specificity_llama:.2%}, F1: {f1_llama:.4f}")

# ==================== 1. Distribution Plot - Qwen ====================
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(0, max(same_scores_qwen.max(), diff_scores_qwen.max()) + 0.1, 30)

ax.hist(same_scores_qwen, bins=bins, alpha=0.7, color=same_color, 
        label=f'Same (Qwen -> Qwen derived)\nmu={same_scores_qwen.mean():.3f}', edgecolor='white')
ax.hist(diff_scores_qwen, bins=bins, alpha=0.7, color=diff_color, 
        label=f'Diff (Qwen -> LLaMA derived)\nmu={diff_scores_qwen.mean():.3f}', edgecolor='white')

# Only draw mean lines, not threshold
ax.axvline(same_scores_qwen.mean(), color='green', linestyle=':', linewidth=2)
ax.axvline(diff_scores_qwen.mean(), color='red', linestyle=':', linewidth=2)

ax.set_xlabel('avg_score', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Qwen: Distribution of avg_score (Same vs Diff)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_qwen.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== 2. Distribution Plot - LLaMA ====================
fig, ax = plt.subplots(figsize=(10, 6))

bins = np.linspace(0, max(same_scores_llama.max(), diff_scores_llama.max()) + 0.1, 30)

ax.hist(same_scores_llama, bins=bins, alpha=0.7, color=same_color, 
        label=f'Same (LLaMA -> LLaMA derived)\nmu={same_scores_llama.mean():.3f}', edgecolor='white')
ax.hist(diff_scores_llama, bins=bins, alpha=0.7, color=diff_color, 
        label=f'Diff (LLaMA -> Qwen derived)\nmu={diff_scores_llama.mean():.3f}', edgecolor='white')

# Only draw mean lines, not threshold
ax.axvline(same_scores_llama.mean(), color='green', linestyle=':', linewidth=2)
ax.axvline(diff_scores_llama.mean(), color='red', linestyle=':', linewidth=2)

ax.set_xlabel('avg_score', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('LLaMA: Distribution of avg_score (Same vs Diff)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_llama.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== 3. ROC Curves ====================
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(fpr_qwen, tpr_qwen, color='#FF6B6B', linewidth=2, 
        label=f'Qwen (AUC = {roc_auc_qwen:.4f})')
ax.plot(fpr_llama, tpr_llama, color='#4ECDC4', linewidth=2, 
        label=f'LLaMA (AUC = {roc_auc_llama:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

ax.scatter([fpr_qwen[best_idx_qwen]], [tpr_qwen[best_idx_qwen]], 
           s=100, c='#FF6B6B', edgecolors='black', zorder=5)
ax.scatter([fpr_llama[best_idx_llama]], [tpr_llama[best_idx_llama]], 
           s=100, c='#4ECDC4', edgecolors='black', zorder=5)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve: Qwen vs LLaMA (Text-based Fingerprint)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== 4. Confusion Matrices ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Qwen Confusion Matrix (row percentage)
ax = axes[0]
# Calculate row percentage
row0_total_q = tn_q + fp_q
row1_total_q = fn_q + tp_q
cm_percent_q = np.array([
    [tn_q/row0_total_q*100, fp_q/row0_total_q*100],
    [fn_q/row1_total_q*100, tp_q/row1_total_q*100]
])

im = ax.imshow(cm_percent_q, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
ax.set_title(f'Qwen Confusion Matrix\nThreshold={best_threshold_qwen:.4f}, AUC={roc_auc_qwen:.4f}', 
             fontsize=11, fontweight='bold')

labels = [
    [f'TN\n{tn_q} ({cm_percent_q[0,0]:.1f}%)', f'FP\n{fp_q} ({cm_percent_q[0,1]:.1f}%)'],
    [f'FN\n{fn_q} ({cm_percent_q[1,0]:.1f}%)', f'TP\n{tp_q} ({cm_percent_q[1,1]:.1f}%)']
]

for i in range(2):
    for j in range(2):
        text_color = 'white' if cm_percent_q[i,j] > 50 else 'black'
        ax.text(j, i, labels[i][j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Diff', 'Same'], fontsize=11)
ax.set_yticklabels(['Diff', 'Same'], fontsize=11)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

# LLaMA Confusion Matrix (row percentage)
ax = axes[1]
row0_total_l = tn_l + fp_l
row1_total_l = fn_l + tp_l
cm_percent_l = np.array([
    [tn_l/row0_total_l*100, fp_l/row0_total_l*100],
    [fn_l/row1_total_l*100, tp_l/row1_total_l*100]
])

im = ax.imshow(cm_percent_l, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
ax.set_title(f'LLaMA Confusion Matrix\nThreshold={best_threshold_llama:.4f}, AUC={roc_auc_llama:.4f}', 
             fontsize=11, fontweight='bold')

labels = [
    [f'TN\n{tn_l} ({cm_percent_l[0,0]:.1f}%)', f'FP\n{fp_l} ({cm_percent_l[0,1]:.1f}%)'],
    [f'FN\n{fn_l} ({cm_percent_l[1,0]:.1f}%)', f'TP\n{tp_l} ({cm_percent_l[1,1]:.1f}%)']
]

for i in range(2):
    for j in range(2):
        text_color = 'white' if cm_percent_l[i,j] > 50 else 'black'
        ax.text(j, i, labels[i][j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Diff', 'Same'], fontsize=11)
ax.set_yticklabels(['Diff', 'Same'], fontsize=11)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Save results to CSV
results = pd.DataFrame({
    'Model': ['Qwen', 'LLaMA'],
    'AUC': [roc_auc_qwen, roc_auc_llama],
    'Threshold': [best_threshold_qwen, best_threshold_llama],
    'Accuracy': [accuracy_qwen, accuracy_llama],
    'Precision': [precision_qwen, precision_llama],
    'Recall': [recall_qwen, recall_llama],
    'Specificity': [specificity_qwen, specificity_llama],
    'F1': [f1_qwen, f1_llama],
    'Same_Mean': [same_scores_qwen.mean(), same_scores_llama.mean()],
    'Diff_Mean': [diff_scores_qwen.mean(), diff_scores_llama.mean()]
})
results.to_csv('roc_analysis_results.csv', index=False)

print("\n=== Analysis Complete ===")
print("Generated files:")
print("- distribution_qwen.png")
print("- distribution_llama.png")
print("- roc_comparison.png")
print("- confusion_matrix_comparison.png")
print("- roc_analysis_results.csv")
