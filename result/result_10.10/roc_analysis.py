import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Load data
diff1 = pd.read_csv('lineage_scores_qwen2_0.5_diff2.csv')
diff2 = pd.read_csv('lineage_scores_qwen2_0.5b_diff.csv')
same1 = pd.read_csv('lineage_scores_qwen2_0.5b_same.csv')
same2 = pd.read_csv('lineage_scores_qwen2_0.5b_same2.csv')

# Merge data
diff_combined = pd.concat([diff1, diff2], ignore_index=True)
diff_combined = diff_combined.dropna(subset=['score_mean'])

same_combined = pd.concat([same1, same2], ignore_index=True)
same_combined = same_combined.dropna(subset=['score_mean'])

print(f"Same sample count: {len(same_combined)}")
print(f"Diff sample count: {len(diff_combined)}")

# Prepare data for ROC analysis
# Same = 1 (positive class, same lineage), Diff = 0 (negative class, different lineage)
same_scores = same_combined['score_mean'].values
diff_scores = diff_combined['score_mean'].values

y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
y_scores = np.concatenate([same_scores, diff_scores])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find best threshold (Youden's J statistic: maximize TPR - FPR)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"\n=== ROC Analysis Results ===")
print(f"AUC: {roc_auc:.4f}")
print(f"Best threshold (Youden's J): {best_threshold:.4f}")
print(f"At best threshold: TPR = {tpr[best_idx]:.4f}, FPR = {fpr[best_idx]:.4f}")

# Plot ROC curve
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
ax1 = axes[0]
ax1.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
ax1.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5, 
            label=f'Best threshold = {best_threshold:.4f}')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve: Same vs Diff (Qwen2-0.5B)', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Classify using best threshold
y_pred = (y_scores >= best_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== Confusion Matrix (threshold = {best_threshold:.4f}) ===")
print(f"True Positive (TP): {tp} - Same samples correctly identified as Same")
print(f"False Positive (FP): {fp} - Diff samples incorrectly identified as Same")
print(f"True Negative (TN): {tn} - Diff samples correctly identified as Diff")
print(f"False Negative (FN): {fn} - Same samples incorrectly identified as Diff")

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n=== Performance Metrics ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (TPR): {recall:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
ax2 = axes[1]
im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
ax2.set_title(f'Confusion Matrix (threshold = {best_threshold:.4f})', fontsize=13, fontweight='bold')

# Add text annotations
thresh_cm = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax2.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh_cm else "black",
                fontsize=16)

ax2.set_ylabel('True Label', fontsize=12)
ax2.set_xlabel('Predicted Label', fontsize=12)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Diff (0)', 'Same (1)'])
ax2.set_yticklabels(['Diff (0)', 'Same (1)'])

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.savefig('roc_analysis_qwen2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# Detailed analysis of misclassified samples
print("\n=== Misclassified Sample Analysis ===")

# Create full dataframe
same_combined_with_label = same_combined.copy()
same_combined_with_label['true_label'] = 'same'
same_combined_with_label['predicted'] = (same_combined_with_label['score_mean'] >= best_threshold).map({True: 'same', False: 'diff'})

diff_combined_with_label = diff_combined.copy()
diff_combined_with_label['true_label'] = 'diff'
diff_combined_with_label['predicted'] = (diff_combined_with_label['score_mean'] >= best_threshold).map({True: 'same', False: 'diff'})

# False Negatives (Same misclassified as Diff)
fn_samples = same_combined_with_label[same_combined_with_label['predicted'] == 'diff']
print(f"\nFalse Negatives (FN) - Same misclassified as Diff ({len(fn_samples)} samples):")
if len(fn_samples) > 0:
    print(fn_samples[['base_model_name', 'score_mean']].sort_values('score_mean').to_string())

# False Positives (Diff misclassified as Same)
fp_samples = diff_combined_with_label[diff_combined_with_label['predicted'] == 'same']
print(f"\nFalse Positives (FP) - Diff misclassified as Same ({len(fp_samples)} samples):")
if len(fp_samples) > 0:
    print(fp_samples[['base_model_name', 'score_mean']].sort_values('score_mean', ascending=False).to_string())

# Save analysis results
results_summary = pd.DataFrame({
    'Metric': ['AUC', 'Best Threshold', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 
               'TP', 'FP', 'TN', 'FN'],
    'Value': [roc_auc, best_threshold, accuracy, precision, recall, specificity, f1,
              tp, fp, tn, fn]
})
results_summary.to_csv('roc_analysis_results.csv', index=False)
print("\nResults saved to roc_analysis_results.csv")
