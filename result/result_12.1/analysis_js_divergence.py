import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
diff_qwen = pd.read_csv('lineage_bottomk_scores_diff_qwen.csv')
same_qwen = pd.read_csv('lineage_bottomk_scores_same_qwen.csv')

print(f"Same Qwen: {len(same_qwen)}")
print(f"Diff Qwen: {len(diff_qwen)}")

# Color configuration
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Select the most discriminative metric: avg_js_divergence
# Note: For JS divergence, Same should have smaller values (more similar)
# So we use 1 - js_divergence as "similarity score"

same_scores = 1 - same_qwen['avg_js_divergence'].values  # Convert to similarity
diff_scores = 1 - diff_qwen['avg_js_divergence'].values  # Convert to similarity

# Prepare data for ROC analysis
# Same = 1 (positive class), Diff = 0 (negative class)
y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
y_scores = np.concatenate([same_scores, diff_scores])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Find best threshold (Youden's J statistic)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

print(f"\n=== Using avg_js_divergence (converted to similarity) ===")
print(f"AUC: {roc_auc:.4f}")
print(f"Best threshold (similarity): {best_threshold:.4f}")
print(f"Corresponding JS divergence threshold: {1 - best_threshold:.4f}")

# Classify using best threshold
y_pred = (y_scores >= best_threshold).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate various metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Classification Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (TPR): {recall:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

print(f"\n=== Confusion Matrix ===")
print(f"TN (Diff correct): {tn}")
print(f"FP (Diff misclassified as Same): {fp}")
print(f"FN (Same misclassified as Diff): {fn}")
print(f"TP (Same correct): {tp}")

# ==================== Plotting ====================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Distribution plot (using original JS divergence)
ax1 = axes[0]
bins = np.linspace(0, 0.7, 30)
ax1.hist(same_qwen['avg_js_divergence'], bins=bins, alpha=0.7, 
         color=same_color, label=f'Same (n={len(same_qwen)})', density=True)
ax1.hist(diff_qwen['avg_js_divergence'], bins=bins, alpha=0.7, 
         color=diff_color, label=f'Diff (n={len(diff_qwen)})', density=True)
ax1.axvline(1 - best_threshold, color='black', linestyle='--', linewidth=2, 
            label=f'Threshold: {1 - best_threshold:.4f}')
ax1.axvline(same_qwen['avg_js_divergence'].mean(), color='green', linestyle=':', 
            linewidth=2, label=f'Same mean: {same_qwen["avg_js_divergence"].mean():.4f}')
ax1.axvline(diff_qwen['avg_js_divergence'].mean(), color='red', linestyle=':', 
            linewidth=2, label=f'Diff mean: {diff_qwen["avg_js_divergence"].mean():.4f}')
ax1.set_xlabel('avg_js_divergence', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Distribution of JS Divergence\nSame vs Diff', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. ROC curve
ax2 = axes[1]
ax2.plot(fpr, tpr, color='#4169E1', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax2.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
            label=f'Best threshold\n(FPR={fpr[best_idx]:.3f}, TPR={tpr[best_idx]:.3f})')
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve - JS Divergence', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Confusion matrix (row percentage)
ax3 = axes[2]

# Calculate row percentage
row_sums = np.array([[tn + fp], [fn + tp]])
cm_percent = np.array([[tn, fp], [fn, tp]]) / row_sums * 100

im = ax3.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)

# Add text annotations
labels = [
    [f'TN\n{tn}\n({cm_percent[0, 0]:.1f}%)', f'FP\n{fp}\n({cm_percent[0, 1]:.1f}%)'],
    [f'FN\n{fn}\n({cm_percent[1, 0]:.1f}%)', f'TP\n{tp}\n({cm_percent[1, 1]:.1f}%)']
]

for i in range(2):
    for j in range(2):
        color = 'white' if cm_percent[i, j] > 50 else 'black'
        ax3.text(j, i, labels[i][j], ha='center', va='center', 
                fontsize=12, fontweight='bold', color=color)

ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Pred: Diff', 'Pred: Same'])
ax3.set_yticklabels(['Actual: Diff', 'Actual: Same'])
ax3.set_title(f'Confusion Matrix\nAccuracy: {accuracy*100:.1f}%', fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax3, shrink=0.8)

plt.tight_layout()
plt.savefig('roc_analysis_js_divergence.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlot saved: roc_analysis_js_divergence.png")

# ==================== Save results ====================
results = {
    'metric': ['avg_js_divergence'],
    'auc': [roc_auc],
    'threshold_similarity': [best_threshold],
    'threshold_js_div': [1 - best_threshold],
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'specificity': [specificity],
    'f1_score': [f1],
    'same_mean': [same_qwen['avg_js_divergence'].mean()],
    'diff_mean': [diff_qwen['avg_js_divergence'].mean()]
}

results_df = pd.DataFrame(results)
results_df.to_csv('roc_analysis_results.csv', index=False)
print("Results saved: roc_analysis_results.csv")
