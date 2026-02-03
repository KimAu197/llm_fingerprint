"""
Visualize complete threshold analysis including False Positives.

Creates multiple plots:
1. TPR vs FPR (ROC-style) - Before vs After filtering
2. Accuracy vs Threshold - Before vs After filtering
3. Precision vs Recall - Before vs After filtering
4. F1-Score vs Threshold - Before vs After filtering
5. Confusion Matrix heatmaps at optimal thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
ROC_DIR = SCRIPT_DIR / "roc_analysis"
OUTPUT_DIR = ROC_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
qwen_before = pd.read_csv(ROC_DIR / "complete_threshold_table_qwen_before.csv")
qwen_after = pd.read_csv(ROC_DIR / "complete_threshold_table_qwen_after.csv")
llama_before = pd.read_csv(ROC_DIR / "complete_threshold_table_llama_before.csv")
llama_after = pd.read_csv(ROC_DIR / "complete_threshold_table_llama_after.csv")

print("=" * 80)
print("Creating Visualizations for Complete Threshold Analysis")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# Plot 1: ROC Curve (TPR vs FPR)
# ============================================================================
print("\n1. Creating ROC curves (TPR vs FPR)...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Qwen ROC
ax = axes[0]
ax.plot(qwen_before['fpr'], qwen_before['tpr'], 'b-o', linewidth=2, markersize=6, 
        label='Before filtering (all models)', alpha=0.7)
ax.plot(qwen_after['fpr'], qwen_after['tpr'], 'r-s', linewidth=2, markersize=6, 
        label='After filtering (downloads > 10)', alpha=0.7)

# Mark optimal points
optimal_before_idx = qwen_before['f1'].idxmax()
optimal_after_idx = qwen_after['f1'].idxmax()
ax.plot(qwen_before.loc[optimal_before_idx, 'fpr'], 
        qwen_before.loc[optimal_before_idx, 'tpr'], 
        'b*', markersize=20, label=f"Best before (threshold={qwen_before.loc[optimal_before_idx, 'threshold']:.3f})")
ax.plot(qwen_after.loc[optimal_after_idx, 'fpr'], 
        qwen_after.loc[optimal_after_idx, 'tpr'], 
        'r*', markersize=20, label=f"Best after (threshold={qwen_after.loc[optimal_after_idx, 'threshold']:.3f})")

# Diagonal line (random classifier)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12, fontweight='bold')
ax.set_title('Qwen: ROC Curve (TPR vs FPR)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

# LLaMA ROC
ax = axes[1]
ax.plot(llama_before['fpr'], llama_before['tpr'], 'b-o', linewidth=2, markersize=6, 
        label='Before filtering (all models)', alpha=0.7)
ax.plot(llama_after['fpr'], llama_after['tpr'], 'r-s', linewidth=2, markersize=6, 
        label='After filtering (downloads > 10)', alpha=0.7)

# Mark optimal points
optimal_before_idx = llama_before['f1'].idxmax()
optimal_after_idx = llama_after['f1'].idxmax()
ax.plot(llama_before.loc[optimal_before_idx, 'fpr'], 
        llama_before.loc[optimal_before_idx, 'tpr'], 
        'b*', markersize=20, label=f"Best before (threshold={llama_before.loc[optimal_before_idx, 'threshold']:.3f})")
ax.plot(llama_after.loc[optimal_after_idx, 'fpr'], 
        llama_after.loc[optimal_after_idx, 'tpr'], 
        'r*', markersize=20, label=f"Best after (threshold={llama_after.loc[optimal_after_idx, 'threshold']:.3f})")

# Diagonal line
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random classifier')

ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12, fontweight='bold')
ax.set_title('LLaMA: ROC Curve (TPR vs FPR)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves_tpr_vs_fpr.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'roc_curves_tpr_vs_fpr.png'}")
plt.close()

# ============================================================================
# Plot 2: Accuracy vs Threshold
# ============================================================================
print("\n2. Creating Accuracy vs Threshold plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Qwen Accuracy
ax = axes[0]
ax.plot(qwen_before['threshold'], qwen_before['accuracy'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['accuracy'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark best accuracy
best_before_idx = qwen_before['accuracy'].idxmax()
best_after_idx = qwen_after['accuracy'].idxmax()
ax.axvline(qwen_before.loc[best_before_idx, 'threshold'], color='b', 
           linestyle='--', alpha=0.3)
ax.axvline(qwen_after.loc[best_after_idx, 'threshold'], color='r', 
           linestyle='--', alpha=0.3)

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Qwen: Accuracy vs Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])
ax.set_ylim([0.4, 1.05])

# LLaMA Accuracy
ax = axes[1]
ax.plot(llama_before['threshold'], llama_before['accuracy'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['accuracy'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark best accuracy
best_before_idx = llama_before['accuracy'].idxmax()
best_after_idx = llama_after['accuracy'].idxmax()
ax.axvline(llama_before.loc[best_before_idx, 'threshold'], color='b', 
           linestyle='--', alpha=0.3)
ax.axvline(llama_after.loc[best_after_idx, 'threshold'], color='r', 
           linestyle='--', alpha=0.3)

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('LLaMA: Accuracy vs Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])
ax.set_ylim([0.4, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_vs_threshold.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'accuracy_vs_threshold.png'}")
plt.close()

# ============================================================================
# Plot 3: Precision-Recall Curve
# ============================================================================
print("\n3. Creating Precision-Recall curves...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Qwen PR
ax = axes[0]
ax.plot(qwen_before['tpr'], qwen_before['precision'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(qwen_after['tpr'], qwen_after['precision'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark optimal F1
optimal_before_idx = qwen_before['f1'].idxmax()
optimal_after_idx = qwen_after['f1'].idxmax()
ax.plot(qwen_before.loc[optimal_before_idx, 'tpr'], 
        qwen_before.loc[optimal_before_idx, 'precision'], 
        'b*', markersize=20, label=f"Best F1 before: {qwen_before.loc[optimal_before_idx, 'f1']:.3f}")
ax.plot(qwen_after.loc[optimal_after_idx, 'tpr'], 
        qwen_after.loc[optimal_after_idx, 'precision'], 
        'r*', markersize=20, label=f"Best F1 after: {qwen_after.loc[optimal_after_idx, 'f1']:.3f}")

ax.set_xlabel('Recall (TPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Qwen: Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

# LLaMA PR
ax = axes[1]
ax.plot(llama_before['tpr'], llama_before['precision'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(llama_after['tpr'], llama_after['precision'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark optimal F1
optimal_before_idx = llama_before['f1'].idxmax()
optimal_after_idx = llama_after['f1'].idxmax()
ax.plot(llama_before.loc[optimal_before_idx, 'tpr'], 
        llama_before.loc[optimal_before_idx, 'precision'], 
        'b*', markersize=20, label=f"Best F1 before: {llama_before.loc[optimal_before_idx, 'f1']:.3f}")
ax.plot(llama_after.loc[optimal_after_idx, 'tpr'], 
        llama_after.loc[optimal_after_idx, 'precision'], 
        'r*', markersize=20, label=f"Best F1 after: {llama_after.loc[optimal_after_idx, 'f1']:.3f}")

ax.set_xlabel('Recall (TPR)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('LLaMA: Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'precision_recall_curves.png'}")
plt.close()

# ============================================================================
# Plot 4: F1-Score vs Threshold
# ============================================================================
print("\n4. Creating F1-Score vs Threshold plots...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Qwen F1
ax = axes[0]
ax.plot(qwen_before['threshold'], qwen_before['f1'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['f1'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark best F1
best_before_idx = qwen_before['f1'].idxmax()
best_after_idx = qwen_after['f1'].idxmax()
ax.axvline(qwen_before.loc[best_before_idx, 'threshold'], color='b', 
           linestyle='--', alpha=0.3, 
           label=f"Best before: {qwen_before.loc[best_before_idx, 'threshold']:.3f}")
ax.axvline(qwen_after.loc[best_after_idx, 'threshold'], color='r', 
           linestyle='--', alpha=0.3,
           label=f"Best after: {qwen_after.loc[best_after_idx, 'threshold']:.3f}")

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Qwen: F1-Score vs Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])
ax.set_ylim([0.3, 1.05])

# LLaMA F1
ax = axes[1]
ax.plot(llama_before['threshold'], llama_before['f1'], 'b-o', linewidth=2, 
        markersize=6, label='Before filtering', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['f1'], 'r-s', linewidth=2, 
        markersize=6, label='After filtering', alpha=0.7)

# Mark best F1
best_before_idx = llama_before['f1'].idxmax()
best_after_idx = llama_after['f1'].idxmax()
ax.axvline(llama_before.loc[best_before_idx, 'threshold'], color='b', 
           linestyle='--', alpha=0.3,
           label=f"Best before: {llama_before.loc[best_before_idx, 'threshold']:.3f}")
ax.axvline(llama_after.loc[best_after_idx, 'threshold'], color='r', 
           linestyle='--', alpha=0.3,
           label=f"Best after: {llama_after.loc[best_after_idx, 'threshold']:.3f}")

ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('LLaMA: F1-Score vs Threshold', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])
ax.set_ylim([0.3, 1.05])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_score_vs_threshold.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'f1_score_vs_threshold.png'}")
plt.close()

# ============================================================================
# Plot 5: Multi-metric Comparison (4 subplots per family)
# ============================================================================
print("\n5. Creating multi-metric comparison plots...")

# Qwen multi-metric
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TPR
ax = axes[0, 0]
ax.plot(qwen_before['threshold'], qwen_before['tpr'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['tpr'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
ax.set_title('TPR (Recall / Sensitivity)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# FPR
ax = axes[0, 1]
ax.plot(qwen_before['threshold'], qwen_before['fpr'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['fpr'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
ax.set_title('FPR (1 - Specificity)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# Precision
ax = axes[1, 0]
ax.plot(qwen_before['threshold'], qwen_before['precision'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['precision'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax.set_title('Precision (TP / (TP + FP))', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# F1
ax = axes[1, 1]
ax.plot(qwen_before['threshold'], qwen_before['f1'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(qwen_after['threshold'], qwen_after['f1'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title('F1-Score (Harmonic Mean)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

fig.suptitle('Qwen: All Metrics vs Threshold', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "qwen_all_metrics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'qwen_all_metrics.png'}")
plt.close()

# LLaMA multi-metric
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# TPR
ax = axes[0, 0]
ax.plot(llama_before['threshold'], llama_before['tpr'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['tpr'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
ax.set_title('TPR (Recall / Sensitivity)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# FPR
ax = axes[0, 1]
ax.plot(llama_before['threshold'], llama_before['fpr'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['fpr'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
ax.set_title('FPR (1 - Specificity)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# Precision
ax = axes[1, 0]
ax.plot(llama_before['threshold'], llama_before['precision'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['precision'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax.set_title('Precision (TP / (TP + FP))', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

# F1
ax = axes[1, 1]
ax.plot(llama_before['threshold'], llama_before['f1'], 'b-o', linewidth=2, label='Before', alpha=0.7)
ax.plot(llama_after['threshold'], llama_after['f1'], 'r-s', linewidth=2, label='After', alpha=0.7)
ax.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title('F1-Score (Harmonic Mean)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.3])

fig.suptitle('LLaMA: All Metrics vs Threshold', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "llama_all_metrics.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'llama_all_metrics.png'}")
plt.close()

# ============================================================================
# Plot 6: Confusion Matrix Heatmaps at Optimal Thresholds
# ============================================================================
print("\n6. Creating confusion matrix heatmaps...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Helper function to plot confusion matrix
def plot_confusion_matrix(ax, tp, fn, fp, tn, title, threshold, accuracy):
    # Convert to integers
    tp, fn, fp, tn = int(tp), int(fn), int(fp), int(tn)
    cm = np.array([[tn, fp], [fn, tp]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                square=True, ax=ax, annot_kws={'size': 16, 'weight': 'bold'},
                vmin=0, vmax=max(tp, fn, fp, tn))
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\nThreshold={threshold:.3f}, Accuracy={accuracy:.1%}', 
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Different', 'Same'], fontsize=11)
    ax.set_yticklabels(['Different', 'Same'], fontsize=11)

# Qwen before (best F1)
best_idx = qwen_before['f1'].idxmax()
row = qwen_before.loc[best_idx]
plot_confusion_matrix(axes[0, 0], row['tp'], row['fn'], row['fp'], row['tn'],
                     'Qwen - Before Filtering', row['threshold'], row['accuracy'])

# Qwen after (best F1)
best_idx = qwen_after['f1'].idxmax()
row = qwen_after.loc[best_idx]
plot_confusion_matrix(axes[0, 1], row['tp'], row['fn'], row['fp'], row['tn'],
                     'Qwen - After Filtering', row['threshold'], row['accuracy'])

# LLaMA before (best F1)
best_idx = llama_before['f1'].idxmax()
row = llama_before.loc[best_idx]
plot_confusion_matrix(axes[1, 0], row['tp'], row['fn'], row['fp'], row['tn'],
                     'LLaMA - Before Filtering', row['threshold'], row['accuracy'])

# LLaMA after (best F1)
best_idx = llama_after['f1'].idxmax()
row = llama_after.loc[best_idx]
plot_confusion_matrix(axes[1, 1], row['tp'], row['fn'], row['fp'], row['tn'],
                     'LLaMA - After Filtering', row['threshold'], row['accuracy'])

fig.suptitle('Confusion Matrices at Optimal Thresholds (Best F1)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")
plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("\nGenerated plots:")
print("  1. roc_curves_tpr_vs_fpr.png - ROC curves (TPR vs FPR)")
print("  2. accuracy_vs_threshold.png - Accuracy comparison")
print("  3. precision_recall_curves.png - Precision-Recall curves")
print("  4. f1_score_vs_threshold.png - F1-Score comparison")
print("  5. qwen_all_metrics.png - Qwen multi-metric dashboard")
print("  6. llama_all_metrics.png - LLaMA multi-metric dashboard")
print("  7. confusion_matrices.png - Confusion matrices at optimal thresholds")
print("=" * 80)
