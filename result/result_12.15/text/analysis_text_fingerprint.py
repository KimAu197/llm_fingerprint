import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
diff_llama = pd.read_csv('lineage_bottomk_text_diff_llama.csv')
diff_qwen = pd.read_csv('lineage_bottomk_text_diff_qwen.csv')
same_llama = pd.read_csv('lineage_bottomk_text_same_llama.csv')
same_qwen = pd.read_csv('lineage_bottomk_text_same_qwen.csv')

# Filter out invalid data
diff_llama = diff_llama[diff_llama['avg_score'] >= 0]
diff_qwen = diff_qwen[diff_qwen['avg_score'] >= 0]
same_llama = same_llama[same_llama['avg_score'] >= 0]
same_qwen = same_qwen[same_qwen['avg_score'] >= 0]

print(f"LLaMA: Same={len(same_llama)}, Diff={len(diff_llama)}")
print(f"Qwen: Same={len(same_qwen)}, Diff={len(diff_qwen)}")

# Color configuration
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

metrics = ['avg_pal_k', 'avg_lev_sim', 'avg_lcs_ratio', 'avg_score']
metric_names = ['PAL-K', 'Levenshtein Similarity', 'LCS Ratio', 'Overall Score']

# ==================== 1. Distribution Plot - Qwen (all metrics in one figure) ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    same_scores = same_qwen[metric].dropna()
    diff_scores = diff_qwen[metric].dropna()
    
    max_val = max(same_scores.max(), diff_scores.max()) if len(same_scores) > 0 and len(diff_scores) > 0 else 1
    bins = np.linspace(0, max_val + 0.1, 30)
    
    ax.hist(same_scores, bins=bins, alpha=0.7, color=same_color, 
            label=f'Same (Qwen -> Qwen derived)\nmu={same_scores.mean():.3f}', edgecolor='white')
    ax.hist(diff_scores, bins=bins, alpha=0.7, color=diff_color, 
            label=f'Diff (Qwen -> LLaMA derived)\nmu={diff_scores.mean():.3f}', edgecolor='white')
    
    # Mean lines
    ax.axvline(same_scores.mean(), color='green', linestyle=':', linewidth=2)
    ax.axvline(diff_scores.mean(), color='red', linestyle=':', linewidth=2)
    
    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of {metric_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Qwen: Text-based Fingerprint Score Distribution\n(Same vs Diff)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_qwen.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_qwen.png")

# ==================== 2. Distribution Plot - LLaMA (all metrics in one figure) ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    same_scores = same_llama[metric].dropna()
    diff_scores = diff_llama[metric].dropna()
    
    max_val = max(same_scores.max(), diff_scores.max()) if len(same_scores) > 0 and len(diff_scores) > 0 else 1
    bins = np.linspace(0, max_val + 0.1, 30)
    
    ax.hist(same_scores, bins=bins, alpha=0.7, color=same_color, 
            label=f'Same (LLaMA -> LLaMA derived)\nmu={same_scores.mean():.3f}', edgecolor='white')
    ax.hist(diff_scores, bins=bins, alpha=0.7, color=diff_color, 
            label=f'Diff (LLaMA -> Qwen derived)\nmu={diff_scores.mean():.3f}', edgecolor='white')
    
    # Mean lines
    ax.axvline(same_scores.mean(), color='green', linestyle=':', linewidth=2)
    ax.axvline(diff_scores.mean(), color='red', linestyle=':', linewidth=2)
    
    ax.set_xlabel(metric_name, fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Distribution of {metric_name}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('LLaMA: Text-based Fingerprint Score Distribution\n(Same vs Diff)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_llama.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_llama.png")

# ==================== 3. ROC Analysis using avg_score ====================
metric = 'avg_score'
roc_results = []

# Qwen ROC
same_scores_qwen = same_qwen[metric].dropna().values
diff_scores_qwen = diff_qwen[metric].dropna().values

y_true_qwen = np.concatenate([np.ones(len(same_scores_qwen)), np.zeros(len(diff_scores_qwen))])
y_scores_qwen = np.concatenate([same_scores_qwen, diff_scores_qwen])

fpr_qwen, tpr_qwen, thresholds_qwen = roc_curve(y_true_qwen, y_scores_qwen)
auc_qwen = auc(fpr_qwen, tpr_qwen)

# Find best threshold
youden_qwen = tpr_qwen - fpr_qwen
best_idx_qwen = np.argmax(youden_qwen)
best_threshold_qwen = thresholds_qwen[best_idx_qwen]

roc_results.append({
    'model': 'Qwen',
    'fpr': fpr_qwen, 'tpr': tpr_qwen, 'auc': auc_qwen,
    'threshold': best_threshold_qwen,
    'y_true': y_true_qwen, 'y_scores': y_scores_qwen
})

# LLaMA ROC
same_scores_llama = same_llama[metric].dropna().values
diff_scores_llama = diff_llama[metric].dropna().values

y_true_llama = np.concatenate([np.ones(len(same_scores_llama)), np.zeros(len(diff_scores_llama))])
y_scores_llama = np.concatenate([same_scores_llama, diff_scores_llama])

fpr_llama, tpr_llama, thresholds_llama = roc_curve(y_true_llama, y_scores_llama)
auc_llama = auc(fpr_llama, tpr_llama)

# Find best threshold
youden_llama = tpr_llama - fpr_llama
best_idx_llama = np.argmax(youden_llama)
best_threshold_llama = thresholds_llama[best_idx_llama]

roc_results.append({
    'model': 'LLaMA',
    'fpr': fpr_llama, 'tpr': tpr_llama, 'auc': auc_llama,
    'threshold': best_threshold_llama,
    'y_true': y_true_llama, 'y_scores': y_scores_llama
})

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 8))

colors = ['#2E86AB', '#E94F37']
for idx, result in enumerate(roc_results):
    ax.plot(result['fpr'], result['tpr'], color=colors[idx], linewidth=2,
            label=f"{result['model']} (AUC = {result['auc']:.4f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison\n(Qwen vs LLaMA) - avg_score', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_comparison.png")

# ==================== 4. Confusion Matrices ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, result in enumerate(roc_results):
    ax = axes[idx]
    
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Row percentage
    row0_total = tn + fp
    row1_total = fn + tp
    tn_p = tn / row0_total * 100 if row0_total > 0 else 0
    fp_p = fp / row0_total * 100 if row0_total > 0 else 0
    fn_p = fn / row1_total * 100 if row1_total > 0 else 0
    tp_p = tp / row1_total * 100 if row1_total > 0 else 0
    
    cm_percent = np.array([[tn_p, fp_p], [fn_p, tp_p]])
    
    im = ax.imshow(cm_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f"{result['model']} Confusion Matrix\nThreshold = {result['threshold']:.4f}", 
                 fontsize=11, fontweight='bold')
    
    labels = [
        [f'TN\n{tn} ({tn_p:.1f}%)', f'FP\n{fp} ({fp_p:.1f}%)'],
        [f'FN\n{fn} ({fn_p:.1f}%)', f'TP\n{tp} ({tp_p:.1f}%)']
    ]
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm_percent[i, j] > 50 else 'black'
            ax.text(j, i, labels[i][j], ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=color)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Diff', 'Pred: Same'])
    ax.set_yticklabels(['Actual: Diff', 'Actual: Same'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.suptitle('Confusion Matrix Comparison (Row Percentage)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_comparison.png")

# ==================== 5. Print Results ====================
print("\n" + "="*60)
print("Result Summary (using avg_score)")
print("="*60)

for result in roc_results:
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{result['model']}:")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  Best Threshold: {result['threshold']:.4f}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall (TPR): {recall:.2%}")
    print(f"  Specificity (TNR): {specificity:.2%}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

# Save results
results_df = pd.DataFrame([{
    'Model': r['model'],
    'AUC': r['auc'],
    'Threshold': r['threshold'],
    'Accuracy': (np.sum(r['y_true'] == (r['y_scores'] >= r['threshold']).astype(int))) / len(r['y_true'])
} for r in roc_results])
results_df.to_csv('roc_analysis_results.csv', index=False)
print("\nSaved: roc_analysis_results.csv")
