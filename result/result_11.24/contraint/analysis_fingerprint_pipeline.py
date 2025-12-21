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

# ==================== 1. Distribution Plot - Qwen ====================
fig, ax = plt.subplots(figsize=(10, 6))

same_scores = same_qwen['avg_overlap_ratio'].dropna()
diff_scores = diff_qwen['avg_overlap_ratio'].dropna()

bins = np.linspace(0, 1, 31)
ax.hist(same_scores, bins=bins, alpha=0.6, color=same_color, 
        label=f'Same (Qwen->Qwen derived, n={len(same_scores)})', edgecolor='white')
ax.hist(diff_scores, bins=bins, alpha=0.6, color=diff_color, 
        label=f'Diff (Qwen->LLaMA derived, n={len(diff_scores)})', edgecolor='white')

ax.axvline(same_scores.mean(), color='green', linestyle='--', linewidth=2, 
           label=f'Same Mean: {same_scores.mean():.3f}')
ax.axvline(diff_scores.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Diff Mean: {diff_scores.mean():.3f}')

ax.set_xlabel('Overlap Ratio', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Qwen: Same vs Diff Distribution\n(Bottom-k Fingerprint Pipeline)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_qwen.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_qwen.png")

# ==================== 2. Distribution Plot - LLaMA ====================
fig, ax = plt.subplots(figsize=(10, 6))

same_scores = same_llama['avg_overlap_ratio'].dropna()
diff_scores = diff_llama['avg_overlap_ratio'].dropna()

bins = np.linspace(0, 1, 31)
ax.hist(same_scores, bins=bins, alpha=0.6, color=same_color, 
        label=f'Same (LLaMA->LLaMA derived, n={len(same_scores)})', edgecolor='white')
ax.hist(diff_scores, bins=bins, alpha=0.6, color=diff_color, 
        label=f'Diff (LLaMA->Qwen derived, n={len(diff_scores)})', edgecolor='white')

ax.axvline(same_scores.mean(), color='green', linestyle='--', linewidth=2, 
           label=f'Same Mean: {same_scores.mean():.3f}')
ax.axvline(diff_scores.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Diff Mean: {diff_scores.mean():.3f}')

ax.set_xlabel('Overlap Ratio', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('LLaMA: Same vs Diff Distribution\n(Bottom-k Fingerprint Pipeline)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_llama.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_llama.png")

# ==================== 3. ROC Analysis ====================
def compute_roc_metrics(same_df, diff_df, model_name):
    same_scores = same_df['avg_overlap_ratio'].dropna().values
    diff_scores = diff_df['avg_overlap_ratio'].dropna().values
    
    y_scores = np.concatenate([same_scores, diff_scores])
    y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find best threshold (Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    return {
        'model': model_name,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'threshold': best_threshold,
        'y_true': y_true,
        'y_scores': y_scores
    }

roc_qwen = compute_roc_metrics(same_qwen, diff_qwen, 'Qwen')
roc_llama = compute_roc_metrics(same_llama, diff_llama, 'LLaMA')

# Plot ROC curves
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(roc_qwen['fpr'], roc_qwen['tpr'], color='#2196F3', lw=2,
        label=f"Qwen (AUC = {roc_qwen['auc']:.4f}, thresh = {roc_qwen['threshold']:.4f})")
ax.plot(roc_llama['fpr'], roc_llama['tpr'], color='#FF9800', lw=2,
        label=f"LLaMA (AUC = {roc_llama['auc']:.4f}, thresh = {roc_llama['threshold']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Qwen vs LLaMA\n(Bottom-k Fingerprint Pipeline)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_comparison.png")

# ==================== 4. Confusion Matrix (Row Percentage) ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, result in enumerate([roc_qwen, roc_llama]):
    ax = axes[idx]
    
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate row percentage
    actual_neg = tn + fp
    actual_pos = fn + tp
    
    tn_p = tn / actual_neg * 100 if actual_neg > 0 else 0
    fp_p = fp / actual_neg * 100 if actual_neg > 0 else 0
    fn_p = fn / actual_pos * 100 if actual_pos > 0 else 0
    tp_p = tp / actual_pos * 100 if actual_pos > 0 else 0
    
    cm_row_percent = np.array([[tn_p, fp_p], [fn_p, tp_p]])
    
    im = ax.imshow(cm_row_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f"{result['model']} Confusion Matrix\nThreshold = {result['threshold']:.4f}",
                 fontsize=12, fontweight='bold')
    
    labels = [
        [f'TN\n{tn} ({tn_p:.1f}%)', f'FP\n{fp} ({fp_p:.1f}%)'],
        [f'FN\n{fn} ({fn_p:.1f}%)', f'TP\n{tp} ({tp_p:.1f}%)']
    ]
    
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_row_percent[i, j] > 50 else 'black'
            ax.text(j, i, labels[i][j], ha='center', va='center', 
                    fontsize=14, color=text_color, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted: Diff', 'Predicted: Same'])
    ax.set_yticklabels(['Actual: Diff', 'Actual: Same'])
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('Actual Label', fontsize=11)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_comparison.png")

# ==================== 5. Detailed Statistics ====================
print("\n" + "="*60)
print("Analysis Results Summary")
print("="*60)

for result in [roc_qwen, roc_llama]:
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{result['model']}:")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  Best Threshold: {result['threshold']:.4f}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall (TPR): {recall:.2%}")
    print(f"  Specificity (TNR): {specificity:.2%}")
    print(f"  F1 Score: {f1:.4f}")

# Save results to CSV
results_df = []
for result in [roc_qwen, roc_llama]:
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results_df.append({
        'Model': result['model'],
        'AUC': result['auc'],
        'Threshold': result['threshold'],
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1': f1,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    })

pd.DataFrame(results_df).to_csv('roc_analysis_results.csv', index=False)
print("\nSaved: roc_analysis_results.csv")
