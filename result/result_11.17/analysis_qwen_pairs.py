import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load data
diff5 = pd.read_csv('lineage_scores_qwen_diff5.csv').dropna(subset=['score_mean'])
diff10 = pd.read_csv('lineage_scores_qwen_diff10.csv').dropna(subset=['score_mean'])
diff20 = pd.read_csv('lineage_scores_qwen_diff20.csv').dropna(subset=['score_mean'])
same5 = pd.read_csv('lineage_scores_qwen_same5.csv').dropna(subset=['score_mean'])
same10 = pd.read_csv('lineage_scores_qwen_same10.csv').dropna(subset=['score_mean'])
same20 = pd.read_csv('lineage_scores_qwen_same20.csv').dropna(subset=['score_mean'])

print(f"num_pairs=5:  Same={len(same5)}, Diff={len(diff5)}")
print(f"num_pairs=10: Same={len(same10)}, Diff={len(diff10)}")
print(f"num_pairs=20: Same={len(same20)}, Diff={len(diff20)}")

# Color configuration
same_color = '#90EE90'  # Light green
diff_color = '#F08080'  # Coral red

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ==================== 1. Distribution Plots ====================
metrics = ['pal_k_mean', 'lev_sim_mean', 'lcs_ratio_mean', 'score_mean']
titles = ['PAL K Mean', 'Levenshtein Similarity', 'LCS Ratio', 'Overall Score']

fig, axes = plt.subplots(4, 3, figsize=(16, 14))

pairs_data = [
    (same5, diff5, 'num_pairs=5'),
    (same10, diff10, 'num_pairs=10'),
    (same20, diff20, 'num_pairs=20')
]

for row, (metric, title) in enumerate(zip(metrics, titles)):
    for col, (same_df, diff_df, pair_label) in enumerate(pairs_data):
        ax = axes[row, col]
        
        same_data = same_df[metric].dropna()
        diff_data = diff_df[metric].dropna()
        
        # Calculate bins
        all_data = pd.concat([same_data, diff_data])
        bins = np.linspace(0, min(all_data.max() * 1.05, 1.0), 25)
        
        ax.hist(same_data, bins=bins, alpha=0.7, label=f'Same (n={len(same_data)})', 
                color=same_color, edgecolor='darkgreen', linewidth=0.5)
        ax.hist(diff_data, bins=bins, alpha=0.7, label=f'Diff (n={len(diff_data)})', 
                color=diff_color, edgecolor='darkred', linewidth=0.5)
        
        # Mean lines
        same_mean = same_data.mean()
        diff_mean = diff_data.mean()
        ax.axvline(same_mean, color='green', linestyle='--', linewidth=2)
        ax.axvline(diff_mean, color='darkred', linestyle='--', linewidth=2)
        
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{title} ({pair_label})', fontsize=11, fontweight='bold')
        ax.legend([f'Same', f'Diff', f'Same mean: {same_mean:.3f}', f'Diff mean: {diff_mean:.3f}'],
                  loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

plt.suptitle('Distribution of Metrics: Same vs Diff (Qwen2.5-0.5B) - Different num_pairs', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('distribution_all_pairs.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Distribution plots saved: distribution_all_pairs.png")

# ==================== 2. ROC Analysis ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

roc_results = []

for idx, (same_df, diff_df, pair_label) in enumerate(pairs_data):
    ax = axes[idx]
    
    same_scores = same_df['score_mean'].values
    diff_scores = diff_df['score_mean'].values
    
    y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
    y_scores = np.concatenate([same_scores, diff_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Best threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    roc_results.append({
        'pair_label': pair_label,
        'auc': roc_auc,
        'threshold': best_threshold,
        'tpr': tpr[best_idx],
        'fpr': fpr[best_idx],
        'y_true': y_true,
        'y_scores': y_scores
    })
    
    ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5, 
               label=f'Best threshold = {best_threshold:.4f}')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve ({pair_label})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('ROC Analysis: Same vs Diff (Qwen2.5-0.5B)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('roc_all_pairs.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("ROC curves saved: roc_all_pairs.png")

# ==================== 3. Confusion Matrix (Row Percentage) ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, result in enumerate(roc_results):
    ax = axes[idx]
    
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate row percentage (percentage within each true class)
    # Row 1: Diff class -> TN + FP = 100%
    # Row 2: Same class -> FN + TP = 100%
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row_percent = cm / row_sums * 100
    
    tn_p = cm_row_percent[0, 0]  # TN / (TN+FP) * 100 = Specificity
    fp_p = cm_row_percent[0, 1]  # FP / (TN+FP) * 100 = FPR
    fn_p = cm_row_percent[1, 0]  # FN / (FN+TP) * 100 = FNR
    tp_p = cm_row_percent[1, 1]  # TP / (FN+TP) * 100 = Recall/TPR
    
    im = ax.imshow(cm_row_percent, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.set_title(f'Confusion Matrix ({result["pair_label"]})\nThreshold = {result["threshold"]:.4f}', 
                 fontsize=11, fontweight='bold')
    
    # Text annotations
    labels = [
        [f'TN\n{tn} ({tn_p:.1f}%)', f'FP\n{fp} ({fp_p:.1f}%)'],
        [f'FN\n{fn} ({fn_p:.1f}%)', f'TP\n{tp} ({tp_p:.1f}%)']
    ]
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                   ha="center", va="center",
                   color="white" if cm_row_percent[i, j] > 50 else "black",
                   fontsize=10)
    
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Diff (0)', 'Same (1)'])
    ax.set_yticklabels(['Diff (0)', 'Same (1)'])
    
    plt.colorbar(im, ax=ax, label='Row Percentage (%)')

plt.suptitle('Confusion Matrix (Row %): Same vs Diff (Qwen2.5-0.5B)\nEach row sums to 100%', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrix_all_pairs.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Confusion matrices saved: confusion_matrix_all_pairs.png")

# ==================== 4. Print Detailed Results ====================
print("\n" + "="*60)
print("ROC Analysis Detailed Results")
print("="*60)

for result in roc_results:
    y_pred = (result['y_scores'] >= result['threshold']).astype(int)
    cm = confusion_matrix(result['y_true'], y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(result['y_true'])
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{result['pair_label']}:")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  Best Threshold: {result['threshold']:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall (TPR): {recall:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp} ({tp/total*100:.1f}%), FP={fp} ({fp/total*100:.1f}%)")
    print(f"    FN={fn} ({fn/total*100:.1f}%), TN={tn} ({tn/total*100:.1f}%)")

# Save results to CSV
results_df = pd.DataFrame([{
    'num_pairs': r['pair_label'],
    'AUC': r['auc'],
    'Best_Threshold': r['threshold'],
    'TPR_at_threshold': r['tpr'],
    'FPR_at_threshold': r['fpr']
} for r in roc_results])
results_df.to_csv('roc_analysis_results.csv', index=False)
print("\nResults saved to: roc_analysis_results.csv")
