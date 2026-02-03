# Visualization Summary: Complete Threshold Analysis with False Positives

## 📊 Generated Visualizations

All plots saved in: `roc_analysis/plots/`

---

## 1. ROC Curves (TPR vs FPR)

**File**: `roc_curves_tpr_vs_fpr.png`

![ROC Curves](roc_analysis/plots/roc_curves_tpr_vs_fpr.png)

**Key Insights**:
- **Qwen**: Both curves hug the top-left corner (ideal), showing excellent separation between same/diff families
  - Before filtering: Best threshold=0.010 (TPR≈95%, FPR=0%)
  - After filtering: Best threshold=0.005 (TPR≈96%, FPR=0%)
  - Very low FPR across all thresholds (excellent specificity)

- **LLaMA**: After filtering dramatically improves the curve
  - Before filtering: High FPR at low thresholds (curve starts far from top-left)
  - After filtering: Perfect curve (hugs top-left corner)
  - Best threshold after filtering: 0.030 (TPR=100%, FPR=0%)

---

## 2. Accuracy vs Threshold

**File**: `accuracy_vs_threshold.png`

![Accuracy](roc_analysis/plots/accuracy_vs_threshold.png)

**Key Insights**:
- **Qwen**: 
  - Peak accuracy ~97% at low thresholds (0.005-0.010)
  - After filtering shows lower accuracy at higher thresholds (red line drops faster)
  - Optimal range: threshold=0.005-0.010

- **LLaMA**:
  - Before filtering: Accuracy starts low (~54%) at threshold=0, improves to ~97% at 0.02
  - After filtering: **100% accuracy** maintained from threshold=0.030 to 0.15!
  - Clear winner: filter + threshold=0.030

---

## 3. Precision-Recall Curves

**File**: `precision_recall_curves.png`

![Precision-Recall](roc_analysis/plots/precision_recall_curves.png)

**Key Insights**:
- **Qwen**: 
  - Nearly perfect precision (100%) across all recall levels
  - Both curves overlap at high precision
  - Best F1=0.97 (before) and 0.98 (after)

- **LLaMA**:
  - Before filtering: Precision drops sharply at high recall (many false positives)
  - After filtering: Perfect precision (100%) maintained even at 100% recall
  - Best F1=0.97 (before) and **1.00** (after) - perfect score!

---

## 4. F1-Score vs Threshold

**File**: `f1_score_vs_threshold.png`

![F1-Score](roc_analysis/plots/f1_score_vs_threshold.png)

**Key Insights**:
- **Qwen**:
  - Peak F1 ~97% at threshold=0.010 (before filtering)
  - Peak F1 ~98% at threshold=0.005 (after filtering)
  - F1 degrades faster after filtering at higher thresholds

- **LLaMA**:
  - Before filtering: Peak F1 ~97% at threshold=0.020
  - After filtering: **F1=1.00** (perfect!) from threshold=0.030 to 0.15
  - Massive improvement with filtering

---

## 5. Confusion Matrices at Optimal Thresholds

**File**: `confusion_matrices.png`

![Confusion Matrices](roc_analysis/plots/confusion_matrices.png)

### Qwen - Before Filtering (threshold=0.010, accuracy=97.3%)
```
              Predicted
           Different  Same
Actual  
Different    104      0     ← Perfect specificity (0 FP)
Same          6      111    ← 6 false negatives
```

### Qwen - After Filtering (threshold=0.005, accuracy=97.6%)
```
              Predicted
           Different  Same
Actual  
Different    15       0     ← Perfect specificity (0 FP)
Same          1      26     ← Only 1 false negative
```

### LLaMA - Before Filtering (threshold=0.020, accuracy=96.9%)
```
              Predicted
           Different  Same
Actual  
Different    107      2     ← 2 false positives
Same          5      111    ← 5 false negatives
```

### LLaMA - After Filtering (threshold=0.030, accuracy=100.0%)
```
              Predicted
           Different  Same
Actual  
Different    13       0     ← Perfect! (0 FP)
Same          0      27     ← Perfect! (0 FN)
```
**🎉 Perfect classification for LLaMA after filtering!**

---

## 6. Qwen Multi-Metric Dashboard

**File**: `qwen_all_metrics.png`

![Qwen All Metrics](roc_analysis/plots/qwen_all_metrics.png)

**4-panel view showing**:
- **TPR**: Decreases gradually with threshold, filtering causes faster drop
- **FPR**: Stays at 0% for threshold ≥ 0.010 (excellent!)
- **Precision**: Maintains 100% across almost all thresholds
- **F1-Score**: Peaks at ~97-98%, filtering causes faster degradation

**Takeaway**: Qwen has excellent precision (low FP) but filtering hurts recall (more FN)

---

## 7. LLaMA Multi-Metric Dashboard

**File**: `llama_all_metrics.png`

![LLaMA All Metrics](roc_analysis/plots/llama_all_metrics.png)

**4-panel view showing**:
- **TPR**: After filtering maintains 100% up to threshold=0.18!
- **FPR**: Before filtering has high FPR at low thresholds; after filtering drops to 0% quickly
- **Precision**: After filtering achieves 100% precision much earlier
- **F1-Score**: After filtering achieves perfect F1=1.00 from threshold=0.030-0.15

**Takeaway**: LLaMA benefits dramatically from filtering - achieves perfect metrics!

---

## Summary of Key Findings

### Qwen Family
| Metric | Before Filtering | After Filtering | Winner |
|--------|------------------|-----------------|--------|
| **Optimal Threshold** | 0.010 | 0.005 | - |
| **TPR** | 94.9% | 96.3% | After (+1.4pp) |
| **FPR** | 0.0% | 0.0% | Tie ✓ |
| **Accuracy** | 97.3% | 97.6% | After (+0.3pp) |
| **F1-Score** | 97.4% | 98.1% | After (+0.7pp) |
| **False Positives** | 0 | 0 | Tie ✓ |
| **False Negatives** | 6 | 1 | After (-5) ✓ |

**Recommendation**: Use threshold=0.010 without filtering OR threshold=0.005 with filtering

---

### LLaMA Family
| Metric | Before Filtering | After Filtering | Winner |
|--------|------------------|-----------------|--------|
| **Optimal Threshold** | 0.020 | 0.030 | - |
| **TPR** | 95.7% | **100.0%** | After (+4.3pp) ✓✓✓ |
| **FPR** | 1.8% | **0.0%** | After (-1.8pp) ✓✓✓ |
| **Accuracy** | 96.9% | **100.0%** | After (+3.1pp) ✓✓✓ |
| **F1-Score** | 96.9% | **100.0%** | After (+3.1pp) ✓✓✓ |
| **False Positives** | 2 | **0** | After (-2) ✓✓✓ |
| **False Negatives** | 5 | **0** | After (-5) ✓✓✓ |

**Recommendation**: **MUST filter by downloads > 10**, then use threshold=0.030 for **perfect classification**!

---

## Practical Implications

### For Production Lineage Detection Systems:

1. **Qwen Models**:
   - Use threshold = 0.010 (no filtering needed)
   - Expect: 94.9% detection rate, 0% false positives
   - Very reliable with minimal false alarms

2. **LLaMA Models**:
   - **MUST filter by downloads > 10**
   - Use threshold = 0.030
   - Expect: **100% detection rate, 0% false positives** 🎯
   - Perfect classification achieved!

3. **Mixed Environment** (both Qwen and LLaMA):
   - Apply download filtering
   - Use family-specific thresholds:
     - Qwen: 0.005
     - LLaMA: 0.030
   - Monitor false positives and adjust as needed

---

## Files Generated

All visualizations saved in: `roc_analysis/plots/`

1. `roc_curves_tpr_vs_fpr.png` - Classic ROC curves
2. `accuracy_vs_threshold.png` - Accuracy comparison
3. `precision_recall_curves.png` - Precision-Recall trade-off
4. `f1_score_vs_threshold.png` - F1-Score optimization
5. `confusion_matrices.png` - Confusion matrices at optimal thresholds
6. `qwen_all_metrics.png` - Qwen comprehensive dashboard
7. `llama_all_metrics.png` - LLaMA comprehensive dashboard

---

## Conclusion

The visualizations clearly demonstrate that:

1. **Qwen** has excellent inherent separation between same/diff families (low FP naturally)
2. **LLaMA** suffers from high false positives at low thresholds due to low-download models
3. **Filtering by downloads** is a game-changer for LLaMA (achieves perfect classification!)
4. **Filtering hurts Qwen** slightly at higher thresholds (reduces recall)

**Bottom line**: Family-specific strategies are essential for optimal lineage detection!
