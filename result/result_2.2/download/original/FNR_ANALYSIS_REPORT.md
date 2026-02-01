# FNR Analysis: Effect of Download Filtering

## PhD Question
What is the false negative rate of identifying the correct base model before and after filtering for low download rates? Does it change significantly when filtering?

## Method
- **Data**: Original data from result_1.26 (wordlist lineage with downloads)
- **Threshold**: Best_Threshold from roc_analysis_results.csv (12.15 wordlist analysis)
  - Qwen: 0.0082
  - LLaMA: 0.0254
- **Decision rule**: overlap >= threshold → Same base model (correct identification)
- **FNR** = FN / Total = proportion of same-family models incorrectly predicted as different
- **Filtering**: derived_model_downloads > 10

## Results

| Family | Threshold | Before FNR | After FNR | FNR Change | Before N | After N |
|--------|-----------|------------|-----------|------------|----------|---------|
| Qwen | 0.0082 | 2.91% | 0.00% | -2.91pp | 103 | 13 |
| LLaMA | 0.0254 | 4.81% | 0.00% | -4.81pp | 104 | 15 |

## Conclusion
Filtering for downloads > 10 **decreases** FNR by ~3.9 percentage points. Removing low-download models improves identification accuracy.
