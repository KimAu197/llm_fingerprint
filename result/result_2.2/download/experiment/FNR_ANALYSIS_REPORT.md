# FNR Analysis: Effect of Download Filtering

## PhD Question
What is the false negative rate of identifying the correct base model before and after filtering for low download rates? Does it change significantly when filtering?

## Method
- **Data**: Merged dataset (original wordlist lineage + experiment popular models)
- **Threshold**: Best_Threshold from roc_analysis_results.csv (12.15 wordlist analysis)
  - Qwen: 0.0082
  - LLaMA: 0.0254
- **Decision rule**: overlap >= threshold → Same base model (correct identification)
- **FNR** = FN / Total = proportion of same-family models incorrectly predicted as different
- **Filtering**: derived_model_downloads > 10

## Results

| Family | Threshold | Before FNR | After FNR | FNR Change | Before N | After N |
|--------|-----------|------------|-----------|------------|----------|---------|
| Qwen | 0.0082 | 4.27% | 7.41% | +3.13pp | 117 | 27 |
| LLaMA | 0.0254 | 4.31% | 0.00% | -4.31pp | 116 | 27 |

## Conclusion
Filtering for downloads > 10 **changes FNR differently by model family**:

- **Qwen**: FNR **increases** from 4.27% to 7.41% (+3.13pp). The merged dataset includes popular high-download Qwen models (from experiment) that have low overlap—these are same-family but incorrectly predicted as different. Filtering does not help Qwen.

- **LLaMA**: FNR **decreases** to 0% (-4.31pp). All FN cases are among low-download models; filtering eliminates them entirely.

**Key finding**: Download filtering is **family-dependent**. For LLaMA, filtering improves identification. For Qwen, popular models include heavily fine-tuned variants with low overlap, so filtering does not reduce (and may increase) FNR.
