# Download Analysis - English Documentation

## Overview

This directory contains analysis of the relationship between model downloads and overlap ratios for Qwen and TinyLlama model families.

---

## File Structure

```
download/
├── README.md                    - This file
├── ANALYSIS_REPORT.md          - Main analysis report (English) ⭐
├── 分析报告.md                  - Original report (Chinese)
│
├── experiment/                  - Detailed experiment results
│   ├── SUMMARY.md              - Experiment summary (English) ⭐
│   ├── ANALYSIS_SUMMARY.md     - Research implications (English) ⭐
│   ├── compare_qwen_vs_llama.py - Comparison script
│   ├── fetch_hf_downloads.py   - Data fetching script
│   │
│   ├── data/                   - Raw data
│   │   ├── popular_models_qwen.csv
│   │   └── popular_models_tinyllama.csv
│   │
│   ├── output/                 - Processed data
│   │   ├── qwen_overlap_downloads.csv
│   │   ├── llama_overlap_downloads.csv
│   │   ├── lineage_bottomk_text_same_qwen.csv
│   │   └── lineage_bottomk_text_same_llama.csv
│   │
│   └── visualizations/
│       ├── qwen_vs_llama_comparison.png
│       └── overlap_vs_downloads_analysis.png
│
└── original/                   - Original analysis files
    └── download_overlap_analysis.png
```

---

## Key Documents (English)

### 1. ANALYSIS_REPORT.md ⭐
**Main analysis report**

**Contents**:
- Hypothesis validation results
- Detailed statistics by download groups
- Key insights and recommendations
- Practical application examples
- Statistical significance tests

**Key Findings**:
- ✅ Hypothesis validated: Low download models show more noise
- ✅ Same family models: overlap = 40-70%
- ✅ Different family models: overlap < 1%
- ✅ Clear distinction regardless of download count

---

### 2. experiment/SUMMARY.md ⭐
**Experiment summary**

**Contents**:
- Data overview for Qwen and TinyLlama
- Overlap statistics (mean, median, std dev)
- Distribution comparison
- Model characteristics analysis

**Key Statistics**:
| Model Family | Valid Models | Mean Overlap | Median | Invalid Rate |
|--------------|--------------|--------------|--------|--------------|
| Qwen2.5-0.5B | 29 | 28.47% | 24.15% | 46.3% |
| TinyLlama-1.1B | 28 | 64.75% | 69.03% | 31.7% |

---

### 3. experiment/ANALYSIS_SUMMARY.md ⭐
**Research implications**

**Contents**:
- Executive summary
- Correlation analysis (Pearson, Spearman)
- Download group analysis
- Research recommendations
- Paper writing tips

**Key Correlations**:
- Qwen: Spearman = 0.43 (moderate positive, p < 0.05)
- TinyLlama: Spearman = -0.16 (no correlation, p > 0.05)

---

## Quick Start

### 1. View Main Findings
```bash
# Read the main analysis report
cat ANALYSIS_REPORT.md

# Or view experiment summary
cat experiment/SUMMARY.md
```

### 2. Run Analysis
```bash
cd experiment/
python compare_qwen_vs_llama.py
```

### 3. Load Data
```python
import pandas as pd

# Load merged data
qwen = pd.read_csv('experiment/output/qwen_overlap_downloads.csv')
llama = pd.read_csv('experiment/output/llama_overlap_downloads.csv')

# Basic statistics
print(f"Qwen mean overlap: {qwen['avg_overlap_ratio'].mean():.4f}")
print(f"TinyLlama mean overlap: {llama['avg_overlap_ratio'].mean():.4f}")
```

---

## Key Findings Summary

### Hypothesis Validation ✅

**Original Hypothesis**: Low download models (personal use, uncontrolled data) should show noise characteristics

**Result**: **Strongly validated!**

### Evidence

#### Low Download Models (≤10)
- LLama same family: 0.53 overlap (high)
- LLama different family: 0.007 overlap (very low)
- **Ratio**: 71.8x difference

- Qwen same family: 0.48 overlap (high)
- Qwen different family: 0.004 overlap (very low)
- **Ratio**: 125.2x difference

#### High Download Models (>100)
- More stable (smaller std dev)
- More reliable (overlap meets expectations)
- Better quality (official/well-known teams)

---

## Filtering Recommendations

### Conservative (High Quality)
```python
# Keep only downloads > 100
filtered = df[df['derived_model_downloads'] > 100]
```
- Pros: Most reliable
- Cons: Small sample size

### Balanced (Recommended) ⭐
```python
# Keep downloads > 10
filtered = df[df['derived_model_downloads'] > 10]
```
- Pros: Good balance of quality and quantity
- Cons: Some noise remains

### Aggressive (Maximum Data)
```python
# Keep all, but flag quality
df['quality'] = df['derived_model_downloads'].apply(
    lambda x: 'high' if x > 100 else ('medium' if x > 10 else 'low')
)
```
- Pros: All data retained
- Cons: Need quality-aware analysis

---

## Visualizations

### 1. qwen_vs_llama_comparison.png
**4-in-1 comparison chart**
- Histogram comparison
- Box plot comparison
- CDF comparison
- Distribution by ranges

**Location**: `experiment/qwen_vs_llama_comparison.png`

### 2. overlap_vs_downloads_analysis.png
**6-in-1 detailed analysis**
- Qwen: Linear scatter, log scatter, box plot
- TinyLlama: Linear scatter, log scatter, box plot

**Location**: `experiment/overlap_vs_downloads_analysis.png`

### 3. download_overlap_analysis.png
**Original 4-subplot analysis**
- LLama different family
- LLama same family
- Qwen different family
- Qwen same family

**Location**: `original/download_overlap_analysis.png`

---

## Data Files

### Raw Data
- `experiment/data/popular_models_qwen.csv` - 54 Qwen models
- `experiment/data/popular_models_tinyllama.csv` - 41 TinyLlama models

### Processed Data
- `experiment/output/qwen_overlap_downloads.csv` - 29 valid Qwen models ⭐
- `experiment/output/llama_overlap_downloads.csv` - 28 valid TinyLlama models ⭐
- `experiment/output/lineage_bottomk_text_same_qwen.csv` - 54 Qwen models (includes invalid)
- `experiment/output/lineage_bottomk_text_same_llama.csv` - 41 TinyLlama models (includes invalid)

---

## Scripts

### compare_qwen_vs_llama.py
**Main comparison script**

Features:
- Compare distributions
- Calculate statistics
- Generate visualizations

Usage:
```bash
cd experiment/
python compare_qwen_vs_llama.py
```

### fetch_hf_downloads.py
**Data fetching script**

Features:
- Fetch from Hugging Face API
- Get downloads, likes, metadata
- Filter by base model

Usage:
```bash
python fetch_hf_downloads.py --base-model "Qwen/Qwen2.5-0.5B" --min-downloads 10
```

---

## For Paper Writing

### Key Points to Include

1. **Hypothesis Validation**
   - Low download models show more noise (validated)
   - Can use downloads as quality indicator

2. **Method Effectiveness**
   - Clear distinction: same family (40-70%) vs different family (<1%)
   - Works even with noisy low-download models (p < 0.001)

3. **Model Family Differences**
   - Qwen: Downloads correlate with overlap (Spearman = 0.43)
   - TinyLlama: No correlation (Spearman = -0.16)
   - TinyLlama has 2.27x higher mean overlap

### Figures to Use

1. `qwen_vs_llama_comparison.png` - Main comparison
2. `overlap_vs_downloads_analysis.png` - Detailed analysis
3. Tables from `SUMMARY.md` - Statistics

### Statistics to Cite

- Same vs Different family: p < 0.001 (highly significant)
- Qwen-downloads correlation: r = 0.43, p < 0.05
- TinyLlama-downloads correlation: r = -0.16, p > 0.05

---

## Next Steps

1. **Re-run lineage recovery** with downloads > 10 threshold
2. **Add weighted accuracy** (higher weight for high-download models)
3. **Analyze anomalies** (high downloads + low overlap, etc.)
4. **Combine metadata** (creation time, author info, etc.)

---

## Citation

If you use this analysis in your research:

```bibtex
@misc{download_overlap_analysis_2026,
  title={Model Lineage Recovery: Download Count and Overlap Analysis},
  author={[Your Name]},
  year={2026},
  note={Analysis of Qwen and TinyLlama model families}
}
```

---

## Contact

For questions about this analysis, please refer to:
- Main report: `ANALYSIS_REPORT.md`
- Experiment details: `experiment/SUMMARY.md`
- Research implications: `experiment/ANALYSIS_SUMMARY.md`

---

**Last Updated**: 2026-01-26  
**Language**: English  
**Data Source**: Hugging Face Model Hub  
**Analysis Tool**: Python + Pandas + Matplotlib
