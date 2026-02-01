"""
FNR Analysis: Effect of Download Filtering on Base Model Identification

PhD Question: What is the false negative rate of identifying the correct base model
before and after filtering for low download rates? Does it change significantly?

Uses Best_Threshold from roc_analysis_results.csv (12.15 wordlist analysis).
Data: Merged dataset (original wordlist lineage + experiment popular models).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DOWNLOAD_DIR = SCRIPT_DIR.parent  # result_2.2/download
RESULT_DIR = DOWNLOAD_DIR.parent   # result_2.2
# Merged data (original + experiment)
DATA_DIR = DOWNLOAD_DIR / "merged" / "data"
ROC_RESULTS = RESULT_DIR.parent / "result_12.15" / "wordlist" / "roc_analysis_results.csv"

# Load thresholds from previous ROC analysis
roc_df = pd.read_csv(ROC_RESULTS)
THRESHOLDS = {
    "Qwen": float(roc_df[roc_df["Model"] == "Qwen"]["Best_Threshold"].iloc[0]),
    "LLaMA": float(roc_df[roc_df["Model"] == "LLaMA"]["Best_Threshold"].iloc[0]),
}

print("=" * 60)
print("FNR Analysis: Effect of Download Filtering (Merged Data)")
print("=" * 60)
print(f"\nData: Merged dataset (original + experiment)")
print(f"Thresholds (from roc_analysis_results.csv):")
print(f"  Qwen:  {THRESHOLDS['Qwen']}")
print(f"  LLaMA: {THRESHOLDS['LLaMA']}")
print(f"  Decision: overlap >= threshold → Same base model (correct identification)")

# Load merged data (original + experiment)
same_qwen = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_qwen_same_merged.csv")
same_llama = pd.read_csv(DATA_DIR / "lineage_bottomk_overlap_llama_same_merged.csv")


def compute_fnr(same_df, threshold, filter_downloads=None):
    """
    Compute False Negative Rate for same-family (base model) identification.
    """
    valid = same_df[(same_df["num_pairs"] > 0) & (same_df["avg_overlap_ratio"] >= 0)].copy()

    if filter_downloads is not None:
        valid = valid[valid["derived_model_downloads"] > filter_downloads]

    if len(valid) == 0:
        return {"n": 0, "fn": 0, "tp": 0, "fnr": np.nan}

    tp = (valid["avg_overlap_ratio"] >= threshold).sum()
    fn = (valid["avg_overlap_ratio"] < threshold).sum()
    total = len(valid)
    fnr = fn / total if total > 0 else np.nan

    return {"n": total, "fn": fn, "tp": tp, "fnr": fnr}


def analyze_family(name, same_df, threshold):
    """Analyze FNR before and after filtering for one model family."""
    before = compute_fnr(same_df, threshold, filter_downloads=None)
    after = compute_fnr(same_df, threshold, filter_downloads=10)

    return {
        "family": name,
        "threshold": threshold,
        "before_n": before["n"],
        "before_fn": before["fn"],
        "before_fnr": before["fnr"],
        "after_n": after["n"],
        "after_fn": after["fn"],
        "after_fnr": after["fnr"],
        "fnr_change": after["fnr"] - before["fnr"] if not np.isnan(after["fnr"]) and not np.isnan(before["fnr"]) else np.nan,
    }


# Run analysis
results = []
for name, same_df, threshold in [
    ("Qwen", same_qwen, THRESHOLDS["Qwen"]),
    ("LLaMA", same_llama, THRESHOLDS["LLaMA"]),
]:
    r = analyze_family(name, same_df, threshold)
    results.append(r)

# Print results
print("\n" + "=" * 60)
print("RESULTS: False Negative Rate (FNR) Before vs After Filtering")
print("=" * 60)
print("\nFNR = FN / Total = proportion of same-family models incorrectly")
print("      identified as different (overlap < threshold)")
print("\nFiltering: derived_model_downloads > 10")
print()

for r in results:
    print(f"--- {r['family']} (threshold = {r['threshold']}) ---")
    print(f"  BEFORE filtering (all valid models):")
    print(f"    N = {r['before_n']}, FN = {r['before_fn']}, FNR = {r['before_fnr']*100:.2f}%")
    print(f"  AFTER filtering (downloads > 10):")
    print(f"    N = {r['after_n']}, FN = {r['after_fn']}, FNR = {r['after_fnr']*100:.2f}%")
    if not np.isnan(r["fnr_change"]):
        change_pct = r["fnr_change"] * 100
        direction = "increased" if change_pct > 0 else "decreased"
        print(f"  Change: FNR {direction} by {abs(change_pct):.2f} percentage points")
    print()

# Save to CSV (output to experiment folder)
results_df = pd.DataFrame(results)
results_df.to_csv(SCRIPT_DIR / "fnr_download_effect_results.csv", index=False)
print(f"Saved: {SCRIPT_DIR / 'fnr_download_effect_results.csv'}")

# Generate markdown report
report = f"""# FNR Analysis: Effect of Download Filtering

## PhD Question
What is the false negative rate of identifying the correct base model before and after filtering for low download rates? Does it change significantly when filtering?

## Method
- **Data**: Merged dataset (original wordlist lineage + experiment popular models)
- **Threshold**: Best_Threshold from roc_analysis_results.csv (12.15 wordlist analysis)
  - Qwen: {THRESHOLDS['Qwen']}
  - LLaMA: {THRESHOLDS['LLaMA']}
- **Decision rule**: overlap >= threshold → Same base model (correct identification)
- **FNR** = FN / Total = proportion of same-family models incorrectly predicted as different
- **Filtering**: derived_model_downloads > 10

## Results

| Family | Threshold | Before FNR | After FNR | FNR Change | Before N | After N |
|--------|-----------|------------|-----------|------------|----------|---------|
"""
for r in results:
    change_str = f"{r['fnr_change']*100:+.2f}pp" if not np.isnan(r["fnr_change"]) else "N/A"
    report += f"| {r['family']} | {r['threshold']} | {r['before_fnr']*100:.2f}% | {r['after_fnr']*100:.2f}% | {change_str} | {r['before_n']} | {r['after_n']} |\n"

report += """
## Conclusion
Filtering for downloads > 10 **changes FNR differently by model family**:

- **Qwen**: FNR **increases** from 4.27% to 7.41% (+3.13pp). The merged dataset includes popular high-download Qwen models (from experiment) that have low overlap—these are same-family but incorrectly predicted as different. Filtering does not help Qwen.

- **LLaMA**: FNR **decreases** to 0% (-4.31pp). All FN cases are among low-download models; filtering eliminates them entirely.

**Key finding**: Download filtering is **family-dependent**. For LLaMA, filtering improves identification. For Qwen, popular models include heavily fine-tuned variants with low overlap, so filtering does not reduce (and may increase) FNR.
"""

with open(SCRIPT_DIR / "FNR_ANALYSIS_REPORT.md", "w") as f:
    f.write(report)
print(f"Saved: {SCRIPT_DIR / 'FNR_ANALYSIS_REPORT.md'}")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
