"""
Merge original (wordlist lineage) + experiment (popular models) datasets.

Strategy:
- Original: result_1.26/download/original/data (lineage discovery)
- Experiment: result_1.26/download/experiment/output (popular models by downloads)
- For duplicates (same derived_model_name): keep original (has base_model_downloads)
- For experiment-only models: add with derived_model_downloads = downloads
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULT_1_26 = SCRIPT_DIR.parent.parent / "result_1.26"
ORIGINAL_DATA = RESULT_1_26 / "download" / "original" / "data"
EXPERIMENT_OUTPUT = RESULT_1_26 / "download" / "experiment" / "output"
MERGED_OUTPUT = SCRIPT_DIR / "merged" / "data"
MERGED_OUTPUT.mkdir(parents=True, exist_ok=True)


def merge_same_family(original_path, experiment_path, output_path, family_name):
    """Merge same-family data from original and experiment.
    For duplicates: prefer experiment's derived_model_downloads (more recent, popular models).
    """
    orig = pd.read_csv(original_path)
    exp = pd.read_csv(experiment_path)

    # Standardize experiment: rename 'downloads' -> 'derived_model_downloads'
    exp_std = exp.rename(columns={"downloads": "derived_model_downloads"})

    # Target columns (match original schema)
    target_cols = ["base_model_name", "derived_model_name", "base_model_downloads", "derived_model_downloads",
                  "num_pairs", "avg_overlap_ratio", "bottom_k_vocab_size", "pair_scores_json"]

    orig_sub = orig[[c for c in target_cols if c in orig.columns]].copy()
    for c in target_cols:
        if c not in orig_sub.columns and c in orig.columns:
            orig_sub[c] = orig[c]

    # Build exp_sub with same schema
    exp_sub = exp_std[["base_model_name", "derived_model_name", "num_pairs", "avg_overlap_ratio",
                       "bottom_k_vocab_size", "pair_scores_json"]].copy()
    exp_sub["derived_model_downloads"] = exp_std["derived_model_downloads"]
    exp_sub["base_model_downloads"] = orig["base_model_downloads"].iloc[0] if "base_model_downloads" in orig.columns and len(orig) > 0 else pd.NA

    # Merge: for duplicates, use max(original, experiment) for derived_model_downloads
    # so we don't lose models due to stale original data
    exp_downloads = exp_sub.set_index("derived_model_name")["derived_model_downloads"]
    orig_sub["derived_model_downloads"] = orig_sub.apply(
        lambda row: max(row["derived_model_downloads"], exp_downloads.get(row["derived_model_name"], row["derived_model_downloads"]))
        if row["derived_model_name"] in exp_downloads.index else row["derived_model_downloads"],
        axis=1
    )

    # Concat: original (with updated downloads) + experiment-only models
    orig_models = set(orig_sub["derived_model_name"])
    exp_new = exp_sub[~exp_sub["derived_model_name"].isin(orig_models)]
    combined = pd.concat([orig_sub, exp_new], ignore_index=True)

    merged = combined.drop_duplicates(subset=["derived_model_name"], keep="first")
    merged.to_csv(output_path, index=False)

    n_new = len(exp_new)
    print(f"  {family_name}: original={len(orig_sub)}, experiment_total={len(exp_sub)}, new_from_exp={n_new}, merged={len(merged)}")
    return merged


def main():
    print("=" * 60)
    print("Merging Original + Experiment Datasets")
    print("=" * 60)

    # Qwen same
    merge_same_family(
        ORIGINAL_DATA / "lineage_bottomk_overlap_qwen_same_with_downloads.csv",
        EXPERIMENT_OUTPUT / "qwen_overlap_downloads.csv",
        MERGED_OUTPUT / "lineage_bottomk_overlap_qwen_same_merged.csv",
        "Qwen",
    )

    # LLaMA same
    merge_same_family(
        ORIGINAL_DATA / "lineage_bottomk_overlap_llama_same_with_downloads.csv",
        EXPERIMENT_OUTPUT / "llama_overlap_downloads.csv",
        MERGED_OUTPUT / "lineage_bottomk_overlap_llama_same_merged.csv",
        "LLaMA",
    )

    # Diff data: experiment doesn't have diff, keep original only
    for fname in ["lineage_bottomk_overlap_qwen_diff_with_downloads.csv", "lineage_bottomk_overlap_llama_diff_with_downloads.csv"]:
        orig = pd.read_csv(ORIGINAL_DATA / fname)
        out_name = fname.replace("_with_downloads", "_merged")
        orig.to_csv(MERGED_OUTPUT / out_name, index=False)
        print(f"  {fname}: copied (experiment has no diff data)")

    print(f"\nMerged data saved to: {MERGED_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
