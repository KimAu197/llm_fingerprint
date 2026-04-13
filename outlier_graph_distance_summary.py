import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTLIER_PATTERN = re.compile(r"^(.+?)\(([\d.]+)\)\s*$")


def parse_outliers_cell(cell):
    if pd.isna(cell) or not str(cell).strip():
        return []
    items = []
    for part in str(cell).split("|"):
        part = part.strip()
        m = OUTLIER_PATTERN.match(part)
        if m:
            items.append((m.group(1).strip(), float(m.group(2))))
        elif part:
            items.append((part, np.nan))
    return items


def main():
    base = Path(__file__).resolve().parent / "result" / "result_3.30"
    tukey_path = base / "tukey_fence_eval.csv"
    dist_path = base / "distance_matrix.csv"

    tukey = pd.read_csv(tukey_path)
    dist = pd.read_csv(dist_path, index_col=0)

    index_set = set(dist.index.astype(str))
    col_set = set(dist.columns.astype(str))

    pair_rows = []
    summary_rows = []

    for _, row in tukey.iterrows():
        model = str(row["model"])
        outliers = parse_outliers_cell(row.get("outliers"))

        distances = []
        missing = 0
        missing_outlier_names = []
        for outlier_name, overlap_score in outliers:
            on = str(outlier_name)
            if model not in index_set or on not in col_set:
                d = np.nan
                missing += 1
                missing_outlier_names.append(on)
            else:
                d = dist.loc[model, on]
                if pd.isna(d):
                    missing += 1
                    missing_outlier_names.append(on)
                else:
                    distances.append(float(d))

            pair_rows.append(
                {
                    "model": model,
                    "outlier": on,
                    "overlap_score": overlap_score,
                    "graph_distance": float(d) if not pd.isna(d) else np.nan,
                }
            )

        n_total = len(outliers)
        n_defined = len(distances)
        if missing > 0 or n_total == 0:
            mean_d = np.nan
        else:
            mean_d = float(np.mean(distances)) if distances else np.nan

        summary_rows.append(
            {
                "model": model,
                "outlier_count_csv": int(row.get("outlier_count", n_total)),
                "n_outliers_parsed": n_total,
                "n_graph_distance_defined": n_defined,
                "n_graph_distance_missing": missing,
                "outliers_missing_graph_distance": " | ".join(missing_outlier_names)
                if missing_outlier_names
                else "",
                "mean_graph_distance_to_outliers": mean_d,
            }
        )

    df_pairs = pd.DataFrame(pair_rows)
    df_summary = pd.DataFrame(summary_rows)
    col = "mean_graph_distance_to_outliers"
    df_summary[col] = df_summary[col].apply(
        lambda x: round(float(x), 6) if pd.notna(x) else np.nan
    )

    df_pairs.to_csv(base / "tukey_outliers_with_graph_distance.csv", index=False)
    df_summary.to_csv(base / "model_mean_outlier_graph_distance.csv", index=False)

    print(f"Pair rows: {len(df_pairs)}")
    print(f"Models: {len(df_summary)}")
    valid = df_summary["mean_graph_distance_to_outliers"].dropna()
    print(f"Models with at least one defined distance: {len(valid)}")
    if len(valid):
        print(valid.describe())
    print(f"\nWrote {base / 'tukey_outliers_with_graph_distance.csv'}")
    print(f"Wrote {base / 'model_mean_outlier_graph_distance.csv'}")


if __name__ == "__main__":
    main()
