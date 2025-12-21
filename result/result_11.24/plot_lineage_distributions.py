# -*- coding: utf-8 -*-
# Usage:
#   python plot_lineage_distributions_by_num.py [--overall-col "overal score"] file1.csv file2.csv ...

import sys, os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["pal_k_mean", "lev_sim_mean", "lcs_ratio_mean"]
# Common "overall score" candidates (fuzzy match, ignoring case/spaces/underscores)
OVERALL_CANDIDATES = [
    "overall_score","overal_score","overall","overal","lineage_score",
    "final_score","fingerprint_score","index","score_mean"
]

def read_csv_robust(p):
    for kw in [dict(), dict(sep=";"), dict(sep=None, engine="python")]:
        try:
            return pd.read_csv(p, **kw)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {p}")

def infer_label_from_filename(path):
    name = os.path.basename(path).lower()
    if "same" in name: return "same"
    if "diff" in name or "different" in name: return "diff"
    return None

def extract_num_tag(path):
    name = os.path.basename(path)
    m = re.search(r'(?:same|diff)[^\d]*?(\d+)(?:\D+|$)', name, flags=re.I)
    if not m: m = re.search(r'(\d+)(?=\D*$)', name)
    return m.group(1) if m else "misc"

def normalize_label_series(s):
    def _m(x):
        t = str(x).strip().lower()
        if t in {"1","true","yes","y","same"}: return "same"
        if t in {"0","false","no","n","diff","different"}: return "diff"
        return np.nan
    return s.map(_m)

def norm_name(c):  # Normalize column name: lowercase, remove non-alphanumeric
    return re.sub(r'[^a-z0-9]+','', str(c).lower())

def find_overall_column(df, user_col=None):
    norm_map = {norm_name(c): c for c in df.columns}
    if user_col is not None:
        key = norm_name(user_col)
        if key in norm_map:
            col = norm_map[key]
            if pd.to_numeric(df[col], errors="coerce").notna().any():
                return col
    for cand in OVERALL_CANDIDATES:
        key = norm_name(cand)
        if key in norm_map:
            col = norm_map[key]
            if pd.to_numeric(df[col], errors="coerce").notna().any():
                return col
    return None

def load_and_group(files, overall_col_hint=None):
    groups = {}
    for f in files:
        df = read_csv_robust(f)

        # label
        label_col = next((c for c in ["label","y","is_same","target","gt","ground_truth"] if c in df.columns), None)
        if label_col: lbl = normalize_label_series(df[label_col])
        else:
            inf = infer_label_from_filename(f)
            if inf is None:
                print(f"[warn] skip (no label): {f}"); continue
            lbl = pd.Series([inf]*len(df))

        # Record overall score column name (may differ per file)
        overall_col = find_overall_column(df, user_col=overall_col_hint)

        keep = set(METRICS) & set(df.columns)
        out = pd.DataFrame(index=df.index)
        for m in keep:
            out[m] = pd.to_numeric(df[m], errors="coerce")
        if overall_col:
            out["__overall__"] = pd.to_numeric(df[overall_col], errors="coerce")
        out["__label__"] = lbl.values
        out["__file__"]  = os.path.basename(f)
        out["__overall_col_name__"] = overall_col if overall_col else ""

        num = extract_num_tag(f)
        groups.setdefault(num, []).append(out)

    for num in list(groups.keys()):
        g = pd.concat(groups[num], ignore_index=True)
        g = g[g["__label__"].isin(["same","diff"])].copy()
        groups[num] = g
    return groups

def build_overall_score(df):
    # If __overall__ exists and has data, return directly
    if "__overall__" in df.columns and df["__overall__"].notna().any():
        return df["__overall__"], df["__overall_col_name__"].mode().iat[0] if df["__overall_col_name__"].notna().any() else "overall"
    # Otherwise use z-mean of three metrics
    avail = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    if not avail: raise ValueError("No metrics to compose overall score.")
    zs = []
    for m in avail:
        x = df[m].astype(float)
        mu, sd = x.mean(), x.std(ddof=1) or 1.0
        zs.append((x - mu) / sd)
    overall = sum(zs) / len(zs)
    return overall, "z-mean(pal_k_mean,lev_sim_mean,lcs_ratio_mean)"

def plot_hist(same_vals, diff_vals, title, xlabel, outpath):
    plt.figure()
    if len(same_vals): plt.hist(same_vals, bins=40, alpha=0.6, density=True, label="same")
    if len(diff_vals): plt.hist(diff_vals, bins=40, alpha=0.6, density=True, label="diff")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("density"); plt.legend()
    plt.savefig(outpath, bbox_inches="tight"); plt.close()
    print(f"[save] {outpath} (same={len(same_vals)}, diff={len(diff_vals)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overall-col", type=str, default=None,
                    help="Explicitly specify overall score column name (e.g., 'overal score' or 'overall_score').")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    groups = load_and_group(args.files, overall_col_hint=args.overall_col)
    if not groups:
        print("[error] no usable data."); sys.exit(2)

    for num, df in groups.items():
        # Three individual metrics
        for m in METRICS:
            if m not in df.columns: 
                print(f"[info][pair={num}] metric missing: {m}"); continue
            s = df.loc[df["__label__"]=="same", m].dropna().to_numpy()
            d = df.loc[df["__label__"]=="diff", m].dropna().to_numpy()
            plot_hist(s, d, f"Distribution - {m} (pair={num})", m, f"dist_{m}_num{num}.png")

        # Overall score
        overall, used_col = build_overall_score(df)
        s = overall[df["__label__"]=="same"].dropna().to_numpy()
        d = overall[df["__label__"]=="diff"].dropna().to_numpy()
        plot_hist(s, d, f"Distribution - overall score (pair={num})\n[source={used_col}]",
                  "overall_score", f"dist_overall_score_num{num}.png")

if __name__ == "__main__":
    main()
