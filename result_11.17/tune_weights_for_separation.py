# tune_weights_for_separation.py  (fixed print + a clearer sign map)

import sys, os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_METRICS = ["pal_k_mean", "lev_sim_mean", "lcs_ratio_mean"]

def read_csv_robust(p: str) -> pd.DataFrame:
    for kw in [dict(), dict(sep=";"), dict(sep=None, engine="python")]:
        try:
            return pd.read_csv(p, **kw)
        except Exception:
            pass
    raise RuntimeError(f"Cannot read CSV: {p}")

def infer_label_from_filename(path: str):
    name = os.path.basename(path).lower()
    if "same" in name: return "same"
    if "diff" in name or "different" in name: return "diff"
    return None

def normalize_label_series(s: pd.Series) -> pd.Series:
    def _m(x):
        t = str(x).strip().lower()
        if t in {"1","true","yes","y","same"}: return "same"
        if t in {"0","false","no","n","diff","different"}: return "diff"
        return np.nan
    return s.map(_m)

def load_all(files, metrics):
    frames = []
    for f in files:
        df = read_csv_robust(f)
        label_col = next((c for c in ["label","y","is_same","target","gt","ground_truth"] if c in df.columns), None)
        if label_col: lbl = normalize_label_series(df[label_col])
        else:
            lab = infer_label_from_filename(f)
            if lab is None:
                print(f"[warn] skip (no label info): {f}")
                continue
            lbl = pd.Series([lab]*len(df))
        keep = [m for m in metrics if m in df.columns]
        if not keep:
            print(f"[warn] skip (no specified metrics found): {f}")
            continue
        out = df[keep].copy()
        for c in keep:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out["__label__"] = lbl
        out["__file__"]  = os.path.basename(f)
        frames.append(out)
    if not frames:
        raise ValueError("No usable rows from given files.")
    data = pd.concat(frames, ignore_index=True)
    data = data[data["__label__"].isin(["same","diff"])].copy()
    data = data.dropna(subset=metrics, how="all")
    return data

def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = (y_true > 0).astype(int)
    pos = scores[y == 1]; neg = scores[y == 0]
    P, N = len(pos), len(neg)
    if P == 0 or N == 0: return np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    ranks = ranks + 1
    r_pos = ranks[:P].sum()
    U = r_pos - P*(P+1)/2
    return float(U / (P*N))

def best_threshold_youden(y_true: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores)[::-1]
    s = scores[order]; y = y_true[order]
    P, N = y.sum(), len(y) - y.sum()
    bestJ = -1.0; best_t = None
    for t in np.unique(s):
        tp = ((y == 1) & (s >= t)).sum()
        fp = ((y == 0) & (s >= t)).sum()
        fn = int(P - tp)
        tn = int(N - fp)
        tpr = tp / P if P else 0.0
        fpr = fp / N if N else 0.0
        J = tpr - fpr
        if J > bestJ:
            bestJ = J; best_t = float(t)
    return best_t, bestJ

def tune_weights(df: pd.DataFrame, metrics, step=0.02, standardize=True, auto_flip=True):
    X = df[metrics].copy().astype(float)
    y = (df["__label__"] == "same").astype(int).to_numpy()

    signs = np.ones(len(metrics))
    if auto_flip:
        for i, m in enumerate(metrics):
            s_mean = X.loc[df["__label__"]=="same", m].mean()
            d_mean = X.loc[df["__label__"]=="diff", m].mean()
            if pd.notna(s_mean) and pd.notna(d_mean) and s_mean < d_mean:
                X[m] = -X[m]
                signs[i] = -1.0

    if standardize:
        for m in metrics:
            mu = X[m].mean()
            sd = X[m].std(ddof=1)
            X[m] = (X[m] - mu) / (sd if sd and sd > 0 else 1.0)

    Z = X[metrics].to_numpy()
    if Z.shape[1] != 3:
        raise ValueError(f"Expect exactly 3 metrics, got {Z.shape[1]}")

    grid = np.arange(0.0, 1.0 + 1e-9, step)
    best = {"auc": -1.0, "w": None, "scores": None}
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-12: continue
            w3 = 1.0 - w1 - w2
            w = np.array([w1, w2, w3], dtype=float)
            scores = Z @ w
            auc = auc_from_scores(y, scores)
            if np.isnan(auc): continue
            if auc > best["auc"] + 1e-9:
                best.update(auc=auc, w=w.copy(), scores=scores.copy())

    thr, J = best_threshold_youden(y, best["scores"])
    y_pred = (best["scores"] >= thr).astype(int)
    TP = int(((y==1) & (y_pred==1)).sum())
    FP = int(((y==0) & (y_pred==1)).sum())
    TN = int(((y==0) & (y_pred==0)).sum())
    FN = int(((y==1) & (y_pred==0)).sum())
    acc = (TP+TN)/len(y) if len(y) else np.nan
    prec = TP/max(TP+FP,1)
    rec  = TP/max(TP+FN,1)
    f1   = 2*prec*rec/max(prec+rec,1e-12)

    return {
        "metrics": metrics,
        "signs": signs.tolist(),
        "weights": best["w"].tolist(),
        "auc": float(best["auc"]),
        "best_threshold": float(thr),
        "youdenJ": float(J),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "scores": best["scores"],
        "labels": y
    }

def plot_distribution(scores, labels, out_path):
    s_same = scores[labels==1]; s_diff = scores[labels==0]
    plt.figure()
    if s_same.size: plt.hist(s_same, bins=40, alpha=0.6, density=True, label="same")
    if s_diff.size: plt.hist(s_diff, bins=40, alpha=0.6, density=True, label="diff")
    plt.title("Distribution — weighted combined score")
    plt.xlabel("combined_score"); plt.ylabel("density"); plt.legend()
    plt.savefig(out_path, bbox_inches="tight"); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metrics", nargs=3, default=DEFAULT_METRICS)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--no-auto-flip", action="store_true")
    ap.add_argument("--save-fig", action="store_true")
    args = ap.parse_args()

    df = load_all(args.files, args.metrics)
    res = tune_weights(
        df, args.metrics, step=args.step,
        standardize=(not args.no_standardize),
        auto_flip=(not args.no_auto_flip)
    )

    sign_map = {m:s for m,s in zip(res["metrics"], res["signs"])}
    feature_note = "z(metric)" if (not args.no_standardize) else "metric"

    print("\n=== Best linear combination (same ↑, diff ↓) ===")
    print(f"metrics: {res['metrics']}")
    print(f"auto_flip signs: {sign_map}   (actual feature used per metric = sign * {feature_note})")
    print(f"weights (w1,w2,w3, sum=1): {res['weights']}")
    print(f"ROC-AUC: {res['auc']:.4f}")
    print(f"Best threshold (Youden J): {res['best_threshold']:.6f}, J={res['youdenJ']:.4f}")
    print(f"@best τ  Acc={res['accuracy']:.3f}  Prec={res['precision']:.3f}  Rec={res['recall']:.3f}  F1={res['f1']:.3f}")

    if args.save_fig:
        plot_distribution(res["scores"], res["labels"], "dist_weighted_combined.png")
        print("[save] dist_weighted_combined.png")

if __name__ == "__main__":
    main()