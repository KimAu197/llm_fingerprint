#!/usr/bin/env python3
"""Tukey fence (TF) on overlap matrix rows; lineage base_model from model_lineage_data.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paths import result_root


def load_overlap_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip() for c in header[1:] if c.strip()]
        mat: dict[str, dict[str, float]] = {}
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            vals = row[1 : 1 + len(cols)]
            mat[name] = {}
            for j, c in enumerate(cols):
                if j < len(vals) and vals[j].strip() != "":
                    try:
                        mat[name][c] = float(vals[j])
                    except ValueError:
                        mat[name][c] = float("nan")
    return cols, mat


def load_effective_base(lineage_path: Path) -> dict[str, str | None]:
    data = json.loads(lineage_path.read_text(encoding="utf-8"))
    out: dict[str, str | None] = {}
    for m in data.get("models", []):
        name = m.get("model_name")
        if not name:
            continue
        par = m.get("direct_parent")
        if par:
            out[name] = par
            continue
        b = m.get("base_model_from_dataset")
        if isinstance(b, str) and b.strip():
            out[name] = b.strip()
        elif isinstance(b, list) and b:
            first = b[0]
            if isinstance(first, str) and first.strip():
                out[name] = first.strip()
            else:
                out[name] = None
        else:
            out[name] = None
    return out


def quartiles(vals: list[float]) -> tuple[float, float, float]:
    if not vals:
        return 0.0, 0.0, 0.0
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return float(s[0])
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        w = idx - lo
        return float(s[lo] * (1 - w) + s[hi] * w)

    q1 = q(0.25)
    q3 = q(0.75)
    return q1, q3, q3 - q1


def round4(x: float) -> float:
    return round(float(x), 4)


def base_in_top_band(scores_no_self: list[float], base_score: float | None) -> int:
    if base_score is None:
        return 0
    levels = sorted({round4(v) for v in scores_no_self}, reverse=True)
    if not levels:
        return 0
    br = round4(base_score)
    try:
        return levels.index(br) + 1
    except ValueError:
        for i, lv in enumerate(levels):
            if abs(br - lv) < 1e-9:
                return i + 1
        return 0


def main() -> None:
    root = result_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap_csv", type=Path, default=root / "overlap_matrix_300_clean.csv")
    ap.add_argument("--lineage", type=Path, default=root / "model_lineage_data.json")
    ap.add_argument("--out_tukey", type=Path, default=root / "tukey_fence_eval.csv")
    ap.add_argument("--out_long", type=Path, default=root / "model_outliers_long.csv")
    args = ap.parse_args()

    models, mat = load_overlap_matrix(args.overlap_csv)
    model_set = set(models)
    base_of = load_effective_base(args.lineage)

    tukey_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    for m in models:
        row_d = mat.get(m, {})
        pairs: list[tuple[str, float]] = []
        for j in models:
            if j == m:
                continue
            v = row_d.get(j)
            if v is None or v != v:
                continue
            pairs.append((j, float(v)))

        scores = [p[1] for p in pairs]
        q1, q3, iqr = quartiles(scores)
        upper = q3 + 1.5 * iqr

        outliers: list[tuple[str, float]] = [(j, v) for j, v in pairs if v > upper]
        outliers.sort(key=lambda x: -x[1])

        base = base_of.get(m)
        base_in_ts = "yes" if (base and base in model_set) else "no"
        base_score: float | None = None
        if base and base in row_d and row_d[base] == row_d[base]:
            base_score = float(row_d[base])

        bit = base_in_top_band(scores, base_score) if base_in_ts == "yes" else 0

        base_in_ol = (
            "yes"
            if base_in_ts == "yes" and base_score is not None and base_score > upper
            else "no"
        )

        ol_parts = [f"{j}({v:.4f})" for j, v in outliers]
        outliers_str = " | ".join(ol_parts)

        tukey_rows.append(
            {
                "model": m,
                "base_model": base or "",
                "base_in_testset": base_in_ts,
                "base_in_top": bit,
                "Q1": round4(q1),
                "Q3": round4(q3),
                "IQR": round4(iqr),
                "upper_fence": round4(upper),
                "outlier_count": len(outliers),
                "base_score": round4(base_score) if base_score is not None else "",
                "base_in_outliers": base_in_ol,
                "outliers": outliers_str,
            }
        )

        for j, v in outliers:
            long_rows.append(
                {
                    "anchor_model": m,
                    "outlier_model": j,
                    "overlap": round4(v),
                    "upper_fence": round4(upper),
                    "Q1": round4(q1),
                    "Q3": round4(q3),
                }
            )

    args.out_tukey.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "base_model",
        "base_in_testset",
        "base_in_top",
        "Q1",
        "Q3",
        "IQR",
        "upper_fence",
        "outlier_count",
        "base_score",
        "base_in_outliers",
        "outliers",
    ]
    with args.out_tukey.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in tukey_rows:
            w.writerow({k: r[k] for k in fieldnames})

    with args.out_long.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["anchor_model", "outlier_model", "overlap", "upper_fence", "Q1", "Q3"],
        )
        w.writeheader()
        w.writerows(long_rows)

    print(f"Wrote {args.out_tukey} ({len(tukey_rows)} models)")
    print(f"Wrote {args.out_long} ({len(long_rows)} outlier pairs)")


if __name__ == "__main__":
    main()
