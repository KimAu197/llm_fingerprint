#!/usr/bin/env python3
"""Evaluate Tukey fence outliers vs strict (1-hop) or loose (same cluster) lineage."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

OUTLIER_ENTRY_RE = re.compile(r"^(.+)\(([0-9.]+)\)\s*$")


def load_lineage_parents(path: Path) -> dict[str, str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str | None] = {}
    for m in data.get("models", []):
        name = m.get("model_name")
        if name:
            out[name] = m.get("direct_parent")
    return out


def load_cluster_map(path: Path) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    """model -> peers in same cluster; model -> cluster family label."""
    data = json.loads(path.read_text(encoding="utf-8"))
    peers: dict[str, frozenset[str]] = {}
    family: dict[str, str] = {}
    for c in data.get("clusters", []):
        models = frozenset(c.get("models") or [])
        fam = str(c.get("family", ""))
        for mname in models:
            peers[mname] = models
            family[mname] = fam
    return peers, family


def strict_neighbors(parent_of: dict[str, str | None], model: str) -> set[str]:
    s: set[str] = set()
    p = parent_of.get(model)
    if p:
        s.add(p)
    for child, par in parent_of.items():
        if par == model:
            s.add(child)
    return s


def loose_peers_except_self(cluster_map: dict[str, frozenset[str]], model: str) -> set[str]:
    c = cluster_map.get(model)
    if not c:
        return set()
    return set(c) - {model}


def parse_outliers(cell: str) -> list[str]:
    if not cell or not str(cell).strip():
        return []
    names: list[str] = []
    for part in str(cell).split("|"):
        part = part.strip()
        if not part:
            continue
        m = OUTLIER_ENTRY_RE.match(part)
        if m:
            names.append(m.group(1).strip())
        else:
            names.append(part)
    return names


def universe_from_tukey_rows(rows: list[dict[str, str]]) -> set[str]:
    u: set[str] = set()
    for r in rows:
        u.add(r["model"].strip())
        for n in parse_outliers(r.get("outliers", "")):
            u.add(n)
    return u


def write_strict_csv(
    out_path: Path,
    rows: list[dict[str, str]],
    u: set[str],
    parent_of: dict[str, str | None],
) -> tuple[float, float]:
    summary_rows: list[dict[str, object]] = []
    tot_pred = tot_tp = tot_gold = 0
    tot_gold_rows = 0

    for r in rows:
        focus = r["model"].strip()
        predicted = parse_outliers(r.get("outliers", ""))
        gold_all = strict_neighbors(parent_of, focus)
        gold = gold_all & u
        pred_set = set(predicted)
        tp_set = pred_set & gold_all
        tp_in_u = pred_set & gold

        n_pred = len(predicted)
        n_hit = len(tp_set)
        n_gold = len(gold)

        if n_gold > 0:
            rec = len(tp_in_u) / n_gold
            tot_gold_rows += 1
        else:
            rec = float("nan")

        prec = (n_hit / n_pred) if n_pred > 0 else float("nan")
        tot_pred += n_pred
        tot_tp += n_hit
        tot_gold += n_gold

        summary_rows.append(
            {
                "model": focus,
                "base_model": r.get("base_model", ""),
                "n_outliers": n_pred,
                "n_strict_related_in_outliers": n_hit,
                "n_strict_related_in_universe": n_gold,
                "precision_outliers_vs_strict": "" if n_pred == 0 else round(prec, 6),
                "recall_strict_in_universe_captured": ""
                if n_gold == 0
                else round(rec, 6),
                "strict_related_outliers": " | ".join(sorted(tp_set)),
                "strict_related_in_universe_not_in_outliers": " | ".join(
                    sorted(gold - pred_set)
                ),
            }
        )

    micro_prec = (tot_tp / tot_pred) if tot_pred > 0 else float("nan")
    micro_rec = (tot_tp / tot_gold) if tot_gold > 0 else float("nan")
    cols = list(summary_rows[0].keys()) if summary_rows else []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sr in summary_rows:
            w.writerow(sr)
        w.writerow(
            {
                "model": "_SUMMARY_",
                "base_model": "",
                "n_outliers": tot_pred,
                "n_strict_related_in_outliers": tot_tp,
                "n_strict_related_in_universe": tot_gold,
                "precision_outliers_vs_strict": round(micro_prec, 6),
                "recall_strict_in_universe_captured": round(micro_rec, 6),
                "strict_related_outliers": f"micro_TP={tot_tp}_pred={tot_pred}_gold_universe_sum={tot_gold}",
                "strict_related_in_universe_not_in_outliers": f"rows_with_strict_neighbor_in_universe={tot_gold_rows}",
            }
        )
    return micro_prec, micro_rec


def write_loose_csv(
    out_path: Path,
    rows: list[dict[str, str]],
    u: set[str],
    cluster_map: dict[str, frozenset[str]],
    family_of: dict[str, str],
) -> tuple[float, float]:
    summary_rows: list[dict[str, object]] = []
    tot_pred = tot_tp = tot_gold = 0
    tot_gold_rows = 0

    for r in rows:
        focus = r["model"].strip()
        predicted = parse_outliers(r.get("outliers", ""))
        gold_all = loose_peers_except_self(cluster_map, focus)
        gold = gold_all & u
        pred_set = set(predicted)
        tp_set = pred_set & gold_all
        tp_in_u = pred_set & gold

        n_pred = len(predicted)
        n_hit = len(tp_set)
        n_gold = len(gold)

        if n_gold > 0:
            rec = len(tp_in_u) / n_gold
            tot_gold_rows += 1
        else:
            rec = float("nan")

        prec = (n_hit / n_pred) if n_pred > 0 else float("nan")
        tot_pred += n_pred
        tot_tp += n_hit
        tot_gold += n_gold

        summary_rows.append(
            {
                "model": focus,
                "base_model": r.get("base_model", ""),
                "cluster_family": family_of.get(focus, ""),
                "n_outliers": n_pred,
                "n_loosely_related_in_outliers": n_hit,
                "n_loosely_related_in_universe": n_gold,
                "precision_outliers_vs_loose": "" if n_pred == 0 else round(prec, 6),
                "recall_loose_in_universe_captured": ""
                if n_gold == 0
                else round(rec, 6),
                "loosely_related_outliers": " | ".join(sorted(tp_set)),
                "loosely_related_in_universe_not_in_outliers": " | ".join(
                    sorted(gold - pred_set)
                ),
            }
        )

    micro_prec = (tot_tp / tot_pred) if tot_pred > 0 else float("nan")
    micro_rec = (tot_tp / tot_gold) if tot_gold > 0 else float("nan")
    cols = list(summary_rows[0].keys()) if summary_rows else []

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sr in summary_rows:
            w.writerow(sr)
        w.writerow(
            {
                "model": "_SUMMARY_",
                "base_model": "",
                "cluster_family": "",
                "n_outliers": tot_pred,
                "n_loosely_related_in_outliers": tot_tp,
                "n_loosely_related_in_universe": tot_gold,
                "precision_outliers_vs_loose": round(micro_prec, 6),
                "recall_loose_in_universe_captured": round(micro_rec, 6),
                "loosely_related_outliers": f"micro_TP={tot_tp}_pred={tot_pred}_gold_universe_sum={tot_gold}",
                "loosely_related_in_universe_not_in_outliers": f"rows_with_loose_peer_in_universe={tot_gold_rows}",
            }
        )
    return micro_prec, micro_rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", type=Path, required=True)
    ap.add_argument("--tukey", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Strict metrics CSV (default: <tukey_dir>/tukey_outliers_strict_recall_precision.csv)",
    )
    ap.add_argument(
        "--out-loose",
        type=Path,
        default=None,
        help="Loose (same-cluster) metrics CSV (default: <tukey_dir>/tukey_outliers_loose_recall_precision.csv)",
    )
    args = ap.parse_args()
    strict_out = args.out or (
        args.tukey.parent / "tukey_outliers_strict_recall_precision.csv"
    )
    loose_out = args.out_loose or (
        args.tukey.parent / "tukey_outliers_loose_recall_precision.csv"
    )

    parent_of = load_lineage_parents(args.lineage)
    cluster_map, family_of = load_cluster_map(args.lineage)
    with args.tukey.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    u = universe_from_tukey_rows(rows)

    sp, sr = write_strict_csv(strict_out, rows, u, parent_of)
    lp, lr = write_loose_csv(loose_out, rows, u, cluster_map, family_of)

    print(f"Wrote {strict_out}")
    print(f"  Micro precision (strict): {sp:.6f}; recall: {sr:.6f}")
    print(f"Wrote {loose_out}")
    print(f"  Micro precision (loose):  {lp:.6f}; recall:  {lr:.6f}")


if __name__ == "__main__":
    main()
