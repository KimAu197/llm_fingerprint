#!/usr/bin/env python3
"""Write models_singleton_clusters.csv (cluster size == 1) from model_lineage_data.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from paths import experiment_root


def main() -> None:
    root = experiment_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", type=Path, default=root / "model_lineage_data.json")
    ap.add_argument("--out", type=Path, default=root / "models_singleton_clusters.csv")
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text(encoding="utf-8"))
    singleton_cids = {c["cluster_id"] for c in data["clusters"] if int(c.get("size", 0)) == 1}

    model_to_cid: dict[str, int] = {}
    for c in data["clusters"]:
        cid = c["cluster_id"]
        for m in c.get("models") or []:
            model_to_cid[m] = cid

    fields = [
        "model_name",
        "model_short_name",
        "organization",
        "family",
        "cluster_id",
        "direct_parent",
        "base_model_from_dataset",
        "fetch_error",
    ]
    rows: list[dict[str, str]] = []
    for m in data["models"]:
        name = m["model_name"]
        cid = model_to_cid.get(name)
        if cid is None or cid not in singleton_cids:
            continue
        b = m.get("base_model_from_dataset")
        if isinstance(b, list):
            b_str = " | ".join(str(x) for x in b)
        else:
            b_str = (b or "") if b is not None else ""
        rows.append(
            {
                "model_name": name,
                "model_short_name": str(m.get("model_short_name", "")),
                "organization": str(m.get("organization", "")),
                "family": str(m.get("family", "")),
                "cluster_id": str(cid),
                "direct_parent": str(m.get("direct_parent") or ""),
                "base_model_from_dataset": b_str,
                "fetch_error": str(m.get("fetch_error") or ""),
            }
        )

    rows.sort(key=lambda r: r["model_name"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.out} ({len(rows)} singleton-cluster models)")


if __name__ == "__main__":
    main()
