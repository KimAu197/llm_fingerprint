#!/usr/bin/env python3
"""Build model_lineage_data.json from overlap matrix CSV using Hugging Face card base_model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from paths import result_root


def load_models_from_overlap_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
    return [h.strip() for h in header[1:] if h.strip()]


def load_models_from_model_ids_csv(path: Path) -> list[str]:
    """Single-column CSV: optional header model_id, then one repo id per row."""
    out: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row or not row[0].strip():
                continue
            cell = row[0].strip()
            if i == 0 and cell.lower() == "model_id":
                continue
            out.append(cell)
    return out


def normalize_base_model(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict) and "name" in item:
                s = str(item["name"]).strip()
                if s:
                    out.append(s)
        return out
    return []


def fetch_card_bases(api: HfApi, model_id: str, retries: int = 3) -> tuple[list[str], str | None]:
    """Return (all bases from card, error message if any)."""
    last_err = None
    for attempt in range(retries):
        try:
            info = api.model_info(model_id, expand=["cardData"])
            cd = info.card_data
            if cd is None:
                return [], None
            bases = normalize_base_model(cd.base_model)
            return bases, None
        except RepositoryNotFoundError as e:
            return [], f"not_found: {e}"
        except HfHubHTTPError as e:
            last_err = str(e)
            if e.response is not None and e.response.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            return [], f"http_error: {e}"
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    return [], last_err or "unknown_error"


def main() -> None:
    root = result_root()
    ap = argparse.ArgumentParser(
        description="Build model_lineage_data.json from HF card base_model (in-set edges only)."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Overlap matrix CSV (column headers = model ids). Default: overlap_matrix_300_clean.csv",
    )
    ap.add_argument(
        "--model-ids",
        type=Path,
        default=None,
        help="Single-column model_id CSV (e.g. closure/model_ids_hf_card_closure.csv)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output JSON path")
    ap.add_argument("--sleep", type=float, default=0.08, help="Seconds between HF API calls")
    args = ap.parse_args()

    if args.model_ids is not None:
        models = load_models_from_model_ids_csv(args.model_ids)
        source_path = args.model_ids
        desc = "HF model-card base_model lineage for models in --model-ids list (in-set edges only)"
    elif args.csv is not None:
        models = load_models_from_overlap_csv(args.csv)
        source_path = args.csv
        desc = "HF model-card base_model lineage for overlap_matrix_300_clean models"
    else:
        args.csv = root / "overlap_matrix_300_clean.csv"
        models = load_models_from_overlap_csv(args.csv)
        source_path = args.csv
        desc = "HF model-card base_model lineage for overlap_matrix_300_clean models"

    if args.out is None:
        if args.model_ids is not None:
            args.out = args.model_ids.parent / "model_lineage_data.json"
        else:
            args.out = root / "model_lineage_data.json"

    model_set = set(models)
    api = HfApi()

    id_to_bases: dict[str, list[str]] = {}
    id_to_err: dict[str, str] = {}
    for i, mid in enumerate(models):
        bases, err = fetch_card_bases(api, mid)
        id_to_bases[mid] = bases
        if err:
            id_to_err[mid] = err
        if args.sleep and i + 1 < len(models):
            time.sleep(args.sleep)

    id_to_parents_in: dict[str, list[str]] = {}
    for mid, bases in id_to_bases.items():
        id_to_parents_in[mid] = [b for b in bases if b in model_set]

    direct_parent: dict[str, str | None] = {}
    for mid, ps in id_to_parents_in.items():
        direct_parent[mid] = ps[0] if ps else None

    parent_u: dict[str, str] = {}

    def ufind(x: str) -> str:
        if x not in parent_u:
            parent_u[x] = x
        if parent_u[x] != x:
            parent_u[x] = ufind(parent_u[x])
        return parent_u[x]

    def union(a: str, b: str) -> None:
        ra, rb = ufind(a), ufind(b)
        if ra != rb:
            parent_u[ra] = rb

    for mid in models:
        ufind(mid)
    for mid, ps in id_to_parents_in.items():
        for p in ps:
            union(mid, p)

    clusters: dict[str, list[str]] = defaultdict(list)
    for mid in models:
        clusters[ufind(mid)].append(mid)

    def canonical_root(mid: str) -> str | None:
        if not id_to_parents_in[mid] and direct_parent[mid] is None:
            return None
        cur = mid
        seen: set[str] = set()
        while direct_parent[cur] is not None and direct_parent[cur] in model_set:
            p = direct_parent[cur]
            if cur in seen:
                break
            seen.add(cur)
            cur = p
        return cur

    cluster_records = []
    cluster_id = 0
    for rep, members in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        cluster_id += 1
        ms = sorted(members)
        root_models = sorted([m for m in ms if direct_parent[m] is None])

        edges = 0
        for mid in ms:
            for p in id_to_parents_in[mid]:
                if p in model_set and p in ms:
                    edges += 1

        def depth_of(m: str) -> int:
            depth = 0
            cur = m
            seen: set[str] = set()
            while direct_parent.get(cur) is not None and direct_parent[cur] in model_set:
                if cur in seen:
                    break
                seen.add(cur)
                depth += 1
                cur = direct_parent[cur]
            return depth

        max_depth = max((depth_of(m) for m in ms), default=0)

        root_counts: dict[str, int] = defaultdict(int)
        for m in ms:
            r = canonical_root(m)
            if r is not None:
                root_counts[r] += 1
        if root_counts:
            fam_src = max(root_counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        else:
            fam_src = root_models[0] if root_models else ms[0]
        family = fam_src.split("/")[-1] if fam_src else "unknown"

        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "size": len(ms),
                "num_edges": edges,
                "root_models": root_models,
                "max_depth": max_depth,
                "models": ms,
                "family": family,
            }
        )

    model_records = []
    children_count: dict[str, int] = defaultdict(int)
    for mid in models:
        par = direct_parent[mid]
        if par is not None:
            children_count[par] += 1

    def compute_depth(mid: str) -> int:
        seen = set()
        cur = mid
        d = 0
        while cur in model_set and direct_parent[cur] in model_set:
            p = direct_parent[cur]
            if p in seen:
                break
            seen.add(cur)
            d += 1
            cur = p
        return d

    cid_by_model: dict[str, str] = {}
    for c in cluster_records:
        for m in c["models"]:
            cid_by_model[m] = c["family"]

    for mid in sorted(models):
        par = direct_parent[mid]
        org = mid.split("/")[0] if "/" in mid else ""
        short = mid.split("/")[-1]
        bases = id_to_bases[mid]
        root_m = canonical_root(mid)

        if id_to_parents_in[mid]:
            b_dataset = id_to_parents_in[mid][0]
        elif bases:
            b_dataset = bases[0] if len(bases) == 1 else bases
        else:
            b_dataset = None

        rec = {
            "model_name": mid,
            "model_short_name": short,
            "organization": org,
            "family": cid_by_model[mid],
            "direct_parent": par,
            "root_model": root_m,
            "is_root": par is None,
            "depth": compute_depth(mid),
            "num_children": children_count[mid],
            "outlier_count": None,
            "base_model_from_dataset": b_dataset,
        }
        if len(bases) > 1:
            rec["hf_base_models"] = bases
        err = id_to_err.get(mid)
        if err:
            rec["fetch_error"] = err
        model_records.append(rec)

    out_obj = {
        "metadata": {
            "total_models": len(models),
            "total_clusters": len(cluster_records),
            "total_edges": sum(c["num_edges"] for c in cluster_records),
            "description": desc,
            "source_csv": str(source_path.resolve()),
        },
        "clusters": cluster_records,
        "models": model_records,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {args.out} ({len(models)} models, {len(cluster_records)} clusters)")


if __name__ == "__main__":
    main()
