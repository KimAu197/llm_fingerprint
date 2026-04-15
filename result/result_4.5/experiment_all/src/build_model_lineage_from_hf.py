#!/usr/bin/env python3
"""Build model_lineage_data.json from models_union.csv using Hugging Face card base_model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from paths import experiment_root


def load_models_from_union_csv(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "model_id" not in reader.fieldnames:
            raise ValueError(f"Expected column model_id in {path}, got {reader.fieldnames}")
        out: list[str] = []
        for row in reader:
            mid = (row.get("model_id") or "").strip()
            if mid:
                out.append(mid)
    seen: set[str] = set()
    uniq: list[str] = []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


def normalize_base_model(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        bases: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    bases.append(s)
            elif isinstance(item, dict) and "name" in item:
                s = str(item["name"]).strip()
                if s:
                    bases.append(s)
        return bases
    return []


def _retry_after_sleep(response) -> float | None:
    if response is None:
        return None
    h = response.headers.get("Retry-After")
    if not h:
        return None
    try:
        return float(h)
    except ValueError:
        return None


def fetch_card_bases(
    api: HfApi,
    model_id: str,
    max_retries: int = 12,
    backoff_cap_s: float = 180.0,
) -> tuple[list[str], str | None]:
    """Fetch base_model from model card; retry on rate limit / transient HTTP."""
    transient_status = {429, 502, 503, 504}
    last_err = None
    for attempt in range(max_retries):
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
            code = e.response.status_code if e.response is not None else 0
            if code in transient_status:
                ra = _retry_after_sleep(e.response)
                if ra is not None and ra > 0:
                    wait = min(ra + random.uniform(0, 3), 300)
                else:
                    wait = min(6.0 * (2**attempt) + random.uniform(0, 4), backoff_cap_s)
                time.sleep(wait)
                continue
            return [], f"http_error: {e}"
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2.0 * (attempt + 1) + random.uniform(0, 2), 60))
    return [], last_err or "unknown_error"


def main() -> None:
    root = experiment_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_csv", type=Path, default=root / "models_union.csv")
    ap.add_argument("--out", type=Path, default=root / "model_lineage_data.json")
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.55,
        help="Seconds between HF calls (success path); increase if you still see 429",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=12,
        help="Per-request retries for 429/502/503/504",
    )
    ap.add_argument(
        "--extra-passes",
        type=int,
        default=4,
        help="After first full pass, re-fetch only models that still have errors (0=off)",
    )
    ap.add_argument(
        "--between-passes",
        type=float,
        default=60.0,
        help="Pause (seconds) before each extra pass",
    )
    args = ap.parse_args()

    models = load_models_from_union_csv(args.models_csv)
    model_set = set(models)
    api = HfApi()

    id_to_bases: dict[str, list[str]] = {m: [] for m in models}
    id_to_err: dict[str, str] = {}

    def run_fetch_batch(targets: list[str]) -> None:
        for i, mid in enumerate(targets):
            bases, err = fetch_card_bases(api, mid, max_retries=args.max_retries)
            id_to_bases[mid] = bases
            if err:
                id_to_err[mid] = err
            else:
                id_to_err.pop(mid, None)
            if args.sleep and i + 1 < len(targets):
                time.sleep(args.sleep)

    run_fetch_batch(models)
    for p in range(args.extra_passes):
        bad = [m for m in models if m in id_to_err]
        if not bad:
            break
        print(f"Extra pass {p + 1}/{args.extra_passes}: {len(bad)} models still failing, waiting {args.between_passes}s…")
        time.sleep(args.between_passes)
        run_fetch_batch(bad)

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
            "description": "HF model-card base_model lineage for experiment_all/models_union.csv",
            "source_csv": str(args.models_csv.resolve()),
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
