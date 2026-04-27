#!/usr/bin/env python3
"""End-to-end overlap matrix analysis pipeline.

Given a new overlap matrix and a relationship CSV, this script regenerates:
- Tukey outlier CSVs
- strict and loose lineage JSONs
- graph distance matrix
- strict/loose outlier precision-recall summaries
- mean graph distance from each model to its Tukey outliers
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


OUTLIER_ENTRY_RE = re.compile(r"^(.+?)\(([0-9.]+)\)\s*$")
NO_PARENT_VALUES = {"", "none", "null", "nan", "n/a", "na", "-"}
LLAMA31_CANONICAL = "meta-llama/Llama-3.1-8B"
HF_EQUIV_WEIGHT_ZERO = [
    ("meta-llama/Meta-Llama-3-8B", LLAMA31_CANONICAL),
    ("NousResearch/Meta-Llama-3-8B", LLAMA31_CANONICAL),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("meta-llama/Meta-Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
]


def parse_parent_cell(cell: Any) -> list[str]:
    """Parse relationship CSV base_model cells."""
    if cell is None:
        return []
    text = str(cell).strip()
    if text.lower() in NO_PARENT_VALUES:
        return []
    parts = re.split(r"[;,|]", text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = part.strip()
        if not value or value.lower() in NO_PARENT_VALUES:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def read_relationship_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_field(row: dict[str, Any], *names: str) -> str:
    lower_map = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for name in names:
        value = lower_map.get(name.lower())
        if value is not None:
            return str(value).strip()
    return ""


def load_overlap_matrix(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip() for c in header[1:] if c.strip()]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            model = row[0].strip()
            matrix[model] = {}
            for idx, col in enumerate(cols):
                if idx + 1 >= len(row) or row[idx + 1].strip() == "":
                    continue
                try:
                    matrix[model][col] = float(row[idx + 1])
                except ValueError:
                    matrix[model][col] = math.nan
    return cols, matrix


def _canonical_model_id(model_id: str) -> str:
    aliases = {
        "meta-llama/Meta-Llama-3.1-8B": LLAMA31_CANONICAL,
    }
    return aliases.get(model_id, model_id)


def fetch_hf_base_models(model_id: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required when --use-hf is enabled"
        ) from exc

    info = HfApi().model_info(model_id, expand=["cardData"])
    card_data = info.card_data
    if card_data is None:
        return []
    raw = getattr(card_data, "base_model", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return parse_parent_cell(raw)
    if isinstance(raw, list):
        bases: list[str] = []
        for item in raw:
            if isinstance(item, str):
                bases.extend(parse_parent_cell(item))
            elif isinstance(item, dict) and "name" in item:
                bases.extend(parse_parent_cell(item["name"]))
        return bases
    return []


def _relationship_row_model(row: dict[str, Any]) -> str:
    return get_field(row, "model_id", "model", "model_name")


def _relationship_row_parents(row: dict[str, Any]) -> list[str]:
    return parse_parent_cell(get_field(row, "base_model", "base", "parent"))


def build_lineage_from_relationship_rows(
    rows: list[dict[str, Any]],
    source: str,
    extra_models: list[str] | None = None,
    use_hf: bool = False,
    hf_sleep: float = 0.0,
) -> dict[str, Any]:
    ordered_models: list[str] = []
    row_by_model: dict[str, dict[str, Any]] = {}

    for row in rows:
        mid = _canonical_model_id(_relationship_row_model(row))
        if not mid:
            continue
        if mid not in row_by_model:
            ordered_models.append(mid)
        row_by_model[mid] = row

    for mid in extra_models or []:
        mid = _canonical_model_id(mid)
        if mid and mid not in row_by_model:
            ordered_models.append(mid)
            row_by_model[mid] = {"model_id": mid, "base_model": "", "relationship_type": ""}

    model_set = set(ordered_models)
    all_parents: dict[str, list[str]] = {}
    parents_in_set: dict[str, list[str]] = {}
    hf_errors: dict[str, str] = {}

    for idx, mid in enumerate(ordered_models):
        parents = [_canonical_model_id(p) for p in _relationship_row_parents(row_by_model[mid])]
        if use_hf and not parents:
            try:
                parents = [_canonical_model_id(p) for p in fetch_hf_base_models(mid)]
            except Exception as exc:  # Keep pipeline usable when HF has a transient issue.
                hf_errors[mid] = str(exc)
            if hf_sleep and idx + 1 < len(ordered_models):
                time.sleep(hf_sleep)
        all_parents[mid] = parents
        parents_in_set[mid] = [p for p in parents if p in model_set]

    direct_parent = {
        mid: (parents_in_set[mid][0] if len(parents_in_set[mid]) == 1 else None)
        for mid in ordered_models
    }

    union_parent: dict[str, str] = {}

    def find(x: str) -> str:
        union_parent.setdefault(x, x)
        if union_parent[x] != x:
            union_parent[x] = find(union_parent[x])
        return union_parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            union_parent[ra] = rb

    for mid in ordered_models:
        find(mid)
    for mid, parents in parents_in_set.items():
        for parent in parents:
            union(mid, parent)

    grouped: dict[str, list[str]] = defaultdict(list)
    for mid in ordered_models:
        grouped[find(mid)].append(mid)

    def model_roots(mid: str, seen: set[str] | None = None) -> list[str]:
        seen = set() if seen is None else seen
        if mid in seen:
            return []
        seen.add(mid)
        parents = [p for p in parents_in_set.get(mid, []) if p in model_set]
        if not parents:
            return [mid]
        roots: set[str] = set()
        for parent in parents:
            roots.update(model_roots(parent, set(seen)))
        return sorted(roots)

    def depth_of(mid: str) -> int:
        depth = 0
        cur = mid
        seen: set[str] = set()
        parents = [p for p in parents_in_set.get(cur, []) if p in model_set]
        while parents:
            next_parent = parents[0]
            if next_parent in seen:
                break
            seen.add(next_parent)
            depth += 1
            cur = next_parent
            parents = [p for p in parents_in_set.get(cur, []) if p in model_set]
        return depth

    cluster_records: list[dict[str, Any]] = []
    cluster_id_by_model: dict[str, int] = {}
    family_by_model: dict[str, str] = {}

    for cluster_id, (_rep, members) in enumerate(
        sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1
    ):
        models = sorted(members)
        model_subset = set(models)
        roots = sorted([m for m in models if not parents_in_set.get(m)])
        edges = sum(1 for m in models for p in parents_in_set[m] if p in model_subset)
        root_counts: dict[str, int] = defaultdict(int)
        for model in models:
            roots_for_model = model_roots(model)
            for root in roots_for_model:
                root_counts[root] += 1
        if root_counts:
            family_source = max(root_counts.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
        else:
            family_source = roots[0] if roots else models[0]
        family = family_source.split("/")[-1]
        record = {
            "cluster_id": cluster_id,
            "size": len(models),
            "num_edges": edges,
            "root_models": roots,
            "max_depth": max((depth_of(m) for m in models), default=0),
            "models": models,
            "family": family,
        }
        cluster_records.append(record)
        for model in models:
            cluster_id_by_model[model] = cluster_id
            family_by_model[model] = family

    child_count: dict[str, int] = defaultdict(int)
    for mid, parents in parents_in_set.items():
        for parent in parents:
            child_count[parent] += 1

    model_records: list[dict[str, Any]] = []
    for mid in sorted(ordered_models):
        parents = all_parents[mid]
        in_set = parents_in_set[mid]
        parent_value: str | list[str] | None
        if not parents:
            parent_value = None
        elif len(parents) == 1:
            parent_value = parents[0]
        else:
            parent_value = parents
        roots_for_model = model_roots(mid)
        record: dict[str, Any] = {
            "model_name": mid,
            "model_short_name": mid.split("/")[-1],
            "organization": mid.split("/")[0] if "/" in mid else "",
            "family": family_by_model.get(mid, mid.split("/")[-1]),
            "direct_parent": direct_parent[mid],
            "root_model": roots_for_model[0] if len(roots_for_model) == 1 else None,
            "is_root": not in_set,
            "depth": depth_of(mid),
            "num_children": child_count[mid],
            "outlier_count": None,
            "base_model_from_dataset": parent_value,
            "relationship_type": get_field(row_by_model[mid], "relationship_type", "relationship"),
        }
        if len(roots_for_model) > 1:
            record["root_models"] = roots_for_model
        if len(in_set) > 1:
            record["merge_parent_models"] = in_set
        if hf_errors.get(mid):
            record["fetch_error"] = hf_errors[mid]
        model_records.append(record)

    return {
        "metadata": {
            "total_models": len(ordered_models),
            "total_clusters": len(cluster_records),
            "total_edges": sum(c["num_edges"] for c in cluster_records),
            "description": (
                "Strict connected components from relationship CSV base_model edges; "
                "parents outside the current model set are retained as metadata only."
            ),
            "source_csv": source,
            "used_huggingface": use_hf,
        },
        "clusters": cluster_records,
        "models": model_records,
    }


def infer_loose_family(model_id: str, strict_family: str) -> tuple[str, str]:
    blob = f"{model_id} {strict_family}".lower()

    def result(label: str, source: str) -> tuple[str, str]:
        return label, f"relationship:name_rule:{source}"

    if "qwen" in blob or "distill-qwen" in blob:
        return result("Qwen", "qwen")
    if "llama-3.2" in blob or "llama3.2" in blob or "llama-3-2" in blob:
        return result("Meta-Llama-3.2", "llama32")
    if (
        "llama-3.1" in blob
        or "llama3.1" in blob
        or "tulu" in blob
        or ("llama-3" in blob and "3.2" not in blob and "gemma" not in blob)
    ):
        return result("Meta-Llama-3.1", "llama31")
    if "gemma" in blob:
        return result("Gemma", "gemma")
    if "mistral" in blob or "zephyr-7b" in blob:
        return result("Mistral", "mistral")
    if any(
        key in blob
        for key in (
            "beagle",
            "marcoro",
            "omnibeagle",
            "westbeagle",
            "beagsake",
            "beaglesempra",
            "neuralbeagle",
            "neuraltrix",
            "una-thebeagle",
            "sectumsempra",
            "mbx-7b",
            "cultrix",
            "pastiche-crown-clown",
            "ogno-monarch",
            "bardsai/jaskier",
            "aimaven",
            "shadowml/",
            "fblgit/",
            "argilla/",
            "flemmingmiguel/",
            "eren23/",
        )
    ):
        return result("Beagle", "beagle_ecosystem")
    if "vicuna" in blob:
        return result("Vicuna", "vicuna")
    if "tinyllama" in blob:
        return result("TinyLlama", "tinyllama")
    if "pythia" in blob:
        return result("Pythia", "pythia")
    if "smollm" in blob:
        return result("SmolLM", "smollm")
    if "bloom" in blob:
        return result("Bloom", "bloom")
    if "01-ai/" in model_id.lower() or "/yi-" in blob:
        return result("Yi", "yi")
    if "mimo" in blob or "koto-small" in blob:
        return result("MiMo", "mimo")
    if "deepseek" in blob:
        return result("DeepSeek", "deepseek")
    if "seed-coder" in blob or "bytedance-seed" in blob:
        return result("Seed-Coder", "seed")
    if "lfm2" in blob or "liquidai/lfm" in blob:
        return result("LFM2", "lfm2")
    if "apertus" in blob or "swiss-ai/" in blob:
        return result("Apertus", "apertus")
    if "exaone" in blob:
        return result("EXAONE", "exaone")
    if "starcoder" in blob:
        return result("StarCoder", "starcoder")
    if "olmoe" in blob:
        return result("OLMoE", "olmoe")
    if "polycoder" in blob:
        return result("PolyCoder", "polycoder")
    if "biomistral" in blob:
        return result("BioMistral", "biomistral")
    if "biomedgpt" in blob:
        return result("BioMedGPT", "biomedgpt")
    if "medicine-llm" in blob:
        return result("medicine-LLM", "medicine")
    if "anima" in blob:
        return result("ANIMA-Nectar-v2", "anima")
    if "starling" in blob:
        return result("Starling", "starling")
    if "sciphi" in blob:
        return result("SciPhi", "sciphi")
    if "polyglot-ko" in blob:
        return result("polyglot-ko-3.8b", "polyglot")
    if "gpt-neo-2.7b" in blob or "horni" in blob:
        return result("GPT-Neo-2.7B-Horni-LN", "horni")
    if "gpt-neo" in blob:
        return result("gpt-neo-1.3B", "gpt-neo")
    if "pygmalion" in blob:
        return result("pygmalion-2.7b", "pygmalion")
    if "llama" in blob or "meta-llama" in blob:
        return result("Llama", "llama_generic")
    return (strict_family.strip() or model_id.split("/")[-1], "relationship:fallback:strict_family")


def build_loose_lineage(strict_data: dict[str, Any]) -> dict[str, Any]:
    models_list = strict_data.get("models", [])
    clusters_strict = strict_data.get("clusters", [])
    model_to_strict_cid = {
        model: cluster["cluster_id"]
        for cluster in clusters_strict
        for model in cluster.get("models", [])
    }
    by_name = {m["model_name"]: m for m in models_list if m.get("model_name")}
    loose_to_models: dict[str, list[str]] = defaultdict(list)
    loose_source: dict[str, str] = {}

    for model in models_list:
        mid = model["model_name"]
        label, source = infer_loose_family(mid, model.get("family") or "")
        loose_to_models[label].append(mid)
        loose_source[label] = source

    loose_clusters: list[dict[str, Any]] = []
    labels = sorted(loose_to_models, key=lambda label: (-len(loose_to_models[label]), label))
    for loose_id, label in enumerate(labels, start=1):
        members = sorted(loose_to_models[label])
        member_set = set(members)
        roots = sorted({by_name[m].get("root_model") or m for m in members if by_name.get(m)})
        edge_count = 0
        for model_id in members:
            rec = by_name[model_id]
            parents = []
            if rec.get("merge_parent_models"):
                parents.extend(rec["merge_parent_models"])
            elif rec.get("direct_parent"):
                parents.append(rec["direct_parent"])
            edge_count += len({p for p in parents if p in member_set})

        loose_clusters.append(
            {
                "cluster_id": loose_id,
                "family": label,
                "size": len(members),
                "strict_cluster_ids": sorted(
                    {model_to_strict_cid[m] for m in members if m in model_to_strict_cid}
                ),
                "strict_families": sorted({by_name[m].get("family", "") for m in members}),
                "root_models": roots,
                "models": members,
                "num_edges": edge_count,
                "max_depth": max((by_name[m].get("depth") or 0 for m in members), default=0),
                "mapping_sources": [loose_source.get(label, "relationship:name_rule:unknown")],
            }
        )

    loose_id_by_family = {c["family"]: c["cluster_id"] for c in loose_clusters}
    loose_size_by_family = {c["family"]: c["size"] for c in loose_clusters}
    out_models = []
    for model in sorted(models_list, key=lambda x: x["model_name"]):
        row = dict(model)
        label, source = infer_loose_family(row["model_name"], row.get("family") or "")
        row["strict_cluster_id"] = model_to_strict_cid.get(row["model_name"])
        row["strict_family"] = row.get("family") or ""
        row["loose_family_id"] = loose_id_by_family[label]
        row["loose_family"] = label
        row["loose_family_size"] = loose_size_by_family[label]
        row["loose_mapping_source"] = source
        out_models.append(row)

    return {
        "metadata": {
            "total_models": len(models_list),
            "total_loose_families": len(loose_clusters),
            "description": "Loose model families from strict lineage plus keyword family rules.",
            "source_lineage_json": strict_data.get("metadata", {}).get("source_csv", ""),
            "method": "Keyword rules on model_name and strict family; fallback to strict family.",
        },
        "clusters": loose_clusters,
        "models": out_models,
    }


def quartiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)

    def quantile(p: float) -> float:
        if n == 1:
            return float(sorted_values[0])
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        weight = idx - lo
        return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)

    q1 = quantile(0.25)
    q3 = quantile(0.75)
    return q1, q3, q3 - q1


def round4(value: float) -> float:
    return round(float(value), 4)


def load_effective_bases_from_lineage(lineage: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rec in lineage.get("models", []):
        name = rec.get("model_name")
        if not name:
            continue
        if rec.get("merge_parent_models"):
            out[name] = [str(parent) for parent in rec["merge_parent_models"] if parent]
            continue
        if rec.get("direct_parent"):
            out[name] = [str(rec["direct_parent"])]
            continue
        base = rec.get("base_model_from_dataset")
        if isinstance(base, list):
            out[name] = [str(item) for item in base if item]
        elif base:
            out[name] = [str(base)]
        else:
            out[name] = []
    return out


def load_effective_base_from_lineage(lineage: dict[str, Any]) -> dict[str, str]:
    return {
        name: "; ".join(bases)
        for name, bases in load_effective_bases_from_lineage(lineage).items()
    }


def base_in_top_band(scores_no_self: list[float], base_score: float | None) -> int:
    if base_score is None:
        return 0
    levels = sorted({round4(v) for v in scores_no_self}, reverse=True)
    if not levels:
        return 0
    rounded = round4(base_score)
    try:
        return levels.index(rounded) + 1
    except ValueError:
        return 0


def compute_tukey_rows(
    models: list[str], matrix: dict[str, dict[str, float]], lineage: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_set = set(models)
    bases_of = load_effective_bases_from_lineage(lineage)
    tukey_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for model in models:
        row_data = matrix.get(model, {})
        pairs = []
        for other in models:
            if other == model:
                continue
            value = row_data.get(other)
            if value is None or math.isnan(value):
                continue
            pairs.append((other, float(value)))

        scores = [score for _, score in pairs]
        q1, q3, iqr = quartiles(scores)
        upper = q3 + 1.5 * iqr
        outliers = sorted([(other, value) for other, value in pairs if value > upper], key=lambda x: -x[1])

        bases = bases_of.get(model, [])
        bases_in_testset = [base for base in bases if base in model_set]
        base_scores = {
            base: float(row_data[base])
            for base in bases_in_testset
            if base in row_data and not math.isnan(row_data[base])
        }
        best_base_score = max(base_scores.values()) if base_scores else None
        base_in_testset = "yes" if bases and len(bases_in_testset) == len(bases) else "no"
        base_in_outliers = (
            "yes"
            if base_scores and any(score > upper for score in base_scores.values())
            else "no"
        )

        tukey_rows.append(
            {
                "model": model,
                "base_model": "; ".join(bases),
                "base_in_testset": base_in_testset,
                "base_in_top": (
                    base_in_top_band(scores, best_base_score)
                    if best_base_score is not None
                    else 0
                ),
                "Q1": round4(q1),
                "Q3": round4(q3),
                "IQR": round4(iqr),
                "upper_fence": round4(upper),
                "outlier_count": len(outliers),
                "base_score": (
                    " | ".join(
                        f"{base}:{round4(score)}" for base, score in sorted(base_scores.items())
                    )
                    if len(base_scores) > 1
                    else (round4(best_base_score) if best_base_score is not None else "")
                ),
                "base_in_outliers": base_in_outliers,
                "outliers": " | ".join(f"{name}({score:.4f})" for name, score in outliers),
            }
        )
        for other, score in outliers:
            long_rows.append(
                {
                    "anchor_model": model,
                    "outlier_model": other,
                    "overlap": round4(score),
                    "upper_fence": round4(upper),
                    "Q1": round4(q1),
                    "Q3": round4(q3),
                }
            )

    return tukey_rows, long_rows


def parse_outliers(cell: Any) -> list[str]:
    if cell is None or not str(cell).strip():
        return []
    out: list[str] = []
    for part in str(cell).split("|"):
        item = part.strip()
        if not item:
            continue
        match = OUTLIER_ENTRY_RE.match(item)
        out.append(match.group(1).strip() if match else item)
    return out


def lineage_parent_map(lineage: dict[str, Any]) -> dict[str, set[str]]:
    parent_map: dict[str, set[str]] = defaultdict(set)
    for rec in lineage.get("models", []):
        child = rec.get("model_name")
        if not child:
            continue
        if rec.get("merge_parent_models"):
            for parent in rec["merge_parent_models"]:
                if parent:
                    parent_map[child].add(parent)
        elif rec.get("direct_parent"):
            parent_map[child].add(rec["direct_parent"])
    return parent_map


def strict_neighbors(lineage: dict[str, Any], model: str) -> set[str]:
    parent_map = lineage_parent_map(lineage)
    neighbors = set(parent_map.get(model, set()))
    for child, parents in parent_map.items():
        if model in parents:
            neighbors.add(child)
    return neighbors


def cluster_peers(lineage: dict[str, Any]) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    peers: dict[str, frozenset[str]] = {}
    family: dict[str, str] = {}
    for cluster in lineage.get("clusters", []):
        members = frozenset(cluster.get("models") or [])
        label = str(cluster.get("family", ""))
        for model in members:
            peers[model] = members
            family[model] = label
    return peers, family


def universe_from_tukey_rows(rows: list[dict[str, Any]]) -> set[str]:
    universe: set[str] = set()
    for row in rows:
        universe.add(str(row["model"]))
        universe.update(parse_outliers(row.get("outliers", "")))
    return universe


def write_tukey_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    write_dict_csv(path, rows, fieldnames)


def write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(
    path: Path,
    rows: list[dict[str, Any]],
    lineage: dict[str, Any],
    mode: str,
) -> tuple[float, float]:
    universe = universe_from_tukey_rows(rows)
    summary_rows: list[dict[str, Any]] = []
    total_pred = total_tp = total_gold = 0
    total_gold_rows = 0

    if mode == "strict":
        base_from_lineage = load_effective_base_from_lineage(lineage)

        def get_gold(model: str) -> set[str]:
            return strict_neighbors(lineage, model)

        label = "strict"
    else:
        peers, family_of = cluster_peers(lineage)
        base_from_lineage = load_effective_base_from_lineage(lineage)

        def get_gold(model: str) -> set[str]:
            return set(peers.get(model, frozenset())) - {model}

        label = "loose"

    for row in rows:
        focus = str(row["model"])
        predicted = parse_outliers(row.get("outliers", ""))
        pred_set = set(predicted)
        gold_all = get_gold(focus)
        gold = gold_all & universe
        tp_set = pred_set & gold_all
        tp_in_universe = pred_set & gold

        n_pred = len(predicted)
        n_hit = len(tp_set)
        n_gold = len(gold)
        precision = n_hit / n_pred if n_pred else math.nan
        recall = len(tp_in_universe) / n_gold if n_gold else math.nan
        if n_gold:
            total_gold_rows += 1
        total_pred += n_pred
        total_tp += n_hit
        total_gold += n_gold

        out_row = {
            "model": focus,
            "base_model": row.get("base_model") or base_from_lineage.get(focus, ""),
        }
        if mode == "loose":
            out_row["cluster_family"] = family_of.get(focus, "")
        out_row.update(
            {
                "n_outliers": n_pred,
                f"n_{label}_related_in_outliers": n_hit,
                f"n_{label}_related_in_universe": n_gold,
                f"precision_outliers_vs_{label}": "" if n_pred == 0 else round(precision, 6),
                f"recall_{label}_in_universe_captured": "" if n_gold == 0 else round(recall, 6),
                f"{label}_related_outliers": " | ".join(sorted(tp_set)),
                f"{label}_related_in_universe_not_in_outliers": " | ".join(sorted(gold - pred_set)),
            }
        )
        summary_rows.append(out_row)

    micro_precision = total_tp / total_pred if total_pred else math.nan
    micro_recall = total_tp / total_gold if total_gold else math.nan
    summary = {
        "model": "_SUMMARY_",
        "base_model": "",
        "n_outliers": total_pred,
        f"n_{label}_related_in_outliers": total_tp,
        f"n_{label}_related_in_universe": total_gold,
        f"precision_outliers_vs_{label}": round(micro_precision, 6),
        f"recall_{label}_in_universe_captured": round(micro_recall, 6),
        f"{label}_related_outliers": (
            f"micro_TP={total_tp}_pred={total_pred}_gold_universe_sum={total_gold}"
        ),
        f"{label}_related_in_universe_not_in_outliers": (
            f"rows_with_{label}_neighbor_in_universe={total_gold_rows}"
        ),
    }
    if mode == "loose":
        summary["cluster_family"] = ""
    summary_rows.append(summary)
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    write_dict_csv(path, summary_rows, fieldnames)
    return micro_precision, micro_recall


def _graph_add_edge(graph: dict[str, dict[str, float]], a: str, b: str, weight: float) -> None:
    if not a or not b or a == b:
        return
    graph.setdefault(a, {})
    graph.setdefault(b, {})
    graph[a][b] = min(graph[a].get(b, math.inf), weight)
    graph[b][a] = min(graph[b].get(a, math.inf), weight)


def build_distance_matrix(
    strict_lineage: dict[str, Any], loose_lineage: dict[str, Any]
) -> dict[str, dict[str, float]]:
    nodes = {m["model_name"] for m in strict_lineage.get("models", []) if m.get("model_name")}
    graph: dict[str, dict[str, float]] = {node: {} for node in nodes}
    parent_index = lineage_parent_map(strict_lineage)

    for a, b in HF_EQUIV_WEIGHT_ZERO:
        if a in nodes and b in nodes:
            _graph_add_edge(graph, a, b, 0.0)

    for child, parents in parent_index.items():
        nodes.add(child)
        graph.setdefault(child, {})
        for parent in parents:
            nodes.add(parent)
            graph.setdefault(parent, {})
            _graph_add_edge(graph, parent, child, 1.0)

    # Within each loose family, connect the cluster's in-set "heads" (nodes with no parent
    # in this cluster) with weight-10 clique edges. This matches: cross-family glue is
    # between the representative bases, while internal parent/child relations stay weight 1.
    for cluster in loose_lineage.get("clusters", []):
        members = [m for m in cluster.get("models", []) if m in nodes]
        member_set = set(members)
        heads = [
            m
            for m in members
            if not (set(parent_index.get(m, set())) & member_set)
        ]
        for a, b in combinations(sorted(set(heads)), 2):
            _graph_add_edge(graph, a, b, 10.0)

    ordered = sorted(nodes)
    matrix: dict[str, dict[str, float]] = {source: {} for source in ordered}
    for source in ordered:
        distances = dijkstra(graph, source)
        for target in ordered:
            if source == target:
                matrix[source][target] = 0.0
            elif target in distances:
                matrix[source][target] = float(distances[target])
            else:
                matrix[source][target] = math.nan
    return matrix


def dijkstra(graph: dict[str, dict[str, float]], source: str) -> dict[str, float]:
    distances = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance != distances[node]:
            continue
        for neighbor, weight in graph.get(node, {}).items():
            next_distance = distance + weight
            if next_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = next_distance
                heapq.heappush(heap, (next_distance, neighbor))
    return distances


def write_distance_matrix(path: Path, matrix: dict[str, dict[str, float]]) -> None:
    nodes = sorted(matrix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + nodes)
        for source in nodes:
            writer.writerow(
                [source]
                + [
                    "" if math.isnan(matrix[source][target]) else f"{matrix[source][target]:.1f}"
                    for target in nodes
                ]
            )


def write_pairwise_distances(path: Path, matrix: dict[str, dict[str, float]]) -> None:
    nodes = sorted(matrix)
    rows = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            distance = matrix[a][b]
            rows.append(
                {
                    "model1": a,
                    "model2": b,
                    "graph_distance": "" if math.isnan(distance) else distance,
                }
            )
    write_dict_csv(path, rows, ["model1", "model2", "graph_distance"])


def write_outlier_distance_summary(
    summary_path: Path,
    pair_path: Path,
    rows: list[dict[str, Any]],
    matrix: dict[str, dict[str, float]],
) -> None:
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        model = str(row["model"])
        outliers = parse_outliers(row.get("outliers", ""))
        distances: list[float] = []
        missing_names: list[str] = []
        for outlier in outliers:
            distance = matrix.get(model, {}).get(outlier, math.nan)
            if math.isnan(distance):
                missing_names.append(outlier)
            else:
                distances.append(distance)
            pair_rows.append(
                {
                    "model": model,
                    "outlier": outlier,
                    "graph_distance": "" if math.isnan(distance) else distance,
                }
            )
        mean_distance = (
            round(sum(distances) / len(distances), 6)
            if distances and not missing_names
            else ""
        )
        summary_rows.append(
            {
                "model": model,
                "outlier_count_csv": row.get("outlier_count", len(outliers)),
                "n_outliers_parsed": len(outliers),
                "n_graph_distance_defined": len(distances),
                "n_graph_distance_missing": len(missing_names),
                "outliers_missing_graph_distance": " | ".join(missing_names),
                "mean_graph_distance_to_outliers": mean_distance,
            }
        )
    write_dict_csv(pair_path, pair_rows, ["model", "outlier", "graph_distance"])
    write_dict_csv(
        summary_path,
        summary_rows,
        [
            "model",
            "outlier_count_csv",
            "n_outliers_parsed",
            "n_graph_distance_defined",
            "n_graph_distance_missing",
            "outliers_missing_graph_distance",
            "mean_graph_distance_to_outliers",
        ],
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def abbrev_label(model_id: str, max_len: int = 14) -> str:
    label = model_id.split("/")[-1] if "/" in model_id else model_id
    label = label.replace("Instruct", "Ins").replace("instruct", "ins")
    if len(label) > max_len:
        return label[: max_len - 2] + ".."
    return label


def lineage_children_map(lineage: dict[str, Any]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for child, parents in lineage_parent_map(lineage).items():
        for parent in parents:
            children[parent].append(child)
    for parent in children:
        children[parent].sort()
    return children


def lineage_layered_pos(
    nodes: list[str],
    parent_map: dict[str, set[str]],
) -> dict[str, tuple[float, float]]:
    node_set = set(nodes)
    roots = sorted([node for node in nodes if not (parent_map.get(node, set()) & node_set)])
    if not roots and nodes:
        roots = [sorted(nodes)[0]]

    depth: dict[str, int] = {root: 0 for root in roots}
    changed = True
    for _ in range(len(nodes) + 2):
        if not changed:
            break
        changed = False
        for node in nodes:
            parents = parent_map.get(node, set()) & node_set
            if not parents:
                continue
            parent_depths = [depth[parent] for parent in parents if parent in depth]
            if not parent_depths:
                continue
            next_depth = max(parent_depths) + 1
            if depth.get(node) != next_depth:
                depth[node] = next_depth
                changed = True

    for node in nodes:
        depth.setdefault(node, 0)

    by_depth: dict[int, list[str]] = defaultdict(list)
    for node, level in depth.items():
        by_depth[level].append(node)

    pos: dict[str, tuple[float, float]] = {}
    for level, level_nodes in by_depth.items():
        ordered = sorted(level_nodes)
        width = len(ordered)
        for idx, node in enumerate(ordered):
            pos[node] = ((idx - (width - 1) / 2.0) * 1.35, -float(level))
    return pos


def draw_lineage_cluster_ax(
    ax: Any,
    cluster: dict[str, Any],
    parent_map: dict[str, set[str]],
    facecolor: Any,
    label_fs: float = 6,
) -> None:
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("networkx is required to draw strict lineage trees") from exc

    models = list(cluster.get("models", []))
    model_set = set(models)
    graph = nx.DiGraph()
    graph.add_nodes_from(models)
    for child in models:
        for parent in sorted(parent_map.get(child, set()) & model_set):
            graph.add_edge(parent, child)

    roots = cluster.get("root_models") or []
    roots_short = ", ".join(abbrev_label(root, 22) for root in roots[:4])
    if len(roots) > 4:
        roots_short += ", ..."
    title = (
        f"Cluster {cluster['cluster_id']} | n={cluster['size']}, "
        f"edges={cluster['num_edges']} | max_depth={cluster['max_depth']}\n"
        f"Root(s): {roots_short or '-'}"
    )
    ax.set_title(title, fontsize=9, fontweight="normal", loc="left")
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")

    if not models:
        return

    pos = lineage_layered_pos(models, parent_map)
    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edge_color="#777777",
            arrows=True,
            arrowsize=12,
            width=0.8,
            node_size=1,
            min_source_margin=12,
            min_target_margin=12,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=[facecolor] * graph.number_of_nodes(),
        node_size=430,
        linewidths=0.4,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        {node: abbrev_label(node) for node in graph.nodes()},
        ax=ax,
        font_size=label_fs,
        font_family="sans-serif",
    )


def write_singleton_clusters_csv(
    clusters: list[dict[str, Any]],
    models_by_name: dict[str, dict[str, Any]],
    out_path: Path,
) -> int:
    rows = []
    for cluster in clusters:
        if int(cluster.get("size", 0)) != 1:
            continue
        model_id = cluster["models"][0]
        detail = models_by_name.get(model_id, {})
        rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "model_id": model_id,
                "family": cluster.get("family", ""),
                "organization": detail.get("organization", ""),
                "model_short_name": detail.get("model_short_name", ""),
            }
        )
    rows.sort(key=lambda row: int(row["cluster_id"]))
    write_dict_csv(
        out_path,
        rows,
        ["cluster_id", "model_id", "family", "organization", "model_short_name"],
    )
    return len(rows)


def write_lineage_tree_plots(
    lineage: dict[str, Any],
    out_pdf: Path,
    out_png: Path,
    singleton_csv: Path,
    ncols: int = 4,
    nrows: int = 3,
    min_size: int = 1,
    dpi: int = 140,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to draw strict lineage trees") from exc

    clusters_all = list(lineage.get("clusters") or [])
    clusters = [c for c in clusters_all if int(c.get("size", 0)) >= min_size]
    parent_map = lineage_parent_map(lineage)
    models_by_name = {
        m["model_name"]: m for m in lineage.get("models", []) if m.get("model_name")
    }
    write_singleton_clusters_csv(clusters_all, models_by_name, singleton_csv)

    per_page = ncols * nrows
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    def draw_page(axes_flat: Any, page_clusters: list[dict[str, Any]]) -> None:
        max_size = max((int(cluster.get("size", 0)) for cluster in page_clusters), default=1)
        label_fs = 5 if max_size > 14 else 6
        for idx, ax in enumerate(axes_flat):
            ax.clear()
            if idx >= len(page_clusters):
                ax.axis("off")
                continue
            cluster = page_clusters[idx]
            color = cmap((cluster["cluster_id"] - 1) % 20 / 20.0)
            draw_lineage_cluster_ax(ax, cluster, parent_map, color, label_fs=label_fs)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    title = (
        "Strict lineage trees from relationship CSV "
        f"(n={lineage.get('metadata', {}).get('total_models', '?')})"
    )

    with PdfPages(out_pdf) as pdf:
        for start in range(0, len(clusters), per_page):
            chunk = clusters[start : start + per_page]
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.2 * ncols, 3.8 * nrows),
                constrained_layout=True,
            )
            axes_flat = axes.flatten() if per_page > 1 else [axes]
            draw_page(axes_flat, chunk)
            fig.suptitle(title, fontsize=11, y=1.02)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    if clusters:
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.8 * nrows),
            constrained_layout=True,
        )
        axes_flat = axes.flatten() if per_page > 1 else [axes]
        draw_page(axes_flat, clusters[:per_page])
        fig.suptitle(title, fontsize=11, y=1.02)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or args.overlap.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_models, overlap_matrix = load_overlap_matrix(args.overlap)
    relationship_rows = read_relationship_rows(args.relationship)
    strict_lineage = build_lineage_from_relationship_rows(
        relationship_rows,
        source=str(args.relationship.resolve()),
        extra_models=overlap_models,
        use_hf=args.use_hf,
        hf_sleep=args.hf_sleep,
    )
    loose_lineage = build_loose_lineage(strict_lineage)
    write_json(out_dir / "model_lineage_data_clean.json", strict_lineage)
    write_json(out_dir / "model_lineage_data_loose.json", loose_lineage)

    tukey_rows, long_rows = compute_tukey_rows(overlap_models, overlap_matrix, strict_lineage)
    write_tukey_csv(out_dir / "tukey_fence_eval.csv", tukey_rows)
    write_dict_csv(
        out_dir / "model_outliers_long.csv",
        long_rows,
        ["anchor_model", "outlier_model", "overlap", "upper_fence", "Q1", "Q3"],
    )

    distance_matrix = build_distance_matrix(strict_lineage, loose_lineage)
    write_distance_matrix(out_dir / "distance_matrix.csv", distance_matrix)
    write_pairwise_distances(out_dir / "pairwise_graph_distances.csv", distance_matrix)

    strict_precision, strict_recall = write_metrics_csv(
        out_dir / "tukey_outliers_strict_recall_precision.csv",
        tukey_rows,
        strict_lineage,
        mode="strict",
    )
    loose_precision, loose_recall = write_metrics_csv(
        out_dir / "tukey_outliers_loose_recall_precision.csv",
        tukey_rows,
        loose_lineage,
        mode="loose",
    )
    write_outlier_distance_summary(
        out_dir / "model_mean_outlier_graph_distance.csv",
        out_dir / "tukey_outliers_with_graph_distance.csv",
        tukey_rows,
        distance_matrix,
    )
    if not args.skip_plots:
        write_lineage_tree_plots(
            strict_lineage,
            out_pdf=out_dir / "strict_lineage_trees_clean.pdf",
            out_png=out_dir / "strict_lineage_trees_clean.png",
            singleton_csv=out_dir / "singleton_clusters_clean.csv",
            ncols=args.plot_ncols,
            nrows=args.plot_nrows,
            dpi=args.plot_dpi,
        )

    print(f"Wrote outputs to {out_dir}")
    print(f"Models in overlap matrix: {len(overlap_models)}")
    print(f"Strict clusters: {strict_lineage['metadata']['total_clusters']}")
    print(f"Loose families: {loose_lineage['metadata']['total_loose_families']}")
    print(f"Tukey outlier pairs: {len(long_rows)}")
    print(f"Strict micro precision/recall: {strict_precision:.6f} / {strict_recall:.6f}")
    print(f"Loose micro precision/recall: {loose_precision:.6f} / {loose_recall:.6f}")
    if not args.skip_plots:
        print(f"Wrote {out_dir / 'strict_lineage_trees_clean.pdf'}")
        print(f"Wrote {out_dir / 'strict_lineage_trees_clean.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run overlap matrix lineage/outlier analysis.")
    parser.add_argument("--overlap", type=Path, required=True, help="Overlap matrix CSV.")
    parser.add_argument(
        "--relationship",
        type=Path,
        required=True,
        help="Relationship CSV with model_id, base_model, relationship_type columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: the overlap CSV directory.",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Fill missing relationship rows from Hugging Face model card base_model.",
    )
    parser.add_argument(
        "--hf-sleep",
        type=float,
        default=0.08,
        help="Seconds to sleep between Hugging Face API calls when --use-hf is enabled.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip strict lineage PDF/PNG plot generation.",
    )
    parser.add_argument("--plot-ncols", type=int, default=4)
    parser.add_argument("--plot-nrows", type=int, default=3)
    parser.add_argument("--plot-dpi", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
