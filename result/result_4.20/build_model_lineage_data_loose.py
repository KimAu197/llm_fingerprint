#!/usr/bin/env python3
"""Build model_lineage_data_loose.json from model_lineage_data_from_relationship.json (keyword loose families)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def infer_loose_family(model_id: str, strict_family: str) -> tuple[str, str]:
    blob = f"{model_id} {strict_family}".lower()

    def r(label: str, src: str) -> tuple[str, str]:
        return label, f"relationship:name_rule:{src}"

    if "qwen" in blob or "distill-qwen" in blob:
        return r("Qwen", "qwen")
    if "llama-3.2" in blob or "llama-3-2" in blob:
        return r("Meta-Llama-3.2", "llama32")
    if (
        "llama-3.1" in blob
        or "llama3.1" in blob
        or "tulu" in blob
        or ("llama-3" in blob and "3.2" not in blob and "gemma" not in blob)
    ):
        return r("Meta-Llama-3.1", "llama31")
    if "gemma" in blob:
        return r("Gemma", "gemma")
    if "mistral" in blob or "zephyr-7b" in blob:
        return r("Mistral", "mistral")
    if any(
        k in blob
        for k in (
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
        return r("Beagle", "beagle_ecosystem")
    if "vicuna" in blob:
        return r("Vicuna", "vicuna")
    if "tinyllama" in blob:
        return r("TinyLlama", "tinyllama")
    if "pythia" in blob:
        return r("Pythia", "pythia")
    if "smollm" in blob:
        return r("SmolLM", "smollm")
    if "bloom" in blob:
        return r("Bloom", "bloom")
    if "01-ai/" in model_id.lower() or "/yi-" in blob:
        return r("Yi", "yi")
    if "mimo" in blob or "koto-small" in blob:
        return r("MiMo", "mimo")
    if "deepseek" in blob:
        return r("DeepSeek", "deepseek")
    if "seed-coder" in blob or "bytedance-seed" in blob:
        return r("Seed-Coder", "seed")
    if "lfm2" in blob or "liquidai/lfm" in blob:
        return r("LFM2", "lfm2")
    if "apertus" in blob or "swiss-ai/" in blob:
        return r("Apertus", "apertus")
    if "exaone" in blob:
        return r("EXAONE", "exaone")
    if "starcoder" in blob:
        return r("StarCoder", "starcoder")
    if "olmoe" in blob:
        return r("OLMoE", "olmoe")
    if "polycoder" in blob:
        return r("PolyCoder", "polycoder")
    if "biomistral" in blob:
        return r("BioMistral", "biomistral")
    if "biomedgpt" in blob:
        return r("BioMedGPT", "biomedgpt")
    if "medicine-llm" in blob:
        return r("medicine-LLM", "medicine")
    if "anima" in blob:
        return r("ANIMA-Nectar-v2", "anima")
    if "starling" in blob:
        return r("Starling", "starling")
    if "sciphi" in blob:
        return r("SciPhi", "sciphi")
    if "polyglot-ko" in blob:
        return r("polyglot-ko-3.8b", "polyglot")
    if "gpt-neo-2.7b" in blob or "horni" in blob:
        return r("GPT-Neo-2.7B-Horni-LN", "horni")
    if "gpt-neo" in blob:
        return r("gpt-neo-1.3B", "gpt-neo")
    if "pygmalion" in blob:
        return r("pygmalion-2.7b", "pygmalion")
    if "neeto" in blob:
        return r("Neeto-1.0-8b", "neeto")
    if "bootstrap-llm" in blob:
        return r("Bootstrap-LLM", "bootstrap")
    if "shisa" in blob:
        return r("shisa-base-7b-v1", "shisa")
    if "luna" in blob and "vicuna" not in blob:
        return r("Luna", "luna")
    if "dictalm" in blob:
        return r("DictaLM", "dictalm")
    if "elyza" in blob:
        return r("Llama-3-ELYZA-JP-8B", "elyza")
    if "llama" in blob or "meta-llama" in blob:
        return r("Llama", "llama_generic")

    fam = strict_family.strip() or model_id.split("/")[-1]
    return fam, "relationship:fallback:strict_family"


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lineage",
        type=Path,
        default=here / "model_lineage_data_from_relationship.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "model_lineage_data_loose.json",
    )
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text(encoding="utf-8"))
    models_list = data.get("models", [])
    clusters_strict = data.get("clusters", [])

    model_to_strict_cid: dict[str, int] = {}
    for c in clusters_strict:
        cid = c["cluster_id"]
        for m in c.get("models", []):
            model_to_strict_cid[m] = cid

    by_name = {m["model_name"]: m for m in models_list if m.get("model_name")}

    loose_to_models: dict[str, list[str]] = defaultdict(list)
    loose_source: dict[str, str] = {}
    for m in models_list:
        mid = m["model_name"]
        sf = m.get("family") or ""
        label, src = infer_loose_family(mid, sf)
        loose_to_models[label].append(mid)
        loose_source[label] = src

    sorted_labels = sorted(loose_to_models.keys(), key=lambda x: (-len(loose_to_models[x]), x))

    loose_clusters = []
    for lid, label in enumerate(sorted_labels, start=1):
        ms = sorted(loose_to_models[label])
        ms_set = set(ms)

        strict_ids = sorted({model_to_strict_cid[m] for m in ms if m in model_to_strict_cid})
        strict_fams = sorted({by_name[m].get("family", "") for m in ms})

        roots = sorted(
            {rm for m in ms if (rm := by_name[m].get("root_model"))}
        )

        n_edges = 0
        for m in ms:
            rec = by_name[m]
            ps = []
            if rec.get("merge_parent_models"):
                ps.extend(rec["merge_parent_models"])
            if rec.get("direct_parent"):
                ps.append(rec["direct_parent"])
            seen = set()
            for p in ps:
                if p and p in ms_set and p not in seen:
                    seen.add(p)
                    n_edges += 1

        max_depth = max((by_name[m].get("depth") or 0 for m in ms), default=0)

        loose_clusters.append(
            {
                "cluster_id": lid,
                "family": label,
                "size": len(ms),
                "strict_cluster_ids": strict_ids,
                "strict_families": strict_fams,
                "root_models": roots,
                "models": ms,
                "num_edges": n_edges,
                "max_depth": max_depth,
                "mapping_sources": [loose_source.get(label, "relationship:name_rule:unknown")],
            }
        )

    out_models = []
    for m in sorted(models_list, key=lambda x: x["model_name"]):
        mid = m["model_name"]
        sf = m.get("family") or ""
        label, map_src = infer_loose_family(mid, sf)
        loose_id = next(c["cluster_id"] for c in loose_clusters if c["family"] == label)
        loose_sz = next(c["size"] for c in loose_clusters if c["family"] == label)
        scid = model_to_strict_cid.get(mid)

        row = dict(m)
        row["strict_cluster_id"] = scid
        row["strict_family"] = sf
        row["loose_family_id"] = loose_id
        row["loose_family"] = label
        row["loose_family_size"] = loose_sz
        row["loose_mapping_source"] = map_src
        out_models.append(row)

    out = {
        "metadata": {
            "total_models": len(models_list),
            "total_loose_families": len(loose_clusters),
            "description": "Loose families from model_lineage_data_from_relationship.json: keyword grouping on model_id + strict family (Qwen, Meta-Llama-3.x, Gemma, Mistral, Beagle, ...).",
            "source_lineage_json": str(args.lineage.resolve()),
            "method": "Keyword rules on model_name and family; fallback to strict family string.",
        },
        "clusters": loose_clusters,
        "models": out_models,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(loose_clusters)} loose families, {len(models_list)} models)")


if __name__ == "__main__":
    main()
