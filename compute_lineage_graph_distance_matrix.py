import itertools
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

OUTLIER_RE = re.compile(r"^(.+?)\(([\d.]+)\)\s*$")

def get_hf_org(model_id):
    if not model_id or "/" not in str(model_id):
        return "unknown"
    return str(model_id).split("/", 1)[0]

def parse_outliers_from_tukey(df):
    names = set()
    for cell in df.get("outliers", pd.Series(dtype=object)):
        if pd.isna(cell) or not str(cell).strip():
            continue
        for part in str(cell).split("|"):
            part = part.strip()
            m = OUTLIER_RE.match(part)
            if m:
                names.add(m.group(1).strip())
            elif part:
                names.add(part)
    return names

def load_model_set(tukey_path):
    df = pd.read_csv(tukey_path)
    models = set(df["model"].dropna().astype(str).tolist())
    for _, row in df.iterrows():
        b = row.get("base_model")
        if pd.notna(b):
            models.add(str(b))
    models |= parse_outliers_from_tukey(df)
    return models

def ensure_alias_closure(model_set, alias_pairs):
    changed = True
    while changed:
        changed = False
        for a, b in alias_pairs:
            if a in model_set or b in model_set:
                if a not in model_set:
                    model_set.add(a)
                    changed = True
                if b not in model_set:
                    model_set.add(b)
                    changed = True

def expand_model_set_with_gt_parents(gt_df, model_set, max_iter=40):
    for _ in range(max_iter):
        sub = gt_df[gt_df["model_id"].isin(model_set)]
        parents = set(sub["effective_base_model"].dropna().astype(str).tolist())
        parents.discard("")
        old = len(model_set)
        model_set |= parents
        if len(model_set) == old:
            break

def load_ground_truth_finetune(ground_truth_path):
    cols = ["model_id", "effective_base_model", "effective_relationship"]
    df = pd.read_csv(ground_truth_path, usecols=cols, low_memory=False)
    df = df[df["effective_relationship"].astype(str) == "finetune"]
    return df

def load_loose_anchor_nodes(lineage_path, model_set):
    with open(lineage_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchors = set()
    for c in data.get("clusters", []):
        for r in c.get("root_models", []):
            if r in model_set:
                anchors.add(r)
    return anchors

def load_finetune_edges_json(lineage_path):
    with open(lineage_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    edges = []
    for m in data["models"]:
        child = m["model_name"]
        parent = m.get("direct_parent")
        if parent:
            edges.append((parent, child))
    return edges

def finetune_edges_from_gt(gt_df, model_set):
    sub = gt_df[
        gt_df["model_id"].isin(model_set)
        & gt_df["effective_base_model"].isin(model_set)
    ]
    return list(
        zip(
            sub["effective_base_model"].astype(str),
            sub["model_id"].astype(str),
        )
    )

def add_or_min_edge(G, u, v, w):
    if G.has_edge(u, v):
        G[u][v]["weight"] = min(G[u][v]["weight"], w)
    else:
        G.add_edge(u, v, weight=w)

HF_EQUIV_WEIGHT_ZERO = [
    ("meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B"),
    ("NousResearch/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"),
    ("meta-llama/Meta-Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ("meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
]

def add_alias_edges(G, model_set):
    for a, b in HF_EQUIV_WEIGHT_ZERO:
        if a in model_set and b in model_set:
            add_or_min_edge(G, a, b, 0)

def build_graph(model_set, finetune_edges, loose_anchors):
    """
    Undirected graph: FT edges weight 1; loose edges weight 10 between same-HF-org
    lineage roots; optional weight-0 between known equivalent HF base ids.
    """
    G = nx.Graph()
    for n in model_set:
        G.add_node(n)

    add_alias_edges(G, model_set)

    for parent, child in finetune_edges:
        if parent not in model_set or child not in model_set:
            continue
        add_or_min_edge(G, parent, child, 1)

    for b1, b2 in itertools.combinations(sorted(loose_anchors), 2):
        if get_hf_org(b1) != get_hf_org(b2):
            continue
        add_or_min_edge(G, b1, b2, 10)

    return G

def shortest_path_matrix(G, ordered_nodes):
    n = len(ordered_nodes)
    idx = {name: i for i, name in enumerate(ordered_nodes)}
    mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mat, 0.0)

    for source in ordered_nodes:
        try:
            lengths = nx.single_source_dijkstra_path_length(G, source, weight="weight")
        except nx.NetworkXNoPath:
            lengths = {}
        i = idx[source]
        for target, dist in lengths.items():
            if target not in idx:
                continue
            j = idx[target]
            mat[i, j] = float(dist)
            mat[j, i] = float(dist)

    return mat

def main():
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "result" / "result_3.30"
    repo_root = base_dir.parent
    tukey_path = result_dir / "tukey_fence_eval.csv"
    lineage_path = result_dir / "model_lineage_data.json"
    gt_path = repo_root / "model_ground_truth.csv"
    out_matrix = result_dir / "distance_matrix.csv"
    out_pairs = result_dir / "pairwise_graph_distances.csv"

    model_set = load_model_set(tukey_path)
    ensure_alias_closure(model_set, HF_EQUIV_WEIGHT_ZERO)

    gt_ft = load_ground_truth_finetune(gt_path)
    expand_model_set_with_gt_parents(gt_ft, model_set)
    ensure_alias_closure(model_set, HF_EQUIV_WEIGHT_ZERO)

    loose_anchors = load_loose_anchor_nodes(lineage_path, model_set)
    edges_json = load_finetune_edges_json(lineage_path)
    edges_gt = finetune_edges_from_gt(gt_ft, model_set)
    finetune_edges = list({tuple(e) for e in edges_json + edges_gt})

    G = build_graph(model_set, finetune_edges, loose_anchors)

    ordered = sorted(model_set)
    mat = shortest_path_matrix(G, ordered)
    df = pd.DataFrame(mat, index=ordered, columns=ordered)
    df.to_csv(out_matrix)

    rows = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            d = df.loc[a, b]
            rows.append(
                {
                    "model1": a,
                    "model2": b,
                    "graph_distance": float(d) if not np.isnan(d) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out_pairs, index=False)

    n_nan = df.isna().sum().sum()
    total = df.shape[0] * df.shape[1]
    print(f"Models (expanded): {len(ordered)}")
    print(f"Unique FT edges (JSON+GT, deduped): {len(finetune_edges)}")
    print(f"Loose anchor roots: {len(loose_anchors)}")
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Matrix cells NaN: {int(n_nan)} / {total}")

    if "Qwen/Qwen2.5-1.5B" in df.index and "Qwen/Qwen2.5-7B" in df.index:
        print(f"Check Q2.5-1.5B <-> Q2.5-7B: {df.loc['Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-7B']}")
    if "Qwen/Qwen2.5-1.5B-Instruct" in df.index and "Qwen/Qwen2.5-7B-Instruct" in df.index:
        print(
            f"Check 1.5B-Instruct <-> 7B-Instruct: {df.loc['Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']}"
        )

    h2 = "NousResearch/Hermes-2-Pro-Llama-3-8B"
    h3 = "NousResearch/Hermes-3-Llama-3.1-8B"
    if h2 in df.index and h3 in df.index:
        print(f"Check Hermes-2-Pro <-> Hermes-3 (Llama family): {df.loc[h2, h3]}")

    print(f"Wrote {out_matrix}")
    print(f"Wrote {out_pairs}")


if __name__ == "__main__":
    main()
