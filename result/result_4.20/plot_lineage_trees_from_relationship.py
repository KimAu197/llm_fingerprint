#!/usr/bin/env python3
"""Plot lineage clusters from model_lineage_data_from_relationship.json (relationship.csv cols 1-2; merge = multi-edge)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages


def load_models_by_name(lineage_path: Path) -> dict[str, dict]:
    data = json.loads(lineage_path.read_text(encoding="utf-8"))
    return {m["model_name"]: m for m in data.get("models", []) if m.get("model_name")}


def parents_in_cluster(rec: dict, cluster_set: set[str]) -> list[str]:
    if rec.get("merge_parent_models"):
        return [p for p in rec["merge_parent_models"] if p in cluster_set]
    p = rec.get("direct_parent")
    if p and p in cluster_set:
        return [p]
    return []


def abbrev_label(mid: str, max_len: int = 14) -> str:
    part = mid.split("/")[-1] if "/" in mid else mid
    part = (
        part.replace("Instruct", "Ins")
        .replace("instruct", "ins")
        .replace("Meta-Llama-", "L-")
    )
    if len(part) > max_len:
        return part[: max_len - 2] + ".."
    return part


def dag_depth_positions(
    nodes: list[str],
    models_by_name: dict[str, dict],
) -> dict[str, tuple[float, float]]:
    s = set(nodes)
    parents_of: dict[str, list[str]] = {}
    for m in nodes:
        rec = models_by_name.get(m, {})
        parents_of[m] = parents_in_cluster(rec, s)

    memo: dict[str, int] = {}

    def depth(m: str) -> int:
        if m in memo:
            return memo[m]
        ps = parents_of.get(m, [])
        if not ps:
            memo[m] = 0
            return 0
        memo[m] = 1 + max(depth(p) for p in ps)
        return memo[m]

    for m in nodes:
        depth(m)

    layers: dict[int, list[str]] = defaultdict(list)
    for m in nodes:
        layers[memo[m]].append(m)
    for lv in layers:
        layers[lv].sort()

    pos: dict[str, tuple[float, float]] = {}
    x_scale = 1.15
    y_step = 1.0
    for y, ns in sorted(layers.items()):
        n = len(ns)
        for i, node in enumerate(ns):
            pos[node] = ((i - (n - 1) / 2.0) * x_scale, -y * y_step)
    return pos


def build_cluster_digraph(
    cluster_models: list[str], models_by_name: dict[str, dict]
) -> nx.DiGraph:
    s = set(cluster_models)
    g = nx.DiGraph()
    for m in cluster_models:
        g.add_node(m)
        rec = models_by_name.get(m, {})
        for p in parents_in_cluster(rec, s):
            g.add_edge(p, m)
    return g


def draw_cluster_ax(
    ax,
    cluster: dict,
    models_by_name: dict[str, dict],
    facecolor: tuple,
    title_fs: float = 9,
    label_fs: float = 6,
) -> None:
    models = cluster["models"]
    g = build_cluster_digraph(models, models_by_name)
    pos = dag_depth_positions(models, models_by_name)

    roots = cluster.get("root_models") or []
    roots_short = ", ".join(abbrev_label(r, 22) for r in roots[:4])
    if len(roots) > 4:
        roots_short += ", …"
    title = (
        f"Cluster {cluster['cluster_id']}  |  n={cluster['size']}, "
        f"edges={cluster['num_edges']}  |  max_depth={cluster['max_depth']}\n"
        f"Root(s): {roots_short or '—'}"
    )
    ax.set_title(title, fontsize=title_fs, fontweight="normal", loc="left")
    ax.margins(0.15)
    ax.axis("off")
    ax.set_aspect("equal", adjustable="datalim")

    if g.number_of_nodes() == 0:
        return

    if g.number_of_edges() > 0:
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edge_color="#888888",
            arrows=True,
            arrowsize=12,
            width=0.8,
            node_size=1,
            min_source_margin=12,
            min_target_margin=12,
        )
    nc = g.number_of_nodes()
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=[facecolor] * nc,
        node_size=420,
        linewidths=0.4,
        edgecolors="#333333",
    )
    labels = {n: abbrev_label(n) for n in g.nodes()}
    nx.draw_networkx_labels(
        g,
        pos,
        labels,
        ax=ax,
        font_size=label_fs,
        font_family="sans-serif",
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lineage",
        type=Path,
        default=here / "model_lineage_data_from_relationship.json",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=here / "strict_lineage_trees_from_relationship.pdf",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=here / "strict_lineage_trees_from_relationship.png",
    )
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--nrows", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text(encoding="utf-8"))
    clusters_all = data.get("clusters") or []
    clusters = [c for c in clusters_all if int(c.get("size", 0)) >= args.min_size]
    models_by_name = load_models_by_name(args.lineage)

    per_page = args.ncols * args.nrows
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    n_models = int(data.get("metadata", {}).get("total_models", 0))
    suptitle = (
        "Lineage clusters from relationship.csv (cols 1-2; merge = multiple parent edges). "
        f"Multi-model clusters only (min size {args.min_size}), n={n_models} models total."
    )

    def draw_page(axes_flat, page_clusters: list[dict]) -> None:
        for i, ax in enumerate(axes_flat):
            ax.clear()
            if i >= len(page_clusters):
                ax.axis("off")
                continue
            c = page_clusters[i]
            color = cmap((c["cluster_id"] - 1) % 20 / 20.0)
            max_sz = max(cl["size"] for cl in page_clusters)
            label_fs = 5 if max_sz > 14 else 6
            draw_cluster_ax(ax, c, models_by_name, facecolor=color, label_fs=label_fs)

    with PdfPages(args.out_pdf) as pdf:
        for start in range(0, len(clusters), per_page):
            chunk = clusters[start : start + per_page]
            fig_w = 4.2 * args.ncols
            fig_h = 3.8 * args.nrows
            fig, axes = plt.subplots(
                args.nrows,
                args.ncols,
                figsize=(fig_w, fig_h),
                constrained_layout=True,
            )
            axes_flat = axes.flatten() if per_page > 1 else [axes]
            draw_page(axes_flat, chunk)
            fig.suptitle(suptitle, fontsize=11, y=1.02)
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)

    if clusters:
        fig_w = 4.2 * args.ncols
        fig_h = 3.8 * args.nrows
        fig, axes = plt.subplots(
            args.nrows,
            args.ncols,
            figsize=(fig_w, fig_h),
            constrained_layout=True,
        )
        axes_flat = axes.flatten() if per_page > 1 else [axes]
        draw_page(axes_flat, clusters[:per_page])
        fig.suptitle(suptitle, fontsize=11, y=1.02)
        fig.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {args.out_pdf} ({len(clusters)} clusters, {per_page} per page)")
    print(f"Wrote {args.out_png} (first page preview)")


if __name__ == "__main__":
    main()
