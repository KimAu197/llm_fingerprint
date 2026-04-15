#!/usr/bin/env python3
"""Plot strict fine-tuning lineage trees from model_lineage_data.json (one subplot per cluster)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

from paths import experiment_root


def load_parent_map(lineage_path: Path) -> dict[str, str | None]:
    data = json.loads(lineage_path.read_text(encoding="utf-8"))
    out: dict[str, str | None] = {}
    for m in data.get("models", []):
        name = m.get("model_name")
        if name:
            out[name] = m.get("direct_parent")
    return out


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


def forest_roots(nodes: list[str], parent_of: dict[str, str | None]) -> list[str]:
    s = set(nodes)
    roots = [m for m in nodes if parent_of.get(m) not in s]
    if not roots and nodes:
        roots = [sorted(nodes)[0]]
    return sorted(roots)


def layered_pos(
    nodes: list[str],
    parent_of: dict[str, str | None],
    x_scale: float = 1.15,
    y_step: float = 1.0,
) -> dict[str, tuple[float, float]]:
    s = set(nodes)
    children: dict[str, list[str]] = defaultdict(list)
    for c in nodes:
        p = parent_of.get(c)
        if p in s:
            children[p].append(c)
    for p in children:
        children[p].sort()

    rts = forest_roots(nodes, parent_of)
    pos: dict[str, tuple[float, float]] = {}
    y = 0
    current = list(rts)
    seen_rounds = 0
    while current:
        seen_rounds += 1
        if seen_rounds > len(nodes) + 2:
            break
        n = len(current)
        for i, node in enumerate(current):
            pos[node] = ((i - (n - 1) / 2.0) * x_scale, -y * y_step)
        nxt: list[str] = []
        for node in current:
            nxt.extend(children.get(node, []))
        current = nxt
        y += 1

    for m in nodes:
        if m not in pos:
            pos[m] = (0.0, 0.0)
    return pos


def build_cluster_graph(
    cluster_models: list[str], parent_of: dict[str, str | None]
) -> nx.DiGraph:
    s = set(cluster_models)
    g = nx.DiGraph()
    for m in cluster_models:
        g.add_node(m)
    for c in cluster_models:
        p = parent_of.get(c)
        if p and p in s:
            g.add_edge(p, c)
    return g


def draw_cluster_ax(
    ax,
    cluster: dict,
    parent_of: dict[str, str | None],
    facecolor: tuple,
    title_fs: float = 9,
    label_fs: float = 6,
) -> None:
    models = cluster["models"]
    g = build_cluster_graph(models, parent_of)
    pos = layered_pos(models, parent_of)

    roots = cluster.get("root_models") or []
    roots_short = ", ".join(abbrev_label(r,22) for r in roots[:4])
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
    root = experiment_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", type=Path, default=root / "model_lineage_data.json")
    ap.add_argument("--out-pdf", type=Path, default=root / "strict_lineage_trees.pdf")
    ap.add_argument(
        "--out-png",
        type=Path,
        default=root / "strict_lineage_trees.png",
        help="First page only (quick preview).",
    )
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--nrows", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=1, help="Skip clusters smaller than this")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    data = json.loads(args.lineage.read_text(encoding="utf-8"))
    clusters_all = data.get("clusters") or []
    clusters = [c for c in clusters_all if int(c.get("size", 0)) >= args.min_size]
    parent_of = load_parent_map(args.lineage)

    per_page = args.ncols * args.nrows
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)

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
            draw_cluster_ax(ax, c, parent_of, facecolor=color, label_fs=label_fs)

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
            fig.suptitle(
                "Strict fine-tuning lineage trees (HF card base_model, in-union edges only)",
                fontsize=12,
                y=1.02,
            )
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
        fig.suptitle(
            "Strict fine-tuning lineage trees (HF card base_model, in-union edges only)",
            fontsize=12,
            y=1.02,
        )
        fig.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {args.out_pdf} ({len(clusters)} clusters, {per_page} per page)")
    print(f"Wrote {args.out_png} (first page preview)")


if __name__ == "__main__":
    main()
