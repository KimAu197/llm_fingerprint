from analyze_overlap_pipeline import (
    build_distance_matrix,
    build_lineage_from_relationship_rows,
    lineage_parent_map,
    parse_parent_cell,
    write_lineage_tree_plots,
)
from pytest import approx


def test_parse_parent_cell_accepts_semicolon_and_comma_separated_parents():
    assert parse_parent_cell("A/base; B/base, C/base") == [
        "A/base",
        "B/base",
        "C/base",
    ]


def test_build_lineage_keeps_only_in_set_relationship_edges():
    rows = [
        {"model_id": "org/base", "base_model": "NONE", "relationship_type": "NONE"},
        {"model_id": "org/child", "base_model": "org/base", "relationship_type": "finetune"},
        {
            "model_id": "org/merge",
            "base_model": "org/base; missing/external",
            "relationship_type": "merge",
        },
    ]

    data = build_lineage_from_relationship_rows(rows, source="unit.csv")
    by_name = {m["model_name"]: m for m in data["models"]}

    assert by_name["org/base"]["direct_parent"] is None
    assert by_name["org/child"]["direct_parent"] == "org/base"
    assert by_name["org/merge"]["direct_parent"] == "org/base"
    assert by_name["org/merge"]["base_model_from_dataset"] == [
        "org/base",
        "missing/external",
    ]
    assert data["metadata"]["total_edges"] == 2


def test_build_lineage_preserves_multiple_in_set_parents():
    rows = [
        {"model_id": "org/base-a", "base_model": "NONE", "relationship_type": "NONE"},
        {"model_id": "org/base-b", "base_model": "NONE", "relationship_type": "NONE"},
        {
            "model_id": "org/merge",
            "base_model": "org/base-a; org/base-b",
            "relationship_type": "merge",
        },
    ]

    data = build_lineage_from_relationship_rows(rows, source="unit.csv")
    by_name = {m["model_name"]: m for m in data["models"]}

    assert by_name["org/merge"]["direct_parent"] is None
    assert by_name["org/merge"]["merge_parent_models"] == ["org/base-a", "org/base-b"]
    assert lineage_parent_map(data)["org/merge"] == {"org/base-a", "org/base-b"}


def test_distance_matrix_uses_tree_edges_and_base_root_edges():
    strict = build_lineage_from_relationship_rows(
        [
            {"model_id": "fam/a-base", "base_model": "NONE", "relationship_type": "NONE"},
            {"model_id": "fam/a-child", "base_model": "fam/a-base", "relationship_type": "finetune"},
            {
                "model_id": "fam/a-grandchild",
                "base_model": "fam/a-child",
                "relationship_type": "finetune",
            },
            {"model_id": "fam/b-base", "base_model": "NONE", "relationship_type": "NONE"},
            {"model_id": "fam/b-child", "base_model": "fam/b-base", "relationship_type": "finetune"},
        ],
        source="unit.csv",
    )
    loose = {
        "clusters": [
            {
                "family": "fam",
                "root_models": ["fam/a-base", "fam/b-base"],
                "models": [
                    "fam/a-base",
                    "fam/a-child",
                    "fam/a-grandchild",
                    "fam/b-base",
                    "fam/b-child",
                ],
            }
        ]
    }

    matrix = build_distance_matrix(strict, loose)

    assert matrix["fam/a-base"]["fam/a-child"] == approx(1.0)
    assert matrix["fam/a-base"]["fam/a-grandchild"] == approx(2.0)
    assert matrix["fam/a-base"]["fam/b-base"] == approx(10.0)
    assert matrix["fam/a-child"]["fam/b-child"] == approx(12.0)


def test_loose_clique_connects_subtrees_at_expected_shortest_path():
    """
    A --1-- A1 --1-- A2     B --1-- B1
    Same loose family: A and B (and all members) pairwise at distance 10.
    Then: A1 to B = 1+10 = 11; A2 to B1 = 1+1+10+1 = 13.
    """
    strict = build_lineage_from_relationship_rows(
        [
            {"model_id": "demo/A", "base_model": "NONE", "relationship_type": "NONE"},
            {"model_id": "demo/B", "base_model": "NONE", "relationship_type": "NONE"},
            {"model_id": "demo/A1", "base_model": "demo/A", "relationship_type": "finetune"},
            {"model_id": "demo/A2", "base_model": "demo/A1", "relationship_type": "finetune"},
            {"model_id": "demo/B1", "base_model": "demo/B", "relationship_type": "finetune"},
        ],
        source="unit.csv",
    )
    loose = {
        "clusters": [
            {
                "family": "demo_loose",
                "models": ["demo/A", "demo/A1", "demo/A2", "demo/B", "demo/B1"],
            }
        ]
    }
    matrix = build_distance_matrix(strict, loose)
    assert matrix["demo/A1"]["demo/B"] == approx(11.0)
    assert matrix["demo/A2"]["demo/B1"] == approx(13.0)


def test_write_lineage_tree_plots_creates_pdf_png_and_singleton_csv(tmp_path):
    lineage = build_lineage_from_relationship_rows(
        [
            {"model_id": "org/base-a", "base_model": "NONE", "relationship_type": "NONE"},
            {"model_id": "org/base-b", "base_model": "NONE", "relationship_type": "NONE"},
            {
                "model_id": "org/merge",
                "base_model": "org/base-a; org/base-b",
                "relationship_type": "merge",
            },
            {"model_id": "solo/model", "base_model": "NONE", "relationship_type": "NONE"},
        ],
        source="unit.csv",
    )

    write_lineage_tree_plots(
        lineage,
        out_pdf=tmp_path / "strict_lineage_trees_clean.pdf",
        out_png=tmp_path / "strict_lineage_trees_clean.png",
        singleton_csv=tmp_path / "singleton_clusters_clean.csv",
        ncols=2,
        nrows=1,
        dpi=80,
    )

    assert (tmp_path / "strict_lineage_trees_clean.pdf").stat().st_size > 0
    assert (tmp_path / "strict_lineage_trees_clean.png").stat().st_size > 0
    assert (tmp_path / "singleton_clusters_clean.csv").read_text(encoding="utf-8").startswith(
        "cluster_id,model_id"
    )
