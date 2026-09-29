"""Unit coverage for `graph_cluster.build_connected_components`.

Asserts the module's own union-find contract: transitive grouping across
chained edges, deterministic cluster-id choice (the lexicographically
smallest node in the component), null-id rows skipped rather than raising,
and the required-column and empty-input guards.
"""

from __future__ import annotations

import polars as pl
import pytest
from company_vectorize.graph_cluster import build_connected_components


def test_raises_when_a_required_column_is_missing():
    edges = pl.DataFrame({"source_id": ["a"], "wrong_target_col": ["b"]})

    with pytest.raises(ValueError, match="target_id"):
        build_connected_components(edges)


def test_empty_edges_returns_empty_typed_frame():
    edges = pl.DataFrame(
        {"source_id": [], "target_id": []},
        schema={"source_id": pl.Utf8, "target_id": pl.Utf8},
    )

    result = build_connected_components(edges)

    assert result.height == 0
    assert result.schema == {"cluster_id": pl.Utf8, "node_id": pl.Utf8}


def test_transitive_chain_forms_a_single_component():
    edges = pl.DataFrame(
        {"source_id": ["a", "b"], "target_id": ["b", "c"]},
    )

    result = build_connected_components(edges)

    assert result.sort("node_id").get_column("node_id").to_list() == ["a", "b", "c"]
    cluster_ids = result.get_column("cluster_id").unique().to_list()
    assert cluster_ids == ["cc:a"]


def test_disjoint_edges_form_separate_components():
    edges = pl.DataFrame(
        {"source_id": ["a", "x"], "target_id": ["b", "y"]},
    )

    result = build_connected_components(edges).sort("node_id")

    assert result.get_column("node_id").to_list() == ["a", "b", "x", "y"]
    assert result.get_column("cluster_id").to_list() == ["cc:a", "cc:a", "cc:x", "cc:x"]


def test_cluster_id_uses_the_lexicographically_smallest_node_regardless_of_edge_order():
    edges = pl.DataFrame(
        {"source_id": ["z", "m"], "target_id": ["m", "a"]},
    )

    result = build_connected_components(edges)

    assert set(result.get_column("cluster_id").to_list()) == {"cc:a"}


def test_rows_with_a_null_id_are_skipped():
    edges = pl.DataFrame(
        {"source_id": ["a", None], "target_id": ["b", "c"]},
    )

    result = build_connected_components(edges).sort("node_id")

    assert result.get_column("node_id").to_list() == ["a", "b"]


def test_custom_column_names_are_honoured():
    edges = pl.DataFrame({"src": ["a"], "dst": ["b"]})

    result = build_connected_components(
        edges, source_id_col="src", target_id_col="dst"
    ).sort("node_id")

    assert result.get_column("node_id").to_list() == ["a", "b"]
    assert set(result.get_column("cluster_id").to_list()) == {"cc:a"}
