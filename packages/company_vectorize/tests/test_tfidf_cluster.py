"""Unit coverage for `tfidf_cluster.build_clustering_text_view`.

Asserts the module's own contract: the `text_view` and required-column
guards, the `"name"` vs `"tokens"` projection shapes, and the
`include_cluster_text` toggle. The view removes nothing: what text a
vectorizer reads is decided before it, by the caller's preprocessing step.
"""

from __future__ import annotations

import polars as pl
import pytest
from company_vectorize.tfidf_cluster import build_clustering_text_view


def test_rejects_an_unknown_text_view():
    frame = pl.DataFrame({"name": ["Acme"]})

    with pytest.raises(ValueError, match="text_view"):
        build_clustering_text_view(frame, text_view="bogus")


def test_rejects_a_missing_name_column():
    frame = pl.DataFrame({"other": ["Acme"]})

    with pytest.raises(ValueError, match="name"):
        build_clustering_text_view(frame, name_col="name")


def test_name_view_strips_and_fills_null_names():
    frame = pl.DataFrame({"name": [" Acme Ltd ", None]})

    result = build_clustering_text_view(frame, text_view="name")

    assert result.get_column("cluster_text").to_list() == ["Acme Ltd", ""]


def test_tokens_view_requires_a_present_token_col():
    frame = pl.DataFrame({"name": ["Acme"]})

    with pytest.raises(ValueError, match="token_col"):
        build_clustering_text_view(frame, text_view="tokens", token_col=None)

    with pytest.raises(ValueError, match="token_col"):
        build_clustering_text_view(frame, text_view="tokens", token_col="missing_col")


def test_tokens_view_builds_cluster_tokens_and_joined_cluster_text():
    frame = pl.DataFrame(
        {"name": ["Acme"], "tokens": [["acme", "ltd"]]},
    )

    result = build_clustering_text_view(frame, text_view="tokens", token_col="tokens")

    assert result.get_column("cluster_tokens").to_list() == [["acme", "ltd"]]
    assert result.get_column("cluster_text").to_list() == ["acme ltd"]


def test_tokens_view_can_omit_the_joined_cluster_text():
    frame = pl.DataFrame(
        {"name": ["Acme"], "tokens": [["acme", "ltd"]]},
    )

    result = build_clustering_text_view(
        frame,
        text_view="tokens",
        token_col="tokens",
        include_cluster_text=False,
    )

    assert "cluster_text" not in result.columns
    assert result.get_column("cluster_tokens").to_list() == [["acme", "ltd"]]


def test_tokens_view_fills_null_token_lists():
    frame = pl.DataFrame(
        {"name": ["Acme"], "tokens": [None]},
        schema={"name": pl.Utf8, "tokens": pl.List(pl.Utf8)},
    )

    result = build_clustering_text_view(frame, text_view="tokens", token_col="tokens")

    assert result.get_column("cluster_tokens").to_list() == [[]]
