import json
from pathlib import Path

import polars as pl
import pytest

from blocking.contracts import BlockingRunConfig, BlockingStrategyConfig
from blocking.inspection import (
    build_inspection_summary,
    compute_cluster_size_distribution,
    filter_residual_pairs,
    summarize_cluster_metrics,
    summarize_pair_outcomes,
    summarize_similarity_distribution,
)
from blocking.loader import load_dataset_descriptor
from blocking.workflow import execute_blocking_run
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

_SIMILARITY_DISTRIBUTION_SCHEMA = {
    "bucket_start": pl.Float64,
    "bucket_end": pl.Float64,
    "edge_count": pl.Int64,
}
_CLUSTERS_SCHEMA = {
    "cluster_id": pl.Utf8,
    "node_id": pl.Utf8,
    "node_name": pl.Utf8,
    "node_role": pl.Utf8,
}


# --- summarize_similarity_distribution -------------------------------------


def test_summarize_similarity_distribution_adds_plot_ready_columns() -> None:
    frame = pl.DataFrame(
        {"bucket_start": [0.0, 0.5], "bucket_end": [0.5, 1.0], "edge_count": [3, 7]},
        schema=_SIMILARITY_DISTRIBUTION_SCHEMA,
    )

    result = summarize_similarity_distribution(frame)

    assert result.columns == [
        "bucket_start",
        "bucket_end",
        "edge_count",
        "bucket_midpoint",
        "bucket_label",
        "edge_share",
        "cumulative_edge_share",
    ]
    assert result.get_column("bucket_midpoint").to_list() == [0.25, 0.75]
    assert result.get_column("bucket_label").to_list() == ["[0.0, 0.5)", "[0.5, 1.0)"]
    assert result.get_column("edge_share").to_list() == pytest.approx([0.3, 0.7])
    assert result.get_column("cumulative_edge_share").to_list() == pytest.approx(
        [0.3, 1.0]
    )


def test_summarize_similarity_distribution_zero_filled_buckets_have_zero_share() -> (
    None
):
    frame = pl.DataFrame(
        {"bucket_start": [0.0, 0.5], "bucket_end": [0.5, 1.0], "edge_count": [0, 0]},
        schema=_SIMILARITY_DISTRIBUTION_SCHEMA,
    )

    result = summarize_similarity_distribution(frame)

    assert result.get_column("edge_share").to_list() == [0.0, 0.0]
    assert result.get_column("cumulative_edge_share").to_list() == [0.0, 0.0]


def test_summarize_similarity_distribution_empty_frame_returns_empty_with_columns() -> (
    None
):
    frame = pl.DataFrame(schema=_SIMILARITY_DISTRIBUTION_SCHEMA)

    result = summarize_similarity_distribution(frame)

    assert result.height == 0
    assert result.columns == [
        "bucket_start",
        "bucket_end",
        "edge_count",
        "bucket_midpoint",
        "bucket_label",
        "edge_share",
        "cumulative_edge_share",
    ]


# --- compute_cluster_size_distribution --------------------------------------


def test_compute_cluster_size_distribution_counts_by_size() -> None:
    clusters = pl.DataFrame(
        {
            "cluster_id": ["a", "a", "b", "c"],
            "node_id": ["1", "2", "3", "4"],
            "node_name": ["x", "y", "z", "w"],
            "node_role": ["source", "target", "source", "target"],
        },
        schema=_CLUSTERS_SCHEMA,
    )

    result = compute_cluster_size_distribution(clusters)

    assert result.columns == [
        "cluster_size",
        "cluster_count",
        "node_count",
        "cluster_share",
        "node_share",
    ]
    rows = {row["cluster_size"]: row for row in result.to_dicts()}
    assert rows[1] == {
        "cluster_size": 1,
        "cluster_count": 2,
        "node_count": 2,
        "cluster_share": 2 / 3,
        "node_share": 0.5,
    }
    assert rows[2] == {
        "cluster_size": 2,
        "cluster_count": 1,
        "node_count": 2,
        "cluster_share": 1 / 3,
        "node_share": 0.5,
    }


def test_compute_cluster_size_distribution_empty_clusters_returns_empty_frame() -> None:
    clusters = pl.DataFrame(schema=_CLUSTERS_SCHEMA)

    result = compute_cluster_size_distribution(clusters)

    assert result.height == 0
    assert result.columns == [
        "cluster_size",
        "cluster_count",
        "node_count",
        "cluster_share",
        "node_share",
    ]


def test_compute_cluster_size_distribution_sorted_by_size() -> None:
    clusters = pl.DataFrame(
        {
            "cluster_id": ["a", "b", "b", "b"],
            "node_id": ["1", "2", "3", "4"],
            "node_name": ["x", "y", "z", "w"],
            "node_role": ["source", "source", "target", "target"],
        },
        schema=_CLUSTERS_SCHEMA,
    )

    result = compute_cluster_size_distribution(clusters)

    assert result.get_column("cluster_size").to_list() == [1, 3]


# --- summarize_cluster_metrics -----------------------------------------------


def _cluster_shape_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country": ["__all__"],
            "total_clusters": [3],
            "singleton_clusters": [1],
            "singleton_ratio": [0.33],
            "mean_cluster_size": [1.5],
            "p95_cluster_size": [2.0],
            "max_cluster_size": [2],
        }
    )


def _directional_coverage_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "country": ["__all__"],
            "source_records": [10],
            "source_records_with_cluster": [8],
            "source_records_clustered_with_target": [5],
            "directional_coverage_ratio": [0.5],
        }
    )


def _pair_truth_eval_frame(*, country: str = "gb") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_system": ["gleif"],
            "target_system": ["gb"],
            "country": [country],
            "labelled_sources": [10],
            "target_rows": [20],
            "pair_universe": [100],
            "truth_pairs": [5],
            "predicted_pairs": [6],
            "tp": [4],
            "fp": [2],
            "fn": [1],
            "precision": [0.66],
            "recall": [0.8],
            "reduction_ratio": [0.9],
        }
    )


def test_summarize_cluster_metrics_melts_all_sources() -> None:
    result = summarize_cluster_metrics(
        cluster_shape=_cluster_shape_frame(),
        directional_coverage=_directional_coverage_frame(),
        pair_truth_eval=_pair_truth_eval_frame(),
        raw_pair_truth_eval=_pair_truth_eval_frame(),
    )

    assert result.columns == [
        "source",
        "country",
        "source_system",
        "target_system",
        "population",
        "metric",
        "value",
    ]
    assert set(result.get_column("source").unique().to_list()) == {
        "cluster_shape",
        "directional_coverage",
        "pair_truth_eval",
        "raw_pair_truth_eval",
    }

    precision_row = result.filter(
        (pl.col("source") == "pair_truth_eval") & (pl.col("metric") == "precision")
    )
    assert precision_row.get_column("value").to_list() == [0.66]
    assert precision_row.get_column("country").to_list() == ["gb"]
    assert precision_row.get_column("source_system").to_list() == ["gleif"]


def test_summarize_cluster_metrics_omits_missing_pair_truth_eval() -> None:
    result = summarize_cluster_metrics(
        cluster_shape=_cluster_shape_frame(),
        directional_coverage=_directional_coverage_frame(),
        pair_truth_eval=None,
        raw_pair_truth_eval=None,
    )

    assert set(result.get_column("source").unique().to_list()) == {
        "cluster_shape",
        "directional_coverage",
    }


def test_summarize_cluster_metrics_all_empty_returns_empty_frame_with_columns() -> None:
    result = summarize_cluster_metrics(
        cluster_shape=pl.DataFrame(),
        directional_coverage=pl.DataFrame(),
        pair_truth_eval=None,
        raw_pair_truth_eval=None,
    )

    assert result.height == 0
    assert result.columns == [
        "source",
        "country",
        "source_system",
        "target_system",
        "population",
        "metric",
        "value",
    ]


# --- filter_residual_pairs ----------------------------------------------------


def _residual_pairs_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_system": ["gleif"] * 4,
            "target_system": ["gb"] * 4,
            "country": ["gb"] * 4,
            "source_id": ["s1", "s2", "s3", "s4"],
            "target_id": ["t1", "t2", "t3", "t4"],
            "source_name": ["a", "b", "c", "d"],
            "source_name_cleansed": ["a", "b", "c", "d"],
            "target_name": ["a", "b2", "c2", "d2"],
            "target_name_cleansed": ["a", "b", "c2", "d2"],
            "name_equality": [
                "raw",
                "cleansed",
                "never",
                "never",
            ],
            "is_truth_pair": [True, True, True, False],
            "found": [True, True, False, None],
            "similarity": [1.0, 0.95, None, 0.4],
            "rank": [1, 1, None, 3],
        }
    )


def test_filter_residual_pairs_defaults_to_the_never_level() -> None:
    result = filter_residual_pairs(_residual_pairs_frame())

    assert result.get_column("source_id").to_list() == ["s3", "s4"]


def test_filter_residual_pairs_bucket_none_returns_every_bucket() -> None:
    result = filter_residual_pairs(_residual_pairs_frame(), bucket=None)

    assert result.height == 4


def test_filter_residual_pairs_isolates_candidate_set_pollution() -> None:
    result = filter_residual_pairs(
        _residual_pairs_frame(), bucket=None, is_truth_pair=False
    )

    assert result.get_column("source_id").to_list() == ["s4"]


def test_filter_residual_pairs_isolates_misses() -> None:
    result = filter_residual_pairs(
        _residual_pairs_frame(), bucket=None, is_truth_pair=True, found=False
    )

    assert result.get_column("source_id").to_list() == ["s3"]


def test_summarize_pair_outcomes_separates_truth_from_predicted_population() -> None:
    matrices = summarize_pair_outcomes(_residual_pairs_frame())

    recall = matrices["recall"].to_dicts()
    precision = matrices["precision"].to_dicts()

    # Truth pairs only: s4 is predicted-but-untrue and carries found=None, so
    # it must not appear in the recall matrix at all. Reported per level --
    # the coarse easy/difficult collapse is gone, since every level now
    # carries its own confusion matrix in `pair_truth_eval`.
    assert recall == [
        {
            "name_equality": "raw",
            "found": 1,
            "missed": 0,
            "total": 1,
            "recall": 1.0,
        },
        {
            "name_equality": "cleansed",
            "found": 1,
            "missed": 0,
            "total": 1,
            "recall": 1.0,
        },
        {
            "name_equality": "never",
            "found": 0,
            "missed": 1,
            "total": 1,
            "recall": 0.0,
        },
    ]
    # Predicted pairs: s3 was missed, so it is absent here even though it is a
    # truth pair; s4 is present as the false pair.
    assert [row["name_equality"] for row in precision] == ["raw", "cleansed", "never"]
    assert [row["false_pair"] for row in precision] == [0, 0, 1]


def test_summarize_pair_outcomes_keeps_levels_in_cascade_order() -> None:
    matrices = summarize_pair_outcomes(_residual_pairs_frame(), by="name_equality")

    assert matrices["recall"].get_column("name_equality").to_list() == [
        "raw",
        "cleansed",
        "never",
    ]
    assert matrices["recall"].get_column("missed").to_list() == [0, 0, 1]


def test_summarize_pair_outcomes_empty_frame_returns_empty_with_columns() -> None:
    empty = _residual_pairs_frame().clear()

    matrices = summarize_pair_outcomes(empty)

    assert matrices["recall"].height == 0
    assert matrices["recall"].columns == [
        "name_equality",
        "found",
        "missed",
        "total",
        "recall",
    ]
    assert matrices["precision"].columns == [
        "name_equality",
        "true_pair",
        "false_pair",
        "total",
        "precision",
    ]


def test_summarize_pair_outcomes_rejects_unknown_by() -> None:
    with pytest.raises(ValueError, match="name_equality"):
        summarize_pair_outcomes(_residual_pairs_frame(), by="similarity")


# --- build_inspection_summary (end-to-end) -----------------------------------


def _write_partition(
    layer_dir_for,
    *,
    system: str,
    layer: str,
    country: str,
    rows: list[dict[str, object]],
) -> Path:
    layer_dir = layer_dir_for(system, layer=layer)
    if layer == "canonical":
        # A target, or a source with no truth, is read from a dated snapshot.
        layer_dir = layer_dir / "2026-01-01"
    partition_dir = layer_partition_dir(layer_dir, value=country)
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(partition_dir / "part-00001.parquet")
    return layer_dir


def _write_match_metadata(
    layer_dir: Path, *, source_system: str, target_systems: list[str]
) -> None:
    metadata = {"source_system": source_system, "target_systems": target_systems}
    (layer_dir / "_match_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )


def _write_gleif_gb_fixture(layer_dir_for) -> None:
    matched_dir = _write_partition(
        layer_dir_for,
        system="gleif",
        layer="matched",
        country="gb",
        rows=[
            {
                "system_uri": "gleif:1",
                "name": "acme limited",
                "jurisdiction_code": "gb",
                "match_uri": "gb:1",
            },
            {
                "system_uri": "gleif:2",
                "name": "beta holdings",
                "jurisdiction_code": "gb",
                "match_uri": None,
            },
        ],
    )
    _write_match_metadata(matched_dir, source_system="gleif", target_systems=["gb"])
    _write_partition(
        layer_dir_for,
        system="gb",
        layer="canonical",
        country="gb",
        rows=[
            {"system_uri": "gb:1", "name": "acme ltd", "jurisdiction_code": "gb"},
            {"system_uri": "gb:2", "name": "omega plc", "jurisdiction_code": "gb"},
        ],
    )


def _build_config(roots: WorkspaceRoots) -> BlockingRunConfig:
    source = load_dataset_descriptor(
        roots=roots, system="gleif", require_ground_truth=True
    )
    target = load_dataset_descriptor(
        roots=roots, system="gb", require_ground_truth=False
    )
    strategy = BlockingStrategyConfig(
        representation="tfidf",
        top_k=2,
        min_similarity=0.2,
        max_candidates_per_source=None,
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
    )
    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=None,
        strategy=strategy,
    )


def test_build_inspection_summary_covers_real_run(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    _write_gleif_gb_fixture(layer_fixture_dir)
    result = execute_blocking_run(_build_config(workspace_roots))

    summary = build_inspection_summary(result)

    assert set(summary.keys()) == {
        "similarity_distribution",
        "cluster_size_distribution",
        "cluster_metrics",
    }
    assert (
        summary["similarity_distribution"].height
        == result.similarity_distribution.height
    )
    assert summary["cluster_size_distribution"].columns == [
        "cluster_size",
        "cluster_count",
        "node_count",
        "cluster_share",
        "node_share",
    ]
    assert "pair_truth_eval" in set(
        summary["cluster_metrics"].get_column("source").to_list()
    )
