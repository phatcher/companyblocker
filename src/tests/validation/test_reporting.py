import polars as pl

from validation.contracts import ARTIFACT_SCHEMAS
from validation.reporting import write_validation_artifacts
from workspace.artifact_layout import validation_artifact_root
from workspace.roots import WorkspaceRoots


def test_write_validation_artifacts_round_trip(
    workspace_roots: WorkspaceRoots,
) -> None:
    output_dir = validation_artifact_root(workspace_roots)

    source_outcomes = pl.DataFrame(
        {
            "source_id": ["gleif:1"],
            "source_name": ["ACME LIMITED"],
            "source_match_uri": ["gb:1"],
            "target_id": ["gb:1"],
            "target_name": ["ACME LTD"],
            "similarity": [0.91],
            "rank": [1],
            "is_true_match": [True],
            "true_match_status": ["true_match"],
            "match_status": ["matched"],
            "reason_code": [None],
        }
    )

    clusters = pl.DataFrame(
        {
            "cluster_id": ["cc:1", "cc:1"],
            "node_id": ["gleif:1", "gb:1"],
            "node_name": ["ACME LIMITED", "ACME LTD"],
            "node_role": ["source", "target"],
        }
    )

    directional_coverage = pl.DataFrame(
        {
            "country": ["gb"],
            "source_records_total": [2],
            "source_records_filtered_out": [0],
            "source_records": [2],
            "source_records_with_cluster": [1],
            "source_records_clustered_with_target": [1],
            "directional_coverage_ratio": [0.5],
        }
    )

    shape_metrics = pl.DataFrame(
        {
            "country": ["gb"],
            "total_clusters": [1],
            "singleton_clusters": [0],
            "singleton_ratio": [0.0],
            "mean_cluster_size": [2.0],
            "p95_cluster_size": [2.0],
            "max_cluster_size": [2],
        }
    )

    pair_truth_eval = pl.DataFrame(
        {
            "source_system": ["gleif"],
            "target_system": ["gb"],
            "country": ["gb"],
            "population": ["universe"],
            "labelled_sources": [1],
            "target_rows": [1],
            "pair_universe": [1],
            "truth_pairs": [1],
            "predicted_pairs": [1],
            "tp": [1],
            "fp": [0],
            "fn": [0],
            "precision": [1.0],
            "recall": [1.0],
            "reduction_ratio": [0.0],
            "candidate_pair_count": [1],
            "source_rows": [1],
            "candidate_set_size_ratio": [1.0],
            "recall_at_k": [1.0],
        },
        schema=ARTIFACT_SCHEMAS["pair_truth_eval"],
    )

    run_metrics = pl.DataFrame(
        {
            "source_system": ["gleif"],
            "target_system": ["gb"],
            "country": ["gb"],
            "source_rows": [1],
            "elapsed_seconds": [0.5],
            "latency_seconds_per_1k_rows": [500.0],
            "candidate_edges": [1],
            "candidate_distinct_targets": [1],
            "candidate_duplication_ratio": [0.0],
        }
    )

    outputs = write_validation_artifacts(
        output_dir=output_dir,
        source_outcomes=source_outcomes,
        clusters=clusters,
        directional_coverage=directional_coverage,
        shape_metrics=shape_metrics,
        pair_truth_eval=pair_truth_eval,
        exceptions=None,
        run_metrics=run_metrics,
    )

    assert outputs["source_outcomes"].exists()
    assert outputs["clusters"].exists()
    assert outputs["directional_coverage"].exists()
    assert outputs["cluster_shape"].exists()
    assert outputs["pair_truth_eval"].exists()
    assert outputs["run_metrics"].exists()

    persisted_edges = pl.read_parquet(outputs["source_outcomes"])
    assert persisted_edges.height == 1
    assert persisted_edges.row(0, named=True)["source_id"] == "gleif:1"

    persisted_run_metrics = pl.read_parquet(outputs["run_metrics"])
    assert persisted_run_metrics.height == 1
    assert (
        persisted_run_metrics.row(0, named=True)["latency_seconds_per_1k_rows"] == 500.0
    )


def test_write_validation_artifacts_omits_run_metrics_when_not_provided(
    workspace_roots: WorkspaceRoots,
) -> None:
    output_dir = validation_artifact_root(workspace_roots)

    empty_source_outcomes = pl.DataFrame(
        {
            "source_id": [],
            "source_name": [],
            "source_match_uri": [],
            "target_id": [],
            "target_name": [],
            "similarity": [],
            "rank": [],
            "is_true_match": [],
            "true_match_status": [],
            "match_status": [],
            "reason_code": [],
        },
        schema={
            "source_id": pl.Utf8,
            "source_name": pl.Utf8,
            "source_match_uri": pl.Utf8,
            "target_id": pl.Utf8,
            "target_name": pl.Utf8,
            "similarity": pl.Float64,
            "rank": pl.Int32,
            "is_true_match": pl.Boolean,
            "true_match_status": pl.Utf8,
            "match_status": pl.Utf8,
            "reason_code": pl.Utf8,
        },
    )
    empty_clusters = pl.DataFrame(
        {"cluster_id": [], "node_id": [], "node_name": [], "node_role": []},
        schema={
            "cluster_id": pl.Utf8,
            "node_id": pl.Utf8,
            "node_name": pl.Utf8,
            "node_role": pl.Utf8,
        },
    )
    empty_coverage = pl.DataFrame(
        {
            "country": [],
            "source_records_total": [],
            "source_records_filtered_out": [],
            "source_records": [],
            "source_records_with_cluster": [],
            "source_records_clustered_with_target": [],
            "directional_coverage_ratio": [],
        },
        schema={
            "country": pl.Utf8,
            "source_records_total": pl.Int64,
            "source_records_filtered_out": pl.Int64,
            "source_records": pl.Int64,
            "source_records_with_cluster": pl.Int64,
            "source_records_clustered_with_target": pl.Int64,
            "directional_coverage_ratio": pl.Float64,
        },
    )
    empty_shape = pl.DataFrame(
        {
            "country": [],
            "total_clusters": [],
            "singleton_clusters": [],
            "singleton_ratio": [],
            "mean_cluster_size": [],
            "p95_cluster_size": [],
            "max_cluster_size": [],
        },
        schema={
            "country": pl.Utf8,
            "total_clusters": pl.Int64,
            "singleton_clusters": pl.Int64,
            "singleton_ratio": pl.Float64,
            "mean_cluster_size": pl.Float64,
            "p95_cluster_size": pl.Float64,
            "max_cluster_size": pl.Int64,
        },
    )
    # Built from the contract: this test's subject is omitting run_metrics, not
    # which columns pair_truth_eval carries.
    empty_pair_truth_eval = pl.DataFrame(schema=ARTIFACT_SCHEMAS["pair_truth_eval"])

    outputs = write_validation_artifacts(
        output_dir=output_dir,
        source_outcomes=empty_source_outcomes,
        clusters=empty_clusters,
        directional_coverage=empty_coverage,
        shape_metrics=empty_shape,
        pair_truth_eval=empty_pair_truth_eval,
        exceptions=None,
    )

    assert "run_metrics" not in outputs
    assert not (output_dir / "run_metrics.parquet").exists()
