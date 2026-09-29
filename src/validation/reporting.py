"""Write a validation run's artefacts, each held to its schema in `contracts.ARTIFACT_SCHEMAS`."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .contracts import validate_artifact_schema


def write_validation_artifacts(
    *,
    output_dir: Path,
    source_outcomes: pl.DataFrame,
    clusters: pl.DataFrame,
    directional_coverage: pl.DataFrame,
    shape_metrics: pl.DataFrame,
    pair_truth_eval: pl.DataFrame,
    exceptions: pl.DataFrame | None,
    run_metrics: pl.DataFrame | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    validate_artifact_schema(source_outcomes, artifact_name="source_outcomes")
    validate_artifact_schema(clusters, artifact_name="clusters")
    validate_artifact_schema(directional_coverage, artifact_name="directional_coverage")
    validate_artifact_schema(shape_metrics, artifact_name="cluster_shape")
    validate_artifact_schema(pair_truth_eval, artifact_name="pair_truth_eval")
    if exceptions is not None:
        validate_artifact_schema(exceptions, artifact_name="exceptions")
    if run_metrics is not None:
        validate_artifact_schema(run_metrics, artifact_name="run_metrics")

    source_outcomes_path = output_dir / "source_outcomes.parquet"
    clusters_path = output_dir / "clusters.parquet"
    coverage_path = output_dir / "directional_coverage.parquet"
    shape_path = output_dir / "cluster_shape.parquet"
    pair_truth_eval_path = output_dir / "pair_truth_eval.parquet"

    source_outcomes.write_parquet(source_outcomes_path)
    clusters.write_parquet(clusters_path)
    directional_coverage.write_parquet(coverage_path)
    shape_metrics.write_parquet(shape_path)
    pair_truth_eval.write_parquet(pair_truth_eval_path)

    outputs: dict[str, Path] = {
        "source_outcomes": source_outcomes_path,
        "clusters": clusters_path,
        "directional_coverage": coverage_path,
        "cluster_shape": shape_path,
        "pair_truth_eval": pair_truth_eval_path,
    }

    if run_metrics is not None:
        run_metrics_path = output_dir / "run_metrics.parquet"
        run_metrics.write_parquet(run_metrics_path)
        outputs["run_metrics"] = run_metrics_path

    if exceptions is not None:
        exceptions_path = output_dir / "exceptions.parquet"
        exceptions.write_parquet(exceptions_path)
        outputs["exceptions"] = exceptions_path

    return outputs
