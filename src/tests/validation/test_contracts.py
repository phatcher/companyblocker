import polars as pl
import pytest

from validation.contracts import ARTIFACT_SCHEMAS, validate_artifact_schema


def _empty_frame_for(artifact_name: str) -> pl.DataFrame:
    schema = ARTIFACT_SCHEMAS[artifact_name]
    data: dict[str, list] = {column: [] for column in schema}
    return pl.DataFrame(data, schema=schema)


def test_validate_artifact_schema_accepts_expected_columns() -> None:
    frame = _empty_frame_for("source_outcomes")
    validate_artifact_schema(
        frame, artifact_name="source_outcomes", require_exact_columns=True
    )


def test_validate_artifact_schema_rejects_missing_columns() -> None:
    frame = pl.DataFrame({"run_id": []}, schema={"run_id": pl.Utf8})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_artifact_schema(frame, artifact_name="source_outcomes")


def test_validate_artifact_schema_rejects_extra_columns_in_exact_mode() -> None:
    frame = _empty_frame_for("source_outcomes").with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("extra_col")
    )

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_artifact_schema(
            frame, artifact_name="source_outcomes", require_exact_columns=True
        )


def test_robustness_eval_schema_carries_the_v1_contract() -> None:
    frame = _empty_frame_for("robustness_eval")
    validate_artifact_schema(
        frame, artifact_name="robustness_eval", require_exact_columns=True
    )
    assert "recall_retention_ratio" in ARTIFACT_SCHEMAS["robustness_eval"]
    assert "baseline_recall" in ARTIFACT_SCHEMAS["robustness_eval"]


def test_robustness_eval_rollup_schema_carries_mean_and_spread() -> None:
    frame = _empty_frame_for("robustness_eval_rollup")
    validate_artifact_schema(
        frame, artifact_name="robustness_eval_rollup", require_exact_columns=True
    )
    columns = ARTIFACT_SCHEMAS["robustness_eval_rollup"]
    assert "recall_retention_ratio_mean" in columns
    assert "recall_retention_ratio_std" in columns
    assert "sample_count" in columns
