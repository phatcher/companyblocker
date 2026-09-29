from __future__ import annotations

import polars as pl
import pytest

from validation.perturbation_dataset_contracts import (
    PERTURBATION_DATASET_REQUIRED_COLUMNS,
    validate_perturbation_dataset_schema,
)


def _complete_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {column: [] for column in PERTURBATION_DATASET_REQUIRED_COLUMNS}
    )


def test_validate_perturbation_dataset_schema_accepts_complete_frame() -> None:
    validate_perturbation_dataset_schema(_complete_frame(), profile_id="robustness-v1")


def test_validate_perturbation_dataset_schema_aggregates_missing_columns() -> None:
    frame = _complete_frame().drop("original_name", "changed")

    with pytest.raises(ValueError, match="original_name") as exc_info:
        validate_perturbation_dataset_schema(frame, profile_id="robustness-v1")

    assert "changed" in str(exc_info.value)
    assert "robustness-v1" in str(exc_info.value)
