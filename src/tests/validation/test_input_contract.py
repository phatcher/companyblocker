import polars as pl
import pytest

from validation.input_contract import (
    PERTURBATION_INPUT_COLUMNS,
    VALIDATION_INPUT_SCHEMA,
    ensure_perturbation_columns,
    validate_perturbed_rows_are_complete,
    validate_required_input_columns,
)


def test_validation_input_schema_covers_baseline_and_perturbation_columns() -> None:
    assert set(VALIDATION_INPUT_SCHEMA) == {
        "system_uri",
        "name",
        "country",
        "match_uri",
        "source_uri",
        "profile_id",
        "profile_version",
        "scenario_id",
        "original_name",
        "changed",
    }
    assert set(PERTURBATION_INPUT_COLUMNS) == {
        "source_uri",
        "profile_id",
        "profile_version",
        "scenario_id",
        "original_name",
        "changed",
    }


def test_validate_required_input_columns_passes_when_present() -> None:
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://1"],
            "name_cleansed": ["acme"],
            "jurisdiction_code": ["gb"],
        }
    )
    validate_required_input_columns(
        frame, system="gb", name_col="name_cleansed", country_col="jurisdiction_code"
    )


def test_validate_required_input_columns_aggregates_every_missing_column() -> None:
    frame = pl.DataFrame({"name_cleansed": ["acme"]})
    with pytest.raises(ValueError) as exc_info:
        validate_required_input_columns(
            frame, system="gb", name_col="other_name", country_col="jurisdiction_code"
        )
    message = str(exc_info.value)
    assert "system_uri" in message
    assert "other_name" in message
    assert "jurisdiction_code" in message


def test_ensure_perturbation_columns_backfills_missing_columns() -> None:
    frame = pl.DataFrame(
        {"system_uri": ["gb://1"], "name": ["acme"], "country": ["gb"]}
    )

    result = ensure_perturbation_columns(frame)

    assert set(PERTURBATION_INPUT_COLUMNS).issubset(set(result.columns))
    row = result.row(0, named=True)
    for column in PERTURBATION_INPUT_COLUMNS:
        assert row[column] is None


def test_ensure_perturbation_columns_preserves_present_values() -> None:
    frame = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp/robustness-v1/light-typo"],
            "source_uri": ["gb://1"],
            "profile_id": ["robustness-v1"],
            "profile_version": ["1.0.0"],
            "scenario_id": ["light-typo"],
            "original_name": ["acme ltd"],
            "changed": [True],
        }
    )

    result = ensure_perturbation_columns(frame)
    row = result.row(0, named=True)
    assert row["source_uri"] == "gb://1"
    assert row["original_name"] == "acme ltd"
    assert row["changed"] is True


def test_validate_perturbed_rows_are_complete_passes_for_baseline_rows() -> None:
    frame = ensure_perturbation_columns(
        pl.DataFrame({"system_uri": ["gb://1"], "name": ["acme"], "country": ["gb"]})
    )
    validate_perturbed_rows_are_complete(frame, system="gb")


def test_validate_perturbed_rows_are_complete_passes_when_fully_populated() -> None:
    frame = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp/robustness-v1/light-typo"],
            "source_uri": ["gb://1"],
            "profile_id": ["robustness-v1"],
            "profile_version": ["1.0.0"],
            "scenario_id": ["light-typo"],
            "original_name": ["acme ltd"],
            "changed": [True],
        }
    )
    validate_perturbed_rows_are_complete(frame, system="perturbed:robustness-v1")


def test_validate_perturbed_rows_are_complete_rejects_null_perturbation_column() -> (
    None
):
    frame = pl.DataFrame(
        {
            "system_uri": ["perturbed://fp/robustness-v1/light-typo"],
            "source_uri": ["gb://1"],
            "profile_id": ["robustness-v1"],
            "profile_version": ["1.0.0"],
            "scenario_id": [None],
            "original_name": ["acme ltd"],
            "changed": [True],
        }
    )
    with pytest.raises(ValueError, match="perturbed"):
        validate_perturbed_rows_are_complete(frame, system="perturbed:robustness-v1")
