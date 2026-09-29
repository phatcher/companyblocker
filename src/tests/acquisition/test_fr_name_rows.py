from __future__ import annotations

import polars as pl
import pytest

from acquisition.fr_name_rows import FR_NAME_SOURCE_TYPES, derive_fr_name_rows
from acquisition.registry import get_system_plan
from acquisition.wikidata_name_rows import NAME_ROW_COLUMNS


def test_derive_fr_name_rows_emits_one_row_per_populated_field():
    frame = pl.DataFrame(
        {
            "system_uri": ["fr://552081317"],
            "sigleUniteLegale": ["IBM"],
            "denominationUsuelle1UniteLegale": ["Big Blue"],
            "denominationUsuelle2UniteLegale": [None],
            "denominationUsuelle3UniteLegale": [None],
            "nomUsageUniteLegale": [None],
            "pseudonymeUniteLegale": [None],
        },
        schema_overrides={
            "denominationUsuelle2UniteLegale": pl.Utf8,
            "denominationUsuelle3UniteLegale": pl.Utf8,
            "nomUsageUniteLegale": pl.Utf8,
            "pseudonymeUniteLegale": pl.Utf8,
        },
    )

    result = derive_fr_name_rows(frame)

    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert set(result["name"].to_list()) == {"IBM", "Big Blue"}
    assert set(result["source_type"].to_list()) == {"ACRONYM", "USUAL_NAME"}
    assert result["language_code"].to_list() == [None, None]


def test_derive_fr_name_rows_derives_id_from_system_uri():
    frame = pl.DataFrame(
        {
            "system_uri": ["fr://552081317"],
            "sigleUniteLegale": ["IBM"],
        }
    )

    result = derive_fr_name_rows(frame)

    assert result["id"].to_list() == ["552081317"]


def test_derive_fr_name_rows_drops_blank_and_null_names():
    frame = pl.DataFrame(
        {
            "system_uri": ["fr://1", "fr://2", "fr://3"],
            "sigleUniteLegale": ["Real Sigle", "", None],
        }
    )

    result = derive_fr_name_rows(frame)

    assert result.height == 1
    assert result["name"].to_list() == ["Real Sigle"]


def test_derive_fr_name_rows_drops_the_nd_sentinel():
    frame = pl.DataFrame(
        {
            "system_uri": ["fr://1", "fr://2"],
            "sigleUniteLegale": ["Real Sigle", "[ND]"],
            "pseudonymeUniteLegale": ["[ND]", "Real Pseudonym"],
        }
    )

    result = derive_fr_name_rows(frame)

    assert "[ND]" not in result["name"].to_list()
    assert set(result["name"].to_list()) == {"Real Sigle", "Real Pseudonym"}


def test_derive_fr_name_rows_explodes_all_three_usual_name_slots():
    frame = pl.DataFrame(
        {
            "system_uri": ["fr://1"],
            "denominationUsuelle1UniteLegale": ["Usual One"],
            "denominationUsuelle2UniteLegale": ["Usual Two"],
            "denominationUsuelle3UniteLegale": ["Usual Three"],
        }
    )

    result = derive_fr_name_rows(frame)

    assert result.height == 3
    assert set(result["name"].to_list()) == {"Usual One", "Usual Two", "Usual Three"}
    assert set(result["source_type"].to_list()) == {"USUAL_NAME"}


def test_derive_fr_name_rows_tolerates_a_frame_with_no_known_name_columns():
    frame = pl.DataFrame({"system_uri": ["fr://1"], "sexeUniteLegale": ["M"]})

    result = derive_fr_name_rows(frame)

    assert result.height == 0
    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert all(dtype == pl.Utf8 for dtype in result.dtypes)


def test_derive_fr_name_rows_requires_system_uri():
    frame = pl.DataFrame({"sigleUniteLegale": ["IBM"]})

    with pytest.raises(RuntimeError, match="system_uri"):
        derive_fr_name_rows(frame)


def test_fr_name_source_type_matches_the_catalog_name_variant_type_map():
    plan = get_system_plan("fr")
    assert plan.name_variant_type_map is not None
    catalog_source_types = {
        source_type for source_type, _ in plan.name_variant_type_map
    }
    assert catalog_source_types == FR_NAME_SOURCE_TYPES


def test_fr_name_variant_type_map_targets_distinct_name_types():
    plan = get_system_plan("fr")
    assert plan.name_variant_type_map is not None
    mapped_name_types = {name_type for _, name_type in plan.name_variant_type_map}
    assert mapped_name_types == {"acronym", "usual_name", "usage_name", "pseudonym"}
