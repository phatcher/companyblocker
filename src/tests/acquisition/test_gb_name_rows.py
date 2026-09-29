from __future__ import annotations

import polars as pl
import pytest

from acquisition.gb_name_rows import derive_gb_name_rows
from acquisition.registry import get_system_plan
from acquisition.wikidata_name_rows import NAME_ROW_COLUMNS


def test_derive_gb_name_rows_emits_one_row_per_previous_name_slot():
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "PreviousName_1.CompanyName": ["Alpha Old One Ltd"],
            "PreviousName_1.CONDATE": ["01/01/2020"],
            "PreviousName_2.CompanyName": ["Alpha Older One Ltd"],
            "PreviousName_2.CONDATE": ["01/01/2010"],
        }
    )

    result = derive_gb_name_rows(frame)

    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert result["name"].to_list() == ["Alpha Old One Ltd", "Alpha Older One Ltd"]
    assert result["source_type"].to_list() == ["PREVIOUS_NAME", "PREVIOUS_NAME"]
    assert result["language_code"].to_list() == [None, None]


def test_derive_gb_name_rows_derives_id_from_system_uri():
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://SC012345"],
            "PreviousName_1.CompanyName": ["Old Name Ltd"],
        }
    )

    result = derive_gb_name_rows(frame)

    assert result["id"].to_list() == ["SC012345"]


def test_derive_gb_name_rows_drops_blank_and_null_names():
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://1", "gb://2", "gb://3"],
            "PreviousName_1.CompanyName": ["Real Name Ltd", "", None],
        }
    )

    result = derive_gb_name_rows(frame)

    assert result.height == 1
    assert result["name"].to_list() == ["Real Name Ltd"]


def test_derive_gb_name_rows_does_not_treat_condate_as_a_name_slot():
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://1"],
            "PreviousName_1.CompanyName": ["Alpha Old One Ltd"],
            "PreviousName_1.CONDATE": ["01/01/2020"],
        }
    )

    result = derive_gb_name_rows(frame)

    assert result.height == 1
    assert "01/01/2020" not in result["name"].to_list()


def test_derive_gb_name_rows_orders_slots_numerically_not_lexically():
    frame = pl.DataFrame(
        {
            "system_uri": ["gb://1"],
            "PreviousName_2.CompanyName": ["Second"],
            "PreviousName_10.CompanyName": ["Tenth"],
            "PreviousName_1.CompanyName": ["First"],
        }
    )

    result = derive_gb_name_rows(frame)

    assert result["name"].to_list() == ["First", "Second", "Tenth"]


def test_derive_gb_name_rows_tolerates_a_frame_with_no_previous_name_columns():
    frame = pl.DataFrame(
        {"system_uri": ["gb://1"], "CountryOfOrigin": ["United Kingdom"]}
    )

    result = derive_gb_name_rows(frame)

    assert result.height == 0
    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert all(dtype == pl.Utf8 for dtype in result.dtypes)


def test_derive_gb_name_rows_requires_system_uri():
    frame = pl.DataFrame({"PreviousName_1.CompanyName": ["Old Name Ltd"]})

    with pytest.raises(RuntimeError, match="system_uri"):
        derive_gb_name_rows(frame)


def test_gb_name_source_type_matches_the_catalog_name_variant_type_map():
    plan = get_system_plan("gb")
    assert plan.name_variant_type_map is not None
    catalog_source_types = {
        source_type for source_type, _ in plan.name_variant_type_map
    }
    assert catalog_source_types == {"PREVIOUS_NAME"}


def test_gb_name_variant_type_map_targets_the_shared_previous_vocabulary():
    plan = get_system_plan("gb")
    assert plan.name_variant_type_map is not None
    mapped_name_types = {name_type for _, name_type in plan.name_variant_type_map}
    assert mapped_name_types == {"previous"}
