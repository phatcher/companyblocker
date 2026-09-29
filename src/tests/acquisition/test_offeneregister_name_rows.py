from __future__ import annotations

import polars as pl
import pytest

from acquisition.offeneregister_name_rows import derive_offeneregister_name_rows
from acquisition.registry import get_system_plan
from acquisition.wikidata_name_rows import NAME_ROW_COLUMNS


def test_derive_offeneregister_name_rows_emits_one_row_per_previous_name():
    frame = pl.DataFrame(
        {
            "system_uri": ["offeneregister://R1101_HRB562"],
            "previous_names_list": [["Old Name One GmbH", "Old Name Two GmbH"]],
        }
    )

    result = derive_offeneregister_name_rows(frame)

    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert result["name"].to_list() == ["Old Name One GmbH", "Old Name Two GmbH"]
    assert result["source_type"].to_list() == ["PREVIOUS_NAME", "PREVIOUS_NAME"]
    assert result["language_code"].to_list() == [None, None]


def test_derive_offeneregister_name_rows_derives_id_from_system_uri():
    frame = pl.DataFrame(
        {
            "system_uri": ["offeneregister://R1101_HRB562"],
            "previous_names_list": [["Old Name GmbH"]],
        }
    )

    result = derive_offeneregister_name_rows(frame)

    assert result["id"].to_list() == ["R1101_HRB562"]


def test_derive_offeneregister_name_rows_drops_blank_and_null_names():
    frame = pl.DataFrame(
        {
            "system_uri": ["offeneregister://1", "offeneregister://2"],
            "previous_names_list": [["Real Name GmbH", ""], None],
        }
    )

    result = derive_offeneregister_name_rows(frame)

    assert result.height == 1
    assert result["name"].to_list() == ["Real Name GmbH"]


def test_derive_offeneregister_name_rows_tolerates_a_frame_without_the_column():
    frame = pl.DataFrame(
        {"system_uri": ["offeneregister://1"], "federal_state": ["Bayern"]}
    )

    result = derive_offeneregister_name_rows(frame)

    assert result.height == 0
    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert all(dtype == pl.Utf8 for dtype in result.dtypes)


def test_derive_offeneregister_name_rows_requires_system_uri():
    frame = pl.DataFrame({"previous_names_list": [["Old Name GmbH"]]})

    with pytest.raises(RuntimeError, match="system_uri"):
        derive_offeneregister_name_rows(frame)


def test_offeneregister_name_source_type_matches_the_catalog_name_variant_type_map():
    plan = get_system_plan("offeneregister")
    assert plan.name_variant_type_map is not None
    catalog_source_types = {
        source_type for source_type, _ in plan.name_variant_type_map
    }
    assert catalog_source_types == {"PREVIOUS_NAME"}


def test_offeneregister_name_variant_type_map_targets_the_shared_previous_vocabulary():
    plan = get_system_plan("offeneregister")
    assert plan.name_variant_type_map is not None
    mapped_name_types = {name_type for _, name_type in plan.name_variant_type_map}
    assert mapped_name_types == {"previous"}
