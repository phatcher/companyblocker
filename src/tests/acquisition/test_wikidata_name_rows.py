from __future__ import annotations

import polars as pl

from acquisition.registry import get_system_plan
from acquisition.wikidata_name_rows import (
    NAME_ROW_COLUMNS,
    WIKIDATA_NAME_SOURCE_TYPES,
    derive_wikidata_name_rows,
)


def _rows(frame: pl.DataFrame) -> list[dict[str, object]]:
    return frame.select("name", "source_type", "language_code").to_dicts()


def test_derive_wikidata_name_rows_emits_one_row_per_name_with_native_source_type():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1"],
            "id": ["Q1"],
            "label_en": ["Acme"],
            "aliases_en": [["Acme Inc"]],
            "official_name_variants": [
                [{"value": "Acme Corporation", "language": "en"}]
            ],
            "short_name_variants": [[{"value": "ACME", "language": "en"}]],
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert _rows(result) == [
        {"name": "Acme", "source_type": "LABEL_EN", "language_code": "en"},
        {"name": "Acme Inc", "source_type": "ALIAS_EN", "language_code": "en"},
        {
            "name": "Acme Corporation",
            "source_type": "OFFICIAL_NAME",
            "language_code": "en",
        },
        {"name": "ACME", "source_type": "SHORT_NAME", "language_code": "en"},
    ]


def test_derive_wikidata_name_rows_tags_each_official_name_with_its_real_language():
    # Real Q8093 (Nintendo) shape.
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q8093"],
            "id": ["Q8093"],
            "official_name_variants": [
                [
                    {"value": "株式会社丸福", "language": "ja"},
                    {
                        "value": "任天堂骨牌株式会社",
                        "language": "ja",
                    },
                    {
                        "value": "任天堂株式会社",
                        "language": "ja",
                    },
                    {"value": "Nintendo Co., Ltd.", "language": "en"},
                ]
            ],
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 4
    assert result.filter(pl.col("language_code") == "ja").height == 3
    en_rows = result.filter(pl.col("language_code") == "en")
    assert en_rows.height == 1
    assert en_rows["name"].to_list() == ["Nintendo Co., Ltd."]


def test_derive_wikidata_name_rows_keeps_a_plain_string_variant_with_null_language():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1"],
            "id": ["Q1"],
            "official_name_variants": [[{"value": "Untyped Name", "language": None}]],
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 1
    assert result["language_code"].to_list() == [None]
    assert result["name"].to_list() == ["Untyped Name"]


def test_derive_wikidata_name_rows_drops_empty_lists_and_blank_names():
    variant_dtype = pl.List(pl.Struct({"value": pl.Utf8, "language": pl.Utf8}))
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1", "wikidata:Q2"],
            "id": ["Q1", "Q2"],
            "label_en": [None, None],
            "aliases_en": [[], ["  ", ""]],
            "official_name_variants": [[], []],
            "short_name_variants": [[], []],
        },
        schema_overrides={
            "label_en": pl.Utf8,
            "official_name_variants": variant_dtype,
            "short_name_variants": variant_dtype,
        },
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 0
    assert list(result.columns) == list(NAME_ROW_COLUMNS)


def test_derive_wikidata_name_rows_dedupes_repeated_value_language_pairs():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1"],
            "id": ["Q1"],
            "official_name_variants": [
                [
                    {"value": "Acme Ltd", "language": "en"},
                    {"value": "Acme Ltd", "language": "en"},
                ]
            ],
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 1


def test_derive_wikidata_name_rows_keeps_same_value_under_different_languages():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1"],
            "id": ["Q1"],
            "official_name_variants": [
                [
                    {"value": "Alpha", "language": "en"},
                    {"value": "Alpha", "language": "de"},
                ]
            ],
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 2
    assert sorted(result["language_code"].to_list()) == ["de", "en"]


def test_derive_wikidata_name_rows_returns_typed_empty_frame_for_empty_input():
    frame = pl.DataFrame(
        schema={
            "system_uri": pl.Utf8,
            "id": pl.Utf8,
            "label_en": pl.Utf8,
            "aliases_en": pl.List(pl.Utf8),
            "official_name_variants": pl.List(
                pl.Struct({"value": pl.Utf8, "language": pl.Utf8})
            ),
            "short_name_variants": pl.List(
                pl.Struct({"value": pl.Utf8, "language": pl.Utf8})
            ),
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert result.height == 0
    assert list(result.columns) == list(NAME_ROW_COLUMNS)
    assert all(dtype == pl.Utf8 for dtype in result.dtypes)


def test_derive_wikidata_name_rows_tolerates_a_missing_optional_column():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata:Q1"],
            "id": ["Q1"],
            "label_en": ["Acme"],
            "aliases_en": [["Acme Inc"]],
            "official_name_variants": [
                [{"value": "Acme Corporation", "language": "en"}]
            ],
            # short_name_variants deliberately omitted.
        }
    )

    result = derive_wikidata_name_rows(frame)

    assert set(result["source_type"].to_list()) == {
        "LABEL_EN",
        "ALIAS_EN",
        "OFFICIAL_NAME",
    }


def test_derive_wikidata_name_rows_requires_system_uri_and_id():
    frame = pl.DataFrame({"label_en": ["Acme"]})

    try:
        derive_wikidata_name_rows(frame)
    except RuntimeError as error:
        assert "system_uri" in str(error)
        assert "id" in str(error)
    else:
        raise AssertionError("expected RuntimeError for missing key columns")


def test_wikidata_name_source_types_match_the_catalog_name_variant_type_map():
    plan = get_system_plan("wikidata")
    assert plan.name_variant_type_map is not None
    catalog_source_types = {
        source_type for source_type, _ in plan.name_variant_type_map
    }
    assert WIKIDATA_NAME_SOURCE_TYPES == catalog_source_types


def test_wikidata_name_variant_type_map_targets_the_expected_shared_vocabulary():
    plan = get_system_plan("wikidata")
    assert plan.name_variant_type_map is not None
    mapped_name_types = {name_type for _, name_type in plan.name_variant_type_map}

    assert mapped_name_types == {"label", "alias", "official", "short"}
    # Deliberately not reusing GLEIF's vocabulary.
    assert "primary" not in mapped_name_types
    assert "trading" not in mapped_name_types
    assert "alternative_language" not in mapped_name_types
