from __future__ import annotations

import polars as pl

from acquisition.sharding_offeneregister import flatten_offeneregister_frame


def test_flatten_extracts_previous_names_from_company_name_entries():
    frame = pl.DataFrame(
        {
            "previous_names": [
                [{"company_name": " Old Name GmbH "}, {"company_name": ""}],
                None,
            ],
        }
    )

    result = flatten_offeneregister_frame(frame)

    assert result["previous_names_list"].to_list() == [["Old Name GmbH"], None]
    assert "previous_names" not in result.columns


def test_flatten_extracts_subsequent_registration_identifiers_with_jurisdiction_prefix():
    frame = pl.DataFrame(
        {
            "subsequent_registrations": [
                [
                    {
                        "subsequent_entity": {
                            "entity_properties": {
                                "company_number": "HRB1234",
                                "jurisdiction_code": "DE",
                            }
                        }
                    },
                    {
                        "subsequent_entity": {
                            "entity_properties": {
                                "company_number": "HRB5678",
                                "jurisdiction_code": None,
                            }
                        }
                    },
                    {"subsequent_entity": None},
                ],
            ],
        }
    )

    result = flatten_offeneregister_frame(frame)

    assert result["subsequent_registration_identifiers"].to_list() == [
        ["de:HRB1234", "HRB5678"]
    ]
    assert "subsequent_registrations" not in result.columns


def test_flatten_returns_none_for_subsequent_registrations_with_no_usable_entries():
    frame = pl.DataFrame(
        {
            "subsequent_registrations": [
                [{"subsequent_entity": {"entity_properties": {"company_number": ""}}}],
            ],
        }
    )

    result = flatten_offeneregister_frame(frame)

    assert result["subsequent_registration_identifiers"].to_list() == [None]


def test_flatten_derives_register_fields_and_flags_from_all_attributes():
    frame = pl.DataFrame(
        {
            "all_attributes": [
                {
                    "_registerArt": "hrb",
                    "_registerNummer": "150148",
                    "federal_state": "Bayern",
                    "former_registrar": "Munich",
                    "native_company_number": "DE123",
                    "registered_office": "Munich",
                    "registrar": "AG Munich",
                    "additional_data": {
                        "AD": True,
                        "CD": False,
                        "DK": None,
                    },
                },
            ],
        }
    )

    result = flatten_offeneregister_frame(frame)
    row = result.to_dicts()[0]

    assert row["register_art"] == "hrb"
    assert row["company_type"] == "hrb"
    assert row["register_number"] == "150148"
    assert row["federal_state"] == "Bayern"
    assert row["registrar"] == "AG Munich"
    assert row["register_flag_ad"] is True
    assert row["register_flag_cd"] is False
    assert row["register_flag_dk"] is None
    assert "all_attributes" not in result.columns


def test_flatten_leaves_frame_unchanged_when_optional_columns_are_absent():
    frame = pl.DataFrame({"id": [1, 2]})

    result = flatten_offeneregister_frame(frame)

    assert result.to_dicts() == [{"id": 1}, {"id": 2}]


def test_flatten_drops_officers_column_when_present():
    frame = pl.DataFrame({"id": [1], "officers": [[{"name": "A"}]]})

    result = flatten_offeneregister_frame(frame)

    assert "officers" not in result.columns
    assert result["id"].to_list() == [1]
