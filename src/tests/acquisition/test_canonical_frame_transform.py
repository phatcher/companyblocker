from __future__ import annotations

import polars as pl

from acquisition.canonical_frame_transform import (
    apply_canonical_source_aliases,
    canonicalize_frame,
)

_IDENTITY_COLUMNS = [
    "jurisdiction_code",
    "company_number",
    "name",
    "alternative_names",
    "lei",
    "vat",
    "tax_id",
    "previous_names",
    "company_type",
    "current_status",
    "incorporation_date",
    "dissolution_date",
    "inactive",
    "branch",
    "branch_status",
    "registered_address_in_full",
    "registry_url",
]
_OPENCORPORATES_COLUMNS = ["opencorporates_url"]
_METADATA_COLUMNS = ["match_uri", "system_uri", "metadata_generated_utc"]


def _canonicalize(
    frame: pl.DataFrame,
    *,
    country: str,
    source_company_type_col: str | None = None,
    exclude_company_type_values: set[str] | None = None,
    canonical_source_column_aliases=None,
    system_field_candidates: dict[str, list[str]] | None = None,
    canonical_derived_fields=None,
    company_type_mapping_resolver=lambda _country: {},
) -> pl.DataFrame:
    return canonicalize_frame(
        frame,
        country=country,
        source_company_type_col=source_company_type_col,
        exclude_company_type_values=exclude_company_type_values or set(),
        canonical_source_column_aliases=canonical_source_column_aliases,
        system_field_candidates=system_field_candidates or {},
        canonical_derived_fields=canonical_derived_fields,
        company_type_mapping_resolver=company_type_mapping_resolver,
        project_identity_columns=_IDENTITY_COLUMNS,
        opencorporates_columns=_OPENCORPORATES_COLUMNS,
        metadata_columns=_METADATA_COLUMNS,
    )


def test_apply_canonical_source_aliases_renames_source_to_target():
    frame = pl.DataFrame({"raisonSociale": ["Acme"], "other": [1]})

    result = apply_canonical_source_aliases(frame, aliases=(("raisonSociale", "name"),))

    assert set(result.columns) == {"name", "other"}
    assert result["name"].to_list() == ["Acme"]


def test_apply_canonical_source_aliases_is_a_noop_without_aliases():
    frame = pl.DataFrame({"a": [1]})

    assert apply_canonical_source_aliases(frame, aliases=None) is frame
    assert apply_canonical_source_aliases(frame, aliases=()) is frame


def test_apply_canonical_source_aliases_skips_rename_when_target_already_present():
    frame = pl.DataFrame({"raisonSociale": ["Acme"], "name": ["Existing"]})

    result = apply_canonical_source_aliases(frame, aliases=(("raisonSociale", "name"),))

    assert result["name"].to_list() == ["Existing"]
    assert "raisonSociale" in result.columns


def _default_system_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Acme Ltd"],
            "company_number": ["12345678"],
            "company_type": ["Private Limited Company"],
            "current_status": ["Active"],
            "dissolution_date": [None],
            "incorporation_date": ["2020-01-01"],
            "branch": [None],
            "branch_status": [None],
            "registered_address_in_full": ["1 Main St"],
            "registry_url": ["https://example.test"],
            "match_uri": ["urn:x"],
            "system_uri": ["urn:y"],
        }
    )


def test_canonicalize_frame_default_system_resolves_same_named_columns():
    result = _canonicalize(_default_system_frame(), country="testsys")
    row = result.to_dicts()[0]

    assert row["jurisdiction_code"] == "testsys"
    assert row["company_number"] == "12345678"
    assert row["name"] == "Acme Ltd"
    assert row["company_type"] == "Private Limited Company"
    assert row["current_status"] == "Active"
    assert row["incorporation_date"] == "2020-01-01"
    assert row["registered_address_in_full"] == "1 Main St"
    assert row["registry_url"] == "https://example.test"
    assert row["match_uri"] == "urn:x"
    assert row["system_uri"] == "urn:y"
    assert row["opencorporates_url"] is None
    assert row["inactive"] is False


def test_canonicalize_frame_derives_inactive_from_current_status_when_no_inactive_column():
    frame = _default_system_frame().with_columns(
        pl.lit("Dissolved").alias("current_status")
    )

    result = _canonicalize(frame, country="testsys")

    assert result.to_dicts()[0]["inactive"] is True


def test_canonicalize_frame_applies_company_type_mapping_when_resolver_returns_values():
    result = _canonicalize(
        _default_system_frame(),
        country="testsys",
        company_type_mapping_resolver=lambda _country: {
            "Private Limited Company": "Limited"
        },
    )

    assert result.to_dicts()[0]["company_type"] == "Limited"


def test_canonicalize_frame_falls_back_to_raw_type_when_mapping_resolver_raises_keyerror():
    def raising_resolver(_country: str) -> dict[str, str]:
        raise KeyError("no mapping for this system")

    result = _canonicalize(
        _default_system_frame(),
        country="testsys",
        company_type_mapping_resolver=raising_resolver,
    )

    assert result.to_dicts()[0]["company_type"] == "Private Limited Company"


def test_canonicalize_frame_filters_rows_by_exclude_company_type_values():
    frame = pl.DataFrame(
        {
            "raw_type": ["Excluded", "Kept"],
            "name": ["Bad Co", "Good Co"],
            "company_number": ["1", "2"],
        }
    )

    result = _canonicalize(
        frame,
        country="testsys",
        source_company_type_col="raw_type",
        exclude_company_type_values={"Excluded"},
    )

    assert result["name"].to_list() == ["Good Co"]


def test_canonicalize_frame_drops_rows_with_populated_successor_lei():
    frame = pl.DataFrame(
        {
            "name": ["Superseded Co", "Current Co"],
            "company_number": ["1", "2"],
            "Entity_SuccessorEntity_SuccessorLEI": ["LEI999", ""],
        }
    )

    result = _canonicalize(frame, country="gleif")

    assert result["name"].to_list() == ["Current Co"]


def _wikidata_frame(**columns: object) -> pl.DataFrame:
    base: dict[str, object] = {"label_en": ["Acme Wiki"]}
    base.update(columns)
    return pl.DataFrame(base)


def test_canonicalize_frame_wikidata_defaults_active_status_and_false_inactive():
    result = _canonicalize(_wikidata_frame(), country="wikidata")
    row = result.to_dicts()[0]

    assert row["name"] == "Acme Wiki"
    assert row["current_status"] == "Active"
    assert row["inactive"] is False


def test_canonicalize_frame_wikidata_derives_dissolved_status_from_dissolution_date():
    result = _canonicalize(
        _wikidata_frame(dissolution_date=["2020-01-01T00:00:00Z"]),
        country="wikidata",
    )
    row = result.to_dicts()[0]

    assert row["current_status"] == "Dissolved"
    assert row["inactive"] is True
    assert row["dissolution_date"] == "2020-01-01"


def test_canonicalize_frame_wikidata_nulls_out_item_placeholder_company_type():
    result = _canonicalize(
        _wikidata_frame(company_type=["item"]),
        country="wikidata",
    )

    assert result.to_dicts()[0]["company_type"] is None


def test_canonicalize_frame_wikidata_resolves_jurisdiction_and_scoped_company_number():
    frame = pl.DataFrame(
        {
            "label_en": ["Acme GB", "Acme FR", "Acme IE"],
            "country": [["Q145"], ["Q142"], ["Q27"]],
            "company_number_gb": [["01234567"], [None], [None]],
            "company_number_fr": [[None], ["123456789"], [None]],
            "lei": [["5493001KJTIIGC8Y1R12"], [None], [None]],
        }
    )

    result = _canonicalize(frame, country="wikidata")
    rows = result.to_dicts()

    assert [row["jurisdiction_code"] for row in rows] == ["GB", "FR", "IE"]
    assert [row["company_number"] for row in rows] == ["01234567", "123456789", None]
    assert rows[0]["lei"] == "5493001KJTIIGC8Y1R12"
    assert rows[1]["lei"] is None


def test_canonicalize_frame_gleif_ra_filtered_company_number_pads_scopes_and_nulls():
    frame = pl.DataFrame(
        {
            "name": ["GB Short", "GB Invalid Authority", "DE Scoped"],
            "Entity_LegalJurisdiction": ["GB", "GB", "DE"],
            "Entity_RegistrationAuthority_RegistrationAuthorityID": [
                "RA000585",
                "RA999999",
                "RA000197",
            ],
            "company_number": ["1234567", "1234567", "150148"],
        }
    )

    result = _canonicalize(frame, country="gleif")
    rows = result.to_dicts()

    assert rows[0]["company_number"] == "01234567"  # zero-padded to GB's width
    assert rows[1]["company_number"] is None  # disallowed registration authority
    assert rows[2]["company_number"] == "RA000197|150148"  # DE authority-scoped


def test_canonicalize_frame_offeneregister_authority_scoped_company_number():
    # A `company_number` with no `_`-separated court prefix (the last row)
    # falls back to the unscoped value per
    # `_resolve_offeneregister_authority_scoped_company_number_expr`'s own
    # docstring. Polars evaluates `.list.get(-2)`/`.list.get(-1)` for every
    # row before the `pl.when(n_segments >= 2)` guard applies, so this shape
    # must tolerate the out-of-bounds index rather than raise.
    frame = pl.DataFrame(
        {
            "name": ["Scoped", "Placeholder Range", "Unmapped Court", "No Prefix"],
            "jurisdiction_code": ["DE", "DE", "DE", "DE"],
            "company_number": [
                "B8534_HRB150148",
                "B8534_HRB205000",
                "ZZZZZ_HRB123",
                "HRB150148",
            ],
        }
    )

    result = _canonicalize(frame, country="offeneregister")

    assert result["company_number"].to_list() == [
        "RA000351|HRB150148",
        None,
        None,
        "HRB150148",
    ]


def test_canonicalize_frame_selects_only_the_requested_output_columns():
    result = _canonicalize(_default_system_frame(), country="testsys")

    assert result.columns == [
        *_IDENTITY_COLUMNS,
        *_OPENCORPORATES_COLUMNS,
        *_METADATA_COLUMNS,
    ]
