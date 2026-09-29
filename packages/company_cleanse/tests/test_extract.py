import pytest
from company_cleanse.extract import (
    _build_noise_word_sets,
    _derive_acronym,
    _derive_lead_token_decisions,
    _derive_row_company_type_tokens,
    _derive_short_name_from_cleansed,
    _ensure_non_acronym_short_name_in_cleansed,
    _ensure_quoted_name_in_cleansed,
    _extract_leading_delimited_raw,
    _extract_leading_delimited_tokens,
    _extract_leading_quoted_tokens,
    _extract_leading_token_struct,
    _extract_quoted_type_parenthesized_parts,
    _extract_quoted_type_parenthesized_struct,
    _extract_special_and_lead_struct,
    _initials_excluding_company_type,
    _is_acronym_like_token,
    _is_valid_short_name_token,
    _replace_company_type_with_canonical,
    _split_personal_owner_struct,
    _split_personal_owner_suffix,
)
from company_cleanse.rules import COMPANY_TYPE_MAPPING


@pytest.mark.parametrize(
    "value,expected",
    [
        ("(A.C.R.) UK LIMITED", "A.C.R."),
        ('"FELDA EUROPE"', None),
        ('"FELDA EUROPE" S.R.L.', "FELDA EUROPE"),
        (None, None),
    ],
)
def test_extract_leading_delimited_raw_cases(value, expected):
    assert _extract_leading_delimited_raw(value) == expected


def test_extract_quoted_type_parenthesized_struct_projects_fields():
    actual = _extract_quoted_type_parenthesized_struct(
        '"VARIG" S.A. (VIACAO AEREA RIO GRANDENSE)',
        COMPANY_TYPE_MAPPING,
    )
    assert actual == {
        "special_short_name": "varig",
        "special_company_type": "sa",
        "special_full_name": "VIACAO AEREA RIO GRANDENSE",
    }


def test_derive_lead_token_decisions_handles_quoted_and_override_paths():
    quoted = _derive_lead_token_decisions('"TRIPLE D" SERVICES LTD', None)
    overridden = _derive_lead_token_decisions(
        "(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD", "OVERRIDE"
    )

    assert quoted["short_name"] is None
    assert quoted["quoted_name"] == "triple d"
    assert overridden["short_name"] == "OVERRIDE"


def test_replace_company_type_with_canonical_appends_when_raw_type_missing():
    assert _replace_company_type_with_canonical("ACME", None, "LTD") == "ACME LTD"


def test_derive_acronym_and_non_acronym_short_name_paths():
    assert (
        _derive_acronym("MISG", "MEDICAL INDUSTRY SUPPORT GROUP LTD", "LTD") == "misg"
    )
    assert (
        _ensure_non_acronym_short_name_in_cleansed("LTD", None, "ALPHA CARE LTD")
        == "ltd ALPHA CARE LTD"
    )


def test_split_personal_owner_suffix_extracts_owner_for_marked_suffix():
    business_name, owner = _split_personal_owner_suffix(
        "SOS-DICHTUNGEN E.K. INHABER SIMONE HAGEMEIER-LEMKE",
        ("INHABER", "INH"),
    )
    assert business_name == "SOS-DICHTUNGEN E.K."
    assert owner == "simone hagemeier lemke"


def test_split_personal_owner_suffix_ignores_partial_or_missing_owner_payload():
    assert _split_personal_owner_suffix("ALPHA INHABER", ("INHABER",)) == (
        "ALPHA INHABER",
        None,
    )
    assert _split_personal_owner_suffix("ALPHA LLC", ("INHABER", "INH")) == (
        "ALPHA LLC",
        None,
    )


def test_derive_short_name_from_cleansed_removes_trailing_company_type_and_noise_words():
    assert _derive_short_name_from_cleansed("ACME LTD", "LTD", ()) == "ACME"
    assert _derive_short_name_from_cleansed("ACME XXX LTD", "LTD", ("XXX",)) == "ACME"


def test_derive_short_name_from_cleansed_never_removes_last_token():
    assert (
        _derive_short_name_from_cleansed("XXX SYSTEMS INC", "INC", ("XXX", "SYSTEMS"))
        == "XXX"
    )


def test_derive_short_name_from_cleansed_supports_anywhere_scope_tokens():
    assert (
        _derive_short_name_from_cleansed(
            "THE ACME LTD",
            "LTD",
            ({"token": "THE", "scope": "anywhere"},),
        )
        == "ACME"
    )
    assert (
        _derive_short_name_from_cleansed(
            "THE LTD",
            "LTD",
            ({"token": "THE", "scope": "anywhere"},),
        )
        == "THE"
    )


def test_derive_short_name_from_cleansed_ignores_corpus_noise_words_without_jurisdiction():
    # "Acme Consulting" -- "consulting" is a promoted global corpus noise word,
    # but no jurisdiction_code was supplied, so the mechanism stays dormant
    # (matches every caller that predates it exactly).
    assert (
        _derive_short_name_from_cleansed("acme consulting", None, ())
        == "acme consulting"
    )


def test_derive_short_name_from_cleansed_uses_jurisdiction_specific_corpus_words():
    # "consulting" is a promoted noise word for both "gb" and "global", so this
    # also exercises the jurisdiction-specific (not just global-fallback) path.
    assert (
        _derive_short_name_from_cleansed(
            "acme consulting", None, (), jurisdiction_code="gb"
        )
        == "acme"
    )


def test_derive_short_name_from_cleansed_falls_back_to_global_for_unmapped_jurisdiction():
    # "es" has no promoted system of its own -- falls back to the global list,
    # which also carries "consulting".
    assert (
        _derive_short_name_from_cleansed(
            "acme consulting", None, (), jurisdiction_code="es"
        )
        == "acme"
    )


def test_derive_short_name_from_cleansed_never_mixes_jurisdiction_word_lists():
    # "jean" is not a promoted noise word for "gb" (it never clears the
    # conservative top-30 cutoff even on "fr"), so a gb-tagged name ending in
    # "Jean" must not be stripped -- a French-only corpus word never applies
    # to a name tagged with a different jurisdiction.
    assert (
        _derive_short_name_from_cleansed(
            "boulangerie jean", None, (), jurisdiction_code="gb"
        )
        == "boulangerie jean"
    )


def test_derive_short_name_from_cleansed_respects_include_noise_words_toggle():
    assert (
        _derive_short_name_from_cleansed(
            "acme consulting",
            None,
            (),
            include_noise_words=False,
            jurisdiction_code="gb",
        )
        == "acme consulting"
    )


def test_build_noise_word_sets_splits_scopes():
    anywhere, suffix = _build_noise_word_sets(
        ("GROUP", {"token": "THE", "scope": "anywhere"}, "SERVICES")
    )
    assert "the" in anywhere
    assert "group" in suffix
    assert "services" in suffix


@pytest.mark.parametrize(
    "value,expected",
    [
        ('"TRIPLE D" SERVICES LTD', "triple d"),
        ("ALPHA LTD", None),
        (None, None),
    ],
)
def test_extract_leading_quoted_tokens(value, expected):
    assert _extract_leading_quoted_tokens(value) == expected


def test_extract_leading_delimited_tokens_handles_absent_and_present_values():
    assert _extract_leading_delimited_tokens("(A.C.R.) UK LIMITED") == "acr"
    assert _extract_leading_delimited_tokens("ALPHA LTD") is None


def test_extract_leading_token_struct_handles_empty_and_quoted():
    empty = _extract_leading_token_struct(None)
    quoted = _extract_leading_token_struct('"TRIPLE D" SERVICES LTD')

    assert empty == {"lead_token": None, "lead_quoted_token": None}
    assert quoted["lead_token"] == "triple d"
    assert quoted["lead_quoted_token"] == "triple d"


def test_token_quality_helpers_cover_edge_branches():
    assert _is_acronym_like_token("AB12") is True
    assert _is_acronym_like_token("A B") is True
    assert _is_valid_short_name_token("A") is False
    assert _is_valid_short_name_token("AB") is True


def test_extract_quoted_type_parenthesized_parts_rejects_non_acronym_quote():
    assert _extract_quoted_type_parenthesized_parts(
        '"Felda Europe" S.R.L. (Some Name)',
        COMPANY_TYPE_MAPPING,
    ) == (None, None, None)


def test_extract_quoted_type_parenthesized_parts_rejects_unknown_middle_type():
    assert _extract_quoted_type_parenthesized_parts(
        '"VARIG" UNKNOWN (VIACAO AEREA RIO GRANDENSE)',
        COMPANY_TYPE_MAPPING,
    ) == (None, None, None)


def test_extract_quoted_type_parenthesized_parts_rejects_short_full_name():
    assert _extract_quoted_type_parenthesized_parts(
        '"VARIG" S.A. (ONE)',
        COMPANY_TYPE_MAPPING,
    ) == (None, None, None)


def test_extract_special_and_lead_struct_uses_special_short_name_override():
    actual = _extract_special_and_lead_struct(
        '"VARIG" S.A. (VIACAO AEREA RIO GRANDENSE)',
        COMPANY_TYPE_MAPPING,
    )
    assert actual["special_short_name"] == "varig"
    assert actual["short_name"] == "varig"


def test_split_personal_owner_struct_projects_expected_fields():
    actual = _split_personal_owner_struct("ALPHA INHABER JOHN DOE", ("INHABER",))
    assert actual == {"stripped_name": "ALPHA", "personal_owner": "john doe"}


def test_replace_company_type_with_canonical_branch_paths():
    assert _replace_company_type_with_canonical("ACME", "LIMITED", "private") == "ACME"
    assert _replace_company_type_with_canonical("ACME", "LIMITED", "NOTATYPE") == "ACME"
    assert _replace_company_type_with_canonical("ACME", None, "LTD") == "ACME LTD"
    assert (
        _replace_company_type_with_canonical("ACME LIMITED", "LIMITED", "LTD")
        == "ACME LTD"
    )
    assert (
        _replace_company_type_with_canonical("ACME   LIMITED", "LIMITED", "LTD")
        == "ACME LTD"
    )


def test_initials_excluding_company_type_handles_tail_removal_and_empty_inputs():
    assert _initials_excluding_company_type(None, "LTD") is None
    assert _initials_excluding_company_type("ACME LIMITED", "LIMITED") == "a"
    assert _initials_excluding_company_type("ACME HOLDINGS", "private") == "ah"


def test_derive_acronym_rejects_blank_short_name():
    assert _derive_acronym("   ", "ACME LIMITED", "LIMITED") is None


def test_derive_row_company_type_tokens_handles_none_and_values():
    assert _derive_row_company_type_tokens(None) == frozenset()
    assert _derive_row_company_type_tokens("Limited Liability") == frozenset(
        {"limited", "liability"}
    )


def test_derive_short_name_from_cleansed_uses_precomputed_noise_word_sets():
    noise_sets = _build_noise_word_sets(
        ({"token": "THE", "scope": "anywhere"}, "GROUP")
    )
    assert (
        _derive_short_name_from_cleansed(
            "THE ACME GROUP LTD",
            "LTD",
            (),
            noise_word_sets=noise_sets,
        )
        == "ACME"
    )


def test_ensure_non_acronym_short_name_in_cleansed_noop_for_non_company_type_token():
    assert (
        _ensure_non_acronym_short_name_in_cleansed("ALPHA", None, "beta ltd")
        == "beta ltd"
    )


def test_ensure_quoted_name_in_cleansed_branch_paths():
    assert _ensure_quoted_name_in_cleansed(None, None, "acme ltd") == "acme ltd"
    assert _ensure_quoted_name_in_cleansed("IBM", "ibm", "acme ltd") == "acme ltd"
    assert _ensure_quoted_name_in_cleansed("Acme", None, "acme ltd") == "acme ltd"
    assert (
        _ensure_quoted_name_in_cleansed("Blue Sky", None, "acme ltd")
        == "blue sky acme ltd"
    )
