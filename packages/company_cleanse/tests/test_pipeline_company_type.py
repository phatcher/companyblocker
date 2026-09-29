import re

import polars as pl
import pytest
from company_cleanse.pipeline import (
    CANONICAL_COMPANY_TYPE_STEP,
    GENERATE_CLEANSED_COMPANY_NAME_STEP,
    STEP_ENGINE_NOOP,
    STEP_ENGINE_POLARS,
    STEP_ENGINE_UDF,
    _process_cleanse_lazyframe,
    _step_extract_canonical_company_type,
    _step_generate_cleansed_company_name,
    _step_resolve_and_unpack_company_type_decision,
)
from company_cleanse.rules import (
    COMPANY_TYPE_MAPPING,
    _build_company_type_suffix_trie,
    get_company_type_rules,
)
from polars.testing import assert_frame_equal


def _run_company_type_pipeline(names: list[str]) -> pl.DataFrame:
    """Run `names` through the default-profile pipeline, returning the extraction columns.

    Shared by the golden tables below so each row only has to state its input and expected
    output, not rebuild the regex/mapping and call `_process_cleanse_lazyframe` itself.
    """
    company_type_regex, company_type_mapping = get_company_type_rules()
    source = pl.DataFrame({"CompanyName": names})
    return (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
        )
        .select(["company_type", "name_cleansed", "company_type_source"])
        .collect()
    )


# Golden table -- exact expected output for a name, run once through the pipeline and recorded
# as a literal below, so an edge case a rule change flips has a case that fails. Each
# row is (name, expected company_type, expected name_cleansed, expected company_type_source).
COMPANY_TYPE_EXTRACTION_GOLDEN_CASES = [
    # A suffix from each of the five most-represented countries in
    # resources/company_type_rules.json.
    ("Acme Cooperatieve vennootschap", "cv", "acme cv", "name_suffix"),  # be
    ("Acme Fondo de Inversion", "fi", "acme fi", "name_suffix"),  # es
    ("Acme Einzelunternehmen", "ek", "acme ek", "name_suffix"),  # de
    ("Acme Co operative", "coop", "acme coop", "name_suffix"),  # au
    ("Acme Amministrazione comunale", "comune", "acme comune", "name_suffix"),  # ch
    # Two legal forms in one name -- only the trailing (English) one matches, not the
    # Liechtenstein "Anstalt" ahead of it.
    ("Anstalt Limited", "ltd", "anstalt ltd", "name_suffix"),
    # Abbreviated forms.
    (
        "Alpha Verwaltungsgesellschaft mbH",
        "gmbh",
        "alpha verwaltungsgesellschaft gmbh",
        "name_suffix",
    ),
    (
        "Pauly & Partner Partnerschaftsgesellschaft m b B",
        "partg",
        "pauly & partner partg",
        "name_suffix",
    ),
    # Non-ASCII form needing transliteration before it matches.
    (
        "Objektgesellschaft Kamper Straße mit beschränkter Haftung",
        "gmbh",
        "objektgesellschaft kamper strasse gmbh",
        "name_suffix",
    ),
    # Punctuated forms.
    ("Acme B.V.", "bv", "acme bv", "name_suffix"),
    ("Acme S.A.R.L.", "sarl", "acme sarl", "name_suffix"),
    # Mixed casing.
    ("AcMe LtD", "ltd", "acme ltd", "name_suffix"),
    # No legal form at all.
    ("Acme Consulting", "private", "acme consulting", "default_private"),
]


@pytest.mark.parametrize(
    "name,expected_company_type,expected_name_cleansed,expected_company_type_source",
    COMPANY_TYPE_EXTRACTION_GOLDEN_CASES,
)
def test_company_type_extraction_golden(
    name, expected_company_type, expected_name_cleansed, expected_company_type_source
):
    actual = _run_company_type_pipeline([name])
    assert actual["company_type"][0] == expected_company_type
    assert actual["name_cleansed"][0] == expected_name_cleansed
    assert actual["company_type_source"][0] == expected_company_type_source


# One measurement per live system (never re-run by the test): for each of gb, fr, ie,
# offeneregister and gleif, the ten most frequent trailing tokens among that system's
# `cleansed/primary` names with `company_type_source == "default_private"`. For each token, a real name ending in it is run once through
# today's pipeline and the result recorded as a literal below -- several no longer come out
# `default_private` at all (the on-disk data predates a rules/pipeline fix), which is exactly
# the point: a future rule change that flips any of these outcomes fails a named case here
# instead of only showing up as a silent shift in aggregate default_private counts.
DEFAULT_PRIVATE_TRAILING_TOKEN_GOLDEN_CASES = [
    # gb
    (
        '"WE THE CHANGE" FOUNDATION',
        "private",
        "we the change foundation",
        "default_private",
    ),
    (
        "10TH ENFIELD BOYS' BRIGADE AND GIRLS' ASSOCIATION",
        "private",
        "10th enfield boys brigade & girls association",
        "default_private",
    ),
    ("'2=1(UK)'", "private", "21 uk", "default_private"),
    ('"HEAD-ON-IN" KIDS CLUB', "private", "head on in kids club", "default_private"),
    ("5 H CENTRE", "private", "5h centre", "default_private"),
    ("3D CHURCH", "private", "3d church", "default_private"),
    ("2000 PARTNERSHIP", "private", "2000 partnership", "default_private"),
    (
        "73082 CAMELOT LOCOMOTIVE SOCIETY",
        "private",
        "73082 camelot locomotive society",
        "default_private",
    ),
    ("1509 GROUP", "private", "1509 group", "default_private"),
    (
        "303 SQUADRON POLISH SATURDAY SCHOOL",
        "private",
        "303 squadron polish saturday school",
        "default_private",
    ),
    # fr
    ("BORFLEX SERVICES", "private", "borflex services", "default_private"),
    ("ESBIT FRANCE", "private", "esbit france", "default_private"),
    ("AUR IMMO", "private", "aur immo", "default_private"),
    ("DECO INVEST", "private", "deco invest", "default_private"),
    ("ALPES CONSULTING", "private", "alpes consulting", "default_private"),
    ("TIVOLI IMMOBILIER", "private", "tivoli immobilier", "default_private"),
    (
        "MAJOR LES SPECIALISTES CONSEIL",
        "private",
        "major les specialistes conseil",
        "default_private",
    ),
    ("GEDIMO HOLDING", "private", "gedimo holding", "default_private"),
    (
        "SOCIETE ANONYME MARCEL CARON ET FILS",
        "private",
        "societe anonyme marcel caron et fils",
        "default_private",
    ),
    (
        "ENTREPRISE GENERALE DU BATIMENT",
        "private",
        "entreprise generale du batiment",
        "default_private",
    ),
    # ie -- several of these now match a real suffix under today's rules, unlike the on-disk
    # data they were sampled from.
    ("ELDAROS LIMITED", "ltd", "eldaros ltd", "name_suffix"),
    ("INESSA INTERNATIONAL COMPANY", "co", "inessa international co", "name_suffix"),
    (
        "DUNDALK CINEMAS, LIMITED TO PRO",
        "private",
        "dundalk cinemas limited to pro",
        "default_private",
    ),
    ("ELLIER HOLDINGS", "private", "ellier holdings", "default_private"),
    ("CAPECOVE INVESTMENTS", "private", "capecove investments", "default_private"),
    ("ASA WOMEN IRELAND", "private", "asa women ireland", "default_private"),
    ("JAVA SALES CORPORATION", "corp", "java sales corp", "name_suffix"),
    (
        "SINGLETON INTERNATIONAL INC.",
        "inc",
        "singleton international inc",
        "name_suffix",
    ),
    ("AVONDALE TRUST", "trust", "avondale trust", "name_suffix"),
    (
        "BRIDGEFOX INTERNATIONAL",
        "private",
        "bridgefox international",
        "default_private",
    ),
    # offeneregister
    (
        "Grand City Property Ltd - Zweigniederlassung Deutschland",
        "private",
        "grand city property ltd zweigniederlassung deutschland",
        "default_private",
    ),
    (
        "Archimedes Treuhand GmbH Steuerberatungsgesellschaft",
        "private",
        "archimedes treuhand gmbh steuerberatungsgesellschaft",
        "default_private",
    ),
    ("Ks, Consult e. Kfm.", "private", "ks consult e kfm", "default_private"),
    (
        "ICOS industrielle Computersysteme GmbH, Berlin",
        "private",
        "icos industrielle computersysteme gmbh berlin",
        "default_private",
    ),
    (
        "Bären-Apotheke, Inh.: Olaf Orthen e.K.",
        "ek",
        "baren apotheke inh olaf orthen ek",
        "name_suffix",
    ),
    (
        "WWS Freight & Trading Service e.Kfr.",
        "private",
        "wws freight & trading service e kfr",
        "default_private",
    ),
    (
        "Hans Egger & Sohn, Inhaber Johann Egger e.K.",
        "ek",
        "hans egger & sohn inhaber johann egger ek",
        "name_suffix",
    ),
    (
        "Wiedmann Gesellschaft mit beschränkter Haftpflicht",
        "private",
        "wiedmann gesellschaft mit beschrankter haftpflicht",
        "default_private",
    ),
    (
        "ALPERS WESSEL DORNBACH GmbH Wirtschaftsprüfungsgesellschaft",
        "private",
        "alpers wessel dornbach gmbh wirtschaftsprufungsgesellschaft",
        "default_private",
    ),
    (
        "Ferntrans GmbH Spedition",
        "private",
        "ferntrans gmbh spedition",
        "default_private",
    ),
    # gleif -- none of these are still `default_private` under today's rules.
    ("Bluewaters Limited", "ltd", "bluewaters ltd", "name_suffix"),
    (
        "Gryphon Corporation FZE LLC",
        "llc",
        "gryphon corporation fze llc",
        "name_suffix",
    ),
    ("Plaza BV", "bv", "plaza bv", "name_suffix"),
    (
        "DCC TRANSPORT LOGISTIK GMBH",
        "gmbh",
        "dcc transport logistik gmbh",
        "name_suffix",
    ),
    ("CONTENIDOS TELESOL SRL", "srl", "contenidos telesol srl", "name_suffix"),
    ("NEIPER HOLDINGS LTD", "ltd", "neiper holdings ltd", "name_suffix"),
    ("ERRANTES SL.", "sl", "errantes sl", "name_suffix"),
    (
        "Gantner Instruments Nordic AB",
        "ab",
        "gantner instruments nordic ab",
        "name_suffix",
    ),
    (
        "Challenger Enhanced Index Fund AS",
        "as",
        "challenger enhanced index fund as",
        "name_suffix",
    ),
]


@pytest.mark.parametrize(
    "name,expected_company_type,expected_name_cleansed,expected_company_type_source",
    DEFAULT_PRIVATE_TRAILING_TOKEN_GOLDEN_CASES,
)
def test_default_private_trailing_token_regression_golden(
    name, expected_company_type, expected_name_cleansed, expected_company_type_source
):
    actual = _run_company_type_pipeline([name])
    assert actual["company_type"][0] == expected_company_type
    assert actual["name_cleansed"][0] == expected_name_cleansed
    assert actual["company_type_source"][0] == expected_company_type_source


def test_source_mapping_non_extension_keeps_company_type_without_suffix():
    source = pl.DataFrame(
        {
            "CompanyName": ["EXAMPLE FUND"],
            "CompanyCategory": ["Investment Company with Variable Capital"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Investment Company with Variable Capital": "Investment Company",
            },
            use_source_company_type=True,
        )
        .select(["company_type", "name_cleansed", "company_type_source"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "company_type": ["investment company"],
            "name_cleansed": ["example fund"],
            "company_type_source": ["source_mapping"],
        }
    )

    assert_frame_equal(actual, expected)


def test_source_mapping_known_extension_appends_suffix():
    source = pl.DataFrame(
        {
            "CompanyName": ["DEEP CLEAN"],
            "CompanyCategory": ["Private Limited Company"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Private Limited Company": "Limited",
            },
            use_source_company_type=True,
        )
        .select(["company_type", "name_cleansed", "company_type_source"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "company_type": ["ltd"],
            "name_cleansed": ["deep clean ltd"],
            "company_type_source": ["source_mapping"],
        }
    )

    assert_frame_equal(actual, expected)


def test_source_mapping_only_uses_explicitly_mapped_keys():
    source = pl.DataFrame(
        {
            "CompanyName": ["DEEP CLEAN"],
            "CompanyCategory": ["Private Limited Company"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col="CompanyCategory",
            source_company_type_mapping={
                "Other Type": "Limited",
            },
            use_source_company_type=True,
        )
        .select(["company_type", "name_cleansed", "company_type_source"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "company_type": ["private"],
            "name_cleansed": ["deep clean"],
            "company_type_source": ["default_private"],
        }
    )

    assert_frame_equal(actual, expected)


def test_step_resolve_and_unpack_company_type_decision_matches_from_normalized_name_body_regex():
    company_type_regex, _company_type_mapping = get_company_type_rules()

    source = pl.DataFrame(
        {
            "_special_company_type": [None],
            "_normalized_name_body": ["ACME LIMITED"],
            "_name_body_reordered": ["ACME LIMITED"],
            "_source_company_type_raw": [None],
        }
    )

    actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=re.compile(company_type_regex),
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )

    assert actual["_matched_company_type"][0] == "LIMITED"
    assert actual["_effective_company_type"][0] == "LIMITED"
    assert actual["company_type_source"][0] == "name_suffix"


def test_step_resolve_and_unpack_company_type_decision_regex_native_path_handles_ascii_suffix():
    company_type_regex, _ = get_company_type_rules()

    source = pl.DataFrame(
        {
            "_special_company_type": [None],
            "_normalized_name_body": ["ALPHA LIMITED"],
            "_name_body_reordered": [None],
            "_source_company_type_raw": [None],
        }
    )

    actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=re.compile(company_type_regex),
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )

    assert actual["_matched_company_type"][0] == "LIMITED"
    assert actual["_effective_company_type"][0] == "LIMITED"
    assert actual["company_type_source"][0] == "name_suffix"


def test_step_resolve_and_unpack_company_type_decision_regex_fallback_transliterates_non_ascii_suffix():
    company_type_regex, _ = get_company_type_rules()

    source = pl.DataFrame(
        {
            "_special_company_type": [None],
            "_normalized_name_body": ["ACME SP ZØO"],
            "_name_body_reordered": ["ACME SP ZØO"],
            "_source_company_type_raw": [None],
        }
    )

    actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=re.compile(company_type_regex),
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )

    assert actual["_matched_company_type"][0] == "sp zoo"
    assert actual["_effective_company_type"][0] == "sp zoo"
    assert actual["company_type_source"][0] == "name_suffix"


def test_step_resolve_and_unpack_company_type_decision_uses_special_company_type_first():
    company_type_regex, _company_type_mapping = get_company_type_rules()

    source = pl.DataFrame(
        {
            "_special_company_type": ["SA"],
            "_normalized_name_body": ["DOES NOT MATCH"],
            "_name_body_reordered": ["DOES NOT MATCH"],
            "_source_company_type_raw": [None],
        }
    )

    actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=re.compile(company_type_regex),
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            ["_matched_company_type", "_effective_company_type", "company_type_source"]
        )
        .collect()
    )

    assert actual["_matched_company_type"][0] == "SA"
    assert actual["_effective_company_type"][0] == "SA"
    assert actual["company_type_source"][0] == "name_suffix"


def test_step_resolve_and_unpack_company_type_decision_fallback_to_source_mapping():
    company_type_regex, _company_type_mapping = get_company_type_rules()

    source = pl.DataFrame(
        {
            "_special_company_type": [None],
            "_normalized_name_body": ["EXAMPLE FUND"],
            "_name_body_reordered": ["EXAMPLE FUND"],
            "_source_company_type_raw": ["Investment Company"],
        }
    )

    actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=re.compile(company_type_regex),
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=True,
            normalized_source_company_type_mapping={
                "Investment Company": "investment firm"
            },
        )
        .select(
            [
                "_matched_company_type",
                "_source_company_type_mapped",
                "_effective_company_type",
                "company_type_source",
            ]
        )
        .collect()
    )

    assert actual["_matched_company_type"][0] is None
    assert actual["_source_company_type_mapped"][0] == "investment firm"
    assert actual["_effective_company_type"][0] == "investment firm"
    assert actual["company_type_source"][0] == "source_mapping"


def test_step_extract_canonical_company_type_maps_to_canonical():
    source = pl.DataFrame(
        {
            "_effective_company_type": ["LIMITED", "LLC", None],
        }
    )

    actual = (
        _step_extract_canonical_company_type(
            source.lazy(),
            company_type_mapping=COMPANY_TYPE_MAPPING,
        )
        .select(["company_type"])
        .collect()
    )

    assert actual["company_type"][0] == "LIMITED"
    assert actual["company_type"][2] == "private"


def test_step_extract_canonical_company_type_native_matches_udf():
    source = pl.DataFrame(
        {
            "_effective_company_type": ["LIMITED", "LLC", "PRIVATE", None, "GMBH"],
        }
    )

    udf_actual = (
        _step_extract_canonical_company_type(
            source.lazy(),
            company_type_mapping=COMPANY_TYPE_MAPPING,
            engine=STEP_ENGINE_UDF,
        )
        .select(["company_type"])
        .collect()
    )

    native_actual = (
        _step_extract_canonical_company_type(
            source.lazy(),
            company_type_mapping=COMPANY_TYPE_MAPPING,
            engine=STEP_ENGINE_POLARS,
        )
        .select(["company_type"])
        .collect()
    )

    assert_frame_equal(native_actual, udf_actual)


def test_step_resolve_company_type_decision_trie_matches_regex():
    source = pl.DataFrame(
        {
            "_special_company_type": [None, None, None, None],
            "_normalized_name_body": [
                "alpha limited",
                "beta public limited company",
                "gamma sp z o o",
                "delta unknown tail",
            ],
            "_name_body_reordered": [
                "alpha limited",
                "beta public limited company",
                "gamma sp z o o",
                "delta unknown tail",
            ],
            "_source_company_type_raw": [None, None, None, None],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    compiled_regex = re.compile(company_type_regex)
    suffix_trie, suffix_trie_max_tokens = _build_company_type_suffix_trie(
        company_type_mapping
    )

    regex_actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="regex",
            company_type_regex=compiled_regex,
            suffix_trie=None,
            suffix_trie_max_tokens=0,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            [
                "_matched_company_type",
                "_effective_company_type",
                "company_type_source",
            ]
        )
        .collect()
    )

    trie_actual = (
        _step_resolve_and_unpack_company_type_decision(
            source.lazy(),
            matcher_mode="trie",
            company_type_regex=None,
            suffix_trie=suffix_trie,
            suffix_trie_max_tokens=suffix_trie_max_tokens,
            and_tokens=("AND",),
            use_source_company_type=False,
            normalized_source_company_type_mapping=None,
        )
        .select(
            [
                "_matched_company_type",
                "_effective_company_type",
                "company_type_source",
            ]
        )
        .collect()
    )

    assert_frame_equal(trie_actual, regex_actual)


def test_process_cleanse_lazyframe_accepts_step_engine_overrides():
    source = pl.DataFrame(
        {
            "CompanyName": ["Acme Ltd"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
            step_engines={CANONICAL_COMPANY_TYPE_STEP: STEP_ENGINE_POLARS},
        )
        .select(["company_type", "name_cleansed"])
        .collect()
    )

    assert actual["company_type"][0] == "ltd"
    assert actual["name_cleansed"][0] == "acme ltd"


def test_step_generate_cleansed_company_name_replaces_type_and_normalizes():
    source = pl.DataFrame(
        {
            "_normalized_name_body": ["ACME LIMITED"],
            "_matched_company_type": ["LIMITED"],
            "company_type": ["ltd"],
        }
    )

    actual = (
        _step_generate_cleansed_company_name(
            source.lazy(),
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] is not None
    assert len(actual["name_cleansed"][0]) > 0


def test_step_generate_cleansed_company_name_normalizes_ampersand_to_connector_symbol():
    source = pl.DataFrame(
        {
            "_normalized_name_body": ["SMITH & JONES LIMITED"],
            "_matched_company_type": ["LIMITED"],
            "company_type": ["ltd"],
        }
    )

    actual = (
        _step_generate_cleansed_company_name(
            source.lazy(),
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert "&" in actual["name_cleansed"][0]


def test_step_generate_cleansed_company_name_recognizes_casefold_like_lowercase():
    # A `casefold`-only chain (no `lowercase`, and deliberately no `transliterate`
    # forcing everything to ASCII first) has to be recognized as a case-folding
    # operation the same way `lowercase` is, or the polars engine's `lowercase_enabled`
    # gate treats it as "no case operation selected" and mixes a mixed-case
    # `company_type` value into an otherwise-lowered name -- previously corrupting the
    # `.str.slice()` length arithmetic below it into producing "td" instead of
    # "acme ltd" for this exact input.
    source = pl.DataFrame(
        {
            "_normalized_name_body": ["ACME LIMITED"],
            "_matched_company_type": ["LIMITED"],
            "company_type": ["Ltd"],
        }
    )

    actual = (
        _step_generate_cleansed_company_name(
            source.lazy(),
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            normalization_operations=("casefold",),
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] == "acme ltd"


def test_step_generate_cleansed_company_name_native_matches_udf():
    source = pl.DataFrame(
        {
            "_normalized_name_body": [
                "ACME LIMITED",
                "SMITH & JONES LIMITED",
                "EXAMPLE PRIVATE",
                "OMEGA GMBH",
            ],
            "_matched_company_type": ["LIMITED", "LIMITED", None, "GMBH"],
            "company_type": ["ltd", "ltd", "private", "gmbh"],
        }
    )

    udf_actual = (
        _step_generate_cleansed_company_name(
            source.lazy(),
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            engine=STEP_ENGINE_UDF,
        )
        .select(["name_cleansed"])
        .collect()
    )

    native_actual = (
        _step_generate_cleansed_company_name(
            source.lazy(),
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            engine=STEP_ENGINE_POLARS,
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert_frame_equal(native_actual, udf_actual)


def test_process_cleanse_lazyframe_accepts_generate_cleansed_step_engine_overrides():
    source = pl.DataFrame(
        {
            "CompanyName": ["Acme Limited"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
            step_engines={GENERATE_CLEANSED_COMPANY_NAME_STEP: STEP_ENGINE_POLARS},
        )
        .select(["name_cleansed", "company_type"])
        .collect()
    )

    assert actual["company_type"][0] == "ltd"
    assert actual["name_cleansed"][0] == "acme ltd"


def test_process_cleanse_lazyframe_defaults_generate_cleansed_step_to_polars():
    source = pl.DataFrame(
        {
            "CompanyName": ["Acme Limited"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
        )
        .select(["name_cleansed", "company_type"])
        .collect()
    )

    assert actual["company_type"][0] == "ltd"
    assert actual["name_cleansed"][0] == "acme ltd"


def test_process_cleanse_lazyframe_accepts_generate_cleansed_noop_step_engine_override():
    source = pl.DataFrame(
        {
            "CompanyName": ["Acme Limited"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = (
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
            step_engines={GENERATE_CLEANSED_COMPANY_NAME_STEP: STEP_ENGINE_NOOP},
        )
        .select(["name_cleansed", "company_type", "acronym"])
        .collect()
    )

    assert actual["company_type"][0] == "ltd"
    assert actual["name_cleansed"][0] == "acme limited"
    assert actual["acronym"][0] is None


def test_process_cleanse_lazyframe_rejects_invalid_matcher_mode():
    source = pl.DataFrame({"CompanyName": ["Acme Ltd"]})
    company_type_regex, company_type_mapping = get_company_type_rules()

    with pytest.raises(
        ValueError, match="company_type_matcher must be 'regex' or 'trie'"
    ):
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="bogus",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
        ).collect()


def test_process_cleanse_lazyframe_rejects_invalid_step_engine_name():
    source = pl.DataFrame({"CompanyName": ["Acme Ltd"]})
    company_type_regex, company_type_mapping = get_company_type_rules()

    with pytest.raises(ValueError, match="Unsupported step engine 'bogus'"):
        _process_cleanse_lazyframe(
            source.lazy(),
            company_col="CompanyName",
            company_type_regex=company_type_regex,
            company_type_mapping=company_type_mapping,
            company_type_matcher="regex",
            and_tokens=("AND",),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
            step_engines={CANONICAL_COMPANY_TYPE_STEP: "bogus"},
        ).collect()
