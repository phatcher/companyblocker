import polars as pl
import pytest
from company_cleanse.pipeline import (
    DERIVE_ACRONYM_FIELD_STEP,
    ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
    ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    STEP_ENGINE_POLARS,
    STEP_ENGINE_UDF,
    _process_cleanse_lazyframe,
    _step_derive_acronym_field,
    _step_ensure_non_acronym_short_name,
    _step_ensure_quoted_name_in_cleansed,
)
from company_cleanse.rules import get_company_type_rules
from polars.testing import assert_frame_equal


def test_process_cleanse_lazyframe_transforms_rows_before_io():
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
        )
        .select(
            ["short_name", "quoted_name", "company_type", "name_cleansed", "acronym"]
        )
        .collect()
    )

    expected = pl.DataFrame(
        {
            "short_name": pl.Series(["acme"], dtype=pl.Utf8),
            "quoted_name": pl.Series([None], dtype=pl.Utf8),
            "company_type": pl.Series(["ltd"], dtype=pl.Utf8),
            "name_cleansed": pl.Series(["acme ltd"], dtype=pl.Utf8),
            "acronym": pl.Series([None], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(actual, expected)


def test_step_derive_acronym_field_uses_special_short_name_when_present():
    source = pl.DataFrame(
        {
            "_special_short_name": ["IBM"],
            "short_name": ["INTERNATIONAL"],
            "name_cleansed": ["international business machines ltd"],
            "company_type": ["ltd"],
            "_name_cleansed_norm": ["international business machines ltd"],
            "_company_type_norm": ["ltd"],
            "_short_name_norm": ["international"],
        }
    )

    actual = _step_derive_acronym_field(source.lazy()).select(["acronym"]).collect()

    assert actual["acronym"][0] == "IBM"


def test_step_derive_acronym_field_generates_from_short_name_when_available():
    source = pl.DataFrame(
        {
            "_special_short_name": [None],
            "short_name": ["IBM"],
            "name_cleansed": ["international business machines limited"],
            "company_type": ["limited"],
            "_name_cleansed_norm": ["international business machines limited"],
            "_company_type_norm": ["limited"],
            "_short_name_norm": ["ibm"],
        }
    )

    actual = _step_derive_acronym_field(source.lazy()).select(["acronym"]).collect()

    assert actual["acronym"][0] is not None


def test_step_derive_acronym_field_native_matches_udf():
    source = pl.DataFrame(
        {
            "_special_short_name": [None, "IBM", None, None, None],
            "short_name": ["IBM", "INTERNATIONAL", "ABC", "LTD", None],
            "name_cleansed": [
                "international business machines limited",
                "international business machines limited",
                "alpha beta consulting",
                "acme holdings ltd",
                "example company",
            ],
            "company_type": ["limited", "limited", "private", "ltd", "private"],
            "_name_cleansed_norm": [
                "international business machines limited",
                "international business machines limited",
                "alpha beta consulting",
                "acme holdings ltd",
                "example company",
            ],
            "_company_type_norm": ["limited", "limited", "private", "ltd", "private"],
            "_short_name_norm": ["ibm", "international", "abc", "ltd", ""],
        }
    )

    udf_actual = (
        _step_derive_acronym_field(
            source.lazy(),
            engine=STEP_ENGINE_UDF,
        )
        .select(["acronym"])
        .collect()
    )

    native_actual = (
        _step_derive_acronym_field(
            source.lazy(),
            engine=STEP_ENGINE_POLARS,
        )
        .select(["acronym"])
        .collect()
    )

    assert_frame_equal(native_actual, udf_actual)


def test_process_cleanse_lazyframe_accepts_derive_acronym_step_engine_overrides():
    source = pl.DataFrame(
        {
            "CompanyName": ["(IBM) International Business Machines Limited"],
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
            step_engines={DERIVE_ACRONYM_FIELD_STEP: STEP_ENGINE_POLARS},
        )
        .select(["acronym", "name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] == "international business machines ltd"
    assert actual["acronym"][0] == "ibm"


def test_step_ensure_non_acronym_short_name_ensures_uniqueness():
    source = pl.DataFrame(
        {
            "short_name": ["ABC"],
            "acronym": ["ABC"],
            "name_cleansed": ["abc corporation ltd"],
            "_short_name_norm": ["abc"],
            "_name_cleansed_norm": ["abc corporation ltd"],
        }
    )

    actual = (
        _step_ensure_non_acronym_short_name(source.lazy())
        .select(["name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] is not None


def test_step_ensure_non_acronym_short_name_native_matches_udf():
    source = pl.DataFrame(
        {
            "short_name": ["LTD", "ALPHA", "SA", "LLC", "", None],
            "acronym": [None, None, None, "LLC", None, None],
            "name_cleansed": [
                "acme holdings",
                "alpha services",
                "beta corp",
                "gamma llc",
                "delta ltd",
                "epsilon ltd",
            ],
            "_short_name_norm": ["ltd", "alpha", "sa", "llc", "", ""],
            "_name_cleansed_norm": [
                "acme holdings",
                "alpha services",
                "beta corp",
                "gamma llc",
                "delta ltd",
                "epsilon ltd",
            ],
        }
    )

    udf_actual = (
        _step_ensure_non_acronym_short_name(
            source.lazy(),
            engine=STEP_ENGINE_UDF,
        )
        .select(["name_cleansed"])
        .collect()
    )

    native_actual = (
        _step_ensure_non_acronym_short_name(
            source.lazy(),
            engine=STEP_ENGINE_POLARS,
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert_frame_equal(native_actual, udf_actual)


def test_process_cleanse_lazyframe_accepts_non_acronym_step_engine_overrides():
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
            step_engines={ENSURE_NON_ACRONYM_SHORT_NAME_STEP: STEP_ENGINE_POLARS},
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] == "acme ltd"


def test_step_ensure_quoted_name_in_cleansed_includes_quoted_name():
    source = pl.DataFrame(
        {
            "quoted_name": ["IBIS"],
            "acronym": [None],
            "name_cleansed": ["corporation ltd"],
            "_quoted_name_norm": ["ibis"],
            "_name_cleansed_norm": ["corporation ltd"],
        }
    )

    actual = (
        _step_ensure_quoted_name_in_cleansed(source.lazy())
        .select(["name_cleansed"])
        .collect()
    )

    assert actual["name_cleansed"][0] is not None


def test_step_ensure_quoted_name_in_cleansed_native_matches_udf():
    source = pl.DataFrame(
        {
            "quoted_name": ["IBIS", "TRIPLE D", "ALPHA", None, "  ", "ACME"],
            "acronym": [None, None, None, None, None, "acme"],
            "name_cleansed": [
                "corporation ltd",
                "triple d services ltd",
                "alpha services ltd",
                "beta ltd",
                "gamma ltd",
                None,
            ],
            "_quoted_name_norm": ["ibis", "triple d", "alpha", "", "", "acme"],
            "_name_cleansed_norm": [
                "corporation ltd",
                "triple d services ltd",
                "alpha services ltd",
                "beta ltd",
                "gamma ltd",
                "",
            ],
        }
    )

    udf_actual = (
        _step_ensure_quoted_name_in_cleansed(
            source.lazy(),
            engine=STEP_ENGINE_UDF,
        )
        .select(["name_cleansed"])
        .collect()
    )

    native_actual = (
        _step_ensure_quoted_name_in_cleansed(
            source.lazy(),
            engine=STEP_ENGINE_POLARS,
        )
        .select(["name_cleansed"])
        .collect()
    )

    assert_frame_equal(native_actual, udf_actual)


def test_process_cleanse_lazyframe_accepts_quoted_name_step_engine_overrides():
    source = pl.DataFrame(
        {
            "CompanyName": ['"IBIS" LIMITED (INTEGRATED BUSINESS INFORMATION SYSTEMS)'],
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
            step_engines={ENSURE_QUOTED_NAME_IN_CLEANSED_STEP: STEP_ENGINE_POLARS},
        )
        .select(["quoted_name", "name_cleansed"])
        .collect()
    )

    assert actual["quoted_name"][0] == "ibis"
    assert actual["name_cleansed"][0] is not None


# Golden table for the acronym/legal-form guard chain (`derive_acronym_field` ->
# `ensure_non_acronym_short_name` -> `ensure_quoted_name_in_cleansed`, the same
# order `_run_post_company_type_steps` runs them in). Each row is a documented
# interaction between a guard and the acronym derivation that gates it, not a
# single function in isolation: a rule change that flips one of these named
# cases fails a golden row rather than an assertion nobody can trace back to
# the behaviour it protected.
ACRONYM_LEGAL_FORM_GUARD_GOLDEN_CASES = [
    pytest.param(
        # `short_name` is itself a bare legal-form token and no acronym was
        # derived (the initials of "alpha care" don't spell "ltd"), so
        # `ensure_non_acronym_short_name` restores it as a prefix rather than
        # letting it vanish.
        "LTD",
        None,
        "alpha care ltd",
        "ltd",
        None,
        "ltd alpha care ltd",
        id="legal_form_short_name_restored_when_no_acronym",
    ),
    pytest.param(
        # `short_name` "LTD" happens to equal the initials of "line trading
        # destinations", so `derive_acronym_field` accepts it as a genuine
        # acronym. That sets `acronym`, which gates `ensure_non_acronym_short_name`
        # off (its guard only fires when `acronym` is null), so the legal-form
        # token is not also prefixed a second time.
        "LTD",
        None,
        "line trading destinations ltd",
        "ltd",
        "ltd",
        "line trading destinations ltd",
        id="coincidental_acronym_suppresses_legal_form_guard",
    ),
    pytest.param(
        # `short_name` "ALPHA" is not a legal-form token at all, so
        # `ensure_non_acronym_short_name`'s guard never applies to it; the name
        # is left exactly as it was.
        "ALPHA",
        None,
        "beta services ltd",
        "ltd",
        None,
        "beta services ltd",
        id="non_legal_form_short_name_left_untouched",
    ),
    pytest.param(
        # `quoted_name` "IBM" is the same token `derive_acronym_field` already
        # derived as `acronym`, so `ensure_quoted_name_in_cleansed`'s guard
        # skips it rather than prefixing the same brand twice.
        "IBM",
        "IBM",
        "international business machines ltd",
        "ltd",
        "ibm",
        "international business machines ltd",
        id="quoted_name_matching_acronym_not_duplicated",
    ),
    pytest.param(
        # `quoted_name` "BIG BLUE" is a distinct brand from the derived
        # `acronym` "ibm", so `ensure_quoted_name_in_cleansed` still restores
        # it even though an acronym was found.
        "IBM",
        "BIG BLUE",
        "international business machines ltd",
        "ltd",
        "ibm",
        "big blue international business machines ltd",
        id="quoted_name_distinct_from_acronym_still_restored",
    ),
]


def _run_acronym_legal_form_guard_chain(
    *,
    short_name: str,
    quoted_name: str | None,
    name_cleansed: str,
    company_type: str,
    engine: str,
) -> pl.DataFrame:
    schema = {
        "_special_short_name": pl.Utf8,
        "short_name": pl.Utf8,
        "quoted_name": pl.Utf8,
        "name_cleansed": pl.Utf8,
        "company_type": pl.Utf8,
        "_name_cleansed_norm": pl.Utf8,
        "_company_type_norm": pl.Utf8,
        "_short_name_norm": pl.Utf8,
        "_quoted_name_norm": pl.Utf8,
    }
    source = pl.DataFrame(
        {
            "_special_short_name": [None],
            "short_name": [short_name],
            "quoted_name": [quoted_name],
            "name_cleansed": [name_cleansed],
            "company_type": [company_type],
            "_name_cleansed_norm": [name_cleansed],
            "_company_type_norm": [company_type],
            "_short_name_norm": [short_name.lower()],
            "_quoted_name_norm": [(quoted_name or "").lower()],
        },
        schema=schema,
    )

    lf = source.lazy()
    lf = _step_derive_acronym_field(lf, engine=engine)
    lf = _step_ensure_non_acronym_short_name(lf, engine=engine)
    lf = _step_ensure_quoted_name_in_cleansed(lf, engine=engine)
    return lf.select(["acronym", "name_cleansed"]).collect()


@pytest.mark.parametrize(
    (
        "short_name",
        "quoted_name",
        "name_cleansed",
        "company_type",
        "expected_acronym",
        "expected_name_cleansed",
    ),
    ACRONYM_LEGAL_FORM_GUARD_GOLDEN_CASES,
)
@pytest.mark.parametrize("engine", [STEP_ENGINE_POLARS, STEP_ENGINE_UDF])
def test_acronym_legal_form_guard_chain_golden_cases(
    engine,
    short_name,
    quoted_name,
    name_cleansed,
    company_type,
    expected_acronym,
    expected_name_cleansed,
):
    actual = _run_acronym_legal_form_guard_chain(
        short_name=short_name,
        quoted_name=quoted_name,
        name_cleansed=name_cleansed,
        company_type=company_type,
        engine=engine,
    )

    assert actual["acronym"][0] == expected_acronym
    assert actual["name_cleansed"][0] == expected_name_cleansed


def test_process_cleanse_lazyframe_projects_basic_name_and_personal_owner_when_configured():
    source = pl.DataFrame(
        {
            "CompanyName": ["SOS-Dichtungen e.K. Inhaber Simone Hagemeier-Lemke"],
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
            personal_owner_markers=("INHABER", "INH"),
            char_whitelist=r"[^a-z0-9\s!&]",
            source_company_type_col=None,
            source_company_type_mapping=None,
            use_source_company_type=False,
        )
        .select(["name_cleansed_basic", "name_cleansed", "personal_owner"])
        .collect()
    )

    assert (
        actual["name_cleansed_basic"][0]
        == "sos dichtungen ek inhaber simone hagemeier lemke"
    )
    assert actual["name_cleansed"][0] == "sos dichtungen ek"
    assert actual["personal_owner"][0] == "simone hagemeier lemke"


@pytest.mark.parametrize(
    (
        "company_name",
        "expected_short_name",
        "expected_quoted_name",
        "expected_acronym",
        "expected_company_type",
        "expected_basic",
        "expected_cleansed",
        "expected_company_type_source",
        "expected_company_type_missing",
    ),
    [
        (
            "Felda Europe SRL",
            "felda europe",
            None,
            None,
            "srl",
            "felda europe srl",
            "felda europe srl",
            "name_suffix",
            False,
        ),
        (
            "ABC LTD",
            "abc",
            None,
            None,
            "ltd",
            "abc ltd",
            "abc ltd",
            "name_suffix",
            False,
        ),
        (
            '"IBIS" Corporation Ltd',
            "ibis",
            "ibis",
            None,
            "ltd",
            "ibis corporation ltd",
            "ibis corporation ltd",
            "name_suffix",
            False,
        ),
        (
            "Medical Industry Support Group Ltd",
            "medical industry support",
            None,
            None,
            "ltd",
            "medical industry support group ltd",
            "medical industry support group ltd",
            "name_suffix",
            False,
        ),
        (
            "Ltd Alpha Care Ltd",
            "ltd alpha care",
            None,
            None,
            "ltd",
            "ltd alpha care ltd",
            "ltd alpha care ltd",
            "name_suffix",
            False,
        ),
        (
            "Societa a responsabilita limitata",
            "srl",
            None,
            None,
            "srl",
            "societa a responsabilita limitata",
            "srl",
            "name_suffix",
            False,
        ),
        (
            "k line lng shipping uk limited",
            "k line lng shipping",
            None,
            None,
            "ltd",
            "k line lng shipping uk limited",
            "k line lng shipping uk ltd",
            "name_suffix",
            False,
        ),
    ],
)
def test_process_cleanse_lazyframe_short_name_policy_matrix(
    company_name,
    expected_short_name,
    expected_quoted_name,
    expected_acronym,
    expected_company_type,
    expected_basic,
    expected_cleansed,
    expected_company_type_source,
    expected_company_type_missing,
):
    source = pl.DataFrame({"CompanyName": [company_name]})

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
        .select(
            [
                "short_name",
                "quoted_name",
                "acronym",
                "company_type",
                "name_cleansed_basic",
                "name_cleansed",
                "company_type_source",
                "company_type_missing",
            ]
        )
        .collect()
    )

    expected = pl.DataFrame(
        {
            "short_name": pl.Series([expected_short_name], dtype=pl.Utf8),
            "quoted_name": pl.Series([expected_quoted_name], dtype=pl.Utf8),
            "acronym": pl.Series([expected_acronym], dtype=pl.Utf8),
            "company_type": pl.Series([expected_company_type], dtype=pl.Utf8),
            "name_cleansed_basic": pl.Series([expected_basic], dtype=pl.Utf8),
            "name_cleansed": pl.Series([expected_cleansed], dtype=pl.Utf8),
            "company_type_source": pl.Series(
                [expected_company_type_source], dtype=pl.Utf8
            ),
            "company_type_missing": pl.Series(
                [expected_company_type_missing], dtype=pl.Boolean
            ),
        }
    )

    assert_frame_equal(actual, expected)
