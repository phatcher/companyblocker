import inspect
import json

import company_cleanse
import polars as pl
import pytest
from company_cleanse.api import (
    _load_noise_words_override,
    _resolve_company_type_rules,
    _split_short_name_stage_flags,
    cleanse_lazyframe,
    extract_quoted_name,
    strip_company_suffix,
)
from company_cleanse.config import CleanseConfig
from polars.testing import assert_frame_equal


def test_cleanse_lazyframe_uses_default_rules():
    source = pl.DataFrame({"company_name": ["Acme Ltd"]})

    actual = (
        cleanse_lazyframe(source.lazy())
        .select(
            [
                "short_name",
                "quoted_name",
                "company_type",
                "name_cleansed_basic",
                "name_cleansed",
                "personal_owner",
                "acronym",
            ]
        )
        .collect()
    )

    expected = pl.DataFrame(
        {
            "short_name": pl.Series(["acme"], dtype=pl.Utf8),
            "quoted_name": pl.Series([None], dtype=pl.Utf8),
            "company_type": pl.Series(["ltd"], dtype=pl.Utf8),
            "name_cleansed_basic": pl.Series(["acme ltd"], dtype=pl.Utf8),
            "name_cleansed": pl.Series(["acme ltd"], dtype=pl.Utf8),
            "personal_owner": pl.Series([None], dtype=pl.Utf8),
            "acronym": pl.Series([None], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(actual, expected)


def test_cleanse_lazyframe_basic_tier_collapses_spaced_single_char_runs():
    """name_cleansed_basic is not just whitespace/punctuation stripping -- it
     inherits the default normalization profile's singlechar collapse, so a
     spaced-out acronym in the raw source ('I B M') already reads as one real
     token ('ibm') at the basic tier, before any noise-word/company-type
     stripping runs. This matters for raw-vs-basic corpus divergence analysis
    : naive per-token wordfreq scoring of the *raw*, uncollapsed
     fragments ('i', 'b', 'm') would score as ordinary/common English (isolated
     letters are frequent), which is a fragmentation artifact, not evidence
     that raw data is closer to general language than cleansed data is."""
    source = pl.DataFrame({"company_name": ["I B M", "S R L Holdings"]})

    actual = (
        cleanse_lazyframe(source.lazy())
        .select(["company_name", "name_cleansed_basic", "name_cleansed"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "company_name": ["I B M", "S R L Holdings"],
            "name_cleansed_basic": ["ibm", "srl holdings"],
            "name_cleansed": ["ibm", "srl holdings"],
        }
    )

    assert_frame_equal(actual, expected)


def test_cleanse_lazyframe_uses_company_name_default_column():
    source = pl.DataFrame({"company_name": ["Acme Ltd"]})

    actual = (
        cleanse_lazyframe(source.lazy())
        .select(
            [
                "company_type",
                "name_cleansed",
            ]
        )
        .collect()
    )

    expected = pl.DataFrame(
        {
            "company_type": pl.Series(["ltd"], dtype=pl.Utf8),
            "name_cleansed": pl.Series(["acme ltd"], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(actual, expected)


def test_cleanse_lazyframe_uses_profile_selection_for_short_name_noise_words():
    source = pl.DataFrame({"company_name": ["Acme Consulting Ltd"]})

    actual_strict = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(noise_words_profile="strict"),
        )
        .select(["short_name", "name_cleansed"])
        .collect()
    )

    actual_balanced = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(noise_words_profile="balanced"),
        )
        .select(["short_name", "name_cleansed"])
        .collect()
    )

    expected_strict = pl.DataFrame(
        {
            "short_name": pl.Series(["acme consulting"], dtype=pl.Utf8),
            "name_cleansed": pl.Series(["acme consulting ltd"], dtype=pl.Utf8),
        }
    )
    expected_balanced = pl.DataFrame(
        {
            "short_name": pl.Series(["acme"], dtype=pl.Utf8),
            "name_cleansed": pl.Series(["acme consulting ltd"], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(actual_strict, expected_strict)
    assert_frame_equal(actual_balanced, expected_balanced)


def test_cleanse_lazyframe_uses_jurisdiction_code_column_for_corpus_noise_words():
    # "london" is a promoted gb-specific corpus noise word (not in the packaged
    # default noise-word list), so it should only be stripped when a row's own
    # jurisdiction_code is "gb" -- never when jurisdiction_code is absent, and
    # never for a differently-tagged row (proving lists never mix languages).
    no_jurisdiction_column = pl.DataFrame({"company_name": ["Acme London Ltd"]})
    gb_row = pl.DataFrame(
        {"company_name": ["Acme London Ltd"], "jurisdiction_code": ["gb"]}
    )
    fr_row = pl.DataFrame(
        {"company_name": ["Acme London Ltd"], "jurisdiction_code": ["fr"]}
    )

    assert cleanse_lazyframe(no_jurisdiction_column.lazy()).select(
        "short_name"
    ).collect()["short_name"].to_list() == ["acme london"]
    assert cleanse_lazyframe(gb_row.lazy()).select("short_name").collect()[
        "short_name"
    ].to_list() == ["acme"]
    assert cleanse_lazyframe(fr_row.lazy()).select("short_name").collect()[
        "short_name"
    ].to_list() == ["acme london"]


def test_cleanse_lazyframe_jurisdiction_col_none_disables_corpus_noise_words():
    source = pl.DataFrame(
        {"company_name": ["Acme London Ltd"], "jurisdiction_code": ["gb"]}
    )

    actual = (
        cleanse_lazyframe(source.lazy(), config=CleanseConfig(jurisdiction_col=None))
        .select("short_name")
        .collect()["short_name"]
    )

    assert actual.to_list() == ["acme london"]


def test_cleanse_lazyframe_short_name_profile_can_skip_noise_words_stage():
    source = pl.DataFrame({"company_name": ["Acme Systems Plc"]})

    default_short_name = (
        cleanse_lazyframe(source.lazy()).select("short_name").collect()["short_name"]
    )
    no_noise_short_name = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(short_name_profile="default|-noise_words"),
        )
        .select("short_name")
        .collect()["short_name"]
    )

    assert default_short_name.to_list() == ["acme"]
    assert no_noise_short_name.to_list() == ["acme systems"]


def test_cleanse_lazyframe_short_name_profile_can_skip_company_type_stage():
    # "AEIE" is a packaged company-type component but not a noise word (mirrors
    # the single-string strip_company_suffix() behavior for the same input).
    source = pl.DataFrame({"company_name": ["Acme AEIE"]})

    default_short_name = (
        cleanse_lazyframe(source.lazy()).select("short_name").collect()["short_name"]
    )
    no_company_type_short_name = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(short_name_profile="default|-company_type"),
        )
        .select("short_name")
        .collect()["short_name"]
    )

    assert default_short_name.to_list() == ["acme"]
    assert no_company_type_short_name.to_list() == ["acme aeie"]


def test_cleanse_lazyframe_short_name_profile_rejects_unsupported_profile():
    source = pl.DataFrame({"company_name": ["Acme Ltd"]})

    with pytest.raises(ValueError, match="short_name_profile"):
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(short_name_profile="default|-lowercase"),
        ).collect()


def test_package_root_exports_supported_api_only():
    assert company_cleanse.__all__ == [
        "CleanseConfig",
        "get_manual_noise_words",
        "get_profiled_noise_words",
        "get_effective_noise_words",
        "get_corpus_noise_words",
        "cleanse_lazyframe",
        "strip_company_suffix",
        "extract_quoted_name",
        "get_company_type_rules",
        "get_company_type_rules_for_country",
        "UnknownCompanyTypeCountryError",
        "get_ascii_homoglyphs",
        "get_geographic_terms",
        "ShortNameCandidate",
        "derive_short_name_candidate",
    ]


def test_package_root_has_no_untracked_public_names():
    # Pinning __all__ alone only catches a name being removed from or reordered
    # within the list; it does not catch a new `from .module import something`
    # landing in __init__.py without also being added to __all__, which would
    # silently widen the real importable surface (`from company_cleanse import
    # something` still works even when something isn't in __all__). Submodules
    # (`company_cleanse.normalize`, etc.) are always accessible by dotted import
    # regardless of __all__, so they are excluded here rather than tracked.
    public_non_module_names = {
        name
        for name in dir(company_cleanse)
        if not name.startswith("_")
        and not inspect.ismodule(getattr(company_cleanse, name))
    }
    assert public_non_module_names == set(company_cleanse.__all__)


def test_get_effective_noise_words_exported_at_root_matches_config():
    # Users customizing noise_words want this exact tuple as a starting point --
    # it's what strip_company_suffix()/cleanse_lazyframe() use by default.
    from company_cleanse.config import get_effective_noise_words as config_impl

    assert company_cleanse.get_effective_noise_words() == config_impl()

    for name in company_cleanse.__all__:
        assert hasattr(company_cleanse, name)


def test_strip_company_suffix_removes_terminal_company_type_tokens_only():
    assert strip_company_suffix("AG Grid Ltd") == "ag grid"
    assert (
        strip_company_suffix("Grid AG Solutions", noise_words_profile="strict")
        == "grid ag solutions"
    )


def test_strip_company_suffix_uses_jurisdiction_code_for_corpus_noise_words():
    assert strip_company_suffix("Acme London Ltd") == "acme london"
    assert strip_company_suffix("Acme London Ltd", jurisdiction_code="gb") == "acme"
    assert (
        strip_company_suffix("Acme London Ltd", jurisdiction_code="fr") == "acme london"
    )


def test_strip_company_suffix_supports_noise_word_override_path(tmp_path):
    override_path = tmp_path / "noise_override.json"
    override_path.write_text(
        json.dumps({"suffix": ["systems"], "anywhere": []}),
        encoding="utf-8",
    )

    assert (
        strip_company_suffix(
            "Acme Systems",
            noise_words_path=override_path,
        )
        == "acme"
    )


def test_strip_company_suffix_accepts_direct_noise_words_override():
    assert (
        strip_company_suffix(
            "Acme Systems Ltd",
            noise_words=("systems",),
        )
        == "acme"
    )


def test_strip_company_suffix_removes_trailing_ampersand_after_company_type_trim():
    assert strip_company_suffix("ABC & CO") == "abc"


def test_strip_company_suffix_can_skip_noise_words_stage():
    # The common case: still strip the trailing legal form (Plc) but leave other
    # trailing words alone, rather than the default's more aggressive noise trimming.
    assert strip_company_suffix("Acme Systems Plc") == "acme"
    assert (
        strip_company_suffix(
            "Acme Systems Plc", normalization_profile="default|-noise_words"
        )
        == "acme systems"
    )

    assert strip_company_suffix("Acme Systems Ltd", noise_words=("systems",)) == "acme"
    assert (
        strip_company_suffix(
            "Acme Systems Ltd",
            noise_words=("systems",),
            normalization_profile="default|-noise_words",
        )
        == "acme systems"
    )


def test_strip_company_suffix_can_skip_company_type_stage():
    # "AEIE" is a packaged company-type component but not a noise word, so it only
    # gets trimmed by the company_type stage -- disabling it should leave it in place.
    assert strip_company_suffix("Acme AEIE") == "acme"
    assert (
        strip_company_suffix("Acme AEIE", normalization_profile="default|-company_type")
        == "acme aeie"
    )


def test_strip_company_suffix_with_both_stages_disabled_only_tidies():
    assert (
        strip_company_suffix(
            "Acme Systems Ltd",
            noise_words=("systems",),
            normalization_profile="default|-company_type|-noise_words",
        )
        == "acme systems ltd"
    )


def test_a_profile_carries_its_noise_words_level_and_it_outranks_the_argument():
    """`+noise_words:<level>` puts the level in the profile string, so one
    string says everything a record needs. An unknown level in the string is
    refused even though the argument names a good one, which shows the string
    is what is used."""
    flags = _split_short_name_stage_flags("default|+noise_words:balanced")
    assert flags.noise_words_level == "balanced"
    assert flags.include_noise_words and flags.geographic_tiers is None
    assert _split_short_name_stage_flags("default").noise_words_level is None

    with pytest.raises(TypeError, match="'nonsense' not found"):
        strip_company_suffix(
            "Acme Systems Ltd",
            normalization_profile="default|+noise_words:nonsense",
            noise_words_profile="aggressive",
        )


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ("default|+noise_words", "without a level"),
        ("default|+noise_words:", "without a level"),
        ("default|-noise_words|+noise_words:strict", "switches noise words off"),
    ],
)
def test_a_noise_words_level_that_is_missing_or_contradicted_is_refused(
    profile: str, message: str
):
    with pytest.raises(ValueError, match=message):
        strip_company_suffix("Acme Systems Ltd", normalization_profile=profile)


def test_strip_company_suffix_stage_toggles_compose_with_normal_profile_modifiers():
    # "LTD" is trimmed by the (still-enabled) noise_words stage regardless of
    # -company_type, since it happens to be a packaged noise word too -- "AEIE"
    # isolates the -company_type toggle from that overlap (see the skip-stage test).
    assert (
        strip_company_suffix(
            "Acme-AEIE",
            normalization_profile="default|-lowercase|-company_type",
        )
        == "Acme AEIE"
    )


@pytest.mark.parametrize(
    "normalization_profile,expected_profile,expected_company_type,expected_noise_words",
    [
        ("default", "default", True, True),
        ("default|-company_type", "default", False, True),
        ("default|-noise_words", "default", True, False),
        ("default|-company_type|-noise_words", "default", False, False),
        ("default|-lowercase|-company_type", "default|-lowercase", False, True),
        ("-COMPANY_TYPE|-Noise_Words", "default", False, False),
    ],
)
def test_split_short_name_stage_flags_cases(
    normalization_profile,
    expected_profile,
    expected_company_type,
    expected_noise_words,
):
    flags = _split_short_name_stage_flags(normalization_profile)

    assert flags.remaining_profile == expected_profile
    assert flags.include_company_type == expected_company_type
    assert flags.include_noise_words == expected_noise_words
    # The geographic stage is opt-in; none of these profiles name it.
    assert flags.geographic_tiers is None


def test_cleanse_lazyframe_handles_long_german_legal_suffix_chain():
    source = pl.DataFrame(
        {
            "company_name": [
                "neunte grundstucksverwaltung ahg beteiligungs & handelsgesellschaft mbh & co kg"
            ]
        }
    )

    actual = (
        cleanse_lazyframe(source.lazy())
        .select(["short_name", "company_type"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "short_name": pl.Series(
                ["neunte grundstucksverwaltung ahg"], dtype=pl.Utf8
            ),
            "company_type": pl.Series(["kg"], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(actual, expected)


def test_cleanse_lazyframe_casefold_matches_lowercase_because_transliteration_is_forced():
    # Measured against every real offeneregister/gleif name containing a case-fold-
    # divergent character (149,762 + 11,893 rows respectively, when the caseless fold was added):
    # `name_cleansed` is identical between `default` and `default|casefold` for all
    # of them, because `_resolve_runtime_normalization` unconditionally forces
    # `transliterate` ahead of the case operation for this pipeline regardless of
    # profile -- by the time either case operation runs, `ß`/`ς`/`Σ` are already
    # plain ASCII (`ss`/`s`/`s`), where `.lower()` and `.casefold()` agree. This is
    # why `lowercase` staying the default chain's choice is a measured decision, not
    # an unexamined inheritance -- see NORMALIZATION_OPERATION_CASEFOLD's docstring
    # in normalize.py for the full real-data figures.
    source = pl.DataFrame(
        {"company_name": ["Straße Handel GmbH", "ΑΚΜΩΝ ΑΝΩΝΥΜΟΣ ΕΤΑΙΡΕΙΑ"]}
    )

    lowercase_out = (
        cleanse_lazyframe(
            source.lazy(), config=CleanseConfig(normalization_profile="default")
        )
        .select("name_cleansed")
        .collect()
    )
    casefold_out = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(normalization_profile="default|casefold"),
        )
        .select("name_cleansed")
        .collect()
    )

    assert_frame_equal(lowercase_out, casefold_out)


def test_strip_company_suffix_casefold_diverges_from_lowercase_without_transliteration():
    # Unlike `cleanse_lazyframe()` above, `strip_company_suffix()` runs the profile
    # exactly as given -- no forced `transliterate` -- so the two case operations do
    # diverge here, the way normalize.py's module docstring documents.
    assert strip_company_suffix("Straße Consulting GmbH") == "straße"
    assert (
        strip_company_suffix(
            "Straße Consulting GmbH", normalization_profile="default|casefold"
        )
        == "strasse"
    )


def test_resolve_company_type_rules_prefers_explicit_config_values():
    cfg = CleanseConfig(company_type_regex="CUSTOM", company_type_mapping={"x": "y"})
    regex, mapping = _resolve_company_type_rules(cfg)
    assert regex == "CUSTOM"
    assert mapping == {"x": "y"}


def test_cleanse_lazyframe_accepts_file_columns_as_set_and_sequence():
    source = pl.DataFrame(
        {"company_name": ["Acme Ltd"], "company_type_src": ["Limited"]}
    )

    actual_set = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(
                source_company_type_col="company_type_src",
                source_company_type_mapping={"Limited": "Ltd"},
            ),
            file_columns={"company_name", "company_type_src"},
        )
        .select(["company_type"])
        .collect()
    )

    actual_seq = (
        cleanse_lazyframe(
            source.lazy(),
            config=CleanseConfig(
                source_company_type_col="company_type_src",
                source_company_type_mapping={"Limited": "Ltd"},
            ),
            file_columns=["company_name", "company_type_src"],
        )
        .select(["company_type"])
        .collect()
    )

    assert actual_set["company_type"].to_list() == ["ltd"]
    assert actual_seq["company_type"].to_list() == ["ltd"]


def test_load_noise_words_override_supports_list_and_rejects_invalid_payload(tmp_path):
    list_path = tmp_path / "tokens_list.json"
    list_path.write_text(json.dumps(["systems", "holdings"]), encoding="utf-8")
    assert _load_noise_words_override(list_path) == ("systems", "holdings")

    invalid_path = tmp_path / "tokens_invalid.json"
    invalid_path.write_text(json.dumps("bad"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list or an object"):
        _load_noise_words_override(invalid_path)


def test_strip_company_suffix_returns_none_for_empty_normalized_name():
    assert strip_company_suffix(None) is None
    assert strip_company_suffix("   ") is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ('"Acme" Ltd', "acme"),
        ("'Acme' Ltd", "acme"),
        ('"ABC" Ltd (Associated Business Consultants)', "abc"),
        ('"Acme"', None),  # bare quote, nothing after -- not a leading-quote match
        ("Acme Ltd", None),  # no leading quote at all
        ("", None),
        (None, None),
    ],
)
def test_extract_quoted_name_cases(value, expected):
    assert extract_quoted_name(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        '"Acme" Ltd',
        '"ABC" Ltd (Associated Business Consultants)',
        "Acme Ltd",
    ],
)
def test_extract_quoted_name_matches_cleanse_lazyframe_quoted_name_column(value):
    # Single-string result must agree with the batch quoted_name column for the
    # same input -- this is the whole point of exposing it as its own function.
    batch_quoted_name = (
        cleanse_lazyframe(pl.DataFrame({"company_name": [value]}).lazy())
        .select("quoted_name")
        .collect()["quoted_name"][0]
    )
    assert extract_quoted_name(value) == batch_quoted_name
