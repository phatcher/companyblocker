import pytest
from company_cleanse.short_name_candidates import (
    DEFAULT_LAYER_ORDER,
    LAYERS,
    ShortNameCandidate,
    derive_short_name_candidate,
    pattern_exact_initials,
    pattern_prefix_truncation,
    pattern_subsequence_initials,
    pattern_type_prefix_acronym_qualifier,
    tokenize_name,
)


def test_tokenize_name_splits_and_drops_empties():
    assert tokenize_name("  International   Business Machines ") == [
        "International",
        "Business",
        "Machines",
    ]
    assert tokenize_name(None) == []
    assert tokenize_name("") == []


class TestPatternExactInitials:
    def test_multi_token_name(self):
        assert (
            pattern_exact_initials(["International", "Business", "Machines"]) == "IBM"
        )

    def test_single_token_returns_none(self):
        assert pattern_exact_initials(["Enedis"]) is None

    def test_requires_alnum_start_in_every_token(self):
        assert pattern_exact_initials(["Ben", "&", "Jerry's"]) is None


class TestPatternSubsequenceInitials:
    def test_skips_stopword(self):
        # "and" is skipped, giving a distinct 4-letter candidate from a
        # 5-token exact-initials computation.
        tokens = ["National", "Aeronautics", "and", "Space", "Administration"]
        assert pattern_exact_initials(tokens) == "NAASA"
        assert pattern_subsequence_initials(tokens) == "NASA"

    def test_returns_none_when_identical_to_exact_initials(self):
        tokens = ["International", "Business", "Machines"]
        assert pattern_subsequence_initials(tokens) is None

    def test_returns_none_below_two_content_tokens(self):
        assert pattern_subsequence_initials(["The", "Firm"]) is None


class TestPatternPrefixTruncation:
    def test_drops_trailing_company_type(self):
        assert pattern_prefix_truncation(["Deutsche", "Bahn", "AG"]) == "Deutsche Bahn"

    def test_drops_trailing_numeral(self):
        assert pattern_prefix_truncation(["PSV", "Meiningen", "90"]) == "PSV Meiningen"

    def test_returns_none_when_nothing_droppable(self):
        assert pattern_prefix_truncation(["Air", "France"]) is None

    def test_returns_none_when_everything_would_be_dropped(self):
        assert pattern_prefix_truncation(["AG"]) is None


class TestPatternTypePrefixAcronymQualifier:
    def test_hyphen_split_without_decompounding(self):
        tokens = ["Schwimm-Startgemeinschaft", "Leipzig"]
        assert pattern_type_prefix_acronym_qualifier(tokens) == "SS Leipzig"

    def test_hyphen_split_with_decompounding(self):
        tokens = ["Schwimm-Startgemeinschaft", "Leipzig"]

        def decompound(part: str) -> list[str]:
            if part.lower() == "startgemeinschaft":
                return ["Start", "Gemeinschaft"]
            return [part]

        assert (
            pattern_type_prefix_acronym_qualifier(tokens, decompound_fn=decompound)
            == "SSG Leipzig"
        )

    def test_drops_trailing_company_type_from_qualifier(self):
        tokens = ["Schwimm-Startgemeinschaft", "Leipzig", "eV"]
        assert pattern_type_prefix_acronym_qualifier(tokens) == "SS Leipzig"

    def test_no_hyphen_and_no_decompound_fn_returns_none(self):
        assert pattern_type_prefix_acronym_qualifier(["Deutsche", "Bahn"]) is None

    def test_single_token_returns_none(self):
        assert (
            pattern_type_prefix_acronym_qualifier(["Schwimm-Startgemeinschaft"]) is None
        )

    def test_no_remaining_qualifier_returns_bare_acronym(self):
        assert pattern_type_prefix_acronym_qualifier(["Schwimm-Start", "eV"]) == "SS"


# Golden table -- exact expected `ShortNameCandidate` (or `None`) for a name, recorded as a
# literal below, so an edge case a layer or ordering change flips has a case that fails.
# Each row is (name, enabled_layers, expected).
DERIVE_SHORT_NAME_GOLDEN_CASES = [
    # A name that reduces to nothing.
    (None, DEFAULT_LAYER_ORDER, None),
    ("", DEFAULT_LAYER_ORDER, None),
    # Single token, no structural pattern to exploit.
    ("Enedis", DEFAULT_LAYER_ORDER, None),
    # A name that is only a legal form.
    ("GmbH", DEFAULT_LAYER_ORDER, None),
    # One row per pattern layer.
    (
        "National Aeronautics and Space Administration",
        DEFAULT_LAYER_ORDER,
        ShortNameCandidate(value="NASA", pattern="subsequence_initials"),
    ),
    (
        "International Business Machines",
        DEFAULT_LAYER_ORDER,
        ShortNameCandidate(value="IBM", pattern="exact_initials"),
    ),
    (
        # exact_initials would fire first under the default order; restricting
        # enabled_layers lets prefix_truncation answer instead.
        "Deutsche Bahn AG",
        ("prefix_truncation", "exact_initials"),
        ShortNameCandidate(value="Deutsche Bahn", pattern="prefix_truncation"),
    ),
    (
        "Schwimm-Startgemeinschaft Leipzig",
        ("type_prefix_acronym_qualifier",),
        ShortNameCandidate(value="SS Leipzig", pattern="type_prefix_acronym_qualifier"),
    ),
    # A quoted brand that is a single pre-formed token.
    (
        '"DW" Sp. z o.o.',
        DEFAULT_LAYER_ORDER,
        ShortNameCandidate(value="DO", pattern="subsequence_initials"),
    ),
]


@pytest.mark.parametrize("name,enabled_layers,expected", DERIVE_SHORT_NAME_GOLDEN_CASES)
def test_derive_short_name_candidate_golden(name, enabled_layers, expected):
    assert derive_short_name_candidate(name, enabled_layers=enabled_layers) == expected


class TestDeriveShortNameCandidate:
    def test_unknown_layer_name_raises(self):
        with pytest.raises(ValueError):
            derive_short_name_candidate("Air France", enabled_layers=("bogus_layer",))

    def test_decompound_fn_reaches_type_prefix_layer(self):
        def decompound(part: str) -> list[str]:
            if part.lower() == "startgemeinschaft":
                return ["Start", "Gemeinschaft"]
            return [part]

        result = derive_short_name_candidate(
            "Schwimm-Startgemeinschaft Leipzig",
            enabled_layers=("type_prefix_acronym_qualifier",),
            decompound_fn=decompound,
        )
        assert result == ShortNameCandidate(
            value="SSG Leipzig", pattern="type_prefix_acronym_qualifier"
        )


def test_default_layer_order_covers_every_registered_layer():
    assert set(DEFAULT_LAYER_ORDER) == set(LAYERS)
