import pytest
from company_cleanse import normalize
from company_cleanse.normalize import (
    collapse_single_char_sequences,
    get_ascii_homoglyphs,
    normalize_company_type_value,
    normalize_suffix_surface,
    normalize_tokens,
    parse_normalization_profile,
    strip_diacritics,
    transliterate_to_ascii,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Société à responsabilité limitée", "Societe a responsabilite limitee"),
        ("", ""),
        (None, None),
        ("ÇĞÖŞÜ çğöşü", "CGOSU cgosu"),
    ],
)
def test_strip_diacritics_cases(value, expected):
    assert strip_diacritics(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("İstanbul, İzmir, ışık", "Istanbul, Izmir, isik"),
        ("", ""),
        (None, None),
        # Russian: full modern alphabet, including Ы/ы and Э/э which are distinct letters
        # (not decomposable diacritic forms) and previously had no table entry.
        ("Ындыр Электростройсервис", "Yndyr Elektrostroiservis"),
        ("экспресс", "ekspress"),
        # Russian pre-1918 orthography (yat/fita/izhitsa) is out of scope: no jurisdiction
        # in entity_legal_forms_iso20275.json uses it, so it's left untransliterated.
        ("Ѣ", "Ѣ"),
        # Serbian/Macedonian: letters unique to those national Cyrillic alphabets.
        (
            "Ђорђевић Јовановић Љубљана Његош Ћирилица Џез Ѕвезда",
            "Dordevic Jovanovic LJubljana NJegosh Cirilitsa DZHez DZvezda",
        ),
        # Ukrainian-specific Cyrillic (Є/І/Ї/Ґ) is out of scope: no Ukraine entries in
        # entity_legal_forms_iso20275.json, so it's left untransliterated rather than guessed.
        ("Ґ", "Ґ"),
        # Maltese: Ħ has no NFKD decomposition, unlike ċ/à which strip_diacritics handles.
        ("Ħal Qormi", "Hal Qormi"),
        # Coptic letters sharing the Greek Unicode block are out of scope: no Coptic-script
        # data anywhere in this repo, so they're left untransliterated rather than guessed.
        ("Ϣ", "Ϣ"),
    ],
)
def test_transliterate_to_ascii_cases(value, expected):
    assert transliterate_to_ascii(value) == expected


def test_get_ascii_homoglyphs_includes_expected_cross_script_candidates():
    homoglyphs = get_ascii_homoglyphs()
    # Cyrillic 'а' (U+0430) and Greek 'α' (U+03B1) both fold to ASCII "a".
    assert "а" in homoglyphs["a"]
    assert "α" in homoglyphs["a"]
    # Cyrillic 'Ј' (U+0408, added for Serbian/Macedonian coverage) folds to ASCII "J".
    assert "Ј" in homoglyphs["J"]


def test_get_ascii_homoglyphs_excludes_digraph_and_empty_folds():
    homoglyphs = get_ascii_homoglyphs()
    all_sources = {source for sources in homoglyphs.values() for source in sources}
    # Ж/ж fold to the digraph "ZH", not a single ASCII letter -- no single-char target to
    # report them under, even though they're valid transliterate_to_ascii() inputs.
    assert "Ж" not in all_sources
    assert "ж" not in all_sources
    # Ь/ь fold to "" (silent) -- nothing to report as a homoglyph either.
    assert "Ь" not in all_sources
    assert "ь" not in all_sources


def test_get_ascii_homoglyphs_every_candidate_round_trips_through_transliterate():
    homoglyphs = get_ascii_homoglyphs()
    for ascii_char, source_chars in homoglyphs.items():
        for source_char in source_chars:
            assert transliterate_to_ascii(source_char) == ascii_char


def test_get_ascii_homoglyphs_is_deterministic_across_calls():
    first = get_ascii_homoglyphs()
    second = get_ascii_homoglyphs()
    assert first == second
    for source_chars in first.values():
        assert source_chars == tuple(sorted(source_chars))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("S R L", "SRL"),
        ("P & S", "P&S"),
        ("A & B & C", "A&B&C"),
        ("A J Bell", "AJ Bell"),
        ("A Bell C D", "A Bell CD"),
        ("A T & T", "AT & T"),
        ("A T & T & Co", "AT & T & Co"),
        ("AT&T", "AT & T"),
        ("&", "&"),
        (" & ", " & "),
        ("", ""),
        (None, None),
    ],
)
def test_collapse_single_char_sequences_cases(value, expected):
    assert collapse_single_char_sequences(value) == expected


@pytest.mark.parametrize("value", ["AT&T", "AT & T", "A T & T"])
def test_collapse_single_char_sequences_ampersand_spacing_is_standalone_consistent(
    value,
):
    # No dependency on the `and` operation having run first.
    assert collapse_single_char_sequences(value) == "AT & T"


@pytest.mark.parametrize(
    "value,and_tokens,normalization_profile,expected",
    [
        ("Hello.World", None, "default", "hello world"),
        ("Hello!World", None, "default", "hello world"),
        ("Hello!World", None, "default|-punctuation|punctuation_bang", "hello!world"),
        ("A & B", ["UND"], "default", "a&b"),
        ("A & B", None, "default", "a&b"),
        ("P and S", ["and"], "default", "p&s"),
        ("S R L", None, "default", "srl"),
        ("A J Bell", None, "default", "aj bell"),
        ("A_B", None, "default", "ab"),
        ("Hello   World", None, "default", "hello world"),
        ("", None, "default", ""),
        (None, None, "default", None),
    ],
)
def test_normalize_tokens_cases(value, and_tokens, normalization_profile, expected):
    assert (
        normalize_tokens(
            value, and_tokens=and_tokens, normalization_profile=normalization_profile
        )
        == expected
    )


@pytest.mark.parametrize(
    "value,and_tokens,expected",
    [
        ("P & S", None, "p&s"),
        ("P and S", ["and"], "p&s"),
        ("Alpha & B", None, "alpha & b"),
        ("A & Beta", None, "a & beta"),
        ("A & B", None, "a&b"),
        ("A & B & C", None, "a&b&c"),
        ("ABC & CO", None, "abc & co"),
        ("AT&T", None, "at & t"),
        ("AT & T", None, "at & t"),
        ("A T & T", None, "at & t"),
    ],
)
def test_normalize_tokens_single_char_ampersand_policy_matrix(
    value, and_tokens, expected
):
    assert normalize_tokens(value, and_tokens=and_tokens) == expected


@pytest.mark.parametrize(
    "value,and_tokens,normalization_profile,expected",
    [
        ("Société_Anónima & Co., Ltd!", ["AND"], "default", "societe anonima & co ltd"),
        (
            "Société_Anónima & Co., Ltd!",
            ["AND"],
            "default|transliterate|-punctuation|punctuation_bang|singlespace",
            "societe anonima & co ltd!",
        ),
        ("A & B", ["UND"], "default", "a&b"),
        ("P and S", ["AND"], "default", "p&s"),
        ("", ["AND"], "default", ""),
        (None, ["AND"], "default", None),
    ],
)
def test_normalize_suffix_surface_cases(
    value, and_tokens, normalization_profile, expected
):
    assert (
        normalize_suffix_surface(
            value, and_tokens=and_tokens, normalization_profile=normalization_profile
        )
        == expected
    )


@pytest.mark.parametrize(
    "value,transliterate,expected",
    [
        ("Société limitée", False, "societe limitee"),
        ("Α Β Γ", True, "abg"),
        ("", False, ""),
        (None, False, ""),
    ],
)
def test_normalize_company_type_value_cases(value, transliterate, expected):
    assert normalize_company_type_value(value, transliterate=transliterate) == expected


def test_normalize_submodule_exposes_public_helpers():
    assert normalize.strip_diacritics("Ç") == "C"
    assert normalize.transliterate_to_ascii("İ") == "I"


@pytest.mark.parametrize(
    "profile,expected",
    [
        (
            "default",
            (
                "punctuation",
                "singlespace",
                "and",
                "singlechar",
                "diacritics",
                "lowercase",
            ),
        ),
        (
            "default|-lowercase",
            ("punctuation", "singlespace", "and", "singlechar", "diacritics"),
        ),
        (
            "default|transliterate",
            (
                "punctuation",
                "singlespace",
                "and",
                "singlechar",
                "diacritics",
                "transliterate",
                "lowercase",
            ),
        ),
        (
            "default|-lowercase|transliterate",
            (
                "punctuation",
                "singlespace",
                "and",
                "singlechar",
                "diacritics",
                "transliterate",
            ),
        ),
        (
            "punctuation|singlespace|and|diacritics|transliterate|singlechar",
            (
                "punctuation",
                "singlespace",
                "and",
                "diacritics",
                "transliterate",
                "singlechar",
            ),
        ),
    ],
)
def test_parse_normalization_profile_cases(profile, expected):
    assert parse_normalization_profile(profile) == expected


def test_parse_normalization_profile_rejects_invalid_remove_only_chain():
    with pytest.raises(ValueError, match="remove modifiers"):
        parse_normalization_profile("-lowercase")


def test_normalize_tokens_supports_default_remove_modifier():
    assert normalize_tokens("Acme-LTD", normalization_profile="default") == "acme ltd"
    assert (
        normalize_tokens("Acme-LTD", normalization_profile="default|-lowercase")
        == "Acme LTD"
    )


def test_normalize_tokens_supports_explicit_chain_with_transliteration():
    assert (
        normalize_tokens(
            "EØFG",
            normalization_profile="punctuation|and|diacritics|transliterate|lowercase|singlechar",
        )
        == "eofg"
    )


def test_normalize_tokens_and_operation_is_explicit():
    assert normalize_tokens("A and B", normalization_profile="default") == "a&b"
    assert (
        normalize_tokens("A and B", normalization_profile="default|-and") == "a and b"
    )


def test_normalize_tokens_singlespace_operation_is_explicit():
    assert (
        normalize_tokens("Alpha   Beta", normalization_profile="default")
        == "alpha beta"
    )
    assert (
        normalize_tokens("Alpha   Beta", normalization_profile="default|-singlespace")
        == "alpha   beta"
    )


def test_normalize_tokens_diacritics_operation_is_explicit():
    assert normalize_tokens("Société", normalization_profile="default") == "societe"
    assert (
        normalize_tokens("Société", normalization_profile="default|-diacritics")
        == "société"
    )


@pytest.mark.parametrize(
    "value,expected_lower,expected_casefold",
    [
        # German sharp s: `.lower()` is a no-op, `.casefold()` expands to "ss" --
        # one codepoint mapping to two, so any consumer assuming length preservation
        # across `normalize_tokens()` would break on this input (none currently do;
        # none did when the fold was added).
        ("Straße", "straße", "strasse"),
        # Greek final vs. medial sigma: `.lower()` keeps them distinct, `.casefold()`
        # unifies them -- the caseless-match behavior the fold exists to enable.
        ("ΣΣΣς", "σσσς", "σσσσ"),
    ],
)
def test_normalize_tokens_casefold_diverges_from_lowercase_on_real_register_text(
    value, expected_lower, expected_casefold
):
    assert (
        normalize_tokens(value, normalization_profile="default|-diacritics")
        == expected_lower
    )
    assert (
        normalize_tokens(value, normalization_profile="default|-diacritics|casefold")
        == expected_casefold
    )


def test_parse_normalization_profile_casefold_replaces_lowercase_in_default_chain():
    assert parse_normalization_profile("default|casefold") == (
        "punctuation",
        "singlespace",
        "and",
        "singlechar",
        "diacritics",
        "casefold",
    )


def test_parse_normalization_profile_lowercase_and_casefold_are_mutually_exclusive():
    # Selecting `lowercase` again after `casefold` displaces it back, the same way a
    # second `default|punctuation_bang|punctuation` displaces `punctuation_bang`.
    assert parse_normalization_profile("default|casefold|lowercase") == (
        "punctuation",
        "singlespace",
        "and",
        "singlechar",
        "diacritics",
        "lowercase",
    )
    # An explicit chain naming both keeps only the one selected last.
    assert parse_normalization_profile(
        "punctuation|singlespace|diacritics|lowercase|casefold"
    ) == (
        "punctuation",
        "singlespace",
        "diacritics",
        "casefold",
    )


def test_normalize_tokens_casefold_operation_is_explicit():
    assert (
        normalize_tokens("Straße", normalization_profile="default|-diacritics")
        == "straße"
    )
    assert (
        normalize_tokens("Straße", normalization_profile="default|-diacritics|casefold")
        == "strasse"
    )


def test_normalize_tokens_obeys_caller_specified_casefold_diacritics_order():
    # Explicit chains are honored in the order the caller gives, not silently
    # reordered around the `NFD(casefold(NFD(x)))` recipe -- callers that put
    # `casefold` ahead of `diacritics` get exactly that.
    assert (
        normalize_tokens(
            "Straße",
            normalization_profile="diacritics|casefold",
        )
        == normalize_tokens(
            "Straße",
            normalization_profile="casefold|diacritics",
        )
        == "strasse"
    )


def test_resolve_char_whitelist_for_operations_treats_casefold_like_lowercase():
    whitelist = r"[^a-z0-9\s]"
    assert (
        normalize.resolve_char_whitelist_for_profile(whitelist, "default|casefold")
        == whitelist
    )
    assert (
        normalize.resolve_char_whitelist_for_profile(whitelist, "default|-lowercase")
        == r"[^A-Za-z0-9\s]"
    )


def test_with_transliteration_operation_inserts_before_casefold():
    operations = parse_normalization_profile("default|casefold")
    resolved = normalize.with_transliteration_operation(operations)
    assert resolved == (
        "punctuation",
        "singlespace",
        "and",
        "singlechar",
        "diacritics",
        "transliterate",
        "casefold",
    )


def test_normalize_tokens_bang_preserving_punctuation_operation_is_explicit():
    assert (
        normalize_tokens(
            "Hello!World", normalization_profile="punctuation_bang|lowercase"
        )
        == "hello!world"
    )
    assert (
        normalize_tokens("Hello!World", normalization_profile="punctuation|lowercase")
        == "hello world"
    )


def test_normalize_tokens_obeys_operation_order():
    assert (
        normalize_tokens(
            "A-B",
            normalization_profile="punctuation|singlechar|lowercase",
        )
        == "ab"
    )
    assert (
        normalize_tokens(
            "A-B",
            normalization_profile="singlechar|punctuation|lowercase",
        )
        == "a b"
    )


def test_supported_operations_are_registered():
    assert normalize.SUPPORTED_NORMALIZATION_OPERATIONS == frozenset(
        normalize.NORMALIZATION_OPERATION_REGISTRY.keys()
    )


def test_normalize_tokens_with_operations_matches_profile_path():
    operations = parse_normalization_profile("default|-lowercase")
    assert normalize.normalize_tokens_with_operations(
        "Acme-LTD",
        operations=operations,
    ) == normalize_tokens("Acme-LTD", normalization_profile="default|-lowercase")


def test_normalize_suffix_surface_with_operations_matches_profile_path():
    operations = parse_normalization_profile("default|transliterate")
    assert normalize.normalize_suffix_surface_with_operations(
        "Société_Anónima & Co., Ltd!",
        operations=normalize.with_bang_preserving_punctuation_operation(
            (*operations, normalize.NORMALIZATION_OPERATION_SINGLESPACE)
        ),
        and_tokens=("AND",),
    ) == normalize_suffix_surface(
        "Société_Anónima & Co., Ltd!",
        and_tokens=("AND",),
        normalization_profile="default|transliterate|-punctuation|punctuation_bang|singlespace",
    )


def test_normalize_company_type_value_ignores_default_profile_mutation(monkeypatch):
    monkeypatch.setattr(
        normalize,
        "DEFAULT_NORMALIZATION_OPERATIONS",
        (
            normalize.NORMALIZATION_OPERATION_LOWERCASE,
            normalize.NORMALIZATION_OPERATION_TRANSLITERATE,
            normalize.NORMALIZATION_OPERATION_PUNCTUATION,
        ),
    )

    # Company-type normalization should remain independent from default profile wiring.
    assert normalize_company_type_value("ΑΕ", transliterate=False) == "αε"
