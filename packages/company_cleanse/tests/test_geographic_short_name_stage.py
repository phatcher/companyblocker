import polars as pl
import pytest
from company_cleanse import CleanseConfig, cleanse_lazyframe, strip_company_suffix
from company_cleanse.extract import (
    _resolve_short_name_stage_flags,
    _split_short_name_stage_flags,
)


def _short_names(names, config):
    frame = cleanse_lazyframe(pl.LazyFrame({"company_name": names}), config).collect()
    return dict(zip(frame["company_name"], frame["short_name"]))


def test_the_stage_is_off_unless_the_profile_names_it():
    flags = _split_short_name_stage_flags("default")

    assert flags.geographic_tiers is None
    assert flags.include_company_type
    assert flags.include_noise_words
    assert flags.remaining_profile == "default"


@pytest.mark.parametrize("spelling", ["geographic", "geographic_terms"])
def test_either_spelling_selects_the_stage_at_the_country_tier(spelling):
    flags = _split_short_name_stage_flags(f"default|{spelling}")

    assert flags.geographic_tiers == ("country",)
    # The stage is consumed, not forwarded to text normalization.
    assert flags.remaining_profile == "default"


def test_tier_modifiers_add_to_the_country_tier():
    assert _split_short_name_stage_flags(
        "default|geographic|+region"
    ).geographic_tiers == (
        "country",
        "region",
    )
    assert _split_short_name_stage_flags(
        "default|geographic|+region|+city"
    ).geographic_tiers == ("country", "region", "city")


def test_the_stage_composes_with_the_two_existing_toggles():
    flags = _split_short_name_stage_flags("default|-noise_words|geographic|-lowercase")

    assert flags.include_noise_words is False
    assert flags.include_company_type is True
    assert flags.geographic_tiers == ("country",)
    assert flags.remaining_profile == "default|-lowercase"


@pytest.mark.parametrize(
    "profile",
    ["default|+region", "default|-noise_words|+city", "default|+"],
)
def test_a_tier_modifier_without_the_stage_is_rejected(profile):
    with pytest.raises(ValueError):
        _split_short_name_stage_flags(profile)


def test_short_name_profile_accepts_the_stage_but_not_a_text_operation():
    assert _resolve_short_name_stage_flags(
        "default|geographic|+region"
    ).geographic_tiers == ("country", "region")

    with pytest.raises(ValueError, match="short_name_profile only supports"):
        _resolve_short_name_stage_flags("default|lowercase")


@pytest.mark.parametrize(
    ("name", "company_type", "default_short_name", "geographic_short_name"),
    [
        # The headline case: the legal form comes off first, then the term behind it.
        ("Oracle Ireland Ltd", "Ltd", "oracle ireland", "oracle"),
        ("Microsoft Deutschland GmbH", "GmbH", "microsoft deutschland", "microsoft"),
        # Integral terms survive with the stage on, exactly as without it.
        ("Bank of Ireland Plc", "Plc", "bank of ireland", "bank of ireland"),
        ("China Mobile", None, "china mobile", "china mobile"),
    ],
)
def test_strip_company_suffix_honors_the_stage(
    name, company_type, default_short_name, geographic_short_name
):
    assert (
        strip_company_suffix(
            name,
            company_type=company_type,
            noise_words=(),
            normalization_profile="default",
        )
        == default_short_name
    )
    assert (
        strip_company_suffix(
            name,
            company_type=company_type,
            noise_words=(),
            normalization_profile="default|geographic",
        )
        == geographic_short_name
    )


def test_a_row_with_no_legal_form_keeps_two_tokens():
    # `Air France` never had a company-type token trimmed, so the stage refuses to
    # reduce it to one token even though `France` is a country term.
    assert (
        strip_company_suffix(
            "Air France", noise_words=(), normalization_profile="default|geographic"
        )
        == "air france"
    )


def test_the_region_tier_only_applies_when_asked_for():
    assert (
        strip_company_suffix(
            "Acme Systems Alberta Ltd",
            company_type="Ltd",
            noise_words=(),
            normalization_profile="default|geographic",
        )
        == "acme systems alberta"
    )
    assert (
        strip_company_suffix(
            "Acme Systems Alberta Ltd",
            company_type="Ltd",
            noise_words=(),
            normalization_profile="default|geographic|+region",
        )
        == "acme systems"
    )


def test_the_city_tier_is_accepted_and_strips_nothing():
    assert (
        strip_company_suffix(
            "Acme Systems London Ltd",
            company_type="Ltd",
            noise_words=(),
            normalization_profile="default|geographic|+city",
        )
        == "acme systems london"
    )


def test_batch_short_name_profile_matches_the_standalone_path():
    names = [
        "Oracle Ireland Ltd",
        "Bank of Ireland Plc",
        "Acme Systems Alberta Limited",
        "China Mobile",
    ]

    default_names = _short_names(names, CleanseConfig())
    geographic_names = _short_names(
        names, CleanseConfig(short_name_profile="default|geographic")
    )
    region_names = _short_names(
        names, CleanseConfig(short_name_profile="default|geographic|+region")
    )

    assert default_names["Oracle Ireland Ltd"] == "oracle ireland"
    assert geographic_names["Oracle Ireland Ltd"] == "oracle"

    assert geographic_names["Acme Systems Alberta Limited"] == "acme systems alberta"
    assert region_names["Acme Systems Alberta Limited"] == "acme"

    for resolved in (default_names, geographic_names, region_names):
        assert resolved["Bank of Ireland Plc"] == "bank of ireland"
        assert resolved["China Mobile"] == "china mobile"


def test_the_two_profiles_select_the_stage_independently():
    names = ["Oracle Ireland Ltd", "Siemens (UK) Ltd"]

    cleanse_only = cleanse_lazyframe(
        pl.LazyFrame({"company_name": names}),
        CleanseConfig(normalization_profile="default|geographic"),
    ).collect()
    short_name_only = cleanse_lazyframe(
        pl.LazyFrame({"company_name": names}),
        CleanseConfig(short_name_profile="default|geographic"),
    ).collect()

    # The normalization profile reaches `name_cleansed`; the bracketed arm is the one
    # that fires there, because the legal form still holds the trailing position.
    assert cleanse_only["name_cleansed"].to_list() == [
        "oracle ireland ltd",
        "siemens ltd",
    ]
    assert cleanse_only["short_name"].to_list() == ["oracle ireland", "siemens"]

    # The short-name profile reaches `short_name` only, and gets the trailing arm.
    assert short_name_only["name_cleansed"].to_list() == [
        "oracle ireland ltd",
        "siemens uk ltd",
    ]
    assert short_name_only["short_name"].to_list() == ["oracle", "siemens"]


def test_batch_rejects_a_tier_without_the_stage():
    with pytest.raises(ValueError, match="without selecting"):
        cleanse_lazyframe(
            pl.LazyFrame({"company_name": ["Oracle Ireland Ltd"]}),
            CleanseConfig(short_name_profile="default|+region"),
        ).collect()
