"""Direct tests for `company_perturbation.profile_schema`: parsing and shape.

What an authored file may leave out is the contract here, so each default is
asserted on its own: an omitted step number, an omitted weight, and the
difference between a scenario that declares no exclusions (it inherits) and one
that declares empty ones (it opts out). The rules a profile is rejected for are
`test_profile_schema_validation.py`.
"""

from __future__ import annotations

from company_perturbation.chain import ChainStep
from company_perturbation.profile_schema import (
    ExclusionRules,
    PerturbationProfile,
    Scenario,
    effective_exclusions,
    parse_profile,
    serialize_profile,
)

MINIMAL = {
    "profile_id": "light-noise",
    "scenarios": [
        {
            "scenario_id": "one-typo",
            "chain": [{"operator_id": "typo.keyboard_substitution"}],
        }
    ],
}


def _parse(**overrides):
    return parse_profile({**MINIMAL, **overrides})


def _first_step(profile: PerturbationProfile) -> ChainStep:
    return profile.scenarios[0].chain[0]


# --- what an authored file may leave out -----------------------------------


def test_a_step_that_omits_its_probability_always_fires():
    assert _first_step(_parse()).probability == 1.0


def test_a_step_that_omits_its_attempt_count_tries_once():
    assert _first_step(_parse()).attempts == 1


def test_a_scenario_that_omits_its_weight_takes_an_equal_share():
    assert _parse().scenarios[0].weight == 1.0


def test_a_profile_that_omits_its_exclusions_excludes_nothing():
    assert _parse().exclusions == ExclusionRules()


def test_a_scenario_that_declares_no_exclusions_inherits_rather_than_opts_out():
    """`None` and empty rules are different answers, and only `None` inherits."""
    assert _parse().scenarios[0].exclusions is None


def test_a_scenario_declaring_empty_exclusions_opts_out_of_the_profiles():
    profile = _parse(
        scenarios=[{**MINIMAL["scenarios"][0], "exclusions": {}}],
    )

    assert profile.scenarios[0].exclusions == ExclusionRules()


def test_an_exclusion_rule_that_names_no_axis_excludes_nothing_on_that_axis():
    profile = _parse(exclusions={"exclude_systems": ["gb"]})

    assert profile.exclusions == ExclusionRules(exclude_systems=("gb",))


def test_a_description_is_optional():
    assert _parse().description is None


def test_a_profile_that_names_no_seed_blesses_no_sample():
    """`None` rather than a literal: no author has said which sample is standard."""
    assert _parse().default_seed is None


def test_a_profile_can_name_the_sample_a_caller_gets_by_default():
    assert _parse(default_seed=20260910).default_seed == 20260910


# --- what it says ----------------------------------------------------------


def test_a_profile_is_parsed_under_the_id_it_declares():
    profile = _parse()

    assert profile.profile_id == "light-noise"


def test_a_chain_keeps_the_order_it_was_authored_in():
    """Order is the author's and it decides whether a step finds anything."""
    profile = _parse(
        scenarios=[
            {
                "scenario_id": "recognise-then-corrupt",
                "chain": [
                    {"operator_id": "legal_suffix.variant_substitution"},
                    {"operator_id": "typo.keyboard_substitution"},
                ],
            }
        ]
    )

    assert [step.operator_id for step in profile.scenarios[0].chain] == [
        "legal_suffix.variant_substitution",
        "typo.keyboard_substitution",
    ]


# --- serializing -----------------------------------------------------------


def test_a_parsed_profile_serializes_back_to_the_same_profile():
    profile = _parse(
        description="light noise",
        default_seed=20260910,
        exclusions={"exclude_countries": ["ie"]},
        scenarios=[
            {
                "scenario_id": "one-typo",
                "weight": 2.5,
                "exclusions": {"exclude_families": ["phonetic"]},
                "chain": [
                    {
                        "operator_id": "typo.keyboard_substitution",
                        "probability": 0.4,
                        "attempts": 3,
                    }
                ],
            }
        ],
    )

    assert parse_profile(serialize_profile(profile)) == profile


def test_a_scenario_that_inherits_exclusions_serializes_without_them():
    """Writing `{}` back would silently convert an inheriting scenario into an opted-out one."""
    serialized = serialize_profile(_parse())

    assert "exclusions" not in serialized["scenarios"][0]


# --- which rules a scenario is judged by -----------------------------------


def test_a_scenario_with_no_rules_of_its_own_is_judged_by_the_profiles():
    profile = PerturbationProfile(
        profile_id="p",
        scenarios=(),
        exclusions=ExclusionRules(exclude_systems=("gb",)),
    )
    scenario = Scenario(scenario_id="s", chain=())

    assert effective_exclusions(scenario, profile=profile) == profile.exclusions


def test_a_scenarios_own_rules_replace_the_profiles_entirely():
    """Never merged: a scenario naming one axis does not inherit the others."""
    profile = PerturbationProfile(
        profile_id="p",
        scenarios=(),
        exclusions=ExclusionRules(exclude_systems=("gb",), exclude_countries=("ie",)),
    )
    scenario = Scenario(
        scenario_id="s", chain=(), exclusions=ExclusionRules(exclude_countries=("fr",))
    )

    assert effective_exclusions(scenario, profile=profile) == ExclusionRules(
        exclude_countries=("fr",)
    )
