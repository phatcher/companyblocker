"""Direct tests for `company_perturbation.generation`'s scenario draw.

Two functions, asserted separately because they answer different questions:
`applicable_scenarios` says which of a profile's scenarios a record is eligible
for, and `select_scenario` draws exactly one of them by weight.

`select_scenario` is called with an explicit generator throughout. Its fixed
draw count is a property the rest of the walk depends on -- a record with no
applicable scenario has to leave the stream where any other record would -- and
nothing else would notice it breaking.
"""

from __future__ import annotations

from random import Random

from company_perturbation.chain import ChainStep
from company_perturbation.generation import applicable_scenarios, select_scenario
from company_perturbation.profile_schema import ExclusionRules, Scenario

TYPO = ChainStep("typo.keyboard_substitution", 1.0, 1)
SUFFIX = ChainStep("legal_suffix.drop", 1.0, 1)


def _ids(scenarios):
    return [scenario.scenario_id for scenario in scenarios]


# --- drawing one -----------------------------------------------------------


def test_the_only_scenario_is_always_the_one_drawn(make_scenario):
    only = make_scenario("only")

    assert select_scenario((only,), rng=Random(3)) is only


def test_no_scenarios_draws_nothing():
    assert select_scenario((), rng=Random(3)) is None


def test_a_zero_weight_scenario_is_never_drawn(make_scenario):
    never = make_scenario("never", weight=0.0)
    always = make_scenario("always", weight=1.0)

    drawn = {select_scenario((never, always), rng=Random(seed)) for seed in range(50)}

    assert drawn == {always}


def test_weights_that_are_all_zero_draw_nothing(make_scenario):
    """Not a division by zero, and not an arbitrary pick: nothing was asked for."""
    scenarios = (make_scenario("a", weight=0.0), make_scenario("b", weight=0.0))

    assert select_scenario(scenarios, rng=Random(3)) is None


def test_weights_are_relative_shares_rather_than_probabilities(make_scenario):
    """1 and 9 give the same split as 0.1 and 0.9: they are normalised, not read."""
    rare = make_scenario("rare", weight=1.0)
    common = make_scenario("common", weight=9.0)

    drawn = [select_scenario((rare, common), rng=Random(seed)) for seed in range(400)]

    assert 0.8 < sum(scenario is common for scenario in drawn) / 400 < 1.0


def test_a_draw_consumes_exactly_one_value_when_it_finds_a_scenario(make_scenario):
    generator, untouched = Random(4), Random(4)

    select_scenario((make_scenario("one"),), rng=generator)
    untouched.random()

    assert generator.random() == untouched.random()


def test_a_draw_consumes_exactly_one_value_when_there_is_nothing_to_draw():
    """So a record no scenario applies to advances the stream like any other."""
    empty, populated = Random(4), Random(4)

    select_scenario((), rng=empty)
    select_scenario((Scenario(scenario_id="one", chain=(TYPO,)),), rng=populated)

    assert empty.random() == populated.random()


# --- which scenarios a record is eligible for ------------------------------


def test_every_scenario_applies_when_nothing_excludes_the_record(
    make_record, make_profile, make_scenario
):
    profile = make_profile(make_scenario("a"), make_scenario("b"))

    assert _ids(applicable_scenarios(profile, make_record())) == ["a", "b"]


def test_a_scenario_excluding_the_records_system_does_not_apply(
    make_record, make_profile, make_scenario
):
    profile = make_profile(
        make_scenario("not-gb", exclusions=ExclusionRules(exclude_systems=("gb",))),
        make_scenario("anywhere"),
    )

    assert _ids(applicable_scenarios(profile, make_record())) == ["anywhere"]


def test_a_scenario_excluding_the_records_country_does_not_apply(
    make_record, make_profile, make_scenario
):
    profile = make_profile(
        make_scenario("not-gb", exclusions=ExclusionRules(exclude_countries=("gb",))),
        make_scenario("anywhere"),
    )

    assert _ids(applicable_scenarios(profile, make_record())) == ["anywhere"]


def test_a_record_with_no_country_is_not_excluded_by_a_country_rule(
    make_record, make_profile, make_scenario
):
    """An unrecorded country is not evidence that the exclusion applies."""
    profile = make_profile(
        make_scenario("not-gb", exclusions=ExclusionRules(exclude_countries=("gb",)))
    )

    record = make_record(country=None)

    assert _ids(applicable_scenarios(profile, record)) == ["not-gb"]


def test_a_scenario_naming_an_excluded_family_does_not_apply(
    make_record, make_profile, make_scenario
):
    profile = make_profile(
        make_scenario("typos", exclusions=ExclusionRules(exclude_families=("typo",))),
        make_scenario("suffixes", steps=(SUFFIX,)),
    )

    assert _ids(applicable_scenarios(profile, make_record())) == ["suffixes"]


def test_one_excluded_family_anywhere_in_a_chain_disqualifies_the_whole_scenario(
    make_record, make_profile, make_scenario
):
    """A chain is one history; it cannot be run with a step taken out of it."""
    profile = make_profile(
        make_scenario(
            "mixed",
            steps=(SUFFIX, TYPO),
            exclusions=ExclusionRules(exclude_families=("typo",)),
        )
    )

    assert applicable_scenarios(profile, make_record()) == ()


def test_a_profiles_exclusions_apply_to_a_scenario_that_declares_none(
    make_record, make_profile, make_scenario
):
    profile = make_profile(
        make_scenario("inherits"),
        exclusions=ExclusionRules(exclude_systems=("gb",)),
    )

    assert applicable_scenarios(profile, make_record()) == ()


def test_a_scenarios_own_exclusions_replace_the_profiles_rather_than_add_to_them(
    make_record, make_profile, make_scenario
):
    """Declaring an empty rule set is how a scenario opts out of the profile's."""
    profile = make_profile(
        make_scenario("opted-out", exclusions=ExclusionRules()),
        make_scenario("inherits"),
        exclusions=ExclusionRules(exclude_systems=("gb",)),
    )

    assert _ids(applicable_scenarios(profile, make_record())) == ["opted-out"]
