"""Direct tests for `company_perturbation.generation`: the row a record produces.

One record in, exactly one row out, whatever happened in between. That is the
guarantee the perturbed set is joined back to its source by, so the case worth
pinning hardest is the one a "skip what didn't change" implementation would get
wrong: a record whose chain landed nothing still emits.

Seeding is `test_generation_seeding.py`; which scenario is drawn and which apply
is `test_generation_scenarios.py`.
"""

from __future__ import annotations

from company_perturbation.chain import ChainStep
from company_perturbation.generation import perturb_record, perturb_records
from company_perturbation.profile_schema import ExclusionRules

LANDS_NOTHING = ChainStep("legal_suffix.drop", 1.0, 1)
"""Asked to act, and offered nowhere to do it: a name with no suffix to drop."""

SUFFIXLESS_NAME = "acme holdings"


def test_a_row_carries_the_name_it_was_made_from(
    make_record, make_profile, make_scenario
):
    row = perturb_record(make_record(), make_profile(make_scenario("s")), seed=1)

    assert row.original_name == "acme holdings ltd"


def test_a_row_keeps_the_local_id_of_the_record_it_came_from(
    make_record, make_profile, make_scenario
):
    row = perturb_record(make_record("gb-42"), make_profile(make_scenario("s")), seed=1)

    assert row.local_id == "gb-42"


def test_a_row_names_the_scenario_that_produced_it(
    make_record, make_profile, make_scenario
):
    row = perturb_record(
        make_record(), make_profile(make_scenario("light-typo")), seed=1
    )

    assert row.scenario_id == "light-typo"


def test_a_row_carries_the_changes_that_landed(
    make_record, make_profile, make_scenario
):
    row = perturb_record(make_record(), make_profile(make_scenario("s")), seed=1)

    assert [change.operator_id for change in row.changes] == (
        ["typo.keyboard_substitution"]
    )


def test_changed_is_true_when_the_name_moved(make_record, make_profile, make_scenario):
    row = perturb_record(make_record(), make_profile(make_scenario("s")), seed=1)

    assert row.changed is True


# --- a record always emits -------------------------------------------------


def test_a_chain_that_landed_nothing_still_emits_a_row(
    make_record, make_profile, make_scenario
):
    """Some records really are recorded correctly; skipping them breaks the join."""
    profile = make_profile(make_scenario("s", steps=(LANDS_NOTHING,)))

    row = perturb_record(make_record(name=SUFFIXLESS_NAME), profile, seed=1)

    assert row.name == SUFFIXLESS_NAME


def test_a_row_that_landed_nothing_is_marked_unchanged(
    make_record, make_profile, make_scenario
):
    profile = make_profile(make_scenario("s", steps=(LANDS_NOTHING,)))

    row = perturb_record(make_record(name=SUFFIXLESS_NAME), profile, seed=1)

    assert row.changed is False


def test_a_row_that_landed_nothing_says_which_step_did_not(
    make_record, make_profile, make_scenario
):
    """So nothing downstream has to compare strings to find out why."""
    profile = make_profile(make_scenario("s", steps=(LANDS_NOTHING,)))

    row = perturb_record(make_record(name=SUFFIXLESS_NAME), profile, seed=1)

    assert row.steps_that_landed_nothing == ("legal_suffix.drop",)


def test_a_row_that_landed_nothing_still_names_its_scenario(
    make_record, make_profile, make_scenario
):
    profile = make_profile(make_scenario("suffix-drop", steps=(LANDS_NOTHING,)))

    row = perturb_record(make_record(name=SUFFIXLESS_NAME), profile, seed=1)

    assert row.scenario_id == "suffix-drop"


# --- a record no scenario applies to ---------------------------------------


def test_a_record_no_scenario_applies_to_still_emits_a_row(
    make_record, make_profile, make_scenario
):
    profile = make_profile(
        make_scenario("s"), exclusions=ExclusionRules(exclude_systems=("gb",))
    )

    row = perturb_record(make_record(), profile, seed=1)

    assert row.name == row.original_name == "acme holdings ltd"


def test_a_record_no_scenario_applies_to_names_no_scenario(
    make_record, make_profile, make_scenario
):
    """`None` is how a row says nothing was drawn, rather than that nothing landed."""
    profile = make_profile(
        make_scenario("s"), exclusions=ExclusionRules(exclude_systems=("gb",))
    )

    row = perturb_record(make_record(), profile, seed=1)

    assert row.scenario_id is None


def test_a_record_no_scenario_applies_to_reports_no_step_as_landing_nothing(
    make_record, make_profile, make_scenario
):
    """No step ran, so no step failed to land: that is a different case from a no-op."""
    profile = make_profile(
        make_scenario("s"), exclusions=ExclusionRules(exclude_systems=("gb",))
    )

    row = perturb_record(make_record(), profile, seed=1)

    assert row.steps_that_landed_nothing == ()


def test_a_profile_with_no_scenarios_at_all_still_emits_a_row(
    make_record, make_profile
):
    row = perturb_record(make_record(), make_profile(), seed=1)

    assert row.scenario_id is None
    assert row.changed is False


# --- a corpus --------------------------------------------------------------


def test_one_row_comes_out_per_record(make_record, make_profile, make_scenario):
    """5,000 records in, 5,000 rows out -- not one per scenario."""
    records = [make_record(f"gb-{index}") for index in range(5)]
    profile = make_profile(make_scenario("a"), make_scenario("b"), make_scenario("c"))

    assert len(list(perturb_records(records, profile, seed=1))) == 5


def test_rows_come_out_in_the_order_the_records_arrived(
    make_record, make_profile, make_scenario
):
    records = [make_record(f"gb-{index}") for index in (3, 1, 2)]

    rows = perturb_records(records, make_profile(make_scenario("s")), seed=1)

    assert [row.local_id for row in rows] == ["gb-3", "gb-1", "gb-2"]


def test_an_empty_corpus_emits_nothing(make_profile, make_scenario):
    assert list(perturb_records([], make_profile(make_scenario("s")), seed=1)) == []
