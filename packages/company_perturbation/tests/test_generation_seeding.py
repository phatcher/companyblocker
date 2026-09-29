"""Direct tests for `company_perturbation.generation`'s seeding.

A record's generator comes from `(seed, local_id)` together, and each half is
there to rule out a specific failure. Without the seed, a record has exactly one
possible perturbation forever -- the old defect. Without the local id, a
record's outcome depends on how many records preceded it, so scoping a run to
one country silently changes every row in it.

Both are asserted here as properties of the walk rather than as literal digest
values, so the composition can be changed without rewriting the tests that say
what it is for.
"""

from __future__ import annotations

from company_perturbation.generation import (
    _record_seed,
    perturb_record,
    perturb_records,
)


def test_the_same_seed_and_record_give_the_same_row(
    make_record, make_profile, make_scenario
):
    profile = make_profile(make_scenario("s"))

    assert perturb_record(make_record(), profile, seed=7) == (
        perturb_record(make_record(), profile, seed=7)
    )


def test_a_different_seed_gives_an_independent_sample(
    make_record, make_profile, make_scenario
):
    """More than one possible outcome per record, which is what an error bar needs."""
    profile = make_profile(make_scenario("s"))

    names = {
        perturb_record(make_record(), profile, seed=seed).name for seed in range(10)
    }

    assert len(names) > 1


def test_two_records_under_one_seed_are_perturbed_differently(
    make_record, make_profile, make_scenario
):
    """The seed alone does not decide a row, or a corpus would be one repeated edit."""
    profile = make_profile(make_scenario("s"))
    records = [make_record(f"gb-{index}") for index in range(8)]

    rows = list(perturb_records(records, profile, seed=1))

    assert len({row.name for row in rows}) > 1


def test_a_records_row_does_not_depend_on_how_the_corpus_was_sliced(
    make_record, make_profile, make_scenario
):
    """Scoping a run to one country must not change the rows it has in common."""
    profile = make_profile(make_scenario("s"))
    everything = [make_record(f"gb-{index}") for index in range(6)]

    whole = list(perturb_records(everything, profile, seed=1))
    sliced = list(perturb_records([everything[1], everything[4]], profile, seed=1))

    assert sliced == [whole[1], whole[4]]


def test_a_records_row_does_not_depend_on_the_order_it_was_processed_in(
    make_record, make_profile, make_scenario
):
    profile = make_profile(make_scenario("s"))
    records = [make_record(f"gb-{index}") for index in range(4)]

    forwards = list(perturb_records(records, profile, seed=1))
    backwards = list(perturb_records(reversed(records), profile, seed=1))

    assert list(reversed(backwards)) == forwards


def test_a_records_seed_varies_with_the_run_seed():
    assert _record_seed(1, "gb-1") != _record_seed(2, "gb-1")


def test_a_records_seed_varies_with_the_record():
    assert _record_seed(1, "gb-1") != _record_seed(1, "gb-2")


def test_a_local_id_cannot_be_run_together_with_the_seed_beside_it():
    """Length-prefixed, so `(1, '23')` and `(12, '3')` are not one stream.

    A local id is repository data and may contain anything, including whatever
    separator a naive composition would have used.
    """
    assert _record_seed(1, "23") != _record_seed(12, "3")
