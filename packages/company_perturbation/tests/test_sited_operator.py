"""Direct tests for `company_perturbation.sited_operator`.

The runner's contract is small enough to assert exactly: how many trials it
makes, when they fire, and how many values it takes from the generator. That
last one is the property a carried generator depends on and the one nothing
would notice breaking, so it is asserted directly rather than inferred.
"""

from __future__ import annotations

from itertools import pairwise
from random import Random

import pytest
from company_perturbation.families import Family
from company_perturbation.sited_operator import (
    Site,
    SitedOperator,
    apply_operator,
)


def _upper_at(name: str, site: Site, rng: Random, country: str | None) -> str:
    return name[: site.start] + name[site.start : site.end].upper() + name[site.end :]


def _every_character(name: str, country: str | None):
    return tuple(Site(i, i + 1) for i in range(len(name)))


def _no_sites(name: str, country: str | None):
    return ()


UPPER = SitedOperator(
    operator_id="test.upper",
    family=Family.TYPO,
    sites=_every_character,
    change=_upper_at,
)

NOWHERE = SitedOperator(
    operator_id="test.nowhere",
    family=Family.TYPO,
    sites=_no_sites,
    change=_upper_at,
)


def test_probability_zero_never_fires():
    name, applied = apply_operator(
        "acme ltd", UPPER, rng=Random(1), probability=0.0, attempts=5
    )

    assert name == "acme ltd"
    assert applied == ()


def test_probability_one_fires_every_trial():
    name, applied = apply_operator(
        "acme ltd", UPPER, rng=Random(1), probability=1.0, attempts=3
    )

    assert len(applied) == 3
    assert name != "acme ltd"


def test_attempts_caps_the_number_of_changes():
    _, applied = apply_operator(
        "acme ltd", UPPER, rng=Random(1), probability=1.0, attempts=2
    )

    assert len(applied) == 2


def test_edit_count_does_not_scale_with_name_length():
    """The property the site-and-runner split exists to guarantee.

    `attempts` is an absolute count, so a long name is not corrupted harder than a
    short one at the same setting. A site finder that scaled its own output would
    lose this without any signature changing, which is why it is asserted directly.
    """
    short = apply_operator("ab", UPPER, rng=Random(3), probability=1.0, attempts=2)[1]
    long = apply_operator(
        "a much longer company name limited",
        UPPER,
        rng=Random(3),
        probability=1.0,
        attempts=2,
    )[1]

    assert len(short) == len(long) == 2


def test_the_same_generator_state_reproduces_the_same_output():
    first = apply_operator(
        "acme ltd", UPPER, rng=Random(11), probability=0.5, attempts=4
    )
    second = apply_operator(
        "acme ltd", UPPER, rng=Random(11), probability=0.5, attempts=4
    )

    assert first == second


def test_a_different_generator_state_produces_a_different_output():
    """Randomness is controlled by the caller's generator, not by the input's identity."""
    outputs = {
        apply_operator(
            "acme ltd", UPPER, rng=Random(seed), probability=0.5, attempts=4
        )[0]
        for seed in range(12)
    }

    assert len(outputs) > 1


def test_every_trial_consumes_two_draws_whether_or_not_it_fires():
    """What makes one generator safe to carry through a whole chain.

    A step landing no edits and a step landing several must advance the stream
    identically, or changing one step reshuffles every step after it.
    """
    never = Random(5)
    always = Random(5)
    apply_operator("acme ltd", UPPER, rng=never, probability=0.0, attempts=4)
    apply_operator("acme ltd", UPPER, rng=always, probability=1.0, attempts=4)

    assert never.random() == always.random()


def test_an_operator_with_no_sites_runs_no_trials_and_draws_nothing():
    untouched = Random(5)
    name, applied = apply_operator(
        "acme ltd", NOWHERE, rng=untouched, probability=1.0, attempts=4
    )

    assert (name, applied) == ("acme ltd", ())
    assert untouched.random() == Random(5).random()


def test_attempts_is_capped_at_the_number_of_viable_sites():
    """A short name cannot absorb more changes than it has places to change.

    The ceiling comes from the sites the name offers, so a step asking for more
    than that gets what the name can hold rather than an error.
    """
    _, applied = apply_operator("ab", UPPER, rng=Random(1), probability=1.0, attempts=9)

    assert len(applied) == 2


def test_two_changes_never_land_on_the_same_site():
    _, applied = apply_operator(
        "acme ltd", UPPER, rng=Random(2), probability=1.0, attempts=6
    )
    starts = [change.site.start for change in applied]

    assert len(starts) == len(set(starts))


def test_a_chosen_site_removes_the_ones_it_overlaps():
    """How the pair-based families avoid swapping over the same character twice."""

    def _pairs(name: str, country: str | None):
        return tuple(Site(i, i + 2) for i in range(len(name) - 1))

    pair_operator = SitedOperator(
        operator_id="test.pair",
        family=Family.TYPO,
        sites=_pairs,
        change=_upper_at,
    )

    _, applied = apply_operator(
        "abcdef", pair_operator, rng=Random(1), probability=1.0, attempts=5
    )
    spans = sorted((change.site.start, change.site.end) for change in applied)

    for earlier, later in pairwise(spans):
        assert earlier[1] <= later[0]


def test_applied_changes_carry_the_operator_and_the_site():
    _, applied = apply_operator(
        "acme ltd", UPPER, rng=Random(1), probability=1.0, attempts=1
    )

    assert applied[0].operator_id == "test.upper"
    assert applied[0].family is Family.TYPO
    assert 0 <= applied[0].site.start < applied[0].site.end <= len("acme ltd")


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probability_outside_the_unit_interval_is_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        apply_operator(
            "acme", UPPER, rng=Random(1), probability=probability, attempts=1
        )


def test_negative_attempts_is_rejected():
    with pytest.raises(ValueError, match="attempts"):
        apply_operator("acme", UPPER, rng=Random(1), probability=0.5, attempts=-1)
