"""Direct tests for `company_perturbation.chain`.

The properties worth pinning here are the ones the old design got wrong: that the
seed is an input rather than something derived from the record, that one generator
runs the whole chain rather than each step getting its own, and that a step which
landed nothing says so instead of failing silently.
"""

from __future__ import annotations

from random import Random

import company_perturbation.operators  # noqa: F401  (registers the families)
import pytest
from company_perturbation.chain import ChainStep, perturb, run_chain

ALWAYS = {"probability": 1.0, "attempts": 1}


def test_the_same_seed_reproduces_the_same_name():
    steps = [ChainStep("typo.keyboard_substitution", 1.0, 2)]

    assert perturb("acme holdings ltd", steps, seed=7).name == (
        perturb("acme holdings ltd", steps, seed=7).name
    )


def test_a_different_seed_gives_an_independent_sample():
    """The property the old design could not offer: more than one possible output."""
    steps = [ChainStep("typo.keyboard_substitution", 1.0, 2)]
    names = {perturb("acme holdings ltd", steps, seed=seed).name for seed in range(10)}

    assert len(names) > 1


def test_the_output_of_one_step_is_the_input_of_the_next():
    """A chain composes; it does not apply every step to the original name."""
    steps = [
        ChainStep("legal_suffix.drop", **ALWAYS),
        ChainStep("low_salience.token_drop", **ALWAYS),
    ]

    result = perturb("acme systems ltd", steps, seed=1)

    assert "ltd" not in result.name
    assert len(result.outcomes) == 2


def test_one_generator_runs_the_whole_chain():
    """Each step continues the stream rather than restarting it.

    Two steps with the same operator must not produce the same draw twice.
    """
    one_generator = Random(3)
    two_steps = run_chain(
        "acme holdings limited",
        [ChainStep("typo.keyboard_substitution", 1.0, 1)] * 2,
        rng=one_generator,
    )
    restarted = [
        run_chain(
            "acme holdings limited",
            [ChainStep("typo.keyboard_substitution", 1.0, 1)],
            rng=Random(3),
        )
        .changes[0]
        .site.start
        for _ in range(2)
    ]

    starts = [change.site.start for change in two_steps.changes]
    assert starts[0] == restarted[0]
    assert starts != restarted


def test_a_step_that_landed_nothing_is_reported():
    """The silent failure a mis-ordered chain produces.

    `legal_suffix.drop` finds no suffix here, so it makes no change and nothing
    raises. The outcome is what says so.
    """
    result = perturb(
        "acme holdings", [ChainStep("legal_suffix.drop", **ALWAYS)], seed=1
    )

    assert result.name == "acme holdings"
    assert result.steps_that_landed_nothing == ("legal_suffix.drop",)


def test_a_step_that_was_never_asked_to_act_is_not_reported_as_landing_nothing():
    result = perturb("acme holdings", [ChainStep("legal_suffix.drop", 1.0, 0)], seed=1)

    assert result.steps_that_landed_nothing == ()


def test_recognise_then_corrupt_is_what_a_chain_is_for():
    """`Ltd -> Limited -> Limiteda`: substitute the suffix, then mistype the result.

    Reversed, the typo mangles the suffix and the substitution finds nothing --
    which is the ordering hazard, visible through the outcome rather than an error.
    """
    corrupt_first = perturb(
        "acme holdings ltd",
        [
            ChainStep("typo.keyboard_substitution", 1.0, 6),
            ChainStep("legal_suffix.variant_substitution", **ALWAYS),
        ],
        seed=5,
        country="gb",
    )

    assert corrupt_first.steps_that_landed_nothing == (
        "legal_suffix.variant_substitution",
    )


def test_changes_are_collected_across_every_step():
    result = perturb(
        "acme holdings limited",
        [
            ChainStep("typo.keyboard_substitution", 1.0, 2),
            ChainStep("typo.duplication", 1.0, 1),
        ],
        seed=2,
    )

    assert len(result.changes) == 3
    assert {change.operator_id for change in result.changes} == {
        "typo.keyboard_substitution",
        "typo.duplication",
    }


def test_an_unknown_operator_id_is_rejected():
    with pytest.raises(KeyError, match="no operator registered"):
        perturb("acme", [ChainStep("typo.nonexistent", **ALWAYS)], seed=1)


def test_an_empty_chain_returns_the_name_untouched():
    result = perturb("acme holdings ltd", [], seed=1)

    assert result == run_chain("acme holdings ltd", [], rng=Random(1))
    assert result.name == "acme holdings ltd"
