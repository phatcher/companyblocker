"""Direct tests for `company_perturbation.profile_schema.validate_profile`.

Every rejection is asserted separately, and on the message rather than only on
the exception type. A profile is authored by hand, so the thing being tested is
that a run refuses to start on one it could not act on and says which part is
wrong -- one `ValueError` for eleven different mistakes would leave the author
to find it.
"""

from __future__ import annotations

import company_perturbation.operators  # noqa: F401  (registers the families)
import pytest
from company_perturbation.chain import ChainStep
from company_perturbation.profile_schema import (
    ExclusionRules,
    PerturbationProfile,
    Scenario,
    validate_profile,
)

TYPO = ChainStep("typo.keyboard_substitution", 1.0, 1)


def _profile(**overrides) -> PerturbationProfile:
    return PerturbationProfile(
        **{
            "profile_id": "light-noise",
            "scenarios": (Scenario(scenario_id="one-typo", chain=(TYPO,)),),
            **overrides,
        }
    )


def _scenario(**overrides) -> Scenario:
    return Scenario(**{"scenario_id": "one-typo", "chain": (TYPO,), **overrides})


def test_a_well_formed_profile_is_accepted():
    validate_profile(_profile())


# --- naming ----------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_id", ["Light-Noise", "light_noise", "light noise", ""]
)
def test_a_profile_id_that_is_not_kebab_case_is_rejected(profile_id):
    """The id is a filename and a replay key, so it may not vary by case or spacing."""
    with pytest.raises(ValueError, match="must be lowercase kebab-case"):
        validate_profile(_profile(profile_id=profile_id))


def test_a_scenario_id_that_is_not_kebab_case_is_rejected():
    with pytest.raises(ValueError, match="must be lowercase kebab-case"):
        validate_profile(_profile(scenarios=(_scenario(scenario_id="One_Typo"),)))


def test_a_scenario_id_declared_twice_is_rejected():
    """Two scenarios under one name make a row's recorded `scenario_id` ambiguous."""
    with pytest.raises(ValueError, match="is declared twice"):
        validate_profile(_profile(scenarios=(_scenario(), _scenario())))


# --- nothing to run --------------------------------------------------------


def test_a_profile_with_no_scenarios_is_rejected():
    with pytest.raises(ValueError, match="declares no scenarios"):
        validate_profile(_profile(scenarios=()))


def test_a_scenario_with_an_empty_chain_is_rejected():
    with pytest.raises(ValueError, match="has an empty chain"):
        validate_profile(_profile(scenarios=(_scenario(chain=()),)))


def test_a_scenario_that_can_never_be_drawn_is_still_accepted():
    """A zero weight is how an author shelves a scenario without deleting it."""
    validate_profile(_profile(scenarios=(_scenario(weight=0.0),)))


def test_a_negative_weight_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_profile(_profile(scenarios=(_scenario(weight=-1.0),)))


# --- steps -----------------------------------------------------------------


def test_a_step_naming_an_unregistered_operator_is_rejected():
    """Caught before the run rather than partway through it."""
    with pytest.raises(ValueError, match="names unknown operator"):
        validate_profile(
            _profile(scenarios=(_scenario(chain=(ChainStep("typo.nonexistent"),)),))
        )


@pytest.mark.parametrize("probability", [-0.1, 1.5])
def test_a_probability_outside_zero_to_one_is_rejected(probability):
    with pytest.raises(ValueError, match=r"is outside \[0.0, 1.0\]"):
        validate_profile(
            _profile(
                scenarios=(
                    _scenario(
                        chain=(ChainStep("typo.keyboard_substitution", probability, 1),)
                    ),
                )
            )
        )


def test_a_negative_attempt_count_is_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        validate_profile(
            _profile(
                scenarios=(
                    _scenario(
                        chain=(ChainStep("typo.keyboard_substitution", 1.0, -1),)
                    ),
                )
            )
        )


def test_zero_attempts_is_accepted():
    """A step asking for no changes is a deliberate setting, not a malformed one."""
    validate_profile(
        _profile(
            scenarios=(
                _scenario(chain=(ChainStep("typo.keyboard_substitution", 1.0, 0),)),
            )
        )
    )


# --- exclusions ------------------------------------------------------------


def test_a_profile_excluding_an_unknown_family_is_rejected():
    """A misspelt family silently excludes nothing, which is the failure to catch."""
    with pytest.raises(ValueError, match="unknown family"):
        validate_profile(
            _profile(exclusions=ExclusionRules(exclude_families=("typos",)))
        )


def test_a_scenario_excluding_an_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        validate_profile(
            _profile(
                scenarios=(
                    _scenario(exclusions=ExclusionRules(exclude_families=("typos",))),
                )
            )
        )


def test_an_unknown_system_or_country_is_not_rejected():
    """Only families are a closed set here; the other two axes are repository data."""
    validate_profile(
        _profile(
            exclusions=ExclusionRules(
                exclude_systems=("nowhere",), exclude_countries=("zz",)
            )
        )
    )
