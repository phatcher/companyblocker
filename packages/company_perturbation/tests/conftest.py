"""Builders shared by the `generation` aspect suites.

Three files assert three different things about the same walk, so the records,
scenarios and profiles they set up are built here once rather than cloned into
each. The operators are the registered ones: they are deterministic and cost
nothing, so a fake would only make an assertion about the walk depend on a
stand-in instead of on what actually runs.

A scenario's default step always alters the name it is given -- a keyboard
substitution at one site, at probability 1.0 -- so a test that means "this
record was perturbed" does not also have to arrange for it.
"""

from __future__ import annotations

import company_perturbation.operators  # noqa: F401  (registers the families)
import pytest
from company_perturbation.chain import ChainStep
from company_perturbation.generation import SourceRecord
from company_perturbation.profile_schema import PerturbationProfile, Scenario

CHANGES = ChainStep("typo.keyboard_substitution", 1.0, 1)


@pytest.fixture
def make_record():
    """A source record, `gb`/`gb` unless a test cares otherwise."""

    def _make(
        local_id: str = "gb-1",
        *,
        name: str = "acme holdings ltd",
        system: str = "gb",
        country: str | None = "gb",
    ) -> SourceRecord:
        return SourceRecord(
            local_id=local_id, name=name, system=system, country=country
        )

    return _make


@pytest.fixture
def make_scenario():
    """A one-step scenario, changing the name unless another step is given."""

    def _make(scenario_id: str, *, steps=(CHANGES,), **kwargs) -> Scenario:
        return Scenario(scenario_id=scenario_id, chain=tuple(steps), **kwargs)

    return _make


@pytest.fixture
def make_profile():
    def _make(*scenarios: Scenario, **kwargs) -> PerturbationProfile:
        return PerturbationProfile(
            profile_id="test-profile",
            scenarios=scenarios,
            **kwargs,
        )

    return _make
