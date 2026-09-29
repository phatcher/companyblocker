from __future__ import annotations

import pytest

from acquisition import plan_registry
from acquisition.plan_registry import COUNTRY_REGISTRY, SYSTEM_REGISTRY, get_system_plan


def test_get_system_plan_resolves_from_country_registry():
    known_country_code = next(iter(COUNTRY_REGISTRY))

    plan = get_system_plan(known_country_code)

    assert plan is COUNTRY_REGISTRY[known_country_code]


def test_get_system_plan_resolves_from_system_registry_when_not_a_country():
    system_only_codes = set(SYSTEM_REGISTRY) - set(COUNTRY_REGISTRY)
    assert system_only_codes, "expected at least one system-only plan code to exist"
    system_only_code = next(iter(system_only_codes))

    plan = get_system_plan(system_only_code)

    assert plan is SYSTEM_REGISTRY[system_only_code]


def test_get_system_plan_prefers_system_registry_over_country_registry(
    monkeypatch: pytest.MonkeyPatch,
):
    # The real catalog rejects a code registered in both registries at load
    # time, so precedence is asserted with substitute registries instead of
    # a code found this way in the wild.
    system_plan = SYSTEM_REGISTRY[next(iter(SYSTEM_REGISTRY))]
    country_plan = COUNTRY_REGISTRY[next(iter(COUNTRY_REGISTRY))]
    monkeypatch.setattr(plan_registry, "SYSTEM_REGISTRY", {"shared": system_plan})
    monkeypatch.setattr(plan_registry, "COUNTRY_REGISTRY", {"shared": country_plan})

    assert get_system_plan("shared") is system_plan


def test_get_system_plan_normalizes_case_and_whitespace():
    known_country_code = next(iter(COUNTRY_REGISTRY))

    plan = get_system_plan(f"  {known_country_code.upper()}  ")

    assert plan is COUNTRY_REGISTRY[known_country_code]


def test_get_system_plan_raises_key_error_listing_known_values_for_unknown_code():
    with pytest.raises(KeyError, match="Unknown system 'nope'"):
        get_system_plan("nope")
