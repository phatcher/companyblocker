from __future__ import annotations

from .catalog import load_catalog_plans
from .models import SystemPlan

COUNTRY_REGISTRY, SYSTEM_REGISTRY = load_catalog_plans()


def get_system_plan(system_code: str) -> SystemPlan:
    """Retrieve system plan by code."""
    normalized = system_code.strip().lower()
    if normalized in SYSTEM_REGISTRY:
        return SYSTEM_REGISTRY[normalized]

    if normalized in COUNTRY_REGISTRY:
        return COUNTRY_REGISTRY[normalized]

    known = ", ".join(sorted(set(SYSTEM_REGISTRY) | set(COUNTRY_REGISTRY)))
    raise KeyError(f"Unknown system '{system_code}'. Known values: {known}")
