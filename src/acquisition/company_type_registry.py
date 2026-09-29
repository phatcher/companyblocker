from __future__ import annotations

from .company_type_mappings import get_company_type_mapping
from .plan_registry import get_system_plan


def get_system_company_type_mapping(system_code: str) -> dict[str, str]:
    """Retrieve company type mappings for a supported system."""
    plan = get_system_plan(system_code)
    if not plan.company_type_column:
        return {}
    return get_company_type_mapping(plan.code)
