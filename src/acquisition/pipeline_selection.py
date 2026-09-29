from __future__ import annotations

from dataclasses import dataclass

from .plan_registry import COUNTRY_REGISTRY, SYSTEM_REGISTRY, get_system_plan

# Selection keywords, not catalog codes: `resolve_system_selection` treats
# them specially and they never need a catalog entry of their own.
_SELECTION_KEYWORDS = {"all", "global"}


def _normalize_codes(
    codes: str | list[str] | tuple[str, ...], *, label: str
) -> list[str]:
    if isinstance(codes, str):
        raw_items = codes.split(",")
    else:
        raw_items = []
        for item in codes:
            raw_items.extend(str(item).split(","))

    normalized = [item.strip().lower() for item in raw_items if item.strip()]
    if not normalized:
        raise ValueError(f"At least one {label} code must be provided")
    return normalized


def all_target_code_lists() -> tuple[list[str], list[str]]:
    """Codes `--systems all` targets, split into (countries, other systems).

    Read from each plan's own `all_systems_target` declaration. Status is not
    consulted: it records how far onboarding got and why it stopped, which is
    a different question from who a regeneration should sweep, and using it as
    a proxy was wrong in both directions (it excluded `wikidata`, whose
    extractor and cleansed data had landed, and included the synthetic
    `perturbed`, which `company_perturbation` generates rather than acquires).
    """
    country_codes = sorted(
        code for code, plan in COUNTRY_REGISTRY.items() if plan.all_systems_target
    )
    system_codes = sorted(
        code for code, plan in SYSTEM_REGISTRY.items() if plan.all_systems_target
    )
    return country_codes, system_codes


def all_target_system_codes() -> list[str]:
    """Every code `--systems all` expands to, countries and systems together."""
    country_codes, system_codes = all_target_code_lists()
    return sorted(set(country_codes + system_codes))


def _supported_system_codes() -> list[str]:
    """`normalize_systems`'s default `all` resolver -- see
    `all_target_system_codes`, which it defers to."""
    return all_target_system_codes()


@dataclass(frozen=True)
class SystemSelection:
    requested: list[str]
    concrete: list[str]
    include_global_target: bool


def normalize_systems(
    systems: str | list[str] | tuple[str, ...],
    *,
    expand_all: bool = True,
    supported_codes_resolver=_supported_system_codes,
) -> list[str]:
    normalized = _normalize_codes(systems, label="system")
    if not expand_all or "all" not in normalized:
        return normalized

    all_supported = supported_codes_resolver()
    extras = [
        code for code in normalized if code != "all" and code not in all_supported
    ]
    return all_supported + extras


def resolve_system_selection(
    systems: str | list[str] | tuple[str, ...],
    *,
    add_global_for_all: bool = False,
    global_expands_to_all: bool = False,
    normalize_systems_fn=normalize_systems,
) -> SystemSelection:
    requested = normalize_systems_fn(systems, expand_all=False)
    expanded = normalize_systems_fn(systems)

    include_global_target = "global" in requested or (
        add_global_for_all and "all" in requested
    )
    concrete = [code for code in expanded if code != "global"]

    if global_expands_to_all and not concrete and "global" in requested:
        concrete = normalize_systems_fn("all")

    return SystemSelection(
        requested=requested,
        concrete=concrete,
        include_global_target=include_global_target,
    )


def validate_requested_systems(
    requested: list[str],
    *,
    get_system_plan_fn=get_system_plan,
) -> None:
    """Reject a code the catalog does not hold, before any stage runs.

    Checked against exactly what the caller typed (`SystemSelection.requested`,
    ahead of `all` expansion): `all` and `global` are selection keywords, not
    catalog codes, and pass unchecked. Every other code is looked up through
    `get_system_plan_fn`, reusing `get_system_plan`'s own unknown-code message
    -- naming the code and every known value -- so a typo here and a stage
    that still looks a code up directly report the same thing, and a known
    code whose layer is genuinely missing is untouched: that failure still
    comes from the stage that needed the layer.
    """
    for code in requested:
        if code in _SELECTION_KEYWORDS:
            continue
        try:
            get_system_plan_fn(code)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
