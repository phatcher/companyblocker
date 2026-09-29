"""Load and validate the system catalog: one JSON file per system under `catalog/systems/`.

A system is configuration, not orchestration code. Each declares `all_systems_target`, a required boolean saying whether `--systems all` includes it; membership is read from that declaration and never inferred from `status`, which records how far onboarding got. It is required rather than defaulted, so no system is onboarded without someone deciding whether a full regeneration includes it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NoReturn, cast

from ..models import ResearchPlan, SourceResource, SystemPlan

_REQUIRED_PLAN_FIELDS = (
    "registry",
    "code",
    "display_name",
    "region",
    "status",
    "access_mode",
    "info_url",
    "notes",
    "all_systems_target",
)


def _normalize_optional_list_field_to_tuple[T](
    payload: dict[str, object],
    field_name: str,
    *,
    transform_item: Callable[[object], T],
) -> tuple[T, ...] | None:
    field_payload = payload.get(field_name)
    if not isinstance(field_payload, list):
        return None
    return tuple(transform_item(item) for item in field_payload)


def _normalize_optional_map_field_to_tuple[T](
    payload: dict[str, object],
    field_name: str,
    *,
    transform_entry: Callable[[object, object], T],
) -> tuple[T, ...] | None:
    field_payload = payload.get(field_name)
    if not isinstance(field_payload, dict):
        return None
    return tuple(transform_entry(key, value) for key, value in field_payload.items())


def _validate_optional_payload_typed_field(
    payload: dict[str, object],
    path: Path,
    field_name: str,
    *,
    expected_type: type,
    expected_display: str,
) -> object | None:
    field_payload = payload.get(field_name)
    if field_payload is not None and not isinstance(field_payload, expected_type):
        actual = type(field_payload).__name__
        raise ValueError(
            f"Invalid catalog payload in {path}: field '{field_name}' must be {expected_display} or null, got {actual}"
        )
    return field_payload


def _validate_nonempty_string_list_items(
    items: list[object],
    *,
    path: Path,
    field_name: str,
    key_format: str,
) -> None:
    _validate_items(
        enumerate(items),
        validate_item=lambda entry: _require_nonempty_catalog_string(
            entry[1],
            path=path,
            message=(
                f"{key_format.format(field=field_name, index=entry[0])} must be a non-empty string"
            ),
        ),
    )


def _validate_items[T](
    items: Iterable[T],
    *,
    validate_item: Callable[[T], object],
) -> None:
    for item in items:
        validate_item(item)


def _validate_optional_map_field(
    payload: dict[str, object],
    path: Path,
    field_name: str,
    *,
    validate_value,
) -> None:
    _validate_items(
        _iter_validated_optional_map_items(
            payload,
            path,
            field_name,
            expected_display="an object",
        ),
        validate_item=lambda entry: validate_value(entry[0], entry[1]),
    )


def _iter_validated_optional_map_items(
    payload: dict[str, object],
    path: Path,
    field_name: str,
    *,
    expected_display: str,
) -> tuple[tuple[str, object], ...]:
    field_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        field_name,
        expected_type=dict,
        expected_display=expected_display,
    )
    if not isinstance(field_payload, dict):
        return ()

    validated_items: list[tuple[str, object]] = []
    for key, value in field_payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"Invalid catalog payload in {path}: {field_name} keys must be non-empty strings"
            )
        validated_items.append((key, value))
    return tuple(validated_items)


def _validate_optional_map_field_values(
    payload: dict[str, object],
    path: Path,
    field_name: str,
    *,
    value_kind: str,
) -> None:
    validators = {
        "string_list": lambda field_key, value: _validate_map_string_list_value(
            field_name=field_name,
            field_key=field_key,
            value=value,
            path=path,
        ),
        "string": lambda field_key, value: _validate_map_nonempty_string_value(
            field_name=field_name,
            field_key=field_key,
            value=value,
            path=path,
        ),
    }
    validate_map_value = validators.get(value_kind)
    if validate_map_value is None:
        raise ValueError(f"Unsupported map value_kind '{value_kind}'")

    _validate_optional_map_field(
        payload,
        path,
        field_name,
        validate_value=validate_map_value,
    )


def _validate_map_string_list_value(
    *,
    field_name: str,
    field_key: str,
    value: object,
    path: Path,
) -> None:
    list_value = _require_nonempty_catalog_list(
        value,
        path=path,
        message=f"{field_name}['{field_key}'] must be a non-empty list",
    )
    _validate_nonempty_string_list_items(
        list_value,
        path=path,
        field_name=field_name,
        key_format=f"{field_name}['{{field_key}}'][{{index}}]".replace(
            "{field_key}", field_key
        ),
    )


def _validate_map_nonempty_string_value(
    *,
    field_name: str,
    field_key: str,
    value: object,
    path: Path,
) -> None:
    _require_nonempty_catalog_string(
        value,
        path=path,
        message=f"{field_name}['{field_key}'] must be a non-empty string",
    )


def _require_nonempty_catalog_list(
    value: object,
    *,
    path: Path,
    message: str,
) -> list[object]:
    if not isinstance(value, list) or not value:
        _raise_invalid_catalog_payload_value(
            path=path, message=message, actual_value=value
        )
    return value


def _require_nonempty_catalog_string(
    value: object,
    *,
    path: Path,
    message: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        _raise_invalid_catalog_payload_value(
            path=path, message=message, actual_value=value
        )
    return value


def _raise_invalid_catalog_payload_value(
    *,
    path: Path,
    message: str,
    actual_value: object,
) -> NoReturn:
    actual = type(actual_value).__name__
    raise ValueError(f"Invalid catalog payload in {path}: {message}, got {actual}")


def _validate_required_catalog_fields(payload: dict[str, object], path: Path) -> None:
    for field in _REQUIRED_PLAN_FIELDS:
        if field not in payload:
            raise ValueError(
                f"Invalid catalog payload in {path}: missing required field '{field}'"
            )


def _validate_catalog_registry_kind(payload: dict[str, object], path: Path) -> None:
    if payload.get("registry") not in {"country", "system"}:
        raise ValueError(f"Invalid registry kind in {path}: {payload.get('registry')}")


def _validate_catalog_string_fields(payload: dict[str, object], path: Path) -> None:
    string_fields = (
        "code",
        "display_name",
        "region",
        "status",
        "access_mode",
        "info_url",
        "notes",
    )
    for field in string_fields:
        if not isinstance(payload.get(field), str):
            actual = type(payload.get(field)).__name__
            raise TypeError(
                f"Invalid catalog payload in {path}: field '{field}' must be a string, got {actual}"
            )


def _validate_catalog_all_systems_target(
    payload: dict[str, object], path: Path
) -> None:
    if not isinstance(payload.get("all_systems_target"), bool):
        actual = type(payload.get("all_systems_target")).__name__
        raise TypeError(
            f"Invalid catalog payload in {path}: field 'all_systems_target' must be a boolean, got {actual}"
        )


def _validate_catalog_company_type_column(
    payload: dict[str, object], path: Path
) -> None:
    if (
        "company_type_column" in payload
        and payload["company_type_column"] is not None
        and not isinstance(payload["company_type_column"], str)
    ):
        actual = type(payload["company_type_column"]).__name__
        raise ValueError(
            f"Invalid catalog payload in {path}: field 'company_type_column' must be a string or null, got {actual}"
        )


def _validate_catalog_optional_list_fields(
    payload: dict[str, object], path: Path
) -> None:
    system_uri_identifier_candidates_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        "system_uri_identifier_candidates",
        expected_type=list,
        expected_display="a list",
    )
    if isinstance(system_uri_identifier_candidates_payload, list):
        _validate_nonempty_string_list_items(
            system_uri_identifier_candidates_payload,
            path=path,
            field_name="system_uri_identifier_candidates",
            key_format="{field}[{index}]",
        )

    canonical_input_columns_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        "canonical_input_columns",
        expected_type=list,
        expected_display="a list",
    )
    if isinstance(canonical_input_columns_payload, list):
        _validate_nonempty_string_list_items(
            canonical_input_columns_payload,
            path=path,
            field_name="canonical_input_columns",
            key_format="{field}[{index}]",
        )

    primary_name_override_type_priority_payload = (
        _validate_optional_payload_typed_field(
            payload,
            path,
            "primary_name_override_type_priority",
            expected_type=list,
            expected_display="a list",
        )
    )
    if isinstance(primary_name_override_type_priority_payload, list):
        _validate_nonempty_string_list_items(
            primary_name_override_type_priority_payload,
            path=path,
            field_name="primary_name_override_type_priority",
            key_format="{field}[{index}]",
        )

    and_tokens_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        "and_tokens",
        expected_type=list,
        expected_display="a list",
    )
    if isinstance(and_tokens_payload, list):
        _validate_nonempty_string_list_items(
            and_tokens_payload,
            path=path,
            field_name="and_tokens",
            key_format="{field}[{index}]",
        )

    personal_owner_markers_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        "personal_owner_markers",
        expected_type=list,
        expected_display="a list",
    )
    if isinstance(personal_owner_markers_payload, list):
        _validate_nonempty_string_list_items(
            personal_owner_markers_payload,
            path=path,
            field_name="personal_owner_markers",
            key_format="{field}[{index}]",
        )

    supported_countries_payload = _validate_optional_payload_typed_field(
        payload,
        path,
        "supported_countries",
        expected_type=list,
        expected_display="a list",
    )
    if isinstance(supported_countries_payload, list):
        _validate_nonempty_string_list_items(
            supported_countries_payload,
            path=path,
            field_name="supported_countries",
            key_format="{field}[{index}]",
        )


def _validate_catalog_canonical_derived_fields(
    payload: dict[str, object], path: Path
) -> None:
    for field_name, spec in _iter_validated_optional_map_items(
        payload,
        path,
        "canonical_derived_fields",
        expected_display="an object",
    ):
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(
                f"Invalid catalog payload in {path}: canonical_derived_fields keys must be non-empty strings"
            )
        if not isinstance(spec, dict):
            actual = type(spec).__name__
            raise TypeError(
                f"Invalid catalog payload in {path}: canonical_derived_fields['{field_name}'] must be an object, got {actual}"
            )
        if spec.get("type") != "concat":
            raise ValueError(
                f"Invalid catalog payload in {path}: canonical_derived_fields['{field_name}'].type must be 'concat'"
            )
        parts = spec.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(
                f"Invalid catalog payload in {path}: canonical_derived_fields['{field_name}'].parts must be a non-empty list"
            )
        for index, part in enumerate(parts):
            if not isinstance(part, str) or not part.strip():
                actual = type(part).__name__
                raise ValueError(
                    f"Invalid catalog payload in {path}: canonical_derived_fields['{field_name}'].parts[{index}] must be a non-empty string, got {actual}"
                )
        separator = spec.get("separator", " ")
        if not isinstance(separator, str):
            actual = type(separator).__name__
            raise TypeError(
                f"Invalid catalog payload in {path}: canonical_derived_fields['{field_name}'].separator must be a string, got {actual}"
            )


def _validate_catalog_resources_payload(payload: dict[str, object], path: Path) -> None:
    resources_payload = payload.get("resources", [])
    if not isinstance(resources_payload, list):
        raise TypeError(f"Invalid catalog payload in {path}: resources must be a list")

    for index, item in enumerate(resources_payload):
        if not isinstance(item, dict):
            actual = type(item).__name__
            raise TypeError(
                f"Invalid catalog payload in {path}: resources[{index}] must be an object, got {actual}"
            )


def _validate_catalog_research_payload(payload: dict[str, object], path: Path) -> None:
    research_payload = payload.get("research")
    if research_payload is not None and not isinstance(research_payload, dict):
        actual = type(research_payload).__name__
        raise ValueError(
            f"Invalid catalog payload in {path}: research must be an object when provided, got {actual}"
        )


def _validate_catalog_payload(payload: dict[str, object], path: Path) -> None:
    _validate_required_catalog_fields(payload, path)
    _validate_catalog_registry_kind(payload, path)
    _validate_catalog_string_fields(payload, path)
    _validate_catalog_all_systems_target(payload, path)
    _validate_catalog_company_type_column(payload, path)

    _validate_optional_map_field_values(
        payload,
        path,
        "system_field_candidates",
        value_kind="string_list",
    )

    _validate_catalog_optional_list_fields(payload, path)

    _validate_optional_map_field_values(
        payload,
        path,
        "canonical_source_column_aliases",
        value_kind="string",
    )
    _validate_optional_map_field_values(
        payload,
        path,
        "name_variant_type_map",
        value_kind="string",
    )
    _validate_catalog_canonical_derived_fields(payload, path)
    _validate_catalog_resources_payload(payload, path)
    _validate_catalog_research_payload(payload, path)


def _normalize_resource_payload(payload: dict[str, object]) -> dict[str, Any]:
    normalized = dict(payload)

    request_headers = normalized.get("request_headers")
    if isinstance(request_headers, list):
        normalized["request_headers"] = tuple(
            tuple(item) if isinstance(item, list) else item for item in request_headers
        )

    required_env_vars = normalized.get("required_env_vars")
    if isinstance(required_env_vars, list):
        normalized["required_env_vars"] = tuple(required_env_vars)

    return normalized


def _normalize_research_payload(payload: dict[str, object]) -> dict[str, Any]:
    normalized = dict(payload)
    allowed_stages = normalized.get("allowed_stages")
    if isinstance(allowed_stages, list):
        normalized["allowed_stages"] = tuple(allowed_stages)
    return normalized


def _derived_field_spec_entry(
    field_name: object, spec: object
) -> tuple[str, tuple[str, ...], str]:
    spec_dict = cast(dict[str, object], spec)
    parts = cast(list[object], spec_dict.get("parts", []))
    return (
        str(field_name).strip(),
        tuple(str(part).strip() for part in parts),
        str(spec_dict.get("separator", " ")),
    )


def _load_system_plan_from_payload(payload: dict[str, object]) -> SystemPlan:
    resources_payload = cast(list[dict[str, object]], payload.get("resources", []))
    research_payload = payload.get("research")

    resources = tuple(
        SourceResource(**_normalize_resource_payload(item))
        for item in resources_payload
    )
    research = (
        ResearchPlan(**_normalize_research_payload(research_payload))
        if isinstance(research_payload, dict)
        else None
    )
    supported_countries = _normalize_optional_list_field_to_tuple(
        payload,
        "supported_countries",
        transform_item=lambda item: str(item).strip().lower(),
    )
    and_tokens = _normalize_optional_list_field_to_tuple(
        payload,
        "and_tokens",
        transform_item=lambda item: str(item).strip(),
    )
    personal_owner_markers = _normalize_optional_list_field_to_tuple(
        payload,
        "personal_owner_markers",
        transform_item=lambda item: str(item).strip(),
    )
    system_uri_identifier_candidates = _normalize_optional_list_field_to_tuple(
        payload,
        "system_uri_identifier_candidates",
        transform_item=lambda item: str(item).strip(),
    )
    system_field_candidates = _normalize_optional_map_field_to_tuple(
        payload,
        "system_field_candidates",
        transform_entry=lambda field_name, candidates: (
            str(field_name).strip(),
            tuple(str(item).strip() for item in cast(list[object], candidates)),
        ),
    )
    canonical_input_columns = _normalize_optional_list_field_to_tuple(
        payload,
        "canonical_input_columns",
        transform_item=lambda item: str(item).strip(),
    )
    primary_name_override_type_priority = _normalize_optional_list_field_to_tuple(
        payload,
        "primary_name_override_type_priority",
        transform_item=lambda item: str(item).strip(),
    )
    canonical_source_column_aliases = _normalize_optional_map_field_to_tuple(
        payload,
        "canonical_source_column_aliases",
        transform_entry=lambda source_name, target_name: (
            str(source_name).strip(),
            str(target_name).strip(),
        ),
    )
    canonical_derived_fields = _normalize_optional_map_field_to_tuple(
        payload,
        "canonical_derived_fields",
        transform_entry=_derived_field_spec_entry,
    )
    name_variant_type_map = _normalize_optional_map_field_to_tuple(
        payload,
        "name_variant_type_map",
        transform_entry=lambda source_type, name_type: (
            str(source_type).strip(),
            str(name_type).strip(),
        ),
    )

    company_type_column = payload.get("company_type_column")
    return SystemPlan(
        code=str(payload["code"]),
        display_name=str(payload["display_name"]),
        region=str(payload["region"]),
        status=str(payload["status"]),
        access_mode=str(payload["access_mode"]),
        info_url=str(payload["info_url"]),
        notes=str(payload["notes"]),
        all_systems_target=bool(payload["all_systems_target"]),
        and_tokens=and_tokens,
        personal_owner_markers=personal_owner_markers,
        research=research,
        resources=resources,
        company_type_column=(
            company_type_column if isinstance(company_type_column, str) else None
        ),
        supported_countries=supported_countries,
        system_uri_identifier_candidates=system_uri_identifier_candidates,
        system_field_candidates=system_field_candidates,
        canonical_input_columns=canonical_input_columns,
        canonical_source_column_aliases=canonical_source_column_aliases,
        canonical_derived_fields=canonical_derived_fields,
        name_variant_type_map=name_variant_type_map,
        primary_name_override_type_priority=primary_name_override_type_priority,
    )


def load_catalog_plans(
    systems_dir: Path | None = None,
) -> tuple[dict[str, SystemPlan], dict[str, SystemPlan]]:
    """Load plan definitions from acquisition/catalog/systems JSON files.

    Returns a tuple of (country_registry_overrides, system_registry_overrides).
    """
    resolved_systems_dir = systems_dir or (Path(__file__).resolve().parent / "systems")
    country_registry: dict[str, SystemPlan] = {}
    system_registry: dict[str, SystemPlan] = {}

    if not resolved_systems_dir.exists():
        return country_registry, system_registry

    for path in sorted(resolved_systems_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Invalid catalog file format: {path}")

        _validate_catalog_payload(payload, path)

        code = payload["code"]
        if path.stem != code:
            raise ValueError(
                f"Invalid catalog payload in {path}: file name '{path.stem}' must match code '{code}'"
            )

        registry_kind = payload["registry"]

        plan = _load_system_plan_from_payload(payload)

        if plan.code in country_registry or plan.code in system_registry:
            raise ValueError(f"Duplicate catalog code detected in {path}: {plan.code}")

        if registry_kind == "country":
            country_registry[plan.code] = plan
        else:
            system_registry[plan.code] = plan

    return country_registry, system_registry
