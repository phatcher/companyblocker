from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def normalize_company_type_country(country: str, *, error_message: str) -> str:
    normalized = country.strip().lower()
    if not normalized:
        raise ValueError(error_message)
    return normalized


def load_country_scoped_json(
    *,
    country: str,
    base_dir: Path,
    filename_suffix: str,
    error_message: str,
) -> tuple[Path, object | None]:
    normalized = normalize_company_type_country(country, error_message=error_message)
    path = base_dir / f"{normalized}{filename_suffix}"
    return path, load_optional_json(path)


def load_country_scoped_payload[T](
    *,
    country: str,
    base_dir: Path,
    filename_suffix: str,
    error_message: str,
    normalize_payload: Callable[[object | None, Path], T],
) -> T:
    path, raw = load_country_scoped_json(
        country=country,
        base_dir=base_dir,
        filename_suffix=filename_suffix,
        error_message=error_message,
    )
    return normalize_payload(raw, path)


def load_optional_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_string_set_payload(
    raw: object | None,
    *,
    path: Path,
    invalid_container_message: str,
    invalid_item_message: str,
) -> set[str]:
    if raw is None:
        return set()
    if not isinstance(raw, list):
        raise TypeError(f"{invalid_container_message}: {path}")

    values: set[str] = set()
    for value in raw:
        if not isinstance(value, str):
            raise TypeError(f"{invalid_item_message}: {path}")
        cleaned = value.strip()
        if cleaned:
            values.add(cleaned)
    return values


def normalize_string_map_payload(
    raw: object | None,
    *,
    path: Path,
    invalid_container_message: str,
    invalid_item_message: str,
) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"{invalid_container_message}: {path}")

    mapping: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{invalid_item_message}: {path}")
        mapping[key] = value
    return mapping
