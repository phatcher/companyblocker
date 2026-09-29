"""Per-country mappings from a system's raw company-type values to one canonical form, read from the JSON under `company_type_mappings/` and cached."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .company_type_common import (
    load_country_scoped_payload,
    normalize_string_map_payload,
)

_MAPPINGS_DIR = Path(__file__).with_name("company_type_mappings")


@cache
def get_company_type_mapping(country: str) -> dict[str, str]:
    return load_country_scoped_payload(
        country=country,
        base_dir=_MAPPINGS_DIR,
        filename_suffix=".json",
        error_message="Country code is required to load company type mappings.",
        normalize_payload=lambda raw, path: normalize_string_map_payload(
            raw,
            path=path,
            invalid_container_message="Company type mapping file must contain an object",
            invalid_item_message="Company type mapping must be string to string",
        ),
    )
