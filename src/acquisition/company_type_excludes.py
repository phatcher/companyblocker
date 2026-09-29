from __future__ import annotations

from functools import cache
from pathlib import Path

from .company_type_common import (
    load_country_scoped_payload,
    normalize_string_set_payload,
)

_EXCLUDES_DIR = Path(__file__).with_name("company_type_mappings")


@cache
def get_company_type_exclusions(country: str) -> set[str]:
    return load_country_scoped_payload(
        country=country,
        base_dir=_EXCLUDES_DIR,
        filename_suffix="-exclude.json",
        error_message="Country code is required to load company type exclusions.",
        normalize_payload=lambda raw, path: normalize_string_set_payload(
            raw,
            path=path,
            invalid_container_message="Company type exclusion file must contain a JSON array",
            invalid_item_message="Company type exclusions must be strings",
        ),
    )
