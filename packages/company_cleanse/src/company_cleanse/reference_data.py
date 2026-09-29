"""Reference data loaders for ISO 20275 code sets and other standardized vocabularies."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

_RESOURCES_DIR = Path(__file__).with_name("resources")


@cache
def get_iso20275_entity_legal_forms() -> dict[str, dict]:
    """Load ISO 20275 Entity Legal Forms Code List reference data.

    Returns:
        dict mapping ELF codes (e.g., 'LLCF', 'SARL') to entity legal form metadata:
        - country: Country of formation
        - country_code: ISO 3166-1 code (e.g., 'US', 'FR')
        - entity_legal_form_name: Legal form name in local language
        - language: Language name
        - language_code: ISO 639-1 language code
        - transliterated_name: ISO 01-140-10 transliterated form name
        - abbreviations_local: Local language abbreviations
        - abbreviations_transliterated: Transliterated abbreviations
        - status: ACTV (active) or INAC (inactive)
        - date_created: ISO 8601 creation date

        Returns empty dict if reference data not found.

    Examples:
        >>> legal_forms = get_iso20275_entity_legal_forms()
        >>> legal_forms.get('LLCF')
        {'country': 'United States', 'country_code': 'US', ...}

    Source:
        https://www.gleif.org/en/lei-data/code-lists/iso-20275-entity-legal-forms-code-list
        Contains 3,600+ entity legal forms across 200+ jurisdictions.
    """
    legal_forms_path = _RESOURCES_DIR / "entity_legal_forms_iso20275.json"
    if not legal_forms_path.exists():
        return {}

    raw = json.loads(legal_forms_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(
            f"ISO 20275 entity legal forms reference data must contain an object: {legal_forms_path}"
        )

    return raw
