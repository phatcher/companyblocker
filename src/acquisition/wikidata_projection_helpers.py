from __future__ import annotations

from acquisition.io_eta_helpers import _json_dumps_line, _json_loads

_COMPANY_LIKE_INSTANCE_OF_QIDS = {
    "Q783794",  # company
    "Q6881511",  # enterprise
    "Q43229",  # organization
    "Q167037",  # business
    "Q891723",  # public company
    "Q161726",  # multinational corporation
}

_COMPANY_LIKE_INSTANCE_OF_QIDS_FROZEN = frozenset(_COMPANY_LIKE_INSTANCE_OF_QIDS)
_COMPANY_LIKE_INSTANCE_OF_QID_PATTERNS = tuple(
    f'"{qid}"'.encode("ascii") for qid in _COMPANY_LIKE_INSTANCE_OF_QIDS_FROZEN
)

_ENTITY_ID_PROPERTIES = {
    "P17": "country",
    "P1001": "jurisdiction",
    "P159": "headquarters_location",
    "P452": "industry",
    "P1454": "legal_form",
}

_TIME_PROPERTIES = {
    "P571": "inception",
    "P576": "dissolved",
}

_TEXT_PROPERTIES = {
    "P856": "website",
    "P1278": "lei",
    "P946": "isin",
    # Jurisdiction-scoped company-number source claims: unlike `lei`
    # (a single global P1278 property), no single Wikidata property covers a
    # real company registration number -- each jurisdiction has its own. Read
    # here as plain claim-derived list columns, same shape as `lei`/`isin`;
    # canonical-stage code (`_resolve_company_number_expr`'s
    # "wikidata_jurisdiction_scoped" mode) picks the right one per row's own
    # resolved `jurisdiction_code`. The population
    # survey against the real raw dump says why DE/IE either are or aren't
    # covered here.
    "P2622": "company_number_gb",  # Companies House company ID
    "P1616": "company_number_fr",  # SIREN number
    "P12012": "company_number_de",  # record number (Germany)
}

_MULTI_LANG_TEXT_PROPERTIES = {
    "P1448": ("official_name", "official_name_en", "official_name_variants"),
    "P1813": ("short_name", "short_name_en", "short_name_variants"),
}


def _iter_claim_values(entity: dict[str, object], property_id: str):
    """
    Iterate over raw values from a claim property, handling all navigation and type checks.
    Yields: value object from each claim's mainsnak.datavalue.value
    """
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return

    claim_items = claims.get(property_id)
    if not isinstance(claim_items, list):
        return

    for claim in claim_items:
        if not isinstance(claim, dict):
            continue
        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, dict):
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, dict):
            continue
        value = datavalue.get("value")
        if value is not None:
            yield value


def _extract_claim_entity_ids(entity: dict[str, object], property_id: str) -> list[str]:
    values: list[str] = []
    for value in _iter_claim_values(entity, property_id):
        if not isinstance(value, dict):
            continue
        entity_id = value.get("id")
        if isinstance(entity_id, str) and entity_id:
            values.append(entity_id)
    return values


def _extract_time_claim(entity: dict[str, object], property_id: str) -> str | None:
    for value in _iter_claim_values(entity, property_id):
        if not isinstance(value, dict):
            continue
        time_value = value.get("time")
        if isinstance(time_value, str) and time_value:
            return time_value
    return None


def _iter_claim_text_values_with_language(
    entity: dict[str, object],
    property_id: str,
):
    for value in _iter_claim_values(entity, property_id):
        if isinstance(value, str):
            if value:
                yield value, None
            continue

        if not isinstance(value, dict):
            continue
        text_value = value.get("text")
        if not isinstance(text_value, str) or not text_value:
            continue
        language = value.get("language")
        yield text_value, language if isinstance(language, str) else None


def _extract_claim_text_values(
    entity: dict[str, object], property_id: str
) -> list[str]:
    values: list[str] = []
    for text_value, _ in _iter_claim_text_values_with_language(entity, property_id):
        values.append(text_value)

    return values


def _extract_claim_text_values_by_language(
    entity: dict[str, object],
    property_id: str,
    *,
    language: str,
) -> list[str]:
    values: list[str] = []
    for text_value, value_language in _iter_claim_text_values_with_language(
        entity, property_id
    ):
        if value_language == language:
            values.append(text_value)

    return values


def _extract_claim_text_values_with_english(
    entity: dict[str, object],
    property_id: str,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    """Return (all_text_values, english_text_values, tagged_values) in one scan.

    `tagged_values` keeps each value's own language ("en", another ISO code,
    or None for an untyped plain-string datavalue) instead of collapsing it
    to the all/english split.
    """
    all_values: list[str] = []
    english_values: list[str] = []
    tagged_values: list[dict[str, object]] = []
    for text_value, value_language in _iter_claim_text_values_with_language(
        entity, property_id
    ):
        all_values.append(text_value)
        if value_language == "en":
            english_values.append(text_value)
        tagged_values.append({"value": text_value, "language": value_language})

    return all_values, english_values, tagged_values


def _extract_instance_of_fast(entity: dict[str, object]) -> list[str]:
    """Fast path for P31 (instance_of) extraction - just delegates to the generic extractor."""
    return _extract_claim_entity_ids(entity, "P31")


def _get_langstring_field(
    entity: dict[str, object], field_name: str, language: str = "en"
):
    """
    Navigate langstring fields (labels, aliases, descriptions).
    Returns dict for field_name → language, or None if not found.
    For aliases, returns the list; for labels/descriptions, returns the dict.
    """
    field = entity.get(field_name)
    if not isinstance(field, dict):
        return None
    lang_data = field.get(language)
    return lang_data


def _extract_projection_claims(
    entity: dict[str, object],
) -> tuple[dict[str, object], list[str]]:
    result: dict[str, object] = {
        "country": [],
        "jurisdiction": [],
        "headquarters_location": [],
        "industry": [],
        "legal_form": [],
        "inception": None,
        "dissolved": None,
        "website": [],
        "lei": [],
        "isin": [],
        "company_number_gb": [],
        "company_number_fr": [],
        "company_number_de": [],
        "official_name": [],
        "official_name_en": [],
        "official_name_variants": [],
        "short_name": [],
        "short_name_en": [],
        "short_name_variants": [],
    }
    instance_of: list[str] = []

    if not isinstance(entity.get("claims"), dict):
        return result, instance_of

    for property_id, target_key in _ENTITY_ID_PROPERTIES.items():
        result[target_key] = _extract_claim_entity_ids(entity, property_id)

    for property_id, target_key in _TIME_PROPERTIES.items():
        result[target_key] = _extract_time_claim(entity, property_id)

    for property_id, target_key in _TEXT_PROPERTIES.items():
        result[target_key] = _extract_claim_text_values(entity, property_id)

    for property_id, (
        all_key,
        en_key,
        variants_key,
    ) in _MULTI_LANG_TEXT_PROPERTIES.items():
        all_values, english_values, tagged_values = (
            _extract_claim_text_values_with_english(entity, property_id)
        )
        result[all_key] = all_values
        result[en_key] = english_values
        result[variants_key] = tagged_values

    return result, instance_of


def _is_company_like(instance_of: list[str]) -> bool:
    return any(
        entity_id in _COMPANY_LIKE_INSTANCE_OF_QIDS_FROZEN for entity_id in instance_of
    )


def _extract_en_label(entity: dict[str, object]) -> str | None:
    lang_data = _get_langstring_field(entity, "labels", "en")
    if not isinstance(lang_data, dict):
        return None
    value = lang_data.get("value")
    return value if isinstance(value, str) and value else None


def _extract_en_aliases(entity: dict[str, object]) -> list[str]:
    lang_data = _get_langstring_field(entity, "aliases", "en")
    if not isinstance(lang_data, list):
        return []

    values: list[str] = []
    for alias_item in lang_data:
        if not isinstance(alias_item, dict):
            continue
        value = alias_item.get("value")
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _extract_en_description(entity: dict[str, object]) -> str | None:
    lang_data = _get_langstring_field(entity, "descriptions", "en")
    if not isinstance(lang_data, dict):
        return None
    value = lang_data.get("value")
    return value if isinstance(value, str) and value else None


def _extract_sitelinks_count(entity: dict[str, object]) -> int:
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, dict):
        return 0
    return len(sitelinks)


def _project_wikidata_company_record_line(
    line: bytes,
    legal_form_qids: frozenset[str] | None = None,
) -> bytes | None:
    """Project a candidate raw line into the final company record JSONL line.

    ``legal_form_qids`` is the p279 subclass-of-company-legal-form closure (see
    ``p279.json``). When provided, this performs the authoritative company decision
    (root instance-of OR legal-form match) itself, since pass 1's raw-line filter is
    a permissive superset once it also has to admit legal-form candidates. When
    omitted, callers are trusting pass 1 as the sole decision, matching the original
    root-QID-only behavior.
    """
    entity = _json_loads(line)
    if entity is None:
        return None

    entity_id = entity.get("id")
    if not isinstance(entity_id, str) or not entity_id:
        return None

    instance_of = _extract_instance_of_fast(entity)
    claims_projection, _ = _extract_projection_claims(entity)

    matched_company_type_qids: list[str] = []
    if legal_form_qids is not None:
        legal_form_values = claims_projection["legal_form"]
        if isinstance(legal_form_values, list):
            matched_company_type_qids = [
                qid
                for qid in legal_form_values
                if isinstance(qid, str) and qid in legal_form_qids
            ]
        if not _is_company_like(instance_of) and not matched_company_type_qids:
            return None

    lei_values_raw = claims_projection["lei"]
    lei_values = lei_values_raw if isinstance(lei_values_raw, list) else []
    record = {
        "id": entity_id,
        "entity_type": entity.get("type")
        if isinstance(entity.get("type"), str)
        else None,
        "modified": entity.get("modified")
        if isinstance(entity.get("modified"), str)
        else None,
        "label_en": _extract_en_label(entity),
        "description_en": _extract_en_description(entity),
        "aliases_en": _extract_en_aliases(entity),
        "official_name": claims_projection["official_name"],
        "official_name_en": claims_projection["official_name_en"],
        "official_name_variants": claims_projection["official_name_variants"],
        "short_name": claims_projection["short_name"],
        "short_name_en": claims_projection["short_name_en"],
        "short_name_variants": claims_projection["short_name_variants"],
        "instance_of": instance_of,
        "country": claims_projection["country"],
        "jurisdiction": claims_projection["jurisdiction"],
        "inception": claims_projection["inception"],
        "dissolved": claims_projection["dissolved"],
        "headquarters_location": claims_projection["headquarters_location"],
        "industry": claims_projection["industry"],
        "legal_form": claims_projection["legal_form"],
        "website": claims_projection["website"],
        "lei": lei_values,
        # No `company_number` field here -- it used to be a bare copy
        # of `lei` (a global LEI, not a real jurisdiction-specific company
        # registration number). Canonical-stage code derives it from
        # `company_number_gb`/`_fr`/`_de` below instead.
        "company_number_gb": claims_projection["company_number_gb"],
        "company_number_fr": claims_projection["company_number_fr"],
        "company_number_de": claims_projection["company_number_de"],
        "isin": claims_projection["isin"],
        "sitelinks_count": _extract_sitelinks_count(entity),
        "matched_company_type_qids": matched_company_type_qids,
    }
    return _json_dumps_line(record)
