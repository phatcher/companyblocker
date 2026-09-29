"""Extraction from a normalized name: company type, quoted and parenthesized names, short name.

One shape is a special case: an acronym in quotes, a legal form, then a parenthesized full name
(`"ABC" Ltd (Associated Business Consultants)`). It rewrites `name_cleansed` full name first,
`associated business consultants ltd`, and takes `short_name` from the quoted acronym directly,
so that row's `short_name` ignores `short_name_profile`. A one-word parenthetical does not
trigger it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .corpus_noise_words import get_corpus_noise_words
from .geographic_terms import (
    resolve_geographic_term_kinds,
    strip_geographic_terms,
)
from .normalize import (
    DEFAULT_NORMALIZATION_PROFILE,
    GEOGRAPHIC_TERMS_PROFILE_ALIAS,
    GEOGRAPHIC_TIER_SIGIL,
    NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS,
    _normalize_company_type_value,
    _normalize_suffix_surface,
    _normalize_tokens,
)
from .rules import COMPANY_TYPE_TOKENS

_SHORT_NAME_STAGE_COMPANY_TYPE = "company_type"
_SHORT_NAME_STAGE_NOISE_WORDS = "noise_words"
_NOISE_WORDS_LEVEL_PART = f"+{_SHORT_NAME_STAGE_NOISE_WORDS}"
_SHORT_NAME_STAGE_GEOGRAPHIC_SPELLINGS = frozenset(
    {NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS, GEOGRAPHIC_TERMS_PROFILE_ALIAS}
)


@dataclass(frozen=True, slots=True)
class ShortNameStageFlags:
    """Which short-name derivation stages a profile selected, and on what tiers.

    Attributes:
        remaining_profile: The profile parts that are text-normalization operations
            rather than short-name stages, for the caller to hand on to
            `normalize_tokens()`.
        include_company_type: Whether trailing company-type component trimming runs.
            On unless `-company_type` turned it off.
        include_noise_words: Whether noise-word removal runs. On unless
            `-noise_words` turned it off.
        geographic_tiers: The tiers the geographic stage strips, or ``None`` when the
            profile did not name it. Opt-*in*, unlike the two above.
        noise_words_level: The noise-word profile a `+noise_words:<level>` part
            named, or ``None`` when the profile left the level to the caller's own
            `noise_words_profile`.
    """

    remaining_profile: str
    include_company_type: bool
    include_noise_words: bool
    geographic_tiers: tuple[str, ...] | None
    noise_words_level: str | None = None


def _split_short_name_stage_flags(normalization_profile: str) -> ShortNameStageFlags:
    """Pull the short-name stage selections out of a normalization profile string.

    Three stages, using the same three sigils `parse_normalization_profile()` documents:
    `-company_type` and `-noise_words` switch off a stage that is on by default, a bare
    `geographic_terms` (or its `geographic` shorthand) switches on a stage that is off by
    default, and `+region`/`+city` add tiers to that stage.

    None of these are `normalize_tokens()` operations here -- they gate the short-name
    derivation stages run after normalizing (in `strip_company_suffix()` and the
    `cleanse_lazyframe()` batch path), not the text normalization itself. The geographic
    stage in particular is deliberately consumed rather than forwarded: run as a text
    operation it would fire before company-type trimming, and the trailing arm needs the
    legal form to be gone already. The remaining profile parts are returned unchanged for
    the caller to pass on to text normalization.
    """
    include_company_type = True
    include_noise_words = True
    noise_words_level: str | None = None
    geographic_selected = False
    tiers: list[str] = []
    remaining_parts: list[str] = []

    for part in normalization_profile.split("|"):
        stripped = part.strip()
        lowered = stripped.lower()
        if lowered == f"-{_SHORT_NAME_STAGE_COMPANY_TYPE}":
            include_company_type = False
            continue
        if lowered == f"-{_SHORT_NAME_STAGE_NOISE_WORDS}":
            include_noise_words = False
            continue
        if lowered in _SHORT_NAME_STAGE_GEOGRAPHIC_SPELLINGS:
            geographic_selected = True
            continue
        # `+noise_words:<level>` carries the noise-word profile inside the string,
        # so one profile says everything a record needs. Read before the tier
        # branch, which would otherwise take it for a geographic tier.
        if lowered.split(":", 1)[0] == _NOISE_WORDS_LEVEL_PART:
            level = lowered.partition(":")[2].strip()
            if not level:
                raise ValueError(
                    f"Profile {normalization_profile!r} uses "
                    f"'{_NOISE_WORDS_LEVEL_PART}' without a level, for example "
                    f"'{_NOISE_WORDS_LEVEL_PART}:balanced'."
                )
            noise_words_level = level
            continue
        if lowered.startswith(GEOGRAPHIC_TIER_SIGIL):
            tier = lowered[1:].strip()
            if not tier:
                raise ValueError(
                    f"Profile {normalization_profile!r} has an empty "
                    f"'{GEOGRAPHIC_TIER_SIGIL}' tier modifier."
                )
            tiers.append(tier)
            continue
        if stripped:
            remaining_parts.append(stripped)

    if tiers and not geographic_selected:
        listed = ", ".join(f"{GEOGRAPHIC_TIER_SIGIL}{tier}" for tier in tiers)
        raise ValueError(
            f"Profile {normalization_profile!r} uses tier modifier(s) {listed} without "
            f"selecting '{NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS}'."
        )

    if noise_words_level is not None and not include_noise_words:
        raise ValueError(
            f"Profile {normalization_profile!r} both switches noise words off and "
            f"names a level for them."
        )

    remaining_profile = "|".join(remaining_parts) or DEFAULT_NORMALIZATION_PROFILE
    return ShortNameStageFlags(
        remaining_profile=remaining_profile,
        include_company_type=include_company_type,
        include_noise_words=include_noise_words,
        geographic_tiers=(
            resolve_geographic_term_kinds(tuple(tiers)) if geographic_selected else None
        ),
        noise_words_level=noise_words_level,
    )


def _resolve_short_name_stage_flags(short_name_profile: str) -> ShortNameStageFlags:
    """Resolve a `CleanseConfig.short_name_profile` string into stage flags.

    Unlike `strip_company_suffix()`'s `normalization_profile`, this field carries only
    the stage selections -- text canonicalization for the batch pipeline is already
    controlled separately by `CleanseConfig.normalization_profile`, so anything else
    here is a mistake rather than a chain to forward.
    """
    flags = _split_short_name_stage_flags(short_name_profile)
    if flags.remaining_profile.strip().lower() != DEFAULT_NORMALIZATION_PROFILE:
        raise ValueError(
            "short_name_profile only supports 'default' plus '-company_type'/"
            "'-noise_words'/'geographic_terms' (with optional '+region'/'+city') "
            f"toggles, got {short_name_profile!r}."
        )
    return flags


_RE_LEADING_PAREN = re.compile(r"^\(\s*([^)]+)\s*\)")
_RE_LEADING_QUOTED = re.compile(r'^(?:"([^"]+)"|\'([^\']+)\')(?=\s+\S)')
_RE_QUOTED_TYPE_PAREN = re.compile(
    r"^(?:\"([^\"]+)\"|'([^']+)')\s*(?:([^()]+?)\s*)?\(\s*([^)]+)\s*\)\s*$"
)
_RE_ACRONYM_LIKE = re.compile(r"[a-z0-9]{2,12}")


def _build_short_name_company_type_component_tokens() -> frozenset[str]:
    tokens: set[str] = set()
    for company_type_value in COMPANY_TYPE_TOKENS:
        for token in company_type_value.split():
            normalized = _normalize_company_type_value(token, transliterate=True)
            if normalized:
                tokens.add(normalized)
    return frozenset(tokens)


_SHORT_NAME_COMPANY_TYPE_COMPONENT_TOKENS = (
    _build_short_name_company_type_component_tokens()
)


@lru_cache(maxsize=512)
def _derived_company_type_suffix_forms(
    company_type: str | None,
    frozen_company_type_mapping: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    normalized_company_type = _normalize_company_type_value(
        company_type, transliterate=True
    )
    if not normalized_company_type:
        return ()

    suffix_forms: set[str] = {normalized_company_type}
    for source_value, canonical_value in frozen_company_type_mapping:
        normalized_canonical = _normalize_company_type_value(
            canonical_value, transliterate=True
        )
        if normalized_canonical != normalized_company_type:
            continue

        normalized_source = _normalize_company_type_value(
            source_value, transliterate=True
        )
        if normalized_source:
            suffix_forms.add(normalized_source)

    return tuple(
        sorted(suffix_forms, key=lambda value: (-len(value.split()), -len(value)))
    )


def _trim_quoted_name_company_type_suffix(
    quoted_name: str | None,
    company_type: str | None,
    company_type_mapping: dict[str, str],
) -> str | None:
    normalized_quoted_name = _normalize_company_type_value(
        quoted_name, transliterate=True
    )
    if not normalized_quoted_name:
        return None

    suffix_forms = _derived_company_type_suffix_forms(
        company_type,
        tuple(
            sorted(
                (str(key), str(value)) for key, value in company_type_mapping.items()
            )
        ),
    )
    if not suffix_forms:
        return normalized_quoted_name

    for suffix_form in suffix_forms:
        if normalized_quoted_name == suffix_form:
            return None
        if normalized_quoted_name.endswith(f" {suffix_form}"):
            trimmed = normalized_quoted_name[: -len(suffix_form)].rstrip()
            return trimmed or None

    return normalized_quoted_name


def _build_noise_word_sets(
    noise_words: tuple[str | dict[str, str], ...] | list[str | dict[str, str]] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    removable_anywhere_tokens: set[str] = set()
    removable_suffix_tokens: set[str] = set()

    for noise_word in noise_words or ():
        if isinstance(noise_word, dict):
            token_value = noise_word.get("token")
            scope = str(noise_word.get("scope", "suffix")).strip().lower()
        else:
            token_value = noise_word
            scope = "suffix"

        normalized_stop = _normalize_company_type_value(token_value, transliterate=True)
        if not normalized_stop:
            continue

        normalized_parts = [token for token in normalized_stop.split() if token]
        if scope == "anywhere":
            removable_anywhere_tokens.update(normalized_parts)
        else:
            removable_suffix_tokens.update(normalized_parts)

    return frozenset(removable_anywhere_tokens), frozenset(removable_suffix_tokens)


@lru_cache(maxsize=64)
def _derive_corpus_noise_suffix_tokens(jurisdiction_code: str | None) -> frozenset[str]:
    """Return the promoted, corpus-derived noise words for a jurisdiction, normalized.

    A caller with no jurisdiction information at all (``jurisdiction_code``
    is ``None``/blank -- the common case for callers that never opted into
    this feature) gets an empty set here, leaving `short_name` derivation
    exactly as it was before this feature existed. Only a caller that
    actually supplies a jurisdiction code (even one with no promoted list of
    its own) reaches ``get_corpus_noise_words()``, which selects that
    jurisdiction's own list or falls back to the cross-language ``global``
    entry for anything unmapped -- that is the "unknown" this falls back for,
    distinct from "no information supplied at all." Tokens are normalized the
    same way every other suffix-scope noise word is, so matching stays
    consistent regardless of which source contributed a token.
    """
    if not jurisdiction_code or not jurisdiction_code.strip():
        return frozenset()

    normalized: set[str] = set()
    for token in get_corpus_noise_words(jurisdiction_code):
        normalized_token = _normalize_company_type_value(token, transliterate=True)
        if normalized_token:
            normalized.add(normalized_token)
    return frozenset(normalized)


@lru_cache(maxsize=512)
def _derive_row_company_type_tokens(company_type: str | None) -> frozenset[str]:
    normalized_company_type = _normalize_company_type_value(
        company_type, transliterate=True
    )
    if not normalized_company_type:
        return frozenset()
    return frozenset(token for token in normalized_company_type.split() if token)


def _normalize_short_name_tokens(tokens: list[str]) -> list[str]:
    # Fast path for already-normalized tokens from earlier cleanse phases.
    if all(
        token.isascii() and token == token.lower() and "_" not in token
        for token in tokens
    ):
        return tokens.copy()

    return [
        _normalize_company_type_value(token, transliterate=True) for token in tokens
    ]


def _trim_trailing_company_type_tokens(
    tokens: list[str],
    normalized_tokens: list[str],
    row_company_type_tokens: frozenset[str],
) -> tuple[list[str], list[str]]:
    end_index = len(tokens) - 1
    while end_index >= 0 and end_index > 0:
        normalized_token = normalized_tokens[end_index]
        if not normalized_token or (
            normalized_token not in _SHORT_NAME_COMPANY_TYPE_COMPONENT_TOKENS
            and normalized_token not in row_company_type_tokens
        ):
            break
        end_index -= 1

    return tokens[: end_index + 1], normalized_tokens[: end_index + 1]


def _strip_geographic_stage(
    tokens: list[str],
    normalized_tokens: list[str],
    kinds: tuple[str, ...],
    *,
    company_type_trimmed: bool,
) -> tuple[list[str], list[str]]:
    """Run the geographic stage over already-company-type-trimmed short-name tokens.

    ``company_type_trimmed`` is the only evidence available that a trailing geographic
    term was a subsidiary marker rather than part of the name: `Oracle Ireland` (after
    `Ltd` came off) and `Air France` (which never had a legal form) are otherwise the
    same two tokens. So a row that actually lost a company-type token may strip down to
    one surviving token; every other row must leave at least two.
    """
    stripped = strip_geographic_terms(
        " ".join(tokens),
        kinds=kinds,
        min_remaining_tokens=1 if company_type_trimmed else 2,
    )
    stripped_tokens = (stripped or "").split()
    if not stripped_tokens or stripped_tokens == tokens:
        return tokens, normalized_tokens

    keep = len(stripped_tokens)
    if stripped_tokens == tokens[:keep]:
        return tokens[:keep], normalized_tokens[:keep]
    return stripped_tokens, _normalize_short_name_tokens(stripped_tokens)


def _remove_anywhere_noise_words_preserving_one(
    tokens: list[str],
    normalized_tokens: list[str],
    removable_anywhere_tokens: frozenset[str],
) -> tuple[list[str], list[str]]:
    filtered_tokens: list[str] = []
    filtered_normalized_tokens: list[str] = []
    remaining_count = len(tokens)

    for token, normalized_token in zip(tokens, normalized_tokens):
        if (
            normalized_token
            and normalized_token in removable_anywhere_tokens
            and remaining_count > 1
        ):
            remaining_count -= 1
            continue
        filtered_tokens.append(token)
        filtered_normalized_tokens.append(normalized_token)

    return filtered_tokens, filtered_normalized_tokens


def _trim_trailing_suffix_noise_words(
    tokens: list[str],
    normalized_tokens: list[str],
    removable_suffix_tokens: frozenset[str],
) -> list[str]:
    end_index = len(tokens) - 1
    while end_index >= 0 and end_index > 0:
        normalized_token = normalized_tokens[end_index]
        if not normalized_token or normalized_token not in removable_suffix_tokens:
            break
        end_index -= 1

    return tokens[: end_index + 1]


def _extract_leading_delimited_raw(text: str | None) -> str | None:
    """Phase 1: extract raw text from a leading delimited token, or None."""
    if not text:
        return None
    stripped = text.strip()
    m = _RE_LEADING_PAREN.match(stripped)
    if m:
        return m.group(1)
    quoted_raw = _extract_leading_quoted_raw(stripped)
    if quoted_raw is not None:
        return quoted_raw
    return None


def _extract_leading_quoted_raw(text: str | None) -> str | None:
    if not text:
        return None

    stripped = text.strip()
    match = _RE_LEADING_QUOTED.match(stripped)
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _extract_leading_delimited_tokens(text: str | None) -> str | None:
    raw = _extract_leading_delimited_raw(text)
    if raw is None:
        return None
    return _normalize_tokens(raw)


def _extract_leading_quoted_tokens(text: str | None) -> str | None:
    raw = _extract_leading_quoted_raw(text)
    if raw is None:
        return None
    return _normalize_tokens(raw)


def _extract_leading_token_struct(text: str | None) -> dict[str, str | None]:
    if not text:
        return {
            "lead_token": None,
            "lead_quoted_token": None,
        }

    stripped = text.strip()
    lead_raw: str | None = None
    quoted_raw: str | None = None

    m_paren = _RE_LEADING_PAREN.match(stripped)
    if m_paren:
        lead_raw = m_paren.group(1)
    else:
        m_quote = _RE_LEADING_QUOTED.match(stripped)
        if m_quote:
            quoted_raw = next(g for g in m_quote.groups() if g is not None)
            lead_raw = quoted_raw

    lead_token = _normalize_tokens(lead_raw) if lead_raw is not None else None
    lead_quoted_token = (
        _normalize_tokens(quoted_raw) if quoted_raw is not None else None
    )

    return {
        "lead_token": lead_token,
        "lead_quoted_token": lead_quoted_token,
    }


def _is_acronym_like_token(token: str | None) -> bool:
    if not token:
        return False

    normalized = (_normalize_tokens(token) or "").strip().lower()
    if not normalized or " " in normalized:
        return False
    return _RE_ACRONYM_LIKE.fullmatch(normalized) is not None


def _is_valid_short_name_token(token: str | None) -> bool:
    if not token:
        return False

    normalized = (_normalize_tokens(token) or "").strip().lower()
    if not normalized:
        return False

    alnum_count = sum(1 for char in normalized if char.isalnum())
    return alnum_count >= 2


def _derive_lead_token_decisions(
    stripped_name: str | None,
    special_short_name: str | None,
) -> dict[str, str | bool | None]:
    lead_parts = _extract_leading_token_struct(stripped_name)
    lead_token = lead_parts["lead_token"]
    lead_quoted_token = lead_parts["lead_quoted_token"]

    lead_token_valid = _is_valid_short_name_token(lead_token)
    lead_token_is_acronym = _is_acronym_like_token(lead_token)
    lead_quoted_token_valid = _is_valid_short_name_token(lead_quoted_token)

    short_name: str | None = None
    if special_short_name is not None:
        short_name = special_short_name
    elif (
        lead_token is not None
        and lead_token != ""
        and lead_token != "the"
        and lead_token_valid
        and (lead_quoted_token is None or lead_token_is_acronym)
    ):
        short_name = lead_token

    quoted_name: str | None = None
    if lead_quoted_token is not None and lead_quoted_token != "":
        quoted_name = lead_quoted_token

    return {
        "lead_token": lead_token,
        "lead_quoted_token": lead_quoted_token,
        "lead_token_valid": lead_token_valid,
        "lead_token_is_acronym": lead_token_is_acronym,
        "lead_quoted_token_valid": lead_quoted_token_valid,
        "short_name": short_name,
        "quoted_name": quoted_name,
    }


def _extract_quoted_type_parenthesized_parts(
    text: str | None,
    company_type_mapping: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    if not text:
        return (None, None, None)

    stripped = text.strip()
    m = _RE_QUOTED_TYPE_PAREN.match(stripped)
    if not m:
        return (None, None, None)

    quoted = next(g for g in m.groups()[:2] if g is not None)
    middle_type = m.group(3)
    full_name = m.group(4)

    normalized_acronym = (_normalize_tokens(quoted) or "").strip().lower()
    if not _is_acronym_like_token(normalized_acronym):
        return (None, None, None)

    canonical_type: str | None = None
    if middle_type and middle_type.strip():
        normalized_type = _normalize_company_type_value(middle_type, transliterate=True)
        canonical_type = company_type_mapping.get(normalized_type)
        if not canonical_type:
            return (None, None, None)

    normalized_full_name = (_normalize_suffix_surface(full_name) or "").strip()
    if not normalized_full_name or len(normalized_full_name.split()) < 2:
        return (None, None, None)

    return (normalized_acronym, canonical_type, full_name.strip())


def _extract_quoted_type_parenthesized_struct(
    text: str | None,
    company_type_mapping: dict[str, str],
) -> dict[str, str | None]:
    short_name, company_type, full_name = _extract_quoted_type_parenthesized_parts(
        text,
        company_type_mapping,
    )
    return {
        "special_short_name": short_name,
        "special_company_type": company_type,
        "special_full_name": full_name,
    }


def _extract_special_and_lead_struct(
    stripped_name: str | None,
    company_type_mapping: dict[str, str],
) -> dict[str, str | bool | None]:
    """Extract special-case fields and lead-token decisions in one pass."""
    special = _extract_quoted_type_parenthesized_struct(
        stripped_name, company_type_mapping
    )
    lead = _derive_lead_token_decisions(stripped_name, special["special_short_name"])
    return {
        "special_short_name": special["special_short_name"],
        "special_company_type": special["special_company_type"],
        "special_full_name": special["special_full_name"],
        "lead_token": lead["lead_token"],
        "lead_quoted_token_valid": lead["lead_quoted_token_valid"],
        "short_name": lead["short_name"],
        "quoted_name": lead["quoted_name"],
    }


def _split_personal_owner_suffix(
    text: str | None,
    owner_markers: tuple[str, ...] | list[str] | None,
    *,
    marker_regex: re.Pattern[str] | None = None,
) -> tuple[str | None, str | None]:
    if not text:
        return (text, None)

    stripped = re.sub(r"\s+", " ", text).strip()
    if not stripped:
        return (stripped, None)

    if not owner_markers:
        return (stripped, None)

    compiled_marker_regex = marker_regex or _compile_personal_owner_marker_regex(
        owner_markers
    )
    if compiled_marker_regex is None:
        return (stripped, None)

    match = compiled_marker_regex.search(stripped)
    if not match:
        return (stripped, None)

    business_name = stripped[: match.start()].rstrip(" ,;:-")
    owner_text = stripped[match.end() :].strip(" ,;:-")
    if not business_name or not owner_text:
        return (stripped, None)

    normalized_owner = _normalize_suffix_surface(owner_text, and_tokens=())
    normalized_owner = re.sub(r"\s+", " ", (normalized_owner or "")).strip()
    if not normalized_owner:
        return (stripped, None)

    return (business_name, normalized_owner)


def _split_personal_owner_struct(
    text: str | None,
    owner_markers: tuple[str, ...] | list[str] | None,
    *,
    marker_regex: re.Pattern[str] | None = None,
) -> dict[str, str | None]:
    business_name, personal_owner = _split_personal_owner_suffix(
        text,
        owner_markers,
        marker_regex=marker_regex,
    )
    return {
        "stripped_name": business_name,
        "personal_owner": personal_owner,
    }


def _normalize_personal_owner_markers(
    owner_markers: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not owner_markers:
        return ()

    normalized_markers: list[str] = []
    seen: set[str] = set()
    for marker in owner_markers:
        normalized = _normalize_tokens(marker, and_tokens=())
        normalized = (normalized or "").strip().lower()
        if normalized and normalized not in seen:
            normalized_markers.append(normalized)
            seen.add(normalized)

    return tuple(normalized_markers)


def _compile_personal_owner_marker_regex(
    owner_markers: tuple[str, ...] | list[str] | None,
) -> re.Pattern[str] | None:
    normalized_markers = _normalize_personal_owner_markers(owner_markers)
    if not normalized_markers:
        return None

    marker_pattern = "|".join(re.escape(marker) for marker in normalized_markers)
    return re.compile(rf"\b(?:{marker_pattern})\.?\b", flags=re.IGNORECASE)


def _replace_company_type_with_canonical(
    normalized_name: str | None,
    extracted_company_type: str | None,
    canonical_company_type: str | None,
) -> str | None:
    if not normalized_name:
        return normalized_name

    if not canonical_company_type or canonical_company_type.casefold() == "private":
        return normalized_name

    normalized_target = _normalize_company_type_value(
        canonical_company_type, transliterate=True
    )
    if normalized_target not in COMPANY_TYPE_TOKENS:
        return normalized_name

    raw = (extracted_company_type or "").strip()
    if not raw:
        return f"{normalized_name.strip()} {canonical_company_type}".strip()

    if normalized_name.endswith(raw):
        base = normalized_name[: len(normalized_name) - len(raw)].strip()
        return f"{base} {canonical_company_type}".strip()

    escaped = re.escape(raw).replace(r"\ ", r"\s+")
    replaced = re.sub(rf"\s*{escaped}\s*$", "", normalized_name).strip()
    return f"{replaced} {canonical_company_type}".strip()


def _initials_excluding_company_type(
    companyname: str | None,
    company_type: str | None,
) -> str | None:
    if not companyname:
        return None

    name_tokens = companyname.strip().split()
    if not name_tokens:
        return None

    name_tokens_normalized = [token.lower() for token in name_tokens]

    if company_type and company_type.casefold() != "private":
        type_tokens = [token.lower() for token in company_type.strip().split()]
        if type_tokens and len(name_tokens) >= len(type_tokens):
            tail = name_tokens_normalized[-len(type_tokens) :]
            if tail == type_tokens:
                name_tokens_normalized = name_tokens_normalized[: -len(type_tokens)]

    if not name_tokens_normalized:
        return None

    return "".join(token[0] for token in name_tokens_normalized if token)


def _derive_acronym(
    short_name: str | None,
    companyname: str | None,
    company_type: str | None,
) -> str | None:
    """Tag a raw `short_name` as the row's `acronym` only when it is provably one.

    Guard/intended constraint: `short_name` is a source-supplied field with no
    guarantee it is actually an acronym of the company name -- it may be a
    trading name, a legal-form token, or unrelated text. This only accepts it
    as `acronym` when it reproduces, letter for letter and case-insensitively,
    the initials of `companyname` with the legal form excluded
    (`_initials_excluding_company_type`). Any other `short_name` -- including
    one that happens to itself be a legal-form token such as `"Ltd"` -- is
    rejected here and left for `_ensure_non_acronym_short_name_in_cleansed`
    to decide whether it still belongs in `name_cleansed`.
    """
    if not short_name:
        return None

    candidate = short_name.strip()
    if not candidate:
        return None

    initials = _initials_excluding_company_type(companyname, company_type)
    if initials and initials.casefold() == candidate.casefold():
        return candidate.lower()
    return None


def _derive_short_name_from_cleansed(
    name_cleansed: str | None,
    company_type: str | None,
    noise_words: tuple[str | dict[str, str], ...] | list[str | dict[str, str]] | None,
    noise_word_sets: tuple[frozenset[str], frozenset[str]] | None = None,
    *,
    include_company_type: bool = True,
    include_noise_words: bool = True,
    geographic_tiers: tuple[str, ...] | None = None,
    jurisdiction_code: str | None = None,
) -> str | None:
    """Derive `short_name` from an already-cleansed name.

    ``jurisdiction_code``, when given, additionally pulls in the promoted,
    corpus-derived noise words for that jurisdiction (falling back to the
    ``global`` entry when unmapped/unknown) as extra suffix-scope
    noise words, on top of whatever ``noise_words`` already supplies. Only
    ever selects the sub-list matching the record's own jurisdiction -- never
    applies one language's corpus-derived words to another language's names.

    ``geographic_tiers``, when given, runs the geographic stage between the
    company-type and noise-word stages, stripping a trailing geographic term of
    those tiers. It sits there rather than in text normalization because the
    trailing arm cannot see the geographic term until the legal form is gone:
    `Oracle Ireland Ltd` presents `Ltd` in the trailing position first.
    """
    if not name_cleansed:
        return None

    tokens = [token for token in name_cleansed.strip().split() if token]
    if not tokens:
        return None

    # Common case: single-token short names are already final.
    if len(tokens) == 1:
        return tokens[0]

    normalized_tokens = _normalize_short_name_tokens(tokens)

    company_type_trimmed = False
    if include_company_type:
        row_company_type_tokens = _derive_row_company_type_tokens(company_type)
        trimmed_tokens, trimmed_normalized_tokens = _trim_trailing_company_type_tokens(
            tokens,
            normalized_tokens,
            row_company_type_tokens,
        )
        company_type_trimmed = len(trimmed_tokens) < len(tokens)
        tokens, normalized_tokens = trimmed_tokens, trimmed_normalized_tokens

    if geographic_tiers is not None:
        tokens, normalized_tokens = _strip_geographic_stage(
            tokens,
            normalized_tokens,
            geographic_tiers,
            company_type_trimmed=company_type_trimmed,
        )

    if include_noise_words:
        if noise_word_sets is None:
            removable_anywhere_tokens, removable_suffix_tokens = _build_noise_word_sets(
                noise_words
            )
        else:
            removable_anywhere_tokens, removable_suffix_tokens = noise_word_sets

        corpus_suffix_tokens = _derive_corpus_noise_suffix_tokens(jurisdiction_code)
        if corpus_suffix_tokens:
            removable_suffix_tokens = removable_suffix_tokens | corpus_suffix_tokens

        tokens, normalized_tokens = _remove_anywhere_noise_words_preserving_one(
            tokens,
            normalized_tokens,
            removable_anywhere_tokens,
        )
        tokens = _trim_trailing_suffix_noise_words(
            tokens,
            normalized_tokens,
            removable_suffix_tokens,
        )

    if not tokens:
        return None
    return " ".join(tokens)


def _ensure_non_acronym_short_name_in_cleansed(
    short_name: str | None,
    acronym: str | None,
    cleansed_name: str | None,
) -> str | None:
    """Restore a legal-form `short_name` that `_derive_acronym` rejected.

    Guard/intended constraint: this only handles the one edge case where a
    source `short_name` is itself a bare legal-form token (normalizes to an
    entry in `COMPANY_TYPE_TOKENS`, e.g. `"Ltd"`, `"SA"`) -- a value
    `_derive_acronym` never accepts, since a legal form is never a genuine
    acronym of the name. Without this guard that token would simply be
    dropped, losing a real signal the source data supplied. It fires only
    when `acronym` is falsy (a real acronym already covers the short-name
    signal, so nothing more to restore) and only prepends the token when
    `cleansed_name` does not already start with it, so it never duplicates.
    Any other `short_name` (not a legal-form token) is left untouched here --
    that is a distinct name, not this guard's concern.
    """
    if not cleansed_name:
        return cleansed_name

    if not short_name or acronym:
        return cleansed_name

    normalized_short_name = _normalize_company_type_value(
        short_name, transliterate=True
    )
    if not normalized_short_name:
        return cleansed_name

    if normalized_short_name not in COMPANY_TYPE_TOKENS:
        return cleansed_name

    token = short_name.strip().lower()
    normalized_name = cleansed_name.strip().lower()
    if normalized_name == token or normalized_name.startswith(f"{token} "):
        return cleansed_name

    return f"{token} {cleansed_name}".strip()


def _ensure_quoted_name_in_cleansed(
    quoted_name: str | None,
    acronym: str | None,
    cleansed_name: str | None,
) -> str | None:
    """Restore a leading quoted brand generic cleansing dropped from `name_cleansed`.

    Guard/intended constraint: `quoted_name` is the brand captured from a
    leading quoted segment and is prepended to `name_cleansed` so that signal
    survives cleansing, unless `cleansed_name` already starts with it. It is
    skipped when `quoted_name` equals `acronym` (case/normalization-
    insensitively): a case such as `"IBM" International Business Machines
    Ltd` already carries the same token as the derived `acronym`, so
    prefixing it again in `name_cleansed` would duplicate, not add,
    information.
    """
    if not cleansed_name:
        return cleansed_name

    if not quoted_name:
        return cleansed_name

    token = quoted_name.strip().lower()
    if not token:
        return cleansed_name

    normalized_acronym = _normalize_company_type_value(acronym, transliterate=True)
    if normalized_acronym and normalized_acronym == token:
        return cleansed_name

    normalized_name = cleansed_name.strip().lower()
    if normalized_name == token or normalized_name.startswith(f"{token} "):
        return cleansed_name

    return f"{token} {cleansed_name}".strip()


def _recombine_quoted_name_short_name(
    quoted_name: str | None,
    short_name: str | None,
    company_type: str | None,
    cleansed_name: str | None,
) -> str | None:
    normalized_short_name = _normalize_company_type_value(
        short_name, transliterate=True
    )
    if not normalized_short_name:
        return short_name

    normalized_quoted_name = _normalize_company_type_value(
        quoted_name, transliterate=True
    )
    if not normalized_quoted_name or normalized_short_name.startswith(
        f"{normalized_quoted_name} "
    ):
        return short_name
    if normalized_short_name == normalized_quoted_name:
        return short_name

    normalized_company_type = _normalize_company_type_value(
        company_type, transliterate=True
    )
    normalized_cleansed_name = _normalize_company_type_value(
        cleansed_name, transliterate=True
    )
    if not normalized_company_type or not normalized_cleansed_name:
        return short_name
    if normalized_cleansed_name == normalized_company_type:
        return short_name
    if normalized_cleansed_name.endswith(f" {normalized_company_type}"):
        normalized_cleansed_name = normalized_cleansed_name[
            : -(len(normalized_company_type) + 1)
        ]

    if not normalized_cleansed_name.startswith(f"{normalized_quoted_name} "):
        return short_name
    if not normalized_cleansed_name.endswith(normalized_short_name):
        return short_name

    return normalized_cleansed_name
