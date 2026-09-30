"""`CleanseConfig`, and the noise words short-name derivation strips.

Noise words are read at three levels of assembly: `get_manual_noise_words` (the hand-kept list,
by `suffix` and `anywhere` scope), `get_profiled_noise_words` (the corpus-derived `strict`,
`balanced` or `aggressive` profile), and `get_effective_noise_words`, the two combined into the
tuple derivation uses by default. A caller customizing noise words starts from the last, edits
it, and passes it back as `noise_words`. The profiled lists are promoted from
`company_tokenize`'s noise-word workflow, not hand-authored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

_PROFILE_ORDER = ("strict", "balanced", "aggressive")


def _append_unique_stripped_tokens(
    ordered_tokens: list[str],
    seen: set[str],
    values: list[Any] | None,
) -> None:
    if not isinstance(values, list):
        return

    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            ordered_tokens.append(token)
            seen.add(token)


def _profile_tokens_values(profile_payload: dict[str, Any]) -> list[Any] | None:
    token_values = profile_payload.get("tokens")
    if isinstance(token_values, list):
        return token_values

    legacy_values = profile_payload.get("tfidf_descriptor")
    if isinstance(legacy_values, list):
        return legacy_values

    return None


def _load_manual_noise_words_by_scope() -> dict[str, tuple[str, ...]]:
    tokens_path = files("company_cleanse.resources").joinpath("manual_noise_words.json")
    raw_entries = json.loads(tokens_path.read_text(encoding="utf-8"))

    if not isinstance(raw_entries, dict):
        raise TypeError(
            "manual_noise_words.json must be an object with 'suffix' and 'anywhere' lists."
        )

    # 'provenance' is optional, documentary metadata -- it is not
    # a token scope and carries no runtime meaning beyond recording why this
    # file is hand-curated rather than generated.
    unknown_keys = set(raw_entries) - {"suffix", "anywhere", "provenance"}
    if unknown_keys:
        raise ValueError(
            "manual_noise_words.json only supports 'suffix', 'anywhere', and 'provenance' keys."
        )

    suffix_entries = raw_entries.get("suffix", [])
    anywhere_entries = raw_entries.get("anywhere", [])
    if not isinstance(suffix_entries, list) or not isinstance(anywhere_entries, list):
        raise TypeError(
            "manual_noise_words.json 'suffix' and 'anywhere' values must be lists."
        )

    if not all(isinstance(entry, str) for entry in suffix_entries):
        raise ValueError(
            "manual_noise_words.json 'suffix' list entries must be strings."
        )
    if not all(isinstance(entry, str) for entry in anywhere_entries):
        raise ValueError(
            "manual_noise_words.json 'anywhere' list entries must be strings."
        )

    normalized_suffix = tuple(
        entry.strip() for entry in suffix_entries if entry.strip()
    )
    normalized_anywhere = tuple(
        entry.strip() for entry in anywhere_entries if entry.strip()
    )

    return {
        "suffix": normalized_suffix,
        "anywhere": normalized_anywhere,
    }


def validate_profiled_noise_words_payload(
    payload: Any, *, source_label: str = "noise_words.json"
) -> dict[str, Any]:
    """Validate a candidate profiled-noise-word payload's shape.

    This is the same shape check ``get_profiled_noise_words()`` implicitly
    relies on at runtime (a top-level object with a ``profiles`` object, whose
    per-profile entries carry a ``tokens``/``tfidf_descriptor`` list of
    strings and an optional ``seed_legal`` list of strings) -- the exact shape
    ``scripts/generate_noise_words.py`` writes to
    ``artifacts/tokenizers/work/<system>/noise_words.json``.

    Reused both by the packaged-resource loader (against the file actually
    shipped in ``company_cleanse.resources``) and by
    ``scripts/promote_noise_words.py`` (against a promotion candidate file,
    before it is copied over the packaged resource) so the two never drift
    apart.

    Raises ``TypeError``/``ValueError`` describing the first shape problem
    found, using ``source_label`` to identify which file is being checked.
    Returns ``payload`` unchanged on success, for convenient call chaining.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"{source_label} must be an object.")

    profiles_obj = payload.get("profiles")
    if not isinstance(profiles_obj, dict):
        raise TypeError(f"{source_label} must include an object field 'profiles'.")

    if not any(key in profiles_obj for key in _PROFILE_ORDER):
        raise ValueError(
            f"{source_label} 'profiles' must include at least one of "
            f"{', '.join(_PROFILE_ORDER)}."
        )

    for profile_key, profile_payload in profiles_obj.items():
        if not isinstance(profile_payload, dict):
            raise TypeError(
                f"{source_label} profile '{profile_key}' must be an object."
            )

        token_values = _profile_tokens_values(profile_payload)
        if token_values is not None and not all(
            isinstance(value, str) for value in token_values
        ):
            raise ValueError(
                f"{source_label} profile '{profile_key}' 'tokens'/'tfidf_descriptor' "
                "entries must be strings."
            )

        seed_legal_values = profile_payload.get("seed_legal")
        if seed_legal_values is not None and (
            not isinstance(seed_legal_values, list)
            or not all(isinstance(value, str) for value in seed_legal_values)
        ):
            raise ValueError(
                f"{source_label} profile '{profile_key}' 'seed_legal' must be a "
                "list of strings."
            )

    return payload


def summarize_profiled_noise_word_counts(payload: dict[str, Any]) -> dict[str, int]:
    """Return each present profile's token count.

    Counts ``tokens``/``tfidf_descriptor`` plus ``seed_legal``, with deduplication not
    applied: a simple size summary, not the resolved token set itself.

    Intended for reviewing a promotion candidate against the currently-packaged file
    (``scripts/promote_noise_words.py``) without needing to know this module's private
    payload-shape helpers. Assumes ``payload`` has already passed
    ``validate_profiled_noise_words_payload()``.
    """
    profiles_obj = payload.get("profiles")
    if not isinstance(profiles_obj, dict):
        return {}

    counts: dict[str, int] = {}
    for profile_key, profile_payload in profiles_obj.items():
        if not isinstance(profile_payload, dict):
            continue
        token_values = _profile_tokens_values(profile_payload)
        seed_legal_values = profile_payload.get("seed_legal")
        count = len(token_values) if isinstance(token_values, list) else 0
        count += len(seed_legal_values) if isinstance(seed_legal_values, list) else 0
        counts[profile_key] = count
    return counts


def _load_profiled_noise_words_payload() -> dict[str, Any]:
    tokens_path = files("company_cleanse.resources").joinpath("noise_words.json")
    payload = json.loads(tokens_path.read_text(encoding="utf-8"))
    return validate_profiled_noise_words_payload(
        payload, source_label="noise_words.json"
    )


def _resolve_profile_cumulative_tokens(
    *, profiles_obj: dict[str, Any], profile_key: str
) -> tuple[str, ...]:
    if profile_key in _PROFILE_ORDER:
        cutoff = _PROFILE_ORDER.index(profile_key)
        included_profiles = _PROFILE_ORDER[: cutoff + 1]
    else:
        included_profiles = (profile_key,)

    ordered_tokens: list[str] = []
    seen: set[str] = set()

    for key in included_profiles:
        profile_payload = profiles_obj.get(key)
        if not isinstance(profile_payload, dict):
            continue

        _append_unique_stripped_tokens(
            ordered_tokens,
            seen,
            profile_payload.get("seed_legal"),
        )
        _append_unique_stripped_tokens(
            ordered_tokens,
            seen,
            _profile_tokens_values(profile_payload),
        )

    return tuple(ordered_tokens)


def get_profiled_noise_words(
    *,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> tuple[str, ...]:
    """Return profiled noise words from ``noise_words.json``.

    ``noise_words_set_kind='combined'`` accumulates up to the selected profile
    using strict->balanced->aggressive ordering.
    """
    profiles_obj = _DEFAULT_PROFILED_NOISE_WORDS_PAYLOAD.get("profiles")
    if not isinstance(profiles_obj, dict):
        raise TypeError(
            "Profiled noise-word payload must include an object field 'profiles'."
        )

    profile_key = (noise_words_profile or "aggressive").strip().lower()
    set_kind_key = (noise_words_set_kind or "combined").strip().lower()

    profile_payload = profiles_obj.get(profile_key)
    if not isinstance(profile_payload, dict):
        available_profiles = ", ".join(sorted(str(key) for key in profiles_obj))
        raise TypeError(
            f"Noise-word profile '{profile_key}' not found. Available profiles: {available_profiles}."
        )

    if set_kind_key == "combined":
        return _resolve_profile_cumulative_tokens(
            profiles_obj=profiles_obj, profile_key=profile_key
        )

    values = profile_payload.get(set_kind_key)
    if not isinstance(values, list):
        if set_kind_key == "tokens":
            legacy_values = profile_payload.get("tfidf_descriptor")
            if isinstance(legacy_values, list):
                values = legacy_values
        if not isinstance(values, list):
            available_sets = ", ".join(sorted(str(key) for key in profile_payload))
            raise TypeError(
                f"Noise-word set kind '{set_kind_key}' not found in profile '{profile_key}'. "
                f"Available set kinds: {available_sets}."
            )

    normalized_values = [str(value).strip() for value in values if str(value).strip()]
    return tuple(normalized_values)


def get_effective_noise_words(
    *,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> tuple[str | dict[str, str], ...]:
    """Return effective noise words combining the manual list and the profiled defaults.

    The manual noise words from ``manual_noise_words.json`` are always included.
    Profiled tokens from ``noise_words.json`` are appended as suffix-scope tokens.
    """
    combined_entries: list[str | dict[str, str]] = list(MANUAL_NOISE_WORDS)

    existing_suffix_tokens: set[str] = set()
    for entry in combined_entries:
        if isinstance(entry, dict):
            scope = str(entry.get("scope", "suffix")).strip().lower()
            if scope != "suffix":
                continue
            token = str(entry.get("token", "")).strip().lower()
        else:
            token = str(entry).strip().lower()
        if token:
            existing_suffix_tokens.add(token)

    for token in get_profiled_noise_words(
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    ):
        normalized = token.strip()
        normalized_key = normalized.lower()
        if not normalized or normalized_key in existing_suffix_tokens:
            continue
        combined_entries.append(normalized)
        existing_suffix_tokens.add(normalized_key)

    return tuple(combined_entries)


def get_manual_noise_words() -> dict[str, tuple[str, ...]]:
    """Return the immutable manual noise words grouped by scope."""
    return {
        "suffix": _MANUAL_NOISE_WORDS_BY_SCOPE["suffix"],
        "anywhere": _MANUAL_NOISE_WORDS_BY_SCOPE["anywhere"],
    }


def _load_manual_noise_words() -> tuple[str | dict[str, str], ...]:
    scoped_tokens = _MANUAL_NOISE_WORDS_BY_SCOPE
    entries: list[str | dict[str, str]] = []
    entries.extend(scoped_tokens["suffix"])
    entries.extend(
        {"token": entry, "scope": "anywhere"} for entry in scoped_tokens["anywhere"]
    )

    normalized_entries: list[str | dict[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):
            token = entry.strip()
            if token:
                normalized_entries.append(token)
            continue

        if isinstance(entry, dict):
            token_raw: Any = entry.get("token")
            token = str(token_raw).strip() if token_raw is not None else ""
            if not token:
                continue
            scope = str(entry.get("scope", "suffix")).strip().lower()
            if scope not in {"suffix", "anywhere"}:
                raise ValueError(
                    "noise_words.json entry scope must be 'suffix' or 'anywhere'."
                )
            normalized_entries.append({"token": token, "scope": scope})
            continue

        raise ValueError("noise_words.json entries must be strings or objects.")

    return tuple(normalized_entries)


_MANUAL_NOISE_WORDS_BY_SCOPE = _load_manual_noise_words_by_scope()
_DEFAULT_PROFILED_NOISE_WORDS_PAYLOAD = _load_profiled_noise_words_payload()
MANUAL_NOISE_WORDS: tuple[str | dict[str, str], ...] = _load_manual_noise_words()


@dataclass(slots=True)
class CleanseConfig:
    """Public configuration for deterministic company-name cleansing.

    This dataclass controls the supported batch-cleansing knobs exposed by the
    package, including input column selection, normalization tokens, company-type
    matching, and optional source company-type mapping.

    Attributes:
        company_col: Name of the input column holding the raw company name.
        and_tokens: Tokens treated as equivalent to "and" during normalization
            (for example ``("AND", "&")``). ``None`` uses the built-in default set.
        personal_owner_markers: Tokens that mark an embedded owner name to split
            into ``personal_owner``. ``None`` disables owner-name splitting.
        char_whitelist: Regex of characters to strip when building
            ``name_cleansed_basic``. Matched characters are removed.
        company_type_regex: Override regex for company-type suffix matching.
            Must be paired with ``company_type_mapping``; otherwise packaged
            defaults from ``get_company_type_rules()`` are used for both.
        company_type_mapping: Override canonical company-type mapping paired with
            ``company_type_regex``. See ``company_type_regex``.
        source_company_type_col: Optional input column providing a pre-existing
            company-type value to prefer over suffix-derived detection.
        source_company_type_mapping: Mapping from ``source_company_type_col``
            values to canonical company types. Required for
            ``source_company_type_col`` to take effect.
        company_type_matcher: Matching strategy for company-type suffix
            detection. One of ``"trie"`` (default) or ``"regex"``.
        normalization_profile: Name of the normalization operation chain to
            apply, using the composable profile syntax described in the
            package README (for example ``"default"`` or
            ``"default|-diacritics"``).
        noise_words_profile: Short-name noise-word strictness profile. One of
            ``"strict"``, ``"balanced"``, or ``"aggressive"`` (default); see
            ``get_profiled_noise_words()``.
        noise_words_set_kind: Which token set to draw from a noise-word
            profile. ``"combined"`` (default) accumulates
            strict->balanced->aggressive up to ``noise_words_profile``; other
            values select a single named set within that profile.
        short_name_profile: Controls which `short_name` derivation stages run
            for rows that fall back to noise-word-based derivation (rows
            resolved via the quoted/parenthesized special case are unaffected).
            Accepts ``"default"`` (both stages on, matching prior behavior) plus
            optional ``-company_type``/``-noise_words`` toggles, using the same
            syntax as ``strip_company_suffix()``'s ``normalization_profile``
            (for example ``"default|-noise_words"``). A third stage,
            ``geographic_terms``, is off unless named and strips a trailing
            geographic term once the legal form is gone, with ``+region``/``+city``
            adding tiers to its default ``country`` tier (for example
            ``"default|geographic_terms|+region"``). See the package README.
        jurisdiction_col: Optional input column providing a per-row
            jurisdiction code (for example ``"gb"``, ``"fr"``, ``"de"``), used
            to select the promoted, corpus-derived noise-word list for that
            jurisdiction during `short_name` derivation. Only takes
            effect when this column is actually present in the input;
            defaults to the standard ``"jurisdiction_code"`` contract column
            name already used across this repo's acquisition pipeline. Pass
            ``None`` to disable jurisdiction-aware noise-word selection
            entirely (every row falls back to the ``global`` entry, same as
            an unmapped/unknown jurisdiction).
        step_engines: Optional per-step override mapping selecting an
            alternate step-engine implementation (for example native Polars
            vs. UDF-backed) by step name. ``None`` uses packaged defaults.
    """

    company_col: str = "company_name"
    and_tokens: tuple[str, ...] | None = None
    personal_owner_markers: tuple[str, ...] | None = None
    char_whitelist: str = r"[^a-z0-9\s!&]"
    company_type_regex: str | None = None
    company_type_mapping: dict[str, str] | None = None
    source_company_type_col: str | None = None
    source_company_type_mapping: dict[str, str] | None = None
    company_type_matcher: str = "trie"
    normalization_profile: str = "default"
    noise_words_profile: str = "aggressive"
    noise_words_set_kind: str = "combined"
    short_name_profile: str = "default"
    jurisdiction_col: str | None = "jurisdiction_code"
    step_engines: dict[str, str] | None = None
