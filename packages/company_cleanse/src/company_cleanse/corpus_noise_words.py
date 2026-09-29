"""Jurisdiction-scoped noise words derived from each corpus, supplementing the packaged lists.

Read from `resources/short_name_noise_word_candidates.json`, keyed per system or language with a
`global` fallback, and merged into short-name derivation only when a jurisdiction is supplied
(`CleanseConfig.jurisdiction_col`, or `strip_company_suffix(..., jurisdiction_code=)`).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

_RESOURCE_FILENAME = "short_name_noise_word_candidates.json"

_GLOBAL_SYSTEM_KEY = "global"

# Closed, explicit jurisdiction_code -> promoted-artifact system-key mapping.
# Deliberately not a general-purpose jurisdiction/language table -- this only
# needs to cover the systems company_tokenize actually promoted (see
# packages/company_tokenize/src/company_tokenize/resources/noise_word_candidates/),
# and stays a closed vocabulary rather than an open heuristic. A
# jurisdiction_code with no entry here falls back to "global" -- see
# _resolve_noise_word_system_key().
_JURISDICTION_TO_NOISE_WORD_SYSTEM: dict[str, str] = {
    "gb": "gb",
    "fr": "fr",
    "ie": "ie",
    "de": "offeneregister",
}

# Conservative cutoff for *this* consumer (short_name derivation wants to
# avoid stripping anything load-bearing), deliberately smaller than company_tokenize's
# own DEFAULT_MAX_NOISE_WORD_CANDIDATES=200 export bound. Chosen by direct
# inspection of the promoted candidate lists (2026-08-30): across all five
# promoted systems (fr/gb/ie/offeneregister/global), ranks 0-29 are
# consistently generic legal-form/business-descriptor/country-branch-tag
# words (e.g. "sarl", "holdings", "consulting", "france", "deutschland").
# Personal and place names that would be unsafe to strip start appearing
# from rank ~30 onward and get materially more frequent past it (French
# "jean" at rank 37, "paris" at rank 36; Irish "dublin" at rank 40; German
# "hans" at rank 42) -- 30 sits just below where that risk starts, not an
# arbitrary round number.
DEFAULT_SHORT_NAME_NOISE_WORD_MAX_TOKENS = 30


def _require_short_name_noise_word_candidates_shape(
    payload: Any, *, source_label: str
) -> dict[str, Any]:
    """Validate a short-name noise-word candidates payload's shape.

    This is the exact shape ``scripts/promote_short_name_noise_words.py``
    writes (a top-level object with a ``systems`` object, each entry a
    ``{candidates: [{token, document_frequency_pct, idf}, ...]}`` record --
    the same per-candidate shape
    ``company_tokenize.noise_export.validate_noise_word_candidates_payload``
    checks, just nested one level deeper under a system key). Reused by both
    the packaged-resource loader and the promotion script so the two can't
    silently drift.

    Raises ``TypeError``/``ValueError`` describing the first shape problem
    found. Returns ``payload`` unchanged on success.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"{source_label} must be an object.")

    systems = payload.get("systems")
    if not isinstance(systems, dict) or not systems:
        raise ValueError(
            f"{source_label} must include a non-empty object field 'systems'."
        )

    for system_key, system_payload in systems.items():
        if not isinstance(system_payload, dict):
            raise TypeError(
                f"{source_label} systems[{system_key!r}] must be an object."
            )

        candidates = system_payload.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError(
                f"{source_label} systems[{system_key!r}] must include a "
                "'candidates' list."
            )

        for index, entry in enumerate(candidates):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"{source_label} systems[{system_key!r}].candidates[{index}] "
                    "must be an object."
                )
            token = entry.get("token")
            if not isinstance(token, str) or not token:
                raise ValueError(
                    f"{source_label} systems[{system_key!r}].candidates[{index}] "
                    "must include a non-empty 'token' string."
                )

            df_pct = entry.get("document_frequency_pct")
            if not isinstance(df_pct, (int, float)) or isinstance(df_pct, bool):
                raise TypeError(
                    f"{source_label} systems[{system_key!r}].candidates[{index}] "
                    "must include a numeric 'document_frequency_pct'."
                )
            if not (0.0 <= float(df_pct) <= 1.0):
                raise ValueError(
                    f"{source_label} systems[{system_key!r}].candidates[{index}] "
                    "'document_frequency_pct' must be between 0 and 1."
                )

            idf = entry.get("idf")
            if not isinstance(idf, (int, float)) or isinstance(idf, bool):
                raise TypeError(
                    f"{source_label} systems[{system_key!r}].candidates[{index}] "
                    "must include a numeric 'idf'."
                )

    if _GLOBAL_SYSTEM_KEY not in systems:
        raise ValueError(
            f"{source_label} 'systems' must include a '{_GLOBAL_SYSTEM_KEY}' entry "
            "-- it is the required fallback for an unknown/unmapped jurisdiction."
        )

    return payload


def validate_short_name_noise_word_candidates_payload(
    payload: Any, *, source_label: str = "short_name_noise_word_candidates.json"
) -> dict[str, Any]:
    """Public wrapper for `_require_short_name_noise_word_candidates_shape`.

    Reused by ``scripts/promote_short_name_noise_words.py`` to validate a
    promotion candidate before it overwrites the packaged resource file.
    """
    return _require_short_name_noise_word_candidates_shape(
        payload, source_label=source_label
    )


@lru_cache(maxsize=1)
def _load_short_name_noise_word_candidates_payload() -> dict[str, Any]:
    resource_path = files("company_cleanse.resources").joinpath(_RESOURCE_FILENAME)
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    return _require_short_name_noise_word_candidates_shape(
        payload, source_label=_RESOURCE_FILENAME
    )


def _resolve_noise_word_system_key(jurisdiction_code: str | None) -> str:
    normalized = (jurisdiction_code or "").strip().lower()
    return _JURISDICTION_TO_NOISE_WORD_SYSTEM.get(normalized, _GLOBAL_SYSTEM_KEY)


@lru_cache(maxsize=64)
def get_corpus_noise_words(
    jurisdiction_code: str | None = None,
    *,
    max_tokens: int = DEFAULT_SHORT_NAME_NOISE_WORD_MAX_TOKENS,
) -> tuple[str, ...]:
    """Return the promoted, corpus-derived noise words for a jurisdiction.

    Selects the sub-list matching ``jurisdiction_code`` from the packaged
    candidate artifact (``resources/short_name_noise_word_candidates.json``,
    promoted from ``company_tokenize``'s noise-word candidate export), keyed
    per system/language plus a ``global`` entry -- never mixes one language's
    list into another's. An unmapped or ``None`` jurisdiction_code falls back
    to the ``global`` entry: a pooled, cross-language ranking, not any single
    language's words.

    Tokens are already ranked highest-document-frequency/lowest-IDF first in
    the source artifact; ``max_tokens`` bounds how far down that ranking this
    conservative caller reads (see ``DEFAULT_SHORT_NAME_NOISE_WORD_MAX_TOKENS``
    for why 30, specifically, is the default).
    """
    payload = _load_short_name_noise_word_candidates_payload()
    systems = payload["systems"]
    system_key = _resolve_noise_word_system_key(jurisdiction_code)
    system_payload = systems.get(system_key) or systems.get(_GLOBAL_SYSTEM_KEY)
    if system_payload is None:
        return ()

    candidates = system_payload.get("candidates", [])
    bounded = candidates[: max(max_tokens, 0)]
    tokens: list[str] = []
    seen: set[str] = set()
    for entry in bounded:
        if not isinstance(entry, dict):
            continue
        token = str(entry.get("token", "")).strip().lower()
        if token and token not in seen:
            tokens.append(token)
            seen.add(token)
    return tuple(tokens)
