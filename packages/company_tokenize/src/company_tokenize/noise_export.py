"""A bounded, scored list of noise-word candidates, for a consumer choosing its own cutoff.

Where `tfidf`'s profiles are pre-thresholded at strict, balanced and aggressive, this
keeps each token's `document_frequency_pct` and `idf` for the lowest-idf tokens, so a
consumer applies its own threshold. `validate_noise_word_candidates_payload()` checks
the shape both on export and before promotion into this package's
`resources/noise_word_candidates/<system>.json`.
"""

from __future__ import annotations

from typing import Any, TypedDict

import polars as pl

# Cap chosen from the measured score distribution (2026-08-30), not an
# arbitrary round number: `document_frequency_pct`/`idf` are strongly
# rank-correlated (idf is a monotone decreasing function of document
# frequency), so a smoothed slope of idf-per-rank over the lowest-idf
# candidates flattens (drops below 0.003 idf/rank, sustained) somewhere
# between rank ~175 (global) and ~250 (fr) across every system with a
# persisted `token_tfidf_stats.parquet` at the time of writing (ie/gb/fr/
# global/offeneregister). 200 sits inside that observed flattening band for
# every system checked, comfortably above the existing `aggressive`
# noise-word profile's token count (25-60, see
# `scripts/generate_noise_words.py`) so this artifact is a genuinely richer
# candidate pool than the pre-thresholded profiles, while still small enough
# to review in a diff (a few hundred short JSON records).
DEFAULT_MAX_NOISE_WORD_CANDIDATES = 200

_REQUIRED_STATS_COLUMNS = {"token", "document_frequency_pct", "idf"}


class NoiseWordCandidateSelection(TypedDict):
    max_candidates: int
    candidate_count: int
    min_token_length: int
    source_token_count: int
    document_frequency_pct_cutoff: float | None
    idf_cutoff: float | None


class NoiseWordCandidate(TypedDict):
    token: str
    document_frequency_pct: float
    idf: float


class NoiseWordCandidatesPayload(TypedDict):
    generated_utc: str
    system: str
    source_stats_path: str
    tfidf_use_case: str
    tfidf_namespace: str
    selection: NoiseWordCandidateSelection
    candidates: list[NoiseWordCandidate]


def _require_stats_columns(stats: pl.DataFrame) -> None:
    missing = sorted(_REQUIRED_STATS_COLUMNS - set(stats.columns))
    if missing:
        raise ValueError(f"stats is missing required columns: {', '.join(missing)}")


def select_noise_word_candidates(
    stats: pl.DataFrame,
    *,
    max_candidates: int = DEFAULT_MAX_NOISE_WORD_CANDIDATES,
    min_token_length: int = 2,
) -> pl.DataFrame:
    """Select the N lowest-IDF (highest document-frequency) noise-word candidates.

    Bounds the exported artifact to ``max_candidates`` rows, ranked by
    ``document_frequency_pct`` descending / ``idf`` ascending (the same
    ranking convention ``generate_noise_words.py`` already uses to build its
    own strict/balanced/aggressive profiles from the same source stats) --
    the highest document-frequency, lowest-information tokens sort first.

    Unlike ``select_stopword_candidates``, this applies no DF%/IDF threshold
    of its own: it keeps the continuous score for every selected token so a
    downstream consumer can apply its own cutoff, and bounds the *count*
    rather than a threshold so the exported artifact stays small regardless
    of corpus size.
    """
    if max_candidates < 0:
        raise ValueError("max_candidates must be >= 0.")
    if min_token_length <= 0:
        raise ValueError("min_token_length must be greater than zero.")
    _require_stats_columns(stats)

    if stats.height == 0 or max_candidates == 0:
        return stats.clear()

    filtered = stats.filter(
        pl.col("token").str.len_chars() >= pl.lit(min_token_length)
    ).sort(["document_frequency_pct", "idf", "token"], descending=[True, False, False])
    return filtered.head(max_candidates)


def build_noise_word_candidates_payload(
    stats: pl.DataFrame,
    *,
    system: str,
    source_stats_path: str,
    generated_utc: str,
    max_candidates: int = DEFAULT_MAX_NOISE_WORD_CANDIDATES,
    min_token_length: int = 2,
) -> NoiseWordCandidatesPayload:
    """Build the bounded, scored noise-word candidate export payload.

    Reuses the persisted corpus-token TF-IDF stats
    (``token_tfidf_stats.parquet``, produced by ``generate_noise_words.py``)
    rather than recomputing TF-IDF from the raw corpus -- this is a slice of
    an artifact that already exists, not a new computation.
    """
    from .tfidf import expected_tfidf_namespace

    normalized_system = system.strip().lower()
    selected = select_noise_word_candidates(
        stats, max_candidates=max_candidates, min_token_length=min_token_length
    )

    candidates: list[NoiseWordCandidate] = [
        {
            "token": str(row["token"]),
            "document_frequency_pct": float(row["document_frequency_pct"]),
            "idf": float(row["idf"]),
        }
        for row in selected.select(
            ["token", "document_frequency_pct", "idf"]
        ).iter_rows(named=True)
    ]

    selection: NoiseWordCandidateSelection = {
        "max_candidates": int(max_candidates),
        "candidate_count": len(candidates),
        "min_token_length": int(min_token_length),
        "source_token_count": int(stats.height),
        "document_frequency_pct_cutoff": (
            candidates[-1]["document_frequency_pct"] if candidates else None
        ),
        "idf_cutoff": candidates[-1]["idf"] if candidates else None,
    }

    return {
        "generated_utc": generated_utc,
        "system": normalized_system,
        "source_stats_path": source_stats_path,
        "tfidf_use_case": "corpus",
        "tfidf_namespace": expected_tfidf_namespace(use_case="corpus"),
        "selection": selection,
        "candidates": candidates,
    }


def validate_noise_word_candidates_payload(
    payload: Any, *, source_label: str = "noise_word_candidates.json"
) -> dict[str, Any]:
    """Validate a candidate noise-word export payload's shape.

    This is the exact shape ``build_noise_word_candidates_payload()``
    writes: a top-level object with a ``system`` string and a ``candidates``
    list of ``{token, document_frequency_pct, idf}`` records. Reused by both
    ``scripts/export_noise_word_candidates.py`` (defensively, right after
    building the payload) and ``scripts/promote_noise_word_candidates.py``
    (against a promotion candidate file, before it is copied into the
    packaged resource location) so the two can't silently drift -- mirrors
    ``company_cleanse.config.validate_profiled_noise_words_payload``'s role
    for the sibling noise-word artifact.

    Raises ``TypeError``/``ValueError`` describing the first shape problem
    found. Returns ``payload`` unchanged on success, for convenient call
    chaining.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"{source_label} must be an object.")

    system = payload.get("system")
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"{source_label} must include a non-empty 'system' string.")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError(f"{source_label} must include a 'candidates' list.")

    for index, entry in enumerate(candidates):
        if not isinstance(entry, dict):
            raise TypeError(f"{source_label} candidates[{index}] must be an object.")

        token = entry.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(
                f"{source_label} candidates[{index}] must include a non-empty 'token' string."
            )

        df_pct = entry.get("document_frequency_pct")
        if not isinstance(df_pct, (int, float)) or isinstance(df_pct, bool):
            raise TypeError(
                f"{source_label} candidates[{index}] must include a numeric "
                "'document_frequency_pct'."
            )
        if not (0.0 <= float(df_pct) <= 1.0):
            raise ValueError(
                f"{source_label} candidates[{index}] 'document_frequency_pct' must be "
                "between 0 and 1."
            )

        idf = entry.get("idf")
        if not isinstance(idf, (int, float)) or isinstance(idf, bool):
            raise TypeError(
                f"{source_label} candidates[{index}] must include a numeric 'idf'."
            )

    return payload


def summarize_noise_word_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize a validated noise-word candidate payload for promotion review.

    Assumes ``payload`` has already passed
    ``validate_noise_word_candidates_payload()``. Intended for
    ``scripts/promote_noise_word_candidates.py`` to print a human-readable
    diff against the currently-packaged file without needing to know this
    module's payload shape.
    """
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return {"candidate_count": 0, "idf_min": None, "idf_max": None}

    idf_values = [
        float(entry["idf"])
        for entry in candidates
        if isinstance(entry, dict) and isinstance(entry.get("idf"), (int, float))
    ]
    return {
        "candidate_count": len(candidates),
        "idf_min": min(idf_values) if idf_values else None,
        "idf_max": max(idf_values) if idf_values else None,
    }
