"""TF-IDF token statistics, the noise-word candidates drawn from them, and noise-word trimming.

`compute_corpus_token_tfidf_stats()` scores the whitespace-split words of one prepared
corpus parquet file, never a shard directory or glob; `name_col` only picks the
column. `compute_tokenizer_token_tfidf_stats()` scores a trained tokenizer's emitted
tokens instead. Each result carries a namespace tag, `corpus_tfidf` or
`<trainer>_tfidf`, so the two are never conflated. `select_stopword_candidates()` and
`select_rare_token_candidates()` take the frequent and rare ends of that distribution.

Noise words are a text-level decision, made before and independent of any
tokenizer: the same list applies whichever trainer and settings later consume the
text. `resolve_noise_words()` resolves a set in this order:

1. an explicit `noise_words` argument;
2. `noise_words_profile="none"`, which resolves to no words whatever path is given;
3. a `noise_words_path` file, such as one system's list;
4. `company_cleanse`'s packaged profiled defaults (`noise_words_set_kind=combined`).

The packaged default is itself this module's output: `scripts/generate_noise_words.py`
writes a candidate list per system, a caller tries it through `noise_words_path`,
which overrides the default for that run only, and `scripts/promote_noise_words.py
--write` copies a reviewed one into `company_cleanse`, the only step that changes
behaviour for callers passing neither argument.

`pool_token_tfidf_stats()` pools several systems' persisted stats without rescanning
any corpus, under a named rule: `count` weights each system by its share of rows and
reproduces the row-weighted pool; `equal` averages each system's document-frequency
percentage, an absent token counting zero, so corpus size stops deciding the
result. `trim_pooled_corpus_by_system()` trims a mixed corpus row by row with each
row's own system's list.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from math import exp, isfinite
from pathlib import Path
from typing import TypedDict

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from company_cleanse import get_profiled_noise_words
except ModuleNotFoundError:
    # Support monorepo script execution when sibling package paths are not preconfigured.
    sibling_cleanse_src = (
        Path(__file__).resolve().parents[3] / "company_cleanse" / "src"
    )
    if str(sibling_cleanse_src) not in sys.path:
        sys.path.insert(0, str(sibling_cleanse_src))
    from company_cleanse import get_profiled_noise_words


TFIDF_USE_CASE_CORPUS = "corpus"
TFIDF_USE_CASE_TOKENIZER = "tokenizer"
SUPPORTED_TFIDF_USE_CASES = (TFIDF_USE_CASE_CORPUS, TFIDF_USE_CASE_TOKENIZER)

POOLING_RULE_COUNT = "count"
POOLING_RULE_EQUAL = "equal"
SUPPORTED_POOLING_RULES = (POOLING_RULE_COUNT, POOLING_RULE_EQUAL)


def _empty_tfidf_stats_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "token": [],
            "document_frequency": [],
            "document_frequency_pct": [],
            "idf": [],
        },
        schema={
            "token": pl.Utf8,
            "document_frequency": pl.Int64,
            "document_frequency_pct": pl.Float64,
            "idf": pl.Float64,
        },
    )


def _build_tfidf_stats_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return _empty_tfidf_stats_frame()
    return pl.DataFrame(rows).sort(
        ["document_frequency_pct", "idf", "token"], descending=[True, False, False]
    )


def _load_corpus_docs(corpus_path: Path, *, name_col: str) -> list[str]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus parquet not found: {corpus_path}.")

    corpus_df = pl.read_parquet(corpus_path, columns=[name_col])
    return [
        str(value).strip()
        for value in corpus_df.get_column(name_col).to_list()
        if value is not None and str(value).strip()
    ]


def _build_tfidf_stats_from_document_frequencies(
    *,
    n_docs: int,
    document_frequency_by_token: dict[str, int],
    smooth_idf: bool,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for token, document_frequency in document_frequency_by_token.items():
        idf = _compute_idf(
            n_docs=n_docs,
            document_frequency=document_frequency,
            smooth_idf=smooth_idf,
        )
        rows.append(
            {
                "token": token,
                "document_frequency": document_frequency,
                "document_frequency_pct": float(document_frequency) / float(n_docs),
                "idf": idf,
            }
        )

    return _build_tfidf_stats_frame(rows)


def _build_document_frequency_from_vectorizer_docs(
    docs: list[str],
    *,
    lowercase: bool,
    smooth_idf: bool,
    token_pattern: str,
) -> dict[str, int]:
    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        smooth_idf=smooth_idf,
        token_pattern=token_pattern,
        norm=None,
        use_idf=True,
    )
    vectorizer.fit(docs)

    n_docs = len(docs)
    document_frequency_by_token: dict[str, int] = {}
    for token, index in vectorizer.vocabulary_.items():
        idf = float(vectorizer.idf_[index])
        document_frequency_by_token[token] = _estimate_document_frequency_from_idf(
            n_docs=n_docs,
            idf=idf,
            smooth_idf=smooth_idf,
        )
    return document_frequency_by_token


def _build_document_frequency_from_tokenized_docs(
    docs: list[str],
    *,
    encode_tokens: Callable[[str], list[str]],
) -> dict[str, int]:
    document_frequency_by_token: dict[str, int] = {}
    for text in docs:
        unique_tokens = {token for token in encode_tokens(text) if token}
        for token in unique_tokens:
            document_frequency_by_token[token] = (
                document_frequency_by_token.get(token, 0) + 1
            )
    return document_frequency_by_token


def _compute_tfidf_stats_from_docs(
    docs: list[str],
    *,
    smooth_idf: bool,
    document_frequency_builder: Callable[[list[str]], dict[str, int]],
) -> pl.DataFrame:
    if not docs:
        return _empty_tfidf_stats_frame()

    n_docs = len(docs)
    document_frequency_by_token = document_frequency_builder(docs)
    return _build_tfidf_stats_from_document_frequencies(
        n_docs=n_docs,
        document_frequency_by_token=document_frequency_by_token,
        smooth_idf=smooth_idf,
    )


def compute_corpus_token_tfidf_stats(
    corpus_path: Path,
    *,
    name_col: str = "name",
    lowercase: bool = True,
    smooth_idf: bool = True,
    token_pattern: str = r"(?u)\b\w+\b",
) -> pl.DataFrame:
    """Compute corpus-token TF-IDF statistics from a prepared corpus parquet artifact.

    Expected input is a single parquet file (for example
    ``artifacts/tokenizers/work/<scope>/training_corpus.parquet``) with a text column
    selected by ``name_col`` (default ``name``). This function does not discover
    or merge raw shard files/directories.

    Returns one row per text token with document frequency and IDF-derived metrics.
    """
    docs = _load_corpus_docs(corpus_path, name_col=name_col)
    return _compute_tfidf_stats_from_docs(
        docs,
        smooth_idf=smooth_idf,
        document_frequency_builder=lambda values: (
            _build_document_frequency_from_vectorizer_docs(
                values,
                lowercase=lowercase,
                smooth_idf=smooth_idf,
                token_pattern=token_pattern,
            )
        ),
    )


def compute_tokenizer_token_tfidf_stats(
    corpus_path: Path,
    *,
    tokenizer_path: Path,
    trainer: str = "wordpiece",
    name_col: str = "name",
    smooth_idf: bool = True,
) -> pl.DataFrame:
    """Compute tokenizer-token TF-IDF from emitted subword tokens per document.

    Expected input is a single prepared corpus parquet artifact with one row per
    document in ``name_col`` (default ``name``). The tokenizer model is supplied
    separately via ``tokenizer_path``.

    This is separate from corpus-token TF-IDF and should be used when weighting
    tokenizer outputs (for example WordPiece).
    """
    docs = _load_corpus_docs(corpus_path, name_col=name_col)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}.")

    from .training import load_tokenizer_encoder

    encode_tokens, _ = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer=trainer
    )

    return _compute_tfidf_stats_from_docs(
        docs,
        smooth_idf=smooth_idf,
        document_frequency_builder=lambda values: (
            _build_document_frequency_from_tokenized_docs(
                values,
                encode_tokens=encode_tokens,
            )
        ),
    )


def normalize_pooling_rule(pooling: str) -> str:
    normalized = pooling.strip().lower()
    if normalized not in SUPPORTED_POOLING_RULES:
        allowed = ", ".join(SUPPORTED_POOLING_RULES)
        raise ValueError(
            f"Unsupported pooling rule '{pooling}'. Expected one of: {allowed}."
        )
    return normalized


def infer_document_count_from_tfidf_stats(stats: pl.DataFrame) -> int:
    """Recover a persisted TF-IDF stats frame's total document count.

    `document_frequency_pct` is `document_frequency / n_docs` for every
    row, so any row recovers `n_docs`; this uses the row with the largest
    `document_frequency` for the least float-rounding error (the ratio's
    relative precision improves as the numerator grows).
    """
    if stats.height == 0:
        raise ValueError("Cannot infer document count from empty TF-IDF stats.")

    top_row = stats.sort("document_frequency", descending=True).row(0, named=True)
    document_frequency = float(top_row["document_frequency"])
    document_frequency_pct = float(top_row["document_frequency_pct"])
    if document_frequency_pct <= 0.0:
        raise ValueError(
            "Cannot infer document count: top row has a non-positive "
            "document_frequency_pct."
        )
    return round(document_frequency / document_frequency_pct)


class PooledTfidfProvenance(TypedDict):
    pooling_rule: str
    systems: list[str]
    weights: dict[str, float]
    document_counts: dict[str, int]


def pool_token_tfidf_stats(
    stats_by_system: Mapping[str, pl.DataFrame],
    *,
    pooling: str,
    smooth_idf: bool = True,
) -> tuple[pl.DataFrame, PooledTfidfProvenance]:
    """Pool several systems' persisted TF-IDF stats into one corpus-size-independent frame.

    `stats_by_system` maps a system code to that system's own
    `token_tfidf_stats.parquet` frame (`token`, `document_frequency`,
    `document_frequency_pct`, `idf`), so pooling is a join over already
    computed per-corpus stats with no corpus rescan. Two pooling rules:

    - `count` reproduces the row-weighted pool the systems' combined raw
      corpora would themselves carry: since the corpora are disjoint, each
      token's document-frequency *counts* sum directly across systems, and
      dividing by the summed document counts gives back the same
      document-frequency percentages a single rescanned corpus would.
    - `equal` takes the mean of each member's own `document_frequency_pct`
      for a token (an absent token contributing zero), so no member's row
      count skews the result regardless of its size.

    Returns the pooled frame in the same `token`/`document_frequency`/
    `document_frequency_pct`/`idf` schema `compute_corpus_token_tfidf_stats`
    produces, so it flows unchanged through the same threshold and
    cumulative-mass profile selection `scripts/generate_noise_words.py`
    already runs on one system's stats, plus a provenance record naming the
    rule, member systems, their resolved weights and inferred per-system
    document counts.
    """
    normalized_pooling = normalize_pooling_rule(pooling)
    if not stats_by_system:
        raise ValueError("pool_token_tfidf_stats requires at least one system's stats.")

    systems = sorted(stats_by_system)
    document_counts = {
        system: infer_document_count_from_tfidf_stats(stats_by_system[system])
        for system in systems
    }

    per_system_metrics: dict[str, dict[str, tuple[int, float]]] = {}
    for system in systems:
        frame = stats_by_system[system]
        per_system_metrics[system] = {
            row["token"]: (
                int(row["document_frequency"]),
                float(row["document_frequency_pct"]),
            )
            for row in frame.select(
                ["token", "document_frequency", "document_frequency_pct"]
            ).iter_rows(named=True)
        }

    all_tokens = sorted(
        {token for metrics in per_system_metrics.values() for token in metrics}
    )

    rows: list[dict[str, object]]
    if normalized_pooling == POOLING_RULE_COUNT:
        total_docs = sum(document_counts.values())
        weights = {
            system: (document_counts[system] / total_docs if total_docs > 0 else 0.0)
            for system in systems
        }
        rows = []
        for token in all_tokens:
            pooled_document_frequency = sum(
                per_system_metrics[system].get(token, (0, 0.0))[0] for system in systems
            )
            pooled_pct = (
                pooled_document_frequency / total_docs if total_docs > 0 else 0.0
            )
            idf = _compute_idf(
                n_docs=total_docs,
                document_frequency=pooled_document_frequency,
                smooth_idf=smooth_idf,
            )
            rows.append(
                {
                    "token": token,
                    "document_frequency": pooled_document_frequency,
                    "document_frequency_pct": pooled_pct,
                    "idf": idf,
                }
            )
    else:  # POOLING_RULE_EQUAL
        member_count = len(systems)
        reference_document_count = max(
            round(sum(document_counts.values()) / member_count), 1
        )
        weights = dict.fromkeys(systems, 1.0 / member_count)
        rows = []
        for token in all_tokens:
            pooled_pct = (
                sum(
                    per_system_metrics[system].get(token, (0, 0.0))[1]
                    for system in systems
                )
                / member_count
            )
            pooled_document_frequency = round(pooled_pct * reference_document_count)
            idf = _compute_idf(
                n_docs=reference_document_count,
                document_frequency=pooled_document_frequency,
                smooth_idf=smooth_idf,
            )
            rows.append(
                {
                    "token": token,
                    "document_frequency": pooled_document_frequency,
                    "document_frequency_pct": pooled_pct,
                    "idf": idf,
                }
            )

    pooled_frame = _build_tfidf_stats_frame(rows)
    provenance: PooledTfidfProvenance = {
        "pooling_rule": normalized_pooling,
        "systems": systems,
        "weights": weights,
        "document_counts": document_counts,
    }
    return pooled_frame, provenance


def normalize_tfidf_use_case(use_case: str) -> str:
    normalized = use_case.strip().lower()
    if normalized not in SUPPORTED_TFIDF_USE_CASES:
        allowed = ", ".join(SUPPORTED_TFIDF_USE_CASES)
        raise ValueError(
            f"Unsupported TF-IDF use case '{use_case}'. Expected one of: {allowed}."
        )
    return normalized


def expected_tfidf_namespace(*, use_case: str, trainer: str | None = None) -> str:
    normalized_use_case = normalize_tfidf_use_case(use_case)
    if normalized_use_case == TFIDF_USE_CASE_CORPUS:
        return "corpus_tfidf"

    if trainer is None or not trainer.strip():
        raise ValueError("trainer is required when use_case='tokenizer'.")
    return f"{trainer.strip().lower()}_tfidf"


def validate_tfidf_namespace(
    *, use_case: str, namespace: str, trainer: str | None = None
) -> str:
    expected = expected_tfidf_namespace(use_case=use_case, trainer=trainer)
    normalized_namespace = namespace.strip().lower()
    if normalized_namespace != expected:
        raise ValueError(
            "TF-IDF namespace mismatch: "
            + f"expected '{expected}' for use_case='{normalize_tfidf_use_case(use_case)}', "
            + f"got '{namespace}'."
        )
    return expected


def cumulative_document_frequency_mass_cutoff(
    document_frequency_pct_by_rank: Sequence[float],
    *,
    cumulative_df_mass_limit: float,
) -> int:
    """How many leading items reach a cumulative document-frequency-mass limit.

    `document_frequency_pct_by_rank` must already be in the caller's desired
    order (typically descending document frequency) -- this only walks it and
    accumulates. Returns the full length when `cumulative_df_mass_limit` is
    never reached.

    Used by `scripts/generate_noise_words.py`'s noise-word profile selection,
    which combines this with its own per-token prefilter and a minimum-count
    floor -- a different, two-part rule answering a different question (how
    many tokens to trim before tokenization) from
    `count_tokens_above_document_frequency_pct` below.
    """
    if cumulative_df_mass_limit <= 0.0:
        return 0
    cumulative = 0.0
    for index, pct in enumerate(document_frequency_pct_by_rank):
        cumulative += float(pct)
        if cumulative >= cumulative_df_mass_limit:
            return index + 1
    return len(document_frequency_pct_by_rank)


def count_tokens_above_document_frequency_pct(
    document_frequency_pct_by_rank: Sequence[float],
    *,
    min_document_frequency_pct: float,
) -> int:
    """Count of leading items individually at or above a document-frequency share.

    Given items already sorted descending by that share, stops at the first
    item below the floor, since nothing after it can qualify.

    Unlike `cumulative_document_frequency_mass_cutoff`, this asks a simpler
    question directly: is *this token itself* structurally dominant (a
    legal-form marker, a generic connector appearing in a large share of
    names on its own), rather than how much combined mass the leading
    tokens carry together. Used by `analysis.token_zipf` to mark where a
    Zipf rank-frequency curve's head ends.
    """
    count = 0
    for pct in document_frequency_pct_by_rank:
        if pct < min_document_frequency_pct:
            break
        count += 1
    return count


def select_stopword_candidates(
    token_stats: pl.DataFrame,
    *,
    min_document_frequency_pct: float = 0.2,
    max_idf: float | None = None,
    min_token_length: int = 2,
) -> set[str]:
    """Select high-frequency low-information token candidates from TF-IDF stats."""
    if not (0.0 <= min_document_frequency_pct <= 1.0):
        raise ValueError("min_document_frequency_pct must be between 0 and 1.")
    if min_token_length <= 0:
        raise ValueError("min_token_length must be greater than zero.")

    required_columns = {"token", "document_frequency_pct", "idf"}
    missing = sorted(required_columns - set(token_stats.columns))
    if missing:
        raise ValueError(
            f"token_stats is missing required columns: {', '.join(missing)}"
        )

    candidate_frame = token_stats.filter(
        pl.col("document_frequency_pct") >= pl.lit(min_document_frequency_pct)
    )
    if max_idf is not None:
        candidate_frame = candidate_frame.filter(pl.col("idf") <= pl.lit(max_idf))

    candidate_frame = candidate_frame.filter(
        pl.col("token").str.len_chars() >= pl.lit(min_token_length)
    )

    return set(candidate_frame.get_column("token").to_list())


def select_rare_token_candidates(
    token_stats: pl.DataFrame,
    *,
    max_document_frequency: int | None = None,
    max_document_frequency_pct: float | None = None,
    min_token_length: int = 1,
) -> pl.DataFrame:
    """Select low-frequency (rare/unusual) token rows from TF-IDF stats.

    Symmetric counterpart to ``select_stopword_candidates``: instead of the
    high-frequency, low-information end of the distribution, this selects the
    low-frequency end. Returns full stats rows (not just token names) so
    callers retain document-frequency context for reporting.

    When neither ``max_document_frequency`` nor ``max_document_frequency_pct``
    is given, defaults to hapax legomena (``max_document_frequency=1``). When
    both are given, a token must satisfy both to be selected.
    """
    if max_document_frequency is None and max_document_frequency_pct is None:
        max_document_frequency = 1

    if max_document_frequency is not None and max_document_frequency < 1:
        raise ValueError("max_document_frequency must be >= 1.")
    if max_document_frequency_pct is not None and not (
        0.0 <= max_document_frequency_pct <= 1.0
    ):
        raise ValueError("max_document_frequency_pct must be between 0 and 1.")
    if min_token_length <= 0:
        raise ValueError("min_token_length must be greater than zero.")

    required_columns = {"token", "document_frequency", "document_frequency_pct"}
    missing = sorted(required_columns - set(token_stats.columns))
    if missing:
        raise ValueError(
            f"token_stats is missing required columns: {', '.join(missing)}"
        )

    candidate_frame = token_stats
    if max_document_frequency is not None:
        candidate_frame = candidate_frame.filter(
            pl.col("document_frequency") <= pl.lit(max_document_frequency)
        )
    if max_document_frequency_pct is not None:
        candidate_frame = candidate_frame.filter(
            pl.col("document_frequency_pct") <= pl.lit(max_document_frequency_pct)
        )

    candidate_frame = candidate_frame.filter(
        pl.col("token").str.len_chars() >= pl.lit(min_token_length)
    )

    return candidate_frame.sort(["document_frequency", "token"])


NO_NOISE_WORDS_PROFILE = "none"
"""The noise words profile that removes no words."""


def resolve_noise_words(
    *,
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> set[str]:
    """Resolve noise words by precedence: explicit values, path override, packaged profiled default.

    The profile `none` resolves to no words, whatever path is given.
    """
    if noise_words is not None:
        return _normalize_noise_words(noise_words)

    if (noise_words_profile or "").strip().lower() == NO_NOISE_WORDS_PROFILE:
        return set()

    if noise_words_path is not None:
        return _load_noise_words_from_path(
            Path(noise_words_path),
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
        )

    return _load_packaged_profiled_noise_words(
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )


def trim_text_before_tokenization(
    names: Sequence[str | None],
    *,
    # `AbstractSet` rather than `set`: the token set is only read here, and
    # `TokenizerVocabulary.noise_words` hands over a `frozenset`.
    noise_words: AbstractSet[str],
) -> list[str]:
    """Remove whole-word noise words from names before tokenizer encoding."""
    return _trim_names_with_noise_words(
        names,
        noise_words=noise_words,
    )


def _trim_names_with_noise_words(
    names: Sequence[str | None],
    *,
    noise_words: AbstractSet[str],
) -> list[str]:
    if not noise_words:
        return [((name or "").strip()) for name in names]

    lookup = _prepare_noise_word_lookup(noise_words)
    trimmed_names: list[str] = []
    for name in names:
        text = (name or "").strip()
        if not text:
            trimmed_names.append("")
            continue

        raw_tokens = text.split()
        token_keys = _lower_tokens_at_boundary(raw_tokens)

        kept = [
            token
            for token, token_key in zip(raw_tokens, token_keys, strict=True)
            if token_key not in lookup
        ]
        trimmed_names.append(" ".join(kept).strip())

    return trimmed_names


def trim_name_before_tokenization(
    name: str | None,
    *,
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> str:
    """Trim noise words from a single name string before tokenizer encoding."""
    effective_noise_words = resolve_noise_words(
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )
    return _trim_names_with_noise_words(
        [name],
        noise_words=effective_noise_words,
    )[0]


def trim_pooled_corpus_by_system(
    corpus: pl.DataFrame,
    *,
    noise_words_path_by_system: Mapping[str, str | Path],
    system_uri_col: str = "system_uri",
    name_col: str = "name",
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> pl.DataFrame:
    """Trim a pooled training corpus row by row, each row under its own system's list.

    `corpus` carries `system_uri_col` (for example the pooled training
    corpus's `system_uri`); each value is reduced to its short system code
    the same way `optimize.split_system_from_uri` does, then looked up in
    `noise_words_path_by_system` -- a map of system code to a noise-word
    file, resolved the same way `resolve_noise_words`'s path override
    resolves one. A row is trimmed only with its own system's list, so a
    pooled corpus spanning several systems needs no universal list. Returns
    the concatenation of the per-system trimmed frames -- row order is
    grouped by system, not preserved. A system present in `corpus` but
    absent from `noise_words_path_by_system` raises.
    """
    if system_uri_col not in corpus.columns:
        raise ValueError(f"corpus does not contain required column '{system_uri_col}'.")
    if name_col not in corpus.columns:
        raise ValueError(f"corpus does not contain required column '{name_col}'.")
    if corpus.height == 0:
        return corpus

    from .optimize import split_system_from_uri

    system_keys = [
        split_system_from_uri(value)
        for value in corpus.get_column(system_uri_col).to_list()
    ]
    system_key_col = "_pooled_trim_system_key"
    corpus_with_key = corpus.with_columns(pl.Series(system_key_col, system_keys))

    normalized_paths = {
        str(system).strip().lower(): path
        for system, path in noise_words_path_by_system.items()
    }
    present_systems = sorted(set(system_keys))
    missing = sorted(set(present_systems) - set(normalized_paths))
    if missing:
        raise ValueError(
            "corpus contains systems with no noise-word file: " + ", ".join(missing)
        )

    trimmed_frames: list[pl.DataFrame] = []
    for system_key in present_systems:
        subframe = corpus_with_key.filter(pl.col(system_key_col) == system_key).drop(
            system_key_col
        )
        noise_words = resolve_noise_words(
            noise_words_path=normalized_paths[system_key],
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
        )
        trimmed_names = _trim_names_with_noise_words(
            subframe.get_column(name_col).to_list(),
            noise_words=noise_words,
        )
        trimmed_frames.append(subframe.with_columns(pl.Series(name_col, trimmed_names)))

    return pl.concat(trimmed_frames, how="vertical")


def trim_token_column(
    df: pl.DataFrame,
    *,
    noise_words: set[str],
    token_col: str = "name_tokens",
    output_col: str | None = None,
) -> pl.DataFrame:
    """Trim noise words from a token-list column.

    By default, trimming is case-insensitive and writes back to ``token_col``.
    """
    if token_col not in df.columns:
        raise ValueError(f"DataFrame does not contain required column '{token_col}'.")

    target_col = output_col or token_col
    if not noise_words:
        if target_col == token_col:
            return df
        return df.with_columns(pl.col(token_col).alias(target_col))

    stop_lookup = _prepare_noise_word_lookup(noise_words)

    trimmed_rows: list[list[str]] = []
    for tokens in df.get_column(token_col).to_list():
        if not tokens:
            trimmed_rows.append([])
            continue

        token_keys = _lower_tokens_at_boundary(tokens)

        trimmed = [
            token
            for token, token_key in zip(tokens, token_keys, strict=True)
            if token_key not in stop_lookup
        ]
        trimmed_rows.append(trimmed)

    return df.with_columns(pl.Series(target_col, trimmed_rows))


def _estimate_document_frequency_from_idf(
    *, n_docs: int, idf: float, smooth_idf: bool
) -> int:
    if n_docs <= 0:
        return 0
    if not isfinite(idf):
        return 0

    exponent = exp(idf - 1.0)
    if smooth_idf:
        estimated = ((1.0 + float(n_docs)) / exponent) - 1.0
    else:
        estimated = float(n_docs) / exponent

    rounded = round(estimated)
    return max(1, min(rounded, n_docs))


def _compute_idf(*, n_docs: int, document_frequency: int, smooth_idf: bool) -> float:
    if n_docs <= 0:
        return 0.0

    safe_df = max(1, min(int(document_frequency), n_docs))
    if smooth_idf:
        return float(math.log((1.0 + float(n_docs)) / (1.0 + float(safe_df))) + 1.0)
    return float(math.log(float(n_docs) / float(safe_df)) + 1.0)


def _load_packaged_profiled_noise_words(
    *,
    noise_words_profile: str,
    noise_words_set_kind: str,
) -> set[str]:
    # Single source of truth for packaged defaults lives in company_cleanse.
    values = get_profiled_noise_words(
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )
    return _normalize_noise_words(values)


def _load_noise_words_from_path(
    path: Path,
    *,
    noise_words_profile: str,
    noise_words_set_kind: str,
) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Noise-word list not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _resolve_noise_words_json_payload(
            payload=payload,
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
            source_label=str(path),
        )

    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return _normalize_noise_words(lines)

    if suffix in {".csv", ".parquet"}:
        frame = pl.read_parquet(path) if suffix == ".parquet" else pl.read_csv(path)
        if frame.height == 0:
            return set()

        if "token" in frame.columns:
            token_values = frame.get_column("token").to_list()
        else:
            first_col = frame.columns[0]
            token_values = frame.get_column(first_col).to_list()
        return _normalize_noise_words(
            str(value) for value in token_values if value is not None
        )

    raise ValueError(
        "Unsupported noise-word file format. Expected one of: .json, .txt, .csv, .parquet"
    )


def _resolve_noise_words_json_payload(
    *,
    payload: object,
    noise_words_profile: str,
    noise_words_set_kind: str,
    source_label: str,
) -> set[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
        values = payload["tokens"]
    elif isinstance(payload, dict) and isinstance(payload.get("profiles"), dict):
        values = _extract_profiled_noise_words(
            payload=payload,
            noise_words_profile=noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
        )
    else:
        raise TypeError(
            "JSON noise-word payload must be a list, an object with a 'tokens' list, "
            "or an object with profiled 'profiles' noise-word sets "
            f"(source: {source_label})."
        )

    if not all(isinstance(entry, str) for entry in values):
        raise ValueError(
            f"JSON noise-word entries must be strings (source: {source_label})."
        )
    return _normalize_noise_words(values)


def _normalize_noise_words(values: Iterable[object]) -> set[str]:
    normalized: set[str] = set()
    for value in values:
        token = str(value).strip().lower()
        if token:
            normalized.add(token)
    return normalized


def _prepare_noise_word_lookup(
    noise_words: AbstractSet[str],
) -> set[str]:
    # Case normalization belongs at token-loading/lookup boundaries.
    return _normalize_noise_words(noise_words)


def _lower_tokens_at_boundary(tokens: Iterable[object]) -> list[str]:
    # Normalize token keys once at boundary, then reuse for comparisons.
    return [str(token).lower() for token in tokens]


def _extract_profiled_noise_words(
    *,
    payload: dict[str, object],
    noise_words_profile: str,
    noise_words_set_kind: str,
) -> list[str]:
    profiles_obj = payload.get("profiles")
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
        cumulative = _resolve_profile_cumulative_tokens(
            profiles_obj=profiles_obj, profile_key=profile_key
        )
        return sorted(cumulative)

    values = profile_payload.get(set_kind_key)
    if not isinstance(values, list):
        # Backward compatibility for previous field name.
        if set_kind_key == "tokens":
            legacy_values = profile_payload.get("tfidf_descriptor")
            if isinstance(legacy_values, list):
                return legacy_values
        available_sets = ", ".join(sorted(str(key) for key in profile_payload))
        raise ValueError(
            f"Noise-word set kind '{set_kind_key}' not found in profile '{profile_key}'. "
            f"Available set kinds: {available_sets}."
        )

    return values


def _resolve_profile_cumulative_tokens(
    *, profiles_obj: dict[str, object], profile_key: str
) -> set[str]:
    order = ["strict", "balanced", "aggressive"]
    if profile_key in order:
        cutoff = order.index(profile_key)
        included = order[: cutoff + 1]
    else:
        included = [profile_key]

    combined: set[str] = set()
    for key in included:
        payload = profiles_obj.get(key)
        if not isinstance(payload, dict):
            continue

        seed_legal_values = payload.get("seed_legal")
        if isinstance(seed_legal_values, list):
            combined.update(
                str(value).strip().lower()
                for value in seed_legal_values
                if str(value).strip()
            )

        token_values = payload.get("tokens")
        if not isinstance(token_values, list):
            token_values = payload.get("tfidf_descriptor")
        if isinstance(token_values, list):
            combined.update(
                str(value).strip().lower()
                for value in token_values
                if str(value).strip()
            )

    return combined
