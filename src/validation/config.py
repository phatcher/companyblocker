"""A validation run's configuration, and which text column each representation reads.

`ValidationRunConfig` and `validate_run_config` fix what a run was asked to do. `expected_text_view_for_representation` states each representation's text view: `tfidf` and `sbert` read the raw `name` column, `wordpiece` and `sentencepiece` the tokenized `tokens` column, since a dense sentence encoder embeds natural text rather than a re-joined subword list. A representation with no declared view raises rather than falling back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from company_vectorize.clustering_contract import TFIDF_ANALYZERS

from workspace.roots import WorkspaceRoots

_DEFAULT_PREPARED_ROWS_PER_FILE = 1_000_000


_VALID_TEXT_VIEWS = {"name", "tokens"}

# Which text view each representation consumes, as a per-representation
# property rather than a "tfidf versus everything else" branch.
# `"tokens"` is the tokenized `cluster_tokens` list column, which only the
# subword token-list representations can read; every representation that
# vectorizes natural text -- TF-IDF character n-grams and dense sentence
# embeddings alike -- needs the `"name"` view. A dense sentence encoder
# defaulted onto `"tokens"` fails outright (`SbertClusteringStrategy.
# build_target_index` casts the `List(Utf8)` column to `Utf8`), so a new
# representation must be added here deliberately rather than inheriting a
# fallback. `"encoder"` is a caller-supplied dense encoder, which embeds the name as it is.
_REPRESENTATION_TEXT_VIEWS = {
    "tfidf": "name",
    "sbert": "name",
    "encoder": "name",
    "wordpiece": "tokens",
    "sentencepiece": "tokens",
}
_VALID_REPRESENTATIONS = {"tfidf", "wordpiece", "sentencepiece", "sbert"}
_VALID_SIMILARITIES = {"cosine"}
_VALID_CLUSTERING_MODES = {"knn_cc"}
_SUPPORTED_PIPELINES = {
    ("tfidf", "cosine", "knn_cc"),
    ("wordpiece", "cosine", "knn_cc"),
    ("sentencepiece", "cosine", "knn_cc"),
    ("sbert", "cosine", "knn_cc"),
}
_VALID_SIMILARITY_BACKENDS = {"sklearn", "sparse_dot_topn", "svd_rerank"}
_VALID_TOKENIZERS = {"wordpiece", "sentencepiece"}
_VALID_TOKENIZER_SCOPES = {"country", "global"}


@dataclass(frozen=True)
class ValidationRunConfig:
    roots: WorkspaceRoots
    run_date: str | None
    source_systems: tuple[str, ...]
    target_systems: tuple[str, ...]
    countries: tuple[str, ...] | None
    name_col: str
    representation: str
    similarity: str
    clustering: str
    text_view: str | None
    top_k: int
    min_similarity: float
    max_candidates_per_source: int | None
    output_dir: Path
    prepared_base_dir: Path | None
    prepared_rows_per_file: int
    include_reverse_direction: bool
    exception_policy_path: Path | None
    source_name_col: str | None = None
    tfidf_ngram_min: int = 2
    tfidf_ngram_max: int = 3
    tfidf_analyzer: str = "char_wb"
    similarity_backend: str = "sklearn"
    backend_options: dict[str, object] | None = None
    tokenizer: str = "wordpiece"
    tokenizer_scope: str = "country"
    tokenizer_profile: str = "promoted"
    # Opt-in: a tokenizer model file used as it is, a candidate no profile
    # names, in place of the one `tokenizer_profile` resolves.
    tokenizer_path: str | None = None
    noise_words_profile: str = "aggressive"
    noise_words_set_kind: str = "combined"


def _split_system_tokens(value: str) -> list[str]:
    tokens = [
        token.strip().lower() for token in re.split(r"[\s,]+", value) if token.strip()
    ]
    return tokens


def parse_system_list(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_tokens = _split_system_tokens(value)
    else:
        raw_tokens = []
        for item in value:
            raw_tokens.extend(_split_system_tokens(str(item)))

    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
    return tuple(unique_tokens)


def _validate_required_system_lists(config: ValidationRunConfig) -> None:
    if not config.source_systems:
        raise ValueError("source_systems must contain at least one system code")
    if not config.target_systems:
        raise ValueError("target_systems must contain at least one system code")


def _validate_required_text_fields(config: ValidationRunConfig) -> None:
    if config.source_name_col is not None and not str(config.source_name_col).strip():
        raise ValueError("source_name_col must be non-empty")
    if not config.name_col.strip():
        raise ValueError("name_col must be non-empty")


def _validate_enum_values(config: ValidationRunConfig) -> None:
    if config.representation not in _VALID_REPRESENTATIONS:
        allowed = ", ".join(sorted(_VALID_REPRESENTATIONS))
        raise ValueError(f"representation must be one of: {allowed}")

    if config.similarity not in _VALID_SIMILARITIES:
        allowed = ", ".join(sorted(_VALID_SIMILARITIES))
        raise ValueError(f"similarity must be one of: {allowed}")

    if config.clustering not in _VALID_CLUSTERING_MODES:
        allowed = ", ".join(sorted(_VALID_CLUSTERING_MODES))
        raise ValueError(f"clustering must be one of: {allowed}")

    if config.similarity_backend not in _VALID_SIMILARITY_BACKENDS:
        allowed = ", ".join(sorted(_VALID_SIMILARITY_BACKENDS))
        raise ValueError(f"similarity_backend must be one of: {allowed}")

    if config.tokenizer not in _VALID_TOKENIZERS:
        allowed = ", ".join(sorted(_VALID_TOKENIZERS))
        raise ValueError(f"tokenizer must be one of: {allowed}")

    if config.tokenizer_scope not in _VALID_TOKENIZER_SCOPES:
        allowed = ", ".join(sorted(_VALID_TOKENIZER_SCOPES))
        raise ValueError(f"tokenizer_scope must be one of: {allowed}")


def _validate_supported_pipeline(config: ValidationRunConfig) -> None:
    pipeline = (config.representation, config.similarity, config.clustering)
    if pipeline not in _SUPPORTED_PIPELINES:
        supported = ", ".join(
            f"{representation}/{similarity}/{clustering}"
            for representation, similarity, clustering in sorted(_SUPPORTED_PIPELINES)
        )
        raise ValueError(
            "Unsupported representation/similarity/clustering combination: "
            f"{config.representation}/{config.similarity}/{config.clustering}. "
            f"Supported combinations: {supported}"
        )


def _validate_text_view_and_trainer(config: ValidationRunConfig) -> None:
    effective_text_view = resolve_text_view_for_representation(
        representation=config.representation,
        requested_text_view=config.text_view,
    )
    if effective_text_view not in _VALID_TEXT_VIEWS:
        allowed = ", ".join(sorted(_VALID_TEXT_VIEWS))
        raise ValueError(f"text_view must be one of: {allowed}")

    expected_trainer = None
    if config.representation == "wordpiece":
        expected_trainer = "wordpiece"
    elif config.representation == "sentencepiece":
        expected_trainer = "sentencepiece"

    if expected_trainer is not None and config.tokenizer != expected_trainer:
        raise ValueError(
            f"representation '{config.representation}' requires tokenizer='{expected_trainer}'"
        )


def _validate_numeric_limits(config: ValidationRunConfig) -> None:
    if config.top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not (0.0 <= config.min_similarity <= 1.0):
        raise ValueError("min_similarity must be between 0 and 1")

    if (
        config.max_candidates_per_source is not None
        and config.max_candidates_per_source <= 0
    ):
        raise ValueError(
            "max_candidates_per_source must be greater than zero when provided"
        )

    if config.prepared_rows_per_file <= 0:
        raise ValueError("prepared_rows_per_file must be greater than zero")


def _validate_optional_paths_and_collections(config: ValidationRunConfig) -> None:
    if (
        config.prepared_base_dir is not None
        and not str(config.prepared_base_dir).strip()
    ):
        raise ValueError("prepared_base_dir must be non-empty when provided")

    if config.countries is not None and len(config.countries) == 0:
        raise ValueError("countries cannot be empty when provided")

    if (
        config.exception_policy_path is not None
        and config.exception_policy_path.suffix.lower() != ".json"
    ):
        raise ValueError("exception_policy_path must point to a .json file")


def _validate_representation_specific_options(config: ValidationRunConfig) -> None:
    if config.representation != "tfidf":
        return

    if config.tfidf_ngram_min <= 0:
        raise ValueError("tfidf_ngram_min must be greater than zero")
    if config.tfidf_ngram_max <= 0:
        raise ValueError("tfidf_ngram_max must be greater than zero")
    if config.tfidf_ngram_max < config.tfidf_ngram_min:
        raise ValueError(
            "tfidf_ngram_max must be greater than or equal to tfidf_ngram_min"
        )
    if config.tfidf_analyzer not in TFIDF_ANALYZERS:
        raise ValueError(
            f"tfidf_analyzer must be one of {sorted(TFIDF_ANALYZERS)}, "
            f"got {config.tfidf_analyzer!r}"
        )


def _validate_backend_and_tokenizer_profiles(config: ValidationRunConfig) -> None:
    if config.backend_options is not None and not isinstance(
        config.backend_options, dict
    ):
        raise ValueError("backend_options must be a dictionary when provided")

    if not str(config.tokenizer_profile).strip():
        raise ValueError("tokenizer_profile must be non-empty")

    if not str(config.noise_words_profile).strip():
        raise ValueError("noise_words_profile must be non-empty")

    if not str(config.noise_words_set_kind).strip():
        raise ValueError("noise_words_set_kind must be non-empty")


def validate_run_config(config: ValidationRunConfig) -> None:
    _validate_required_system_lists(config)
    _validate_required_text_fields(config)
    _validate_enum_values(config)
    _validate_supported_pipeline(config)
    _validate_text_view_and_trainer(config)
    _validate_numeric_limits(config)
    _validate_optional_paths_and_collections(config)
    _validate_representation_specific_options(config)
    _validate_backend_and_tokenizer_profiles(config)


def get_algorithm_label(config: ValidationRunConfig) -> str:
    return f"{config.representation}_{config.similarity}_{config.clustering}"


def expected_text_view_for_representation(representation: str) -> str:
    """Return the only text view `representation` can be scored on.

    Args:
        representation: A representation name (`"tfidf"`, `"wordpiece"`,
            `"sentencepiece"`, `"sbert"`).

    Raises:
        ValueError: `representation` has no declared text view. Deliberately
            not a fallback: silently defaulting a new representation onto
            `"tokens"` is what forced `"sbert"` onto the subword token-list
            column and made every `"sbert"` run fail.
    """
    try:
        return _REPRESENTATION_TEXT_VIEWS[representation]
    except KeyError:
        known = ", ".join(sorted(_REPRESENTATION_TEXT_VIEWS))
        raise ValueError(
            f"representation '{representation}' has no declared text view; "
            f"known representations: {known}"
        ) from None


def resolve_text_view_for_representation(
    *, representation: str, requested_text_view: str | None
) -> str:
    expected = expected_text_view_for_representation(representation)
    if requested_text_view is None:
        return expected

    requested = str(requested_text_view).strip().lower()
    if not requested or requested == "auto":
        return expected
    if requested not in _VALID_TEXT_VIEWS:
        allowed = ", ".join(sorted(_VALID_TEXT_VIEWS))
        raise ValueError(f"text_view must be one of: {allowed}")
    if requested != expected:
        raise ValueError(
            f"representation '{representation}' requires text_view='{expected}'"
        )
    return requested


def resolve_prepared_rows_per_file(value: int | None) -> int:
    if value is None:
        return _DEFAULT_PREPARED_ROWS_PER_FILE
    return int(value)
