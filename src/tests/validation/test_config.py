from pathlib import Path
from typing import Any

import pytest

from validation.config import (
    ValidationRunConfig,
    expected_text_view_for_representation,
    parse_system_list,
    resolve_text_view_for_representation,
    validate_run_config,
)
from workspace.roots import WorkspaceRoots


def test_parse_system_list_supports_comma_and_space_input() -> None:
    parsed = parse_system_list("gleif, gb fr,ie")
    assert parsed == ("gleif", "gb", "fr", "ie")


def test_parse_system_list_dedupes_preserving_order() -> None:
    parsed = parse_system_list(["gleif", "gb fr", "gleif", "ie"])
    assert parsed == ("gleif", "gb", "fr", "ie")


def test_validate_run_config_accepts_minimal_valid_config(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )

    validate_run_config(config)


@pytest.mark.parametrize(
    "representation,expected",
    [
        ("tfidf", "name"),
        ("sbert", "name"),
        ("wordpiece", "tokens"),
        ("sentencepiece", "tokens"),
    ],
)
def test_resolve_text_view_for_representation_derives_when_unspecified(
    representation: str, expected: str
) -> None:
    assert (
        resolve_text_view_for_representation(
            representation=representation, requested_text_view=None
        )
        == expected
    )


def test_resolve_text_view_accepts_matching_explicit_request_for_sbert() -> None:
    # An explicit text_view="name" for a dense sentence-embedding
    # representation used to be rejected outright, because every non-tfidf
    # representation was assumed to want the subword token-list column.
    assert (
        resolve_text_view_for_representation(
            representation="sbert", requested_text_view="name"
        )
        == "name"
    )


@pytest.mark.parametrize(
    "representation,requested,expected",
    [
        ("sbert", "tokens", "name"),
        ("wordpiece", "name", "tokens"),
        ("tfidf", "tokens", "name"),
    ],
)
def test_resolve_text_view_rejects_mismatched_explicit_request(
    representation: str, requested: str, expected: str
) -> None:
    with pytest.raises(ValueError, match=f"requires text_view='{expected}'"):
        resolve_text_view_for_representation(
            representation=representation, requested_text_view=requested
        )


def test_expected_text_view_rejects_undeclared_representation() -> None:
    # A representation with no declared text view fails loudly rather than
    # inheriting a "tokens" fallback -- that fallback is exactly what broke
    # sbert.
    with pytest.raises(ValueError, match="has no declared text view"):
        expected_text_view_for_representation("brand-new-representation")


@pytest.mark.parametrize(
    "representation,similarity,clustering,text_view,top_k,min_similarity",
    [
        ("bad", "cosine", "knn_cc", "name", 20, 0.8),
        ("tfidf", "bad", "knn_cc", "name", 20, 0.8),
        ("tfidf", "cosine", "bad", "name", 20, 0.8),
        ("tfidf", "cosine", "knn_cc", "bad", 20, 0.8),
        ("tfidf", "cosine", "knn_cc", "name", 0, 0.8),
        ("tfidf", "cosine", "knn_cc", "name", 20, 1.2),
    ],
)
def test_validate_run_config_rejects_invalid_values(
    workspace_roots: WorkspaceRoots,
    representation: str,
    similarity: str,
    clustering: str,
    text_view: str,
    top_k: int,
    min_similarity: float,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation=representation,
        similarity=similarity,
        clustering=clustering,
        text_view=text_view,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
    )

    with pytest.raises(ValueError):
        validate_run_config(config)


def test_validate_run_config_rejects_invalid_tfidf_ngram_bounds(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tfidf_ngram_min=3,
        tfidf_ngram_max=2,
    )

    with pytest.raises(ValueError, match="tfidf_ngram_max"):
        validate_run_config(config)


def test_validate_run_config_rejects_invalid_similarity_backend(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        similarity_backend="invalid_backend",
    )

    with pytest.raises(ValueError, match="similarity_backend"):
        validate_run_config(config)


def test_validate_run_config_rejects_non_dict_backend_options(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        similarity_backend="sklearn",
        backend_options="svd_candidates=200",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="backend_options"):
        validate_run_config(config)


def test_validate_run_config_rejects_invalid_tokenizer(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tokenizer="invalid",
    )

    with pytest.raises(ValueError, match="tokenizer"):
        validate_run_config(config)


def test_validate_run_config_rejects_invalid_tokenizer_scope(
    workspace_roots: WorkspaceRoots,
) -> None:
    config = ValidationRunConfig(
        roots=workspace_roots,
        run_date="2026-07-07",
        source_systems=("gleif",),
        target_systems=("gb",),
        countries=None,
        name_col="name_cleansed",
        representation="tfidf",
        similarity="cosine",
        clustering="knn_cc",
        text_view="name",
        top_k=20,
        min_similarity=0.8,
        max_candidates_per_source=None,
        output_dir=Path("artifacts/validation"),
        prepared_base_dir=None,
        prepared_rows_per_file=1_000_000,
        include_reverse_direction=False,
        exception_policy_path=None,
        tokenizer_scope="invalid",
    )

    with pytest.raises(ValueError, match="tokenizer_scope"):
        validate_run_config(config)


def _token_representation_config(
    roots: WorkspaceRoots, **overrides: Any
) -> ValidationRunConfig:
    base: dict[str, Any] = {
        "roots": roots,
        "run_date": "2026-07-07",
        "source_systems": ("gleif",),
        "target_systems": ("gb",),
        "countries": None,
        "name_col": "name_cleansed",
        "representation": "wordpiece",
        "similarity": "cosine",
        "clustering": "knn_cc",
        "text_view": "tokens",
        "top_k": 20,
        "min_similarity": 0.8,
        "max_candidates_per_source": None,
        "output_dir": Path("artifacts/validation"),
        "prepared_base_dir": None,
        "prepared_rows_per_file": 1_000_000,
        "include_reverse_direction": False,
        "exception_policy_path": None,
    }
    base.update(overrides)
    return ValidationRunConfig(**base)
