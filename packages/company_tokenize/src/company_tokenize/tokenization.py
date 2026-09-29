"""Tokenize a name column in memory, with one tokenizer, a dual pair or several in one pass.

Nothing is removed unless asked: `noise_words_profile` defaults to `"none"`, and naming
a profile trims noise words before encoding, resolved as `tfidf.resolve_noise_words`
says. A `preprocess_profile`, named as a blocking run names it, passes the column
through `name_preprocessing()` first, so the tokens are those of the text a run would
hand the tokenizer. Several calculations in one call share their trim and encoder
caches.

Token lists are variable length and carry no `[PAD]` token; padding belongs to
whatever batches token ids for a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import polars as pl

from .manifest import validate_dual_tokenizer_artifacts
from .name_preprocessing import NAME_PREPROCESSED_COLUMN, name_preprocessing
from .tfidf import resolve_noise_words, trim_text_before_tokenization
from .tokenization_params import (
    CompiledTokenizerCalculationSpec,
    TokenizationRequest,
    TokenizerCalculationSpec,
    coerce_tokenization_request,
    compile_tokenization_request,
)
from .training import load_tokenizer_backend

DEFAULT_TOKENIZER_RESOURCE = "company_wordpiece_tokenizer.json"


@contextmanager
def _resolve_tokenizer_path(
    tokenizer_path: str | Path | None = None,
    *,
    trainer: str = "wordpiece",
):
    if tokenizer_path is not None:
        resolved_path = Path(tokenizer_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {resolved_path}")
        yield resolved_path
        return

    normalized_trainer = trainer.strip().lower()
    if normalized_trainer != "wordpiece":
        raise FileNotFoundError(
            "No bundled tokenizer is available for trainer "
            + f"'{normalized_trainer}'. Provide --tokenizer-path explicitly."
        )

    resource = resources.files(__package__ + ".resources") / DEFAULT_TOKENIZER_RESOURCE
    with resources.as_file(resource) as resolved_path:
        yield resolved_path


def tokenize_name_dataframe(
    df: pl.DataFrame,
    tokenizer_path: str | Path | None = None,
    trainer: str = "wordpiece",
    name_col: str = "name_cleansed",
    token_col: str = "name_tokens",  # nosec B107 - a dataframe column name, not a credential
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
) -> pl.DataFrame:
    request = coerce_tokenization_request(
        tokenizer_specs=[
            TokenizerCalculationSpec(
                token_col=token_col,
                tokenizer_path=tokenizer_path,
                trainer=trainer,
            )
        ],
        name_col=name_col,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        preprocess_profile=preprocess_profile,
    )
    return tokenize_name_dataframe_with_request(df, request=request)


def tokenize_name_dataframe_dual(
    df: pl.DataFrame,
    *,
    country_tokenizer_path: str | Path | None = None,
    global_tokenizer_path: str | Path | None = None,
    country_trainer: str = "wordpiece",
    global_trainer: str = "wordpiece",
    name_col: str = "name_cleansed",
    country_token_col: str = "country_tokens",
    global_token_col: str = "global_tokens",
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
) -> pl.DataFrame:
    if country_tokenizer_path is not None and global_tokenizer_path is not None:
        # Best-effort: only validates when both sides carry a run manifest
        # (see `validate_dual_tokenizer_artifacts`); a `None` path (the
        # bundled default resource) or an artifact without a manifest
        # proceeds unchecked, as it always has.
        validate_dual_tokenizer_artifacts(
            country_tokenizer_path=Path(country_tokenizer_path),
            global_tokenizer_path=Path(global_tokenizer_path),
            country_trainer=country_trainer,
            global_trainer=global_trainer,
        )

    request = coerce_tokenization_request(
        tokenizer_specs=[
            TokenizerCalculationSpec(
                token_col=country_token_col,
                tokenizer_path=country_tokenizer_path,
                trainer=country_trainer,
            ),
            TokenizerCalculationSpec(
                token_col=global_token_col,
                tokenizer_path=global_tokenizer_path,
                trainer=global_trainer,
            ),
        ],
        name_col=name_col,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        preprocess_profile=preprocess_profile,
    )
    return tokenize_name_dataframe_with_request(df, request=request)


def tokenize_name_dataframe_multi(
    df: pl.DataFrame,
    *,
    tokenizer_specs: Sequence[TokenizerCalculationSpec]
    | Sequence[Mapping[str, object]],
    name_col: str = "name_cleansed",
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str | None = None,
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
) -> pl.DataFrame:
    request = coerce_tokenization_request(
        tokenizer_specs=tokenizer_specs,
        name_col=name_col,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        preprocess_profile=preprocess_profile,
    )
    return tokenize_name_dataframe_with_request(df, request=request)


def tokenize_name_dataframe_with_request(
    df: pl.DataFrame,
    *,
    request: TokenizationRequest,
) -> pl.DataFrame:
    compiled_request = compile_tokenization_request(request)
    _validate_name_column(df, name_col=compiled_request.name_col)

    names = _source_names(
        df,
        name_col=compiled_request.name_col,
        preprocess_profile=compiled_request.preprocess_profile,
    )
    default_trimmed_names = _prepare_trimmed_names_from_names(
        names,
        noise_words=compiled_request.noise_words,
        noise_words_path=compiled_request.noise_words_path,
        noise_words_profile=compiled_request.noise_words_profile,
        noise_words_set_kind=compiled_request.noise_words_set_kind,
    )

    runtime_calculations = _build_runtime_calculations(
        compiled_calculations=compiled_request.tokenizer_calculations,
        names=names,
        default_trimmed_names=default_trimmed_names,
        default_profile=compiled_request.noise_words_profile,
        default_set_kind=compiled_request.noise_words_set_kind,
    )

    # One batched encode per calculation rather than one call per row: both
    # backends encode a whole list natively, so this pays the Python-to-Rust
    # boundary cost once per column instead of once per name. Blank rows are
    # held out of the batch and filled back in as `[]`, matching what the
    # per-row path emitted for them.
    output_rows_by_col: dict[str, list[list[str]]] = {}
    for calc in runtime_calculations:
        texts = [text.strip() for text in calc.trimmed_names]
        populated_indices = [index for index, text in enumerate(texts) if text]
        encoded = (
            calc.encode_batch([texts[index] for index in populated_indices])
            if populated_indices
            else []
        )
        rows: list[list[str]] = [[] for _ in texts]
        for position, index in enumerate(populated_indices):
            rows[index] = encoded[position]
        output_rows_by_col[calc.token_col] = rows

    output_columns = [
        pl.Series(calc.token_col, output_rows_by_col[calc.token_col])
        for calc in runtime_calculations
    ]

    return _with_token_output_columns(
        df,
        output_columns=output_columns,
        trimmed_name_col=compiled_request.trimmed_name_col,
        trimmed_names=default_trimmed_names,
    )


def _source_names(
    df: pl.DataFrame, *, name_col: str, preprocess_profile: str | None
) -> list[str]:
    """The text each row is tokenized from.

    That is `name_col`, passed through `name_preprocessing` under
    `preprocess_profile` when one is given.
    """
    if preprocess_profile is None:
        return df[name_col].to_list()
    return [
        value or ""
        for value in name_preprocessing(
            df, name_col=name_col, profile=preprocess_profile
        )[NAME_PREPROCESSED_COLUMN].to_list()
    ]


@dataclass(frozen=True)
class _RuntimeCalculation:
    token_col: str
    trimmed_names: list[str]
    encode_batch: Callable[[list[str]], list[list[str]]]


def _build_runtime_calculations(
    *,
    compiled_calculations: tuple[CompiledTokenizerCalculationSpec, ...],
    names: list[str],
    default_trimmed_names: list[str],
    default_profile: str,
    default_set_kind: str,
) -> list[_RuntimeCalculation]:
    trim_cache: dict[tuple[object, object, str, str], list[str]] = {}
    encoder_cache: dict[tuple[str, str], Callable[[list[str]], list[list[str]]]] = {}
    runtime_calculations: list[_RuntimeCalculation] = []

    for calc in compiled_calculations:
        spec_names = _resolve_spec_names(
            names=names,
            default_trimmed_names=default_trimmed_names,
            calc=calc,
            default_profile=default_profile,
            default_set_kind=default_set_kind,
            trim_cache=trim_cache,
        )
        encode_batch = _resolve_encoder(
            tokenizer_path=calc.tokenizer_path,
            trainer=calc.trainer,
            encoder_cache=encoder_cache,
        )
        runtime_calculations.append(
            _RuntimeCalculation(
                token_col=calc.token_col,
                trimmed_names=spec_names,
                encode_batch=encode_batch,
            )
        )

    return runtime_calculations


def _resolve_spec_names(
    *,
    names: list[str],
    default_trimmed_names: list[str],
    calc: CompiledTokenizerCalculationSpec,
    default_profile: str,
    default_set_kind: str,
    trim_cache: dict[tuple[object, object, str, str], list[str]],
) -> list[str]:
    if calc.noise_words is None and calc.noise_words_path is None:
        return default_trimmed_names

    cache_key = (
        _noise_words_key(calc.noise_words),
        str(calc.noise_words_path) if calc.noise_words_path is not None else None,
        calc.noise_words_profile or default_profile,
        calc.noise_words_set_kind or default_set_kind,
    )
    cached = trim_cache.get(cache_key)
    if cached is not None:
        return cached

    effective_noise_words = resolve_noise_words(
        noise_words=calc.noise_words,
        noise_words_path=calc.noise_words_path,
        noise_words_profile=calc.noise_words_profile,
        noise_words_set_kind=calc.noise_words_set_kind,
    )
    trimmed = trim_text_before_tokenization(
        names,
        noise_words=effective_noise_words,
    )
    trim_cache[cache_key] = trimmed
    return trimmed


def _resolve_encoder(
    *,
    tokenizer_path: str | Path | None,
    trainer: str,
    encoder_cache: dict[tuple[str, str], Callable[[list[str]], list[list[str]]]],
) -> Callable[[list[str]], list[list[str]]]:
    """Resolve one calculation's batched encoder, loading each model once.

    Goes through `load_tokenizer_backend` rather than
    `load_tokenizer_encoder` because the executor needs the backend's
    `encode_batch`; `load_tokenizer_encoder`'s `str -> list[str]` pair
    stays as-is for its own consumers.
    """
    trainer_key = trainer.strip().lower()
    path_key = (
        str(tokenizer_path) if tokenizer_path is not None else "<bundled-default>"
    )
    cache_key = (trainer_key, path_key)
    cached = encoder_cache.get(cache_key)
    if cached is not None:
        return cached

    with _resolve_tokenizer_path(
        tokenizer_path, trainer=trainer_key
    ) as resolved_tokenizer_path:
        backend = load_tokenizer_backend(
            tokenizer_path=resolved_tokenizer_path,
            trainer=trainer_key,
        )

    encoder_cache[cache_key] = backend.encode_batch
    return backend.encode_batch


def _noise_words_key(
    noise_words: set[str] | list[str] | tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    if noise_words is None:
        return None
    if isinstance(noise_words, set):
        return tuple(sorted(noise_words))
    return tuple(noise_words)


def _prepare_trimmed_names(
    df: pl.DataFrame,
    *,
    name_col: str,
    noise_words: set[str] | list[str] | tuple[str, ...] | None,
    noise_words_path: str | Path | None,
    noise_words_profile: str,
    noise_words_set_kind: str,
) -> list[str]:
    _validate_name_column(df, name_col=name_col)
    names = df[name_col].to_list()
    return _prepare_trimmed_names_from_names(
        names,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )


def _prepare_trimmed_names_from_names(
    names: list[str],
    *,
    noise_words: set[str] | list[str] | tuple[str, ...] | None,
    noise_words_path: str | Path | None,
    noise_words_profile: str,
    noise_words_set_kind: str,
) -> list[str]:
    effective_noise_words = resolve_noise_words(
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )
    return trim_text_before_tokenization(
        names,
        noise_words=effective_noise_words,
    )


def _with_token_output_columns(
    df: pl.DataFrame,
    *,
    output_columns: list[pl.Series],
    trimmed_name_col: str | None,
    trimmed_names: list[str],
) -> pl.DataFrame:
    if trimmed_name_col is not None:
        output_columns.append(pl.Series(trimmed_name_col, trimmed_names))
    return df.with_columns(output_columns)


def _validate_name_column(df: pl.DataFrame, *, name_col: str) -> None:
    if name_col not in df.columns:
        raise ValueError(f"DataFrame does not contain required column '{name_col}'.")
