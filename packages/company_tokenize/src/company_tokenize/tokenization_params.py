"""Typed requests for tokenizing one name column through several tokenizers in one pass.

`compile_tokenization_request()` and `coerce_tokenization_request()` validate a
`TokenizationRequest`, or a list of specs or mappings, into a
`CompiledTokenizationRequest`, and raise on duplicate output columns or an empty name
column. Each class's `Attributes:` section documents its fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

NoiseWordsInput = set[str] | list[str] | tuple[str, ...] | None


@dataclass(frozen=True)
class TokenizerCalculationSpec:
    """One tokenizer calculation within a (possibly multi-calculation) request.

    Any ``noise_words*`` field left as ``None`` falls back to the enclosing
    ``TokenizationRequest``'s own default for that field.

    Attributes:
        token_col: Output column name this calculation's token list is
            written to. Must be unique across all calculations in the same
            request; duplicates raise in ``compile_tokenization_request()``.
        tokenizer_path: Path to the trained tokenizer artifact to use.
            Required to be resolvable (non-``None``) before file-based
            tokenization runs; DataFrame tokenization callers may resolve it
            elsewhere.
        trainer: Tokenizer backend that produced/reads ``tokenizer_path``.
            One of ``"wordpiece"`` (default) or ``"sentencepiece"`` (see
            ``SUPPORTED_TRAINERS``).
        label: Human-readable label for this calculation, carried through to
            ``CompiledTokenizerCalculationSpec.label`` for caller bookkeeping.
            Defaults to ``token_col`` when ``None``.
        noise_words: Explicit noise-word set for this calculation only,
            overriding both the request-level default and any profile/path
            resolution. ``None`` defers to the request default.
        noise_words_path: JSON noise-word artifact path for this calculation
            only (for example a per-corpus generated list), overriding the
            packaged profiled default. ``None`` defers to the request default.
        noise_words_profile: Noise-word strictness profile for this
            calculation. One of ``"none"``, ``"strict"``, ``"balanced"``, or
            ``"aggressive"``. ``None`` defers to the request default.
        noise_words_set_kind: Which token set to draw from the resolved
            profile. ``"combined"`` accumulates strict->balanced->aggressive
            up to ``noise_words_profile``; ``"tokens"`` selects only that
            profile's own token list. ``None`` defers to the request default.
    """

    token_col: str
    tokenizer_path: str | Path | None = None
    trainer: str = "wordpiece"
    label: str | None = None
    noise_words: NoiseWordsInput = None
    noise_words_path: str | Path | None = None
    noise_words_profile: str | None = None
    noise_words_set_kind: str | None = None


@dataclass(frozen=True)
class TokenizationRequest:
    """Typed request for tokenizer calculations over one source text column.

    Every calculation in the request runs over that column in a single pass.

    Attributes:
        tokenizer_calculations: One or more per-tokenizer specs to run.
            ``token_col`` values must be unique across the tuple.
        name_col: Source text column read from the input DataFrame/parquet
            file. Defaults to ``"name_cleansed"``, the cleanse-stage output.
        noise_words: Default explicit noise-word set applied to any
            calculation that does not set its own ``noise_words``. ``None``
            defers to path/profile resolution.
        noise_words_path: Default JSON noise-word artifact path applied to
            any calculation that does not set its own. ``None`` defers to
            the packaged profiled default.
        noise_words_profile: Default noise-word strictness profile. One of
            ``"none"`` (default, removes no words), ``"strict"``,
            ``"balanced"``, or ``"aggressive"``.
        noise_words_set_kind: Default token-set selection within the
            resolved profile. ``"combined"`` (default) accumulates
            strict->balanced->aggressive up to ``noise_words_profile``;
            ``"tokens"`` selects only that profile's own token list.
        trimmed_name_col: Optional output column to also persist the
            noise-word-trimmed text alongside the token columns. ``None``
            (default) does not persist trimmed text.
        preprocess_profile: A `name_preprocessing` profile the name column
            passes through before noise words are trimmed and it is
            tokenized, named as a blocking run names it. ``None`` (default)
            tokenizes the column as it is.
    """

    tokenizer_calculations: tuple[TokenizerCalculationSpec, ...]
    name_col: str = "name_cleansed"
    noise_words: NoiseWordsInput = None
    noise_words_path: str | Path | None = None
    noise_words_profile: str = "none"
    noise_words_set_kind: str = "combined"
    trimmed_name_col: str | None = None
    preprocess_profile: str | None = None


@dataclass(frozen=True)
class CompiledTokenizerCalculationSpec:
    token_col: str
    tokenizer_path: Path | None
    trainer: str
    label: str
    noise_words: NoiseWordsInput
    noise_words_path: Path | None
    noise_words_profile: str
    noise_words_set_kind: str


@dataclass(frozen=True)
class CompiledTokenizationRequest:
    tokenizer_calculations: tuple[CompiledTokenizerCalculationSpec, ...]
    name_col: str
    noise_words: NoiseWordsInput
    noise_words_path: Path | None
    noise_words_profile: str
    noise_words_set_kind: str
    trimmed_name_col: str | None
    preprocess_profile: str | None


def _normalize_optional_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    return Path(str(path_value))


def _validate_noise_words_type(noise_words: NoiseWordsInput, *, context: str) -> None:
    if noise_words is not None and not isinstance(noise_words, (set, list, tuple)):
        raise ValueError(f"{context} must be a set, list, or tuple when provided.")


def _coerce_calculation_spec(
    spec: TokenizerCalculationSpec | Mapping[str, object],
    *,
    default_noise_words_profile: str,
    default_noise_words_set_kind: str,
) -> TokenizerCalculationSpec:
    if isinstance(spec, TokenizerCalculationSpec):
        return spec

    token_col = str(spec.get("token_col", "")).strip()
    return TokenizerCalculationSpec(
        token_col=token_col,
        tokenizer_path=cast("str | Path | None", spec.get("tokenizer_path")),
        trainer=str(spec.get("trainer", "wordpiece") or "wordpiece"),
        label=(str(spec.get("label", "")).strip() or None),
        noise_words=(
            cast(NoiseWordsInput, spec.get("noise_words"))
            if "noise_words" in spec
            else None
        ),
        noise_words_path=cast("str | Path | None", spec.get("noise_words_path")),
        noise_words_profile=(
            str(spec.get("noise_words_profile", default_noise_words_profile)).strip()
            or default_noise_words_profile
        ),
        noise_words_set_kind=(
            str(spec.get("noise_words_set_kind", default_noise_words_set_kind)).strip()
            or default_noise_words_set_kind
        ),
    )


def coerce_tokenization_request(
    *,
    tokenizer_specs: Sequence[TokenizerCalculationSpec]
    | Sequence[Mapping[str, object]],
    name_col: str = "name_cleansed",
    noise_words: NoiseWordsInput = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
    preprocess_profile: str | None = None,
) -> TokenizationRequest:
    calculations = tuple(
        _coerce_calculation_spec(
            spec,
            default_noise_words_profile=noise_words_profile,
            default_noise_words_set_kind=noise_words_set_kind,
        )
        for spec in tokenizer_specs
    )
    return TokenizationRequest(
        tokenizer_calculations=calculations,
        name_col=name_col,
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        preprocess_profile=preprocess_profile,
    )


def compile_tokenization_request(
    request: TokenizationRequest,
) -> CompiledTokenizationRequest:
    name_col = request.name_col.strip()
    if not name_col:
        raise ValueError("name_col must be non-empty.")

    if not request.tokenizer_calculations:
        raise ValueError(
            "tokenizer_specs must contain at least one tokenizer specification."
        )

    default_profile = request.noise_words_profile.strip() or "none"
    default_set_kind = request.noise_words_set_kind.strip() or "combined"
    _validate_noise_words_type(request.noise_words, context="noise_words")

    compiled_calculations: list[CompiledTokenizerCalculationSpec] = []
    seen_token_cols: set[str] = set()
    for calc in request.tokenizer_calculations:
        token_col = calc.token_col.strip()
        if not token_col:
            raise ValueError("Each tokenizer spec requires non-empty 'token_col'.")
        if token_col in seen_token_cols:
            raise ValueError(f"Duplicate token column requested: '{token_col}'.")
        seen_token_cols.add(token_col)

        trainer = (calc.trainer or "wordpiece").strip().lower()
        label = (calc.label or token_col).strip() or token_col
        spec_profile = (
            calc.noise_words_profile or default_profile
        ).strip() or default_profile
        spec_set_kind = (
            calc.noise_words_set_kind or default_set_kind
        ).strip() or default_set_kind
        _validate_noise_words_type(
            calc.noise_words, context="Tokenizer spec 'noise_words'"
        )

        compiled_calculations.append(
            CompiledTokenizerCalculationSpec(
                token_col=token_col,
                tokenizer_path=_normalize_optional_path(calc.tokenizer_path),
                trainer=trainer,
                label=label,
                noise_words=calc.noise_words,
                noise_words_path=_normalize_optional_path(calc.noise_words_path),
                noise_words_profile=spec_profile,
                noise_words_set_kind=spec_set_kind,
            )
        )

    return CompiledTokenizationRequest(
        tokenizer_calculations=tuple(compiled_calculations),
        name_col=name_col,
        noise_words=request.noise_words,
        noise_words_path=_normalize_optional_path(request.noise_words_path),
        noise_words_profile=default_profile,
        noise_words_set_kind=default_set_kind,
        trimmed_name_col=request.trimmed_name_col,
        preprocess_profile=request.preprocess_profile,
    )
