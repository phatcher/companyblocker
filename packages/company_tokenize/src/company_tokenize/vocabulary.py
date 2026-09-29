"""A trained tokenizer's vocabulary and noise words as one unit, for callers outside tokenization.

`load_tokenizer_vocabulary()` is for a consumer that needs the whole vocabulary rather
than tokenized text, such as a classifier building its features from the same trained
tokenizer. It shares its loading with `load_tokenizer_encoder()` and its noise-word
resolution with `resolve_noise_words()`, so it agrees with the tokenize path by
construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .tfidf import resolve_noise_words
from .training import load_tokenizer_backend


@dataclass(frozen=True)
class TokenizerVocabulary:
    """A trained tokenizer's subword vocabulary plus its resolved noise-word set.

    This is the documented unit `load_tokenizer_vocabulary()` returns
    for a consumer outside the tokenize stage's file/dataframe entry points
    (`ops.py`/`tokenization.py`, built for `scripts/process_companies.py`) --
    for example `company_classify` building its own feature
    vocabulary from the same trained tokenizer artifact the tokenize stage
    encodes company names against, instead of maintaining an independent one.

    Attributes:
        trainer: normalized trainer backend name (`"wordpiece"` or
            `"sentencepiece"`).
        tokenizer_path: resolved path to the loaded tokenizer artifact.
        vocab: `token -> id` mapping exactly as trained -- the full
            vocabulary the tokenizer backend knows. Noise words are not
            removed from this mapping: noise-word trimming happens on raw
            text *before* encoding (see `noise_words` below and
            `tfidf.trim_text_before_tokenization`), not by filtering
            vocabulary entries after training.
        unk_token: this tokenizer's unknown-token marker -- `"[UNK]"` for
            WordPiece, the model's own configured unk piece for
            SentencePiece (identical resolution to `load_tokenizer_encoder`).
        noise_words: the resolved noise-word set (same precedence and
            packaged defaults as `tfidf.resolve_noise_words`) a caller
            should trim from raw text before encoding/feature-building, so
            results stay consistent with the tokenize stage's own noise
            policy.
        encode: `text -> token list` callable for this artifact/trainer,
            identical to what `load_tokenizer_encoder` returns.
    """

    trainer: str
    tokenizer_path: Path
    vocab: Mapping[str, int]
    unk_token: str
    noise_words: frozenset[str]
    encode: Callable[[str], list[str]]


def load_tokenizer_vocabulary(
    tokenizer_path: str | Path,
    *,
    trainer: str = "wordpiece",
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
) -> TokenizerVocabulary:
    """Load a trained tokenizer's vocabulary and resolved noise-word set as one unit.

    This is the documented, stable entry point for a consumer
    outside the tokenize stage's file/dataframe APIs that needs the same
    trained artifact -- for example `company_classify` building a feature
    vocabulary from it instead of a second, independently-tuned
    `sklearn.TfidfVectorizer`.

    It shares its tokenizer-loading and noise-word-resolution logic with the
    tokenize stage's own path (`training.load_tokenizer_backend`, the same
    loader `load_tokenizer_encoder` uses, and `tfidf.resolve_noise_words`)
    rather than re-implementing either, so the two stay in parity by
    construction -- see
    `test_load_tokenizer_vocabulary_matches_tokenize_stage_encoding` in
    `tests/test_company_tokenize_vocabulary.py` for a fixture proving
    identical output against the same trained artifact.

    Noise-word resolution precedence matches `resolve_noise_words()`
    exactly: explicit `noise_words` > `noise_words_path` > packaged
    profiled defaults (`noise_words_profile`/`noise_words_set_kind`).

    Raises:
        FileNotFoundError: if `tokenizer_path` does not exist.
        ValueError: if `trainer` is not a supported trainer backend name.
    """
    resolved_path = Path(tokenizer_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {resolved_path}")

    backend = load_tokenizer_backend(tokenizer_path=resolved_path, trainer=trainer)
    resolved_noise_words = resolve_noise_words(
        noise_words=noise_words,
        noise_words_path=noise_words_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )

    return TokenizerVocabulary(
        trainer=backend.trainer,
        tokenizer_path=resolved_path,
        vocab=backend.vocab,
        unk_token=backend.unk_token,
        noise_words=frozenset(resolved_noise_words),
        encode=backend.encode,
    )
