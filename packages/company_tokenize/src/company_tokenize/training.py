"""Train WordPiece and SentencePiece tokenizers, load them, and lay out optimize runs.

`train_wordpiece()` and `train_sentencepiece()` train from a prepared parquet corpus
with `system_uri` and `name` columns. Building that corpus, sampling which rows and
which name column, is the caller's.

WordPiece's `continuing_subword_prefix` is fixed to `##` and its `special_tokens` to
`[UNK]`. These are compatibility rules, not tuning knobs: changing either gives a
different token namespace, not comparable with existing tokenizers or token scores.

Training is a greedy approximation, and it is not repeatable. Choosing a vocabulary
is NP-complete, so no trained vocabulary is the optimum. Training the same corpus
twice gives a slightly different one: the trainer's threads merge their partial
counts, and a merge tied with another is decided by whichever thread finished
first. On `ie` (vocabulary 40,000, minimum frequency 1) two runs differed in 5
words while over half the token ids moved; a three-name fixture at vocabulary 60
varies by 15-21%. So a comparison between tokenizers needs an effect larger than
retraining's own spread, and reproducing a result means keeping its model rather
than training again. Compare two vocabularies by their words, never their ids.

`train_sentencepiece()` takes three settings WordPiece has no equivalent of, and each
changes what `fertility` and `unk_rate` mean, so a comparison holds them fixed and
states them:

- `model_type`: `bpe` merges the most frequent adjacent pairs; `unigram` prunes a
  large piece inventory to the one maximizing corpus likelihood. Vocabularies of
  the same size can differ in fertility on identical input.
- `character_coverage`: the share of the corpus's characters guaranteed their own
  piece. `1.0` covers every character seen; lower values suit very large alphabets.
- `byte_fallback`: out-of-vocabulary characters decompose into UTF-8 byte pieces
  instead of the unknown token. It reserves the byte inventory up front, so
  training fails below that floor: 268 pieces in the one case measured. With it on, `unk_rate`
  trends to zero whatever the vocabulary's quality and fertility carries the
  coverage signal instead, so `unk_rate` is not comparable across the setting, or
  against WordPiece, which has no fallback.

An optimize run of one tokenizer sits in `<scope>/<tokenizer id>/optimize/`, under
`<corpus_hash8>/<grid_hash8>/`: a resample changes the corpus hash, a different grid
or SentencePiece setting changes the grid hash, and a different tokenizer changes
the folder, so differently configured runs never overwrite each other. The corpus
hash is of the file's bytes (`compute_corpus_content_hash()`), never its `mtime`.
Seed splits sit at scope level, `<scope>/optimize/<corpus_hash8>/splits/`, shared by
every tokenizer trained on that corpus. `resolve_active_optimize_sweep_dir()` finds the
most recent run through an unhashed pointer file; older runs stay on disk but are
not indexed.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

DEFAULT_WORDPIECE_VOCAB_MULTIPLIER = 1.5
DEFAULT_WORDPIECE_MIN_VOCAB_SIZE = 1000
DEFAULT_WORDPIECE_MAX_VOCAB_SIZE = 200_000
SUPPORTED_TRAINERS = ("wordpiece", "sentencepiece")


@dataclass(frozen=True)
class TrainerOptions:
    """SentencePiece-only training knobs, threaded through regardless of trainer.

    WordPiece training ignores all three fields. See the package README's "SentencePiece
    Knobs" section for how each one changes what `fertility`/`unk_rate` mean.

    Attributes:
        tokenizer_encoding: SentencePiece segmentation algorithm. One of
            ``"bpe"`` (default) or ``"unigram"``.
        sp_character_coverage: Fraction of the training corpus's character
            distribution guaranteed an individually-covered vocabulary slot.
            Must be in the interval ``(0, 1]``; ``1.0`` (default) covers
            every character seen in training.
        sp_byte_fallback: Whether out-of-vocabulary characters decompose
            into raw UTF-8 byte pieces instead of producing ``[UNK]``.
            Reserves vocabulary slots for the full byte inventory up front,
            so training fails below a small vocab-size floor when ``True``.
    """

    tokenizer_encoding: str = "bpe"
    sp_character_coverage: float = 1.0
    sp_byte_fallback: bool = False


def _is_sentencepiece_trainer(normalized_trainer: str) -> bool:
    return normalized_trainer == "sentencepiece"


def resolve_tokenizer_path_for_trainer(
    *, trainer: str, scope_directory: Path, tokenizer_encoding: str | None = None
) -> Path:
    """The model file of one tokenizer in a scope's working directory."""
    from .paths import tokenizer_directory, tokenizer_directory_files

    return tokenizer_directory_files(
        tokenizer_directory(
            scope_directory, trainer=trainer, tokenizer_encoding=tokenizer_encoding
        ),
        trainer=trainer,
    ).model


def resolve_metadata_path_for_trainer(
    *, trainer: str, scope_directory: Path, tokenizer_encoding: str | None = None
) -> Path:
    """The metadata file of one tokenizer in a scope's working directory."""
    from .paths import tokenizer_directory, tokenizer_directory_files

    return tokenizer_directory_files(
        tokenizer_directory(
            scope_directory, trainer=trainer, tokenizer_encoding=tokenizer_encoding
        ),
        trainer=trainer,
    ).metadata


def compute_corpus_content_hash(path: Path) -> str:
    """Content identity for a training corpus file, independent of mtime.

    A filesystem timestamp changes on any touch (a `--mode train` run, a git
    checkout) even when the bytes are identical, which makes anything keyed
    off `mtime` alone (canaries, sweep-directory hashes) spuriously mismatch
    and discard cached candidates for no real reason. Hashing the bytes keys
    reuse off what actually changed. Shared by `src/training/optimize_execution.py`
    and this package's own `candidate_archive.py`/optimize-sweep-dir
    resolution, so it lives here rather than being duplicated per caller.
    """
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OptimizeSweepPaths:
    """Every path belonging to exactly one optimize-sweep configuration.

    `sweep_dir` sits in one tokenizer's own folder and is keyed by a layered
    hash (corpus -> grid, see `resolve_optimize_sweep_paths`), not a single
    flat hash of the whole canary payload -- so a bpe sweep and a unigram
    sweep on the same corpus, or two grids against the same corpus, never
    share a directory and never need to purge each other to proceed.
    """

    sweep_dir: Path
    models_dir: Path
    run_log_dir: Path
    run_log_path: Path
    summary_path: Path
    canary_sidecar_path: Path


OPTIMIZE_DIRNAME = "optimize"
OPTIMIZE_RUN_LOG_FILENAME = "optimize_runs.parquet"
OPTIMIZE_SUMMARY_FILENAME = "optimize_summary.json"
OPTIMIZE_CANARY_POINTER_FILENAME = "optimize_canary.json"


def resolve_optimize_dir(
    *, scope_directory: Path, trainer: str, tokenizer_encoding: str | None = None
) -> Path:
    """Where one tokenizer's sweeps sit: `optimize/` inside its own folder."""
    from .paths import tokenizer_directory

    return (
        tokenizer_directory(
            scope_directory, trainer=trainer, tokenizer_encoding=tokenizer_encoding
        )
        / OPTIMIZE_DIRNAME
    )


def resolve_optimize_sweep_paths(
    *, optimize_dir: Path, corpus_content_hash: str, grid_hash: str
) -> OptimizeSweepPaths:
    """The paths of one sweep beneath a tokenizer's `optimize_dir`.

    The folder already says which tokenizer this is, and the grid hash covers
    every training option the folder does not, so the leaf is the corpus and
    the grid and nothing else.
    """
    sweep_dir = optimize_dir / corpus_content_hash[:8] / grid_hash
    run_log_path = sweep_dir / OPTIMIZE_RUN_LOG_FILENAME
    summary_path = sweep_dir / OPTIMIZE_SUMMARY_FILENAME
    return OptimizeSweepPaths(
        sweep_dir=sweep_dir,
        models_dir=sweep_dir / "models",
        run_log_dir=sweep_dir / "run_logs",
        run_log_path=run_log_path,
        summary_path=summary_path,
        canary_sidecar_path=sweep_dir / "canary.json",
    )


def resolve_optimize_splits_dir(
    *, scope_directory: Path, corpus_content_hash: str
) -> Path:
    """Paths for the seed/train/validation split artifacts of one corpus.

    Those artifacts depend only on corpus identity plus seed/validation_fraction, checked
    independently at the call site (see `resolve_or_create_seed_split_artifacts`), and not
    on trainer, encoding or grid. So they live at scope level, beside the corpus
    they are cut from, shared by every tokenizer's sweeps against the same
    corpus rather than duplicated per tokenizer.
    """
    return scope_directory / OPTIMIZE_DIRNAME / corpus_content_hash[:8] / "splits"


def resolve_active_optimize_sweep_dir(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None,
    profile: str,
    trainer: str,
    tokenizer_encoding: str | None = None,
) -> Path | None:
    """Resolve the most recent optimize-sweep leaf for (trainer, tokenizer_encoding).

    Reads the unhashed pointer file, for a caller that knows only the trainer and model
    type up front rather than the historical grid: a human-facing CLI, or report
    generation triggered by an archive event. Recomputes
    `corpus_content_hash` fresh from the live corpus file rather than
    trusting anything cached, since that's cheap and always correct. Returns
    None if no sweep has ever been recorded for this trainer/model-type, or
    the corpus itself no longer exists.

    Lives here (not in `scripts/`) so both `scripts/plot_optimize_elbow.py`
    and `src/training/tokenizer_corpus_report.py` can resolve the same path
    without `tokenizer_corpus_report.py` (in `src/training`) needing to
    depend on `scripts`, which the module dependency graph doesn't allow.

    Takes an already-resolved `tokenizer_root` rather than deriving
    `artifacts/tokenizers` from a project root itself, matching every other
    resolver in this package: the caller resolves that root (typically via
    `workspace.artifact_layout.tokenizer_artifact_root`) and hands it in.
    """
    from .paths import resolve_tokenizer_paths

    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_root, scope=scope, system=system, profile=profile
    )
    if not tokenizer_paths.corpus_path.exists():
        return None
    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_paths.directory,
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    if not pointer_path.exists():
        return None
    pointer_payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(tokenizer_paths.corpus_path),
        grid_hash=str(pointer_payload["grid_hash"]),
    )
    return sweep_paths.sweep_dir


def resolve_optimize_canary_pointer_path(*, optimize_dir: Path) -> Path:
    """Small, unhashed pointer to the latest sweep in a tokenizer's `optimize_dir`.

    `plot_optimize_elbow.py` and `archive_optimize_candidate.py` know only the
    tokenizer up front, not a historical grid, so they read this file's
    `grid_hash` (recomputing `corpus_content_hash` themselves) rather than needing an
    index of every past sweep. Older sweep directories still exist on disk
    and remain manually discoverable; this is deliberately not an index of
    all of them.
    """
    return optimize_dir / OPTIMIZE_CANARY_POINTER_FILENAME


def resolve_candidate_model_suffix_for_trainer(*, trainer: str) -> str:
    normalized_trainer = _normalize_trainer_name(trainer)
    return "model" if normalized_trainer == "sentencepiece" else "json"


def validate_trainer_options(*, trainer: str, options: TrainerOptions) -> None:
    normalized_trainer = _normalize_trainer_name(trainer)
    if _is_sentencepiece_trainer(normalized_trainer) and not (
        0.0 < options.sp_character_coverage <= 1.0
    ):
        raise ValueError("--sp-character-coverage must be in the interval (0, 1].")


def normalize_optimize_min_frequencies_for_trainer(
    *, trainer: str, min_frequencies: list[int]
) -> list[int]:
    normalized_trainer = _normalize_trainer_name(trainer)
    if _is_sentencepiece_trainer(normalized_trainer):
        if any(value != 1 for value in min_frequencies):
            raise ValueError(
                "SentencePiece optimize currently supports only --min-frequencies 1."
            )
        return [1]
    return min_frequencies


def trainer_params_payload(
    *, trainer: str, options: TrainerOptions
) -> dict[str, object]:
    normalized_trainer = _normalize_trainer_name(trainer)
    if _is_sentencepiece_trainer(normalized_trainer):
        return {
            "tokenizer_encoding": options.tokenizer_encoding,
            "sp_character_coverage": options.sp_character_coverage,
            "sp_byte_fallback": options.sp_byte_fallback,
        }
    return {
        "tokenizer_encoding": None,
        "sp_character_coverage": None,
        "sp_byte_fallback": None,
    }


def _normalize_trainer_name(trainer: str) -> str:
    normalized = trainer.strip().lower()
    if normalized not in SUPPORTED_TRAINERS:
        allowed = ", ".join(SUPPORTED_TRAINERS)
        raise ValueError(
            f"Unsupported trainer '{trainer}'. Expected one of: {allowed}."
        )
    return normalized


def _read_corpus_names(corpus_path: Path) -> list[str]:
    corpus_df = pl.read_parquet(corpus_path, columns=["name"])
    return [
        str(value).strip()
        for value in corpus_df.get_column("name").to_list()
        if value is not None and str(value).strip()
    ]


def estimate_wordpiece_vocab_size(
    corpus_path: Path,
    *,
    multiplier: float = DEFAULT_WORDPIECE_VOCAB_MULTIPLIER,
    min_vocab_size: int = DEFAULT_WORDPIECE_MIN_VOCAB_SIZE,
    max_vocab_size: int = DEFAULT_WORDPIECE_MAX_VOCAB_SIZE,
) -> int:
    """Estimate a practical WordPiece vocabulary size from a prepared parquet corpus.

    The estimate is based on unique whitespace-delimited tokens in the corpus ``name``
    column, scaled by ``multiplier`` and clamped to the provided minimum and maximum.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Training corpus parquet not found: {corpus_path}. Provide a prepared corpus first."
        )
    if multiplier <= 0:
        raise ValueError("multiplier must be greater than zero.")
    if min_vocab_size <= 0:
        raise ValueError("min_vocab_size must be greater than zero.")
    if max_vocab_size < min_vocab_size:
        raise ValueError("max_vocab_size cannot be less than min_vocab_size.")

    corpus_df = pl.read_parquet(corpus_path, columns=["name"])
    unique_tokens: set[str] = set()
    for value in corpus_df.get_column("name").to_list():
        text = "" if value is None else str(value)
        unique_tokens.update(token for token in text.split() if token)

    unique_token_count = max(len(unique_tokens), 1)
    estimated_vocab_size = max(
        unique_token_count, math.ceil(unique_token_count * multiplier)
    )
    return min(max(estimated_vocab_size, min_vocab_size), max_vocab_size)


def train_wordpiece(
    corpus_path: Path,
    tokenizer_path: Path,
    vocab_size: int | None = None,
    min_frequency: int = 1,
    show_progress: bool = True,
) -> int:
    """Train a WordPiece tokenizer from a prepared parquet corpus.

    When ``vocab_size`` is omitted, the function estimates a vocabulary size from the
    corpus and returns the resolved value used for training.

    The trainer currently reserves only ``[UNK]`` as a special token. It does not add
    ``[PAD]`` because this package returns variable-length token lists and does not own
    model-ready fixed-length batching semantics. If a downstream embedding or model
    pipeline later requires padded token-id batches, padding should be introduced at
    that batching layer or added here alongside explicit padding behavior.
    """
    if vocab_size is not None and vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero when provided.")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be greater than zero.")

    resolved_vocab_size = (
        estimate_wordpiece_vocab_size(corpus_path) if vocab_size is None else vocab_size
    )

    names = _read_corpus_names(corpus_path)

    print(
        f"Training WordPiece tokenizer (vocab_size={resolved_vocab_size})"
        + (" [auto]..." if vocab_size is None else "...")
    )

    tokenizer = Tokenizer(
        WordPiece(unk_token="[UNK]")  # nosec B106 - the unknown-token marker, not a credential
    )
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=resolved_vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]"],
        continuing_subword_prefix="##",
        show_progress=show_progress,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as temp_corpus:
        temp_path = Path(temp_corpus.name)
        temp_corpus.write("\n".join(names))

    try:
        tokenizer.train(files=[str(temp_path)], trainer=trainer)
    finally:
        temp_path.unlink(missing_ok=True)

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    print(f"Tokenizer saved -> {tokenizer_path}")
    return resolved_vocab_size


def train_sentencepiece(
    corpus_path: Path,
    tokenizer_path: Path,
    vocab_size: int | None = None,
    min_frequency: int = 1,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
    byte_fallback: bool = False,
) -> int:
    """Train a SentencePiece tokenizer from a prepared parquet corpus.

    SentencePiece does not expose a direct WordPiece-like min-frequency control.
    The parameter is accepted for interface parity and must currently remain 1.
    """
    if vocab_size is not None and vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero when provided.")
    if min_frequency != 1:
        raise ValueError(
            "SentencePiece training currently supports only min_frequency=1."
        )

    normalized_model_type = model_type.strip().lower()
    if normalized_model_type not in {"unigram", "bpe"}:
        raise ValueError("model_type must be either 'unigram' or 'bpe'.")
    if not (0.0 < character_coverage <= 1.0):
        raise ValueError("character_coverage must be in the interval (0, 1].")

    resolved_vocab_size = (
        estimate_wordpiece_vocab_size(corpus_path) if vocab_size is None else vocab_size
    )
    names = _read_corpus_names(corpus_path)

    print(
        f"Training SentencePiece tokenizer (model_type={normalized_model_type}, vocab_size={resolved_vocab_size})"
        + (" [auto]..." if vocab_size is None else "...")
    )

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as temp_corpus:
        temp_path = Path(temp_corpus.name)
        temp_corpus.write("\n".join(names))

    with tempfile.TemporaryDirectory() as temp_model_dir:
        prefix = Path(temp_model_dir) / "sentencepiece"
        try:
            spm.SentencePieceTrainer.Train(
                input=str(temp_path),
                model_prefix=str(prefix),
                model_type=normalized_model_type,
                vocab_size=resolved_vocab_size,
                character_coverage=character_coverage,
                byte_fallback=byte_fallback,
                unk_piece="[UNK]",
                bos_id=-1,
                eos_id=-1,
                hard_vocab_limit=False,
                # The library's per-merge progress lines; warnings and errors still print.
                minloglevel=1,
            )
        finally:
            temp_path.unlink(missing_ok=True)

        model_file = Path(f"{prefix}.model")
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer_path.write_bytes(model_file.read_bytes())

    print(f"Tokenizer saved -> {tokenizer_path}")
    return resolved_vocab_size


def train_tokenizer_with_trainer(
    *,
    trainer: str,
    corpus_path: Path,
    tokenizer_path: Path,
    vocab_size: int | None = None,
    min_frequency: int = 1,
    show_progress: bool = True,
    options: TrainerOptions | None = None,
) -> int:
    normalized_trainer = _normalize_trainer_name(trainer)
    effective_options = options or TrainerOptions()
    validate_trainer_options(trainer=normalized_trainer, options=effective_options)

    if _is_sentencepiece_trainer(normalized_trainer):
        return train_sentencepiece(
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            model_type=effective_options.tokenizer_encoding,
            character_coverage=effective_options.sp_character_coverage,
            byte_fallback=effective_options.sp_byte_fallback,
        )

    return train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )


@dataclass(frozen=True)
class LoadedTokenizerBackend:
    """One trainer-agnostic view of a loaded tokenizer model.

    Shared by `load_tokenizer_encoder` (the encode-only view used by the
    tokenize stage and TF-IDF code) and `vocabulary.py`'s
    `load_tokenizer_vocabulary` (a non-pipeline consumer's entry point that
    additionally needs the trained subword vocabulary). Both
    load through this one function so a trainer backend's model is
    instantiated once, not once per caller.

    Attributes:
        trainer: normalized trainer backend name (`"wordpiece"` or
            `"sentencepiece"`).
        encode: `text -> token list` callable.
        encode_batch: `texts -> token lists` callable, one entry per input
            in the same order. Both backends encode a whole list natively,
            so this crosses the Python-to-Rust (or -to-C++) boundary once
            per batch instead of once per row; prefer it wherever a caller
            already has every string in hand.
        unk_token: this tokenizer's unknown-token marker (`"[UNK]"` for
            WordPiece; the model's own configured unk piece for
            SentencePiece).
        vocab: the full trained `token -> id` vocabulary, exactly as
            trained -- not filtered by any noise-word policy, which is a
            separate, orthogonal concern (see `tfidf.resolve_noise_words`).
    """

    trainer: str
    encode: Callable[[str], list[str]]
    encode_batch: Callable[[list[str]], list[list[str]]]
    unk_token: str
    vocab: dict[str, int]


def load_tokenizer_backend(
    *,
    tokenizer_path: Path,
    trainer: str,
) -> LoadedTokenizerBackend:
    """Load a trained tokenizer artifact once and expose encode/unk/vocab together.

    Internal shared loader -- prefer `load_tokenizer_encoder` when only the
    encode function/unk marker are needed (the tokenize stage's use case),
    or `vocabulary.load_tokenizer_vocabulary` for a non-pipeline consumer
    that also needs the trained vocabulary plus resolved noise words.
    """
    normalized_trainer = _normalize_trainer_name(trainer)
    if normalized_trainer == "wordpiece":
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return LoadedTokenizerBackend(
            trainer=normalized_trainer,
            encode=lambda text: tokenizer.encode(text).tokens,
            encode_batch=lambda texts: [
                encoding.tokens for encoding in tokenizer.encode_batch(texts)
            ],
            unk_token="[UNK]",  # nosec B106 - the unknown-token marker, not a credential
            vocab=tokenizer.get_vocab(),
        )

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    unk_id = processor.unk_id()
    unk_piece = processor.id_to_piece(unk_id) if unk_id >= 0 else "[UNK]"
    vocab = {processor.id_to_piece(i): i for i in range(processor.get_piece_size())}
    return LoadedTokenizerBackend(
        trainer=normalized_trainer,
        encode=lambda text: processor.encode(text, out_type=str),
        encode_batch=lambda texts: processor.encode(texts, out_type=str),
        unk_token=unk_piece,
        vocab=vocab,
    )


def load_tokenizer_encoder(
    *,
    tokenizer_path: Path,
    trainer: str,
) -> tuple[Callable[[str], list[str]], str]:
    """Return a token encoder function and unknown-token marker for a trainer backend."""
    backend = load_tokenizer_backend(tokenizer_path=tokenizer_path, trainer=trainer)
    return backend.encode, backend.unk_token
