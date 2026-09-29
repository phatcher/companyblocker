"""Tokenizer ids, the folders they live in, and the fixed names of the files inside.

A tokenizer is named by its id in `TOKENIZER_IDS`, which is looked up and never split.
Each tokenizer has a folder of its own beneath its scope's working directory, named
by its id, so `bpe` and `unigram` never share a file. The files inside have fixed
names that mention neither tokenizer nor encoding, and nothing about where a folder
sits is read, so a working folder, a stored candidate and a tokenizer shipped as a
package resource are all read alike: a caller holding a directory asks
`tokenizer_directory_files()` or `scope_directory_files()` for a name rather than
appending one itself.

`promoted_candidate_key()` names a stored candidate by the training parameters a
person reads it by, closed by a digest of the model file, so equal content gives an
equal key. Where the working tree is rooted is the caller's: `resolve_tokenizer_paths()`
takes an already-resolved `tokenizer_root`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

TRAINING_CORPUS_FILENAME = "training_corpus.parquet"

METADATA_FILENAME = "metadata.json"
TOKEN_SCORES_FILENAME = "token_scores.json"
METRICS_FILENAME = "metrics.json"
OPTIMIZE_SUMMARY_FILENAME = "optimize_summary.json"
TFIDF_STATS_FILENAME = "token_tfidf_stats.parquet"
TOKEN_SET_FILENAME = "token_set.json"
NOISE_WORDS_FILENAME = "noise_words.json"
NOISE_WORD_CANDIDATES_FILENAME = "noise_word_candidates.json"


@dataclass(frozen=True)
class TokenizerFields:
    """What a tokenizer id names: the tokenizer and, for SentencePiece only, its encoding."""

    tokenizer: str
    tokenizer_encoding: str | None = None


TOKENIZER_IDS: dict[str, TokenizerFields] = {
    "wordpiece": TokenizerFields("wordpiece"),
    "sentencepiece_bpe": TokenizerFields("sentencepiece", "bpe"),
    "sentencepiece_unigram": TokenizerFields("sentencepiece", "unigram"),
}
"""Every tokenizer this package trains, by id. The id is the one-token form for
lists and for a folder name; it is looked up here and never split."""

_ID_BY_FIELDS = {fields: identifier for identifier, fields in TOKENIZER_IDS.items()}


@dataclass(frozen=True)
class TokenizerArtifactPaths:
    """A scope's working directory and the training corpus inside it."""

    scope: str
    directory: Path
    corpus_path: Path


@dataclass(frozen=True)
class TokenizerDirectoryFiles:
    """The files one tokenizer directory holds, by name."""

    directory: Path
    model: Path
    metadata: Path
    token_scores: Path
    metrics: Path
    optimize_summary: Path


@dataclass(frozen=True)
class ScopeDirectoryFiles:
    """The corpus files a scope's working directory holds, by name.

    A scope's corpus is one thing however many tokenizers are trained from
    it, so its statistics sit beside it and not in any tokenizer's folder.
    """

    directory: Path
    corpus: Path
    metadata: Path
    tfidf_stats: Path
    tfidf_stats_csv: Path
    token_set: Path
    noise_words: Path
    noise_word_candidates: Path

    def pooled_noise_words(self, pooling: str) -> Path:
        """The noise words pooled from several systems' statistics by `pooling`."""
        return self.directory / f"noise_words.{pooling}.json"


def tokenizer_directory(
    scope_directory: Path, *, trainer: str, tokenizer_encoding: str | None = None
) -> Path:
    """The folder one tokenizer has to itself beneath a scope's working directory.

    Named by the tokenizer's id, so `bpe` and `unigram` never share a file.
    """
    return scope_directory / tokenizer_id(trainer, tokenizer_encoding)


def tokenizer_directory_files(
    directory: Path, *, trainer: str
) -> TokenizerDirectoryFiles:
    """Name the files of the tokenizer held in `directory`.

    The layout inside a tokenizer directory is this package's, so a caller
    that holds a directory asks here for a file rather than appending a name
    itself. Nothing about where the directory sits is read: a tokenizer's
    folder in a working tree, a promoted candidate in an artifact store and a
    tokenizer shipped as a package resource all answer alike. The names are
    fixed and mention neither tokenizer nor encoding; the trainer decides only
    the model file's extension, which is its library's. A file is named
    whether or not it exists: the metrics and the optimize summary are written
    when a candidate is stored, the summary only for one an optimize run chose, and a
    working folder holds neither.
    """
    from .training import resolve_candidate_model_suffix_for_trainer

    suffix = resolve_candidate_model_suffix_for_trainer(trainer=trainer)
    return TokenizerDirectoryFiles(
        directory=directory,
        model=directory / f"model.{suffix}",
        metadata=directory / METADATA_FILENAME,
        token_scores=directory / TOKEN_SCORES_FILENAME,
        metrics=directory / METRICS_FILENAME,
        optimize_summary=directory / OPTIMIZE_SUMMARY_FILENAME,
    )


def scope_directory_files(directory: Path) -> ScopeDirectoryFiles:
    """Name the corpus files of the scope whose working directory is `directory`.

    The training corpus, the corpus-level tf-idf statistics and what is
    derived from them (the token set, the noise words and the noise word
    candidates) belong to the scope, not to a tokenizer, so a caller that
    holds a scope directory asks here rather than appending a name itself.
    No trainer is involved: none of these files depends on one. A file is
    named whether or not it exists.
    """
    tfidf_stats = directory / TFIDF_STATS_FILENAME
    return ScopeDirectoryFiles(
        directory=directory,
        corpus=directory / TRAINING_CORPUS_FILENAME,
        metadata=directory / METADATA_FILENAME,
        tfidf_stats=tfidf_stats,
        tfidf_stats_csv=tfidf_stats.with_suffix(".csv"),
        token_set=directory / TOKEN_SET_FILENAME,
        noise_words=directory / NOISE_WORDS_FILENAME,
        noise_word_candidates=directory / NOISE_WORD_CANDIDATES_FILENAME,
    )


def tokenizer_id(trainer: str, tokenizer_encoding: str | None = None) -> str:
    """The id in `TOKENIZER_IDS` for a trainer and its encoding.

    A SentencePiece trainer carries its encoding, since `bpe` and `unigram`
    are promoted separately, and WordPiece has none, so an encoding given with
    it is not read. A pair the table does not hold raises `ValueError`.
    """
    from .training import _normalize_trainer_name

    normalized = _normalize_trainer_name(trainer)
    encoding = (
        (tokenizer_encoding or "bpe").strip().lower()
        if normalized == "sentencepiece"
        else None
    )
    try:
        return _ID_BY_FIELDS[TokenizerFields(normalized, encoding)]
    except KeyError:
        allowed = ", ".join(TOKENIZER_IDS)
        raise ValueError(
            f"No tokenizer id for trainer '{normalized}' with encoding "
            f"'{encoding}'. Expected one of: {allowed}."
        ) from None


def promoted_candidate_key(
    *,
    corpus_content_hash: str,
    vocab_size: int | None,
    min_frequency: int,
    model_path: Path,
) -> str:
    """The key a promoted candidate's directory is filed under.

    The training parameters a person reads a candidate by, closed by a digest
    of the model file, which covers every option the readable part leaves
    out. Equal content gives an equal key, so promoting the same model twice
    names one directory.
    """
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    vocab = "auto" if vocab_size is None else str(int(vocab_size))
    return (
        f"{corpus_content_hash[:8]}_v{vocab}_mf{int(min_frequency)}_{model_digest[:8]}"
    )


def write_candidate_directory(
    destination: Path,
    *,
    model: Path,
    metadata: Path,
    token_scores: Path | None = None,
    metrics: Mapping[str, object] | None = None,
    optimize_summary: Mapping[str, object] | None = None,
    trainer: str,
) -> TokenizerDirectoryFiles:
    """Copy one trained tokenizer into `destination` as a whole candidate.

    The directory then holds its model, its metadata and its token scores,
    each under the name this package gives it whatever the source files were
    called. Token scores are copied when they were written. The metrics
    measured for the candidate, and the summary of the optimize run that chose it,
    are written beside them when given, so whoever holds the directory reads
    them with no other lookup.
    """
    for source in (model, metadata):
        if not source.is_file():
            raise FileNotFoundError(f"Tokenizer candidate file not found: {source}")
    written = tokenizer_directory_files(destination, trainer=trainer)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, written.model)
    shutil.copy2(metadata, written.metadata)
    if token_scores is not None and token_scores.is_file():
        shutil.copy2(token_scores, written.token_scores)
    for payload, path in (
        (metrics, written.metrics),
        (optimize_summary, written.optimize_summary),
    ):
        if payload is not None:
            path.write_text(
                json.dumps(dict(payload), indent=2, default=str), encoding="utf-8"
            )
    return written


def _normalize_system(system: str | None) -> str:
    if system is None or not system.strip():
        raise ValueError("A system code is required for system scope.")
    return system.strip().lower()


def _normalize_corpus_filename(corpus_filename: str) -> str:
    normalized = corpus_filename.strip()
    if not normalized:
        raise ValueError("corpus_filename must be a non-empty file name.")
    return normalized


def to_portable_path_str(path: Path, *, project_root: Path | None = None) -> str:
    """Render as a portable, project-relative posix path where possible.

    Absolute paths tie an artifact to one machine/user/OS. When `project_root`
    is given, render `path` relative to it (posix-separated); falls back to
    the absolute string for a path genuinely outside `project_root`, or when
    `project_root` isn't given at all (callers with no project root handy
    keep today's absolute-path behavior unchanged).
    """
    if project_root is None:
        return str(path)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_country_tokenizer_paths(
    tokenizer_root: Path,
    system: str,
    corpus_filename: str = TRAINING_CORPUS_FILENAME,
) -> TokenizerArtifactPaths:
    system_code = _normalize_system(system)
    normalized_corpus_filename = _normalize_corpus_filename(corpus_filename)
    base = tokenizer_root / system_code
    return TokenizerArtifactPaths(
        scope="system",
        directory=base,
        corpus_path=base / normalized_corpus_filename,
    )


def resolve_global_tokenizer_paths(
    tokenizer_root: Path,
    profile: str = "default",
    corpus_filename: str = TRAINING_CORPUS_FILENAME,
) -> TokenizerArtifactPaths:
    _ = profile
    normalized_corpus_filename = _normalize_corpus_filename(corpus_filename)
    base = tokenizer_root / "global"
    return TokenizerArtifactPaths(
        scope="global",
        directory=base,
        corpus_path=base / normalized_corpus_filename,
    )


def resolve_tokenizer_paths(
    *,
    tokenizer_root: Path,
    scope: str,
    system: str | None = None,
    profile: str = "default",
    corpus_filename: str = TRAINING_CORPUS_FILENAME,
) -> TokenizerArtifactPaths:
    normalized_scope = scope.strip().lower()
    if normalized_scope == "country":
        return resolve_country_tokenizer_paths(
            tokenizer_root, system or "", corpus_filename=corpus_filename
        )
    if normalized_scope == "global":
        return resolve_global_tokenizer_paths(
            tokenizer_root, profile, corpus_filename=corpus_filename
        )

    raise ValueError(
        f"Unsupported tokenizer scope '{scope}'. Expected 'country' or 'global'."
    )
