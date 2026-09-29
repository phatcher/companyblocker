from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from .sparse_similarity import TargetSimilarityBackendIndex


@dataclass(frozen=True)
class TargetClusteringIndex:
    """Fitted target-side index, built once and reused across source chunks.

    Returned by `ClusteringStrategy.build_target_index()` and passed into
    `build_backend_index()`/`score_source_chunk()` on the same strategy; not
    normally constructed by hand outside of tests.

    Attributes:
        target_ids: Target row identifiers, aligned by position with the rows
            of `target_matrix` (row `i` of `target_matrix` is `target_ids[i]`).
        vectorizer: The fitted `sklearn`-style vectorizer (for example
            `TfidfVectorizer`) used to build `target_matrix`, reused via
            `.transform()` to project source-side text into the same space.
        target_matrix: Target embedding matrix, one row per `target_ids`
            entry. Sparse (`scipy.sparse.csr_matrix`) for the TF-IDF/token-list
            representations; a dense `numpy.ndarray` for embedding-based
            representations (`"sbert"`) -- see `SbertClusteringStrategy`'s
            docstring for which similarity backends support dense matrices.
    """

    target_ids: list[str]
    vectorizer: Any
    target_matrix: csr_matrix | np.ndarray


TargetIndexStorableParts = dict[str, np.ndarray | pl.DataFrame]
"""What `ClusteringStrategy.target_index_storable_parts()` returns.

Every value is a `numpy.ndarray` or a `polars.DataFrame`, named by what it
holds (`"target_ids"`, `"target_matrix"`, ...). Storage -- where these land
and in what file format -- is the caller's: nothing under `packages/` may
depend on the workspace package, so this type says only what a strategy can
hand over, not how it is kept.
"""


@dataclass(frozen=True)
class SimilarityBackendIndex:
    """Base marker for a built similarity-backend index.

    `ClusteringStrategy.build_backend_index()` returns a subclass of this
    (in practice `TargetSimilarityBackendIndex`) identifying which backend
    implementation should score source chunks against a `TargetClusteringIndex`.

    Attributes:
        backend: Which similarity backend built this index. One of
            `"sklearn"` (brute-force cosine `NearestNeighbors`),
            `"sparse_dot_topn"` (exact sparse top-n cosine via the optional
            `sparse_dot_topn` package, falling back to `"sklearn"` if it is
            not installed), `"svd_rerank"` (dense SVD-projected shortlist
            reranked with an exact sparse score; see the package README's
            Known Issues for why this is a constant-factor speedup and not a
            sub-linear ANN index), or `"hnsw"` (a real sub-linear approximate
            nearest-neighbour index over a *dense* target, on `usearch`; see
            `hnsw_similarity.py`).
        sklearn_nn: Fitted `sklearn.neighbors.NearestNeighbors` index when
            `backend == "sklearn"`; `None` otherwise.
        svd_rerank: Backend-specific SVD-rerank state when
            `backend == "svd_rerank"`; `None` otherwise. Declared as `Any`
            here because subclasses narrow the type (see
            `TargetSimilarityBackendIndex.svd_rerank` in `sparse_similarity.py`).
    """

    backend: str
    sklearn_nn: Any | None = None
    svd_rerank: Any | None = None


@dataclass(frozen=True)
class TargetIndexBuildSettings:
    """Base marker type for per-representation target-index build settings.

    Passed as `build_settings` to `ClusteringStrategy.build_target_index()`;
    callers construct the concrete subclass matching their chosen
    representation directly, or obtain one from
    `resolve_target_index_build_settings()`. Carries no fields itself.
    """


@dataclass(frozen=True)
class TfidfTargetIndexBuildSettings(TargetIndexBuildSettings):
    """Build settings for the `"tfidf"` representation.

    Attributes:
        ngram_min: Minimum n-gram length passed through to
            `TfidfVectorizer(ngram_range=(ngram_min, ngram_max))`. Must be
            greater than zero.
        ngram_max: Maximum n-gram length for the same `ngram_range`. Must be
            greater than or equal to `ngram_min`.
        analyzer: What an n-gram is a run of, passed through as
            `TfidfVectorizer(analyzer=...)`: `"char_wb"` (the default,
            characters within word boundaries), `"char"` or `"word"`. Under
            `"word"` with the default range a one-word name has no features
            at all, which is why the default is characters.
    """

    ngram_min: int
    ngram_max: int
    analyzer: str = "char_wb"


TFIDF_ANALYZERS: frozenset[str] = frozenset({"char_wb", "char", "word"})
"""The `analyzer` values `TfidfTargetIndexBuildSettings` accepts."""


@dataclass(frozen=True)
class WordpieceTargetIndexBuildSettings(TargetIndexBuildSettings):
    """Build settings for the `"wordpiece"` representation.

    Carries no fields today: the WordPiece token-list strategy needs no
    representation-specific tuning beyond the shared TF-IDF-over-tokens
    defaults in `TokenListClusteringStrategy`. Exists as a distinct type so
    `resolve_clustering_strategy()`/`resolve_target_index_build_settings()`
    can validate the settings type matches the chosen representation.
    """


@dataclass(frozen=True)
class SentencepieceTargetIndexBuildSettings(TargetIndexBuildSettings):
    """Build settings for the `"sentencepiece"` representation.

    Carries no fields today, for the same reason as
    `WordpieceTargetIndexBuildSettings`: SentencePiece shares
    `TokenListClusteringStrategy`'s TF-IDF-over-tokens defaults and only
    needs a distinct type for settings-type validation.
    """


@dataclass(frozen=True)
class SbertTargetIndexBuildSettings(TargetIndexBuildSettings):
    """Build settings for the `"sbert"` representation.

    Attributes:
        model_name: A `sentence_transformers.SentenceTransformer` checkpoint
            identifier -- either a hub name (a pretrained baseline, e.g.
            `"sentence-transformers/all-MiniLM-L6-v2"`) or a local filesystem
            path to a custom-trained encoder. Both are the same code path;
            only the checkpoint differs. A registry slug (e.g.
            `"fr-sentence-camembert-base"`) is also accepted and resolved to
            its checkpoint by `sbert_model_registry.resolve_sbert_model_name()`.
        force_untrained_pooling: Load `model_name` even though it declares no
            sentence-embedding configuration. Off by default: see
            `sbert_pooling_gate.py` for why an unverified checkpoint fails
            rather than warns.
    """

    model_name: str
    force_untrained_pooling: bool = False


class EncoderHandle(Protocol):
    """Shape an already-loaded, trained encoder must have for the `"encoder"` representation.

    Duck-typed rather than tied to any one training family's concrete class --
    `company_classify.encoder_models.PooledSubwordContrastiveEncoder.embed()`
    already has exactly this one-name-at-a-time signature, but nothing in
    this package imports that class or any other: a fake test double with
    the same one method satisfies this Protocol equally, and so does any
    other learned family's encoder that produces a vector per name.
    """

    def embed(self, name: str) -> np.ndarray: ...


@dataclass(frozen=True)
class EncoderTargetIndexBuildSettings(TargetIndexBuildSettings):
    """Build settings for the `"encoder"` representation.

    Attributes:
        encoder: Already-loaded encoder handle (see `EncoderHandle`)
            producing one name's vector via `encoder.embed(name)` -- for
            example a persisted `company_classify` pair classifier's
            `PooledSubwordContrastiveEncoder`, loaded by the caller through
            `persistence.load_pair_classifier_model()`. Loading it --
            resolving a checkpoint, an artifact path, or a persisted model
            directory to this handle -- is entirely the caller's job; unlike
            `SbertTargetIndexBuildSettings.model_name`, this strategy never
            resolves a name or path to a handle itself.
        encoder_name: Free-text identifier for `encoder`, carried through
            unchanged so a caller composing a run's production record, or a
            strategy-comparison row, can name the model that produced the
            vectors without inspecting `encoder` itself.
    """

    encoder: EncoderHandle
    encoder_name: str


class ClusteringStrategy(Protocol):
    """Pluggable per-representation clustering/similarity-scoring contract.

    Implemented by `TfidfClusteringStrategy`, `WordpieceClusteringStrategy`,
    `SentencepieceClusteringStrategy`, and `SbertClusteringStrategy`; obtain
    the right implementation via `resolve_clustering_strategy()` rather than
    instantiating a concrete class directly.

    Design invariant -- static target index, streaming/chunked source:
    `build_target_index()`/`build_backend_index()` are meant to be called
    exactly once per run against the *full* target side; `score_source_chunk()`
    is meant to be called repeatedly against that same built index, with a
    different (chunked) slice of the source frame each time. This is what
    lets a caller process a large source dataset in bounded-size batches
    (bounded memory, incremental progress reporting) without rebuilding
    anything between batches -- see `src/blocking/workflow.py`'s
    `_score_country` for a caller that does exactly this. Any implementation
    of this Protocol inherits that contract: the target side must not need
    to change *during* scoring (true streaming/incremental linkage, where
    the target side itself grows or updates mid-run, is out of scope for
    this Protocol as designed).

    Attributes:
        representation: The representation name this strategy handles
            (`"tfidf"`, `"wordpiece"`, `"sentencepiece"`, or `"sbert"`) --
            must match the name passed to `resolve_clustering_strategy()`/
            `resolve_target_index_build_settings()`.
        storage_irrelevant_build_settings_fields: Field names on this
            strategy's `TargetIndexBuildSettings` subclass that do not affect
            `target_index_storable_parts()`'s output -- for `"sbert"`,
            `force_untrained_pooling`, which changes whether loading raises
            rather than what gets embedded. A caller keying a persisted index
            on its build settings can drop these fields from the key. Empty
            for a strategy where every field matters.
    """

    representation: str
    storage_irrelevant_build_settings_fields: frozenset[str]

    def build_target_index(
        self,
        target_frame: pl.DataFrame,
        *,
        target_id_col: str,
        text_col: str,
        build_settings: TargetIndexBuildSettings,
    ) -> TargetClusteringIndex:
        """Fit and return the target-side index for `target_frame`.

        Args:
            target_frame: Target rows to index.
            target_id_col: Column in `target_frame` holding target ids.
            text_col: Column in `target_frame` holding the text (or, for
                token-list representations, list-of-tokens) to vectorize.
            build_settings: The `TargetIndexBuildSettings` subclass matching
                this strategy's `representation` (for example
                `TfidfTargetIndexBuildSettings` for `"tfidf"`).
        """
        ...

    def target_index_storable_parts(
        self, target_index: TargetClusteringIndex
    ) -> TargetIndexStorableParts:
        """Return `target_index`'s storable parts as named arrays and frames.

        The caller -- not this strategy -- decides where and how these are
        kept (a parquet file, a `.npy` array, a memory-mapped file, ...);
        this method only says what the storable parts are. Round-trips
        through `rebuild_target_index()` given the same
        `TargetIndexBuildSettings` `target_index` was built with (modulo
        `storage_irrelevant_build_settings_fields`): the rebuilt index scores
        a source chunk identically to `target_index` itself.

        Args:
            target_index: Index returned by `build_target_index()`.
        """
        ...

    def rebuild_target_index(
        self,
        storable_parts: TargetIndexStorableParts,
        *,
        build_settings: TargetIndexBuildSettings,
    ) -> TargetClusteringIndex:
        """Rebuild the `TargetClusteringIndex` behind a stored-parts result.

        Args:
            storable_parts: A `target_index_storable_parts()` result -- not
                necessarily produced in this process; a caller may have
                persisted it and reloaded it elsewhere, including as a
                memory-mapped array for a part that supports it (see
                `SbertClusteringStrategy`, which accepts its target matrix
                this way without copying it).
            build_settings: The same settings type used to build the
                original index. A registry slug and the checkpoint it
                resolves to (`sbert_model_registry.resolve_sbert_model_name()`)
                rebuild the same `"sbert"` index, since both resolve to the
                same checkpoint before the encoder is constructed.
        """
        ...

    def build_backend_index(
        self,
        target_index: TargetClusteringIndex,
        *,
        backend: str,
        top_k: int,
        max_candidates_per_source: int | None,
        backend_options: dict[str, object] | None,
        min_similarity: float | None = None,
    ) -> TargetSimilarityBackendIndex | None:
        """Build the similarity-backend index used by `score_source_chunk()`.

        Args:
            target_index: Index returned by `build_target_index()`.
            backend: Similarity backend name -- one of `"sklearn"`,
                `"sparse_dot_topn"`, or `"svd_rerank"` (see
                `SimilarityBackendIndex.backend`).
            top_k: Maximum neighbors to retrieve per source row.
            max_candidates_per_source: Optional cap on candidates per source
                row, taken as `min(top_k, max_candidates_per_source)` when set.
            backend_options: Backend-specific options (for example
                `svd_candidates`/`svd_dimensions` for `"svd_rerank"`, or
                `prefix_filter` for `"sklearn"` -- see `prefix_filter.py`);
                `None` uses backend defaults, with no prefix filtering.
            min_similarity: The cosine threshold the run keeps pairs by, which
                the `"sklearn"` prefix filter sizes its prefixes from; required
                when that option is on.
        """
        ...

    def score_source_chunk(
        self,
        source_frame: pl.DataFrame,
        *,
        target_index: TargetClusteringIndex,
        source_id_col: str,
        text_col: str,
        top_k: int,
        min_similarity: float,
        max_candidates_per_source: int | None,
        nn_index: Any | None,
        backend: str,
        backend_index: TargetSimilarityBackendIndex | None,
        backend_options: dict[str, object] | None,
        timings: dict[str, float] | None = None,
    ) -> pl.DataFrame:
        """Score one chunk of source rows against `target_index`.

        Args:
            source_frame: Source rows to score.
            target_index: Index returned by `build_target_index()`.
            source_id_col: Column in `source_frame` holding source ids.
            text_col: Column in `source_frame` holding the text (or tokens)
                to vectorize, matching `target_index`'s representation.
            top_k: Maximum neighbors to retrieve per source row.
            min_similarity: Minimum cosine similarity (`0.0`-`1.0`) a match
                must reach to be kept.
            max_candidates_per_source: Optional cap on candidates per source
                row; see `build_backend_index()`.
            nn_index: Pre-built `sklearn.neighbors.NearestNeighbors` index to
                reuse when `backend_index` is not supplied; `None` builds one.
            backend: Similarity backend name; see `build_backend_index()`.
            backend_index: Index returned by `build_backend_index()`, when
                available; takes precedence over `nn_index`.
            backend_options: Backend-specific options; see
                `build_backend_index()`.
            timings: When given, the seconds this call spent vectorizing the
                chunk and in the backend are added to it under
                `similarity_strategy_helpers`' two step names, so a caller
                scoring many chunks reads each step's total.

        Returns:
            A `DataFrame` with `source_id`, `target_id`, `similarity`, and
            `rank` columns, one row per kept match.
        """
        ...
