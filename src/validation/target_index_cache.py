"""Shared target-index cache, built on the append-only artifact store.

Promoted out of `validation.runner`'s private helpers so a blocking run can
reuse a target index this harness already built, and vice versa, rather than
each area inventing its own cache under `data/<system>/`. Keyed
on the population a run actually consumed (`workspace.identity`), not a file
fingerprint: a cleanse-layer re-run that never touches the column a run
scores leaves every key here unchanged, where the old file-mtime fingerprint
invalidated on any re-run regardless of what changed.

Two storage shapes, chosen by representation:

- Every representation but `"sbert"` gets a whole-index snapshot: built once,
  persisted as `company_vectorize`'s `target_index_storable_parts()` output
  (never a pickled object), and rehydrated through `rebuild_target_index()`
  on a hit -- the same persist-and-rehydrate protocol every `company_vectorize`
  strategy already implements, so a loaded index is the object the strategy
  itself builds, not a stand-in the caller must special-case. One `artifact_archive` candidate
  directory per key; see `_resolve_snapshot_target_index`.
- `"sbert"` gets a per-row store instead: one append-only store per model
  identity (never per population), rows keyed on `system_uri` plus a hash of
  the text actually consumed, so a target that grows or edits a handful of
  rows re-encodes only those rows and reuses every embedding already on
  file. See `_resolve_sbert_target_index`.

`resolve_target_index` is the one entry point; `validation.runner` and
`blocking.workflow` both call it rather than keeping their own copies.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from company_vectorize.clustering_contract import (
    ClusteringStrategy,
    SbertTargetIndexBuildSettings,
    TargetClusteringIndex,
    TargetIndexBuildSettings,
    TargetIndexStorableParts,
)
from company_vectorize.sbert_model_registry import resolve_sbert_model_name

from workspace.artifact_archive import (
    artifact_is_complete,
    begin_artifact_write,
    commit_artifact_write,
    read_artifact_batches,
    read_artifact_manifest,
    resolve_candidate_dir,
    write_artifact_batch,
)
from workspace.artifact_layout import artifact_store_root
from workspace.roots import WorkspaceRoots

TARGET_INDEX_STORE_FACET = "target-index"
SBERT_EMBEDDING_STORE_FACET = "sbert-target-embeddings"
TARGET_NEIGHBOR_STORE_FACET = "target-neighbor-edges"
HNSW_INDEX_STORE_FACET = "hnsw-index"
_TARGET_NEIGHBOR_EDGES_FILENAME = "target_neighbor_edges.parquet"


def target_index_store_root(roots: WorkspaceRoots) -> Path:
    """Where every target-index cache -- snapshot or row-store -- lives."""
    return artifact_store_root(roots)


def build_target_index_cache_settings(
    *,
    population_key: str,
    representation: str,
    text_view: str,
    tfidf_ngram_min: int | None,
    tfidf_ngram_max: int | None,
    tfidf_analyzer: str | None,
    similarity_backend: str,
    backend_options: dict[str, object] | None,
    token_cache_key: str | None,
    tokenizer: str,
    tokenizer_scope: str,
    strategy: str,
) -> dict[str, object]:
    """The snapshot cache's key facets: `population_key` plus every other
    setting that changes what gets built from it. Unused by the `"sbert"`
    row-store path, which is keyed on model identity alone -- see the module
    docstring."""
    return {
        "population_key": population_key,
        "representation": representation,
        "text_view": text_view,
        "tfidf_ngram_min": tfidf_ngram_min,
        "tfidf_ngram_max": tfidf_ngram_max,
        "tfidf_analyzer": tfidf_analyzer,
        "similarity_backend": similarity_backend,
        "backend_options": backend_options or {},
        "token_cache_key": token_cache_key,
        "tokenizer": tokenizer,
        "tokenizer_scope": tokenizer_scope,
        "strategy": strategy,
    }


def resolve_target_index(
    *,
    roots: WorkspaceRoots,
    target_system: str,
    country: str,
    settings: Mapping[str, object],
    clustering_strategy: ClusteringStrategy,
    build_settings: TargetIndexBuildSettings,
    target_frame: pl.DataFrame,
    target_id_col: str,
    text_col: str,
) -> tuple[TargetClusteringIndex, bool, Path]:
    """Resolve `target_frame`'s target index, from cache when possible.

    Returns `(index, cache_hit, cache_location)`. `cache_location` is the
    directory a caller may report in its own progress events; it is not
    guaranteed to exist yet on a miss for the `"sbert"` row-store path (a
    fully-cached target with nothing new to encode never creates it).

    Dispatches on `clustering_strategy.representation`: `"sbert"` goes
    through the per-row embedding store (`_resolve_sbert_target_index`),
    every other representation through the whole-index snapshot
    (`_resolve_snapshot_target_index`). See the module docstring for why the
    two need different shapes.
    """
    if clustering_strategy.representation == "sbert":
        if not isinstance(build_settings, SbertTargetIndexBuildSettings):
            raise TypeError(
                "sbert representation requires SbertTargetIndexBuildSettings"
            )
        return _resolve_sbert_target_index(
            roots=roots,
            clustering_strategy=clustering_strategy,
            build_settings=build_settings,
            target_frame=target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
        )
    return _resolve_snapshot_target_index(
        roots=roots,
        target_system=target_system,
        country=country,
        settings=settings,
        clustering_strategy=clustering_strategy,
        build_settings=build_settings,
        target_frame=target_frame,
        target_id_col=target_id_col,
        text_col=text_col,
    )


def resolve_target_neighbor_edges(
    *,
    roots: WorkspaceRoots,
    target_system: str,
    country: str,
    settings: Mapping[str, object],
    compute: Callable[[], pl.DataFrame],
) -> tuple[pl.DataFrame, bool, Path]:
    """Resolve one run's target-to-target neighbour-edge frame, from cache
    when possible.

    Returns `(edges, cache_hit, cache_location)`, the same three-tuple shape
    `resolve_target_index` returns. `settings` is the caller's own facets --
    `blocking.workflow` passes the target index's own `index_key` plus the
    similarity backend, its options, and the neighbour threshold/cap, so a
    change to any of those resolves a different candidate directory rather
    than silently reusing edges probed under a different setting. `compute`
    is called at most once, and only on a miss: probing every target row
    needs a fitted `NearestNeighbors` or partition/LSH/dense-brute backend
    index built over the target rows, which this cache-only module has no
    reason to know how to build -- the caller supplies it lazily so a cache
    hit never builds it at all, which is the whole point of caching a probe
    expensive enough that the exhaustive self-join it stands in for is out
    of reach.

    Kept beside `resolve_target_index`'s whole-index snapshot rather than
    folded into it: the target index itself is backend-agnostic (the same
    embeddings serve every backend that scores them), while a neighbour edge
    is specific to the backend it was probed with, so the two need different
    keys and the snapshot's cache must not invalidate on a backend change
    that leaves it untouched.
    """
    candidate_dir = resolve_candidate_dir(
        target_index_store_root(roots),
        TARGET_NEIGHBOR_STORE_FACET,
        target_system,
        country,
        settings=settings,
    )

    if artifact_is_complete(candidate_dir):
        edges = pl.read_parquet(candidate_dir / _TARGET_NEIGHBOR_EDGES_FILENAME)
        return edges, True, candidate_dir

    edges = compute()
    temporary_dir = begin_artifact_write(candidate_dir)
    edges.write_parquet(temporary_dir / _TARGET_NEIGHBOR_EDGES_FILENAME)
    commit_artifact_write(candidate_dir, temporary_dir, manifest={})
    return edges, False, candidate_dir


def resolve_hnsw_index_dir(
    *,
    roots: WorkspaceRoots,
    target_system: str,
    country: str,
    index_key: str,
    hnsw_build_settings: Mapping[str, object],
) -> Path:
    """The directory the `"hnsw"` similarity backend saves and reopens its
    built `usearch` graph under, keyed on `index_key` (the target index this
    graph was built over) plus `hnsw_build_settings`
    (`company_vectorize.hnsw_similarity.hnsw_build_settings()`'s output --
    the backend options that change the built graph, never a search-time
    option like `expansion_search`).

    This only resolves *which* directory; everything written inside it --
    the saved index file, its own manifest -- is
    `company_vectorize.hnsw_similarity.build_target_hnsw_index()`'s: that
    package may not depend on this one (see its README's Boundaries
    section), so it takes a directory from its caller and composes no path
    above it, which is what this function is for.
    """
    return resolve_candidate_dir(
        target_index_store_root(roots),
        HNSW_INDEX_STORE_FACET,
        target_system,
        country,
        settings={
            "index_key": index_key,
            "hnsw_build_settings": dict(hnsw_build_settings),
        },
    )


# --- whole-index snapshot (tfidf / wordpiece / sentencepiece) -------------


def _write_storable_parts(
    directory: Path, storable_parts: TargetIndexStorableParts
) -> dict[str, str]:
    kinds: dict[str, str] = {}
    for name, value in storable_parts.items():
        if isinstance(value, pl.DataFrame):
            value.write_parquet(directory / f"{name}.parquet")
            kinds[name] = "dataframe"
        else:
            np.save(directory / f"{name}.npy", np.asarray(value))
            kinds[name] = "ndarray"
    return kinds


def _read_storable_parts(
    directory: Path, kinds: Mapping[str, str]
) -> TargetIndexStorableParts:
    parts: TargetIndexStorableParts = {}
    for name, kind in kinds.items():
        if kind == "dataframe":
            parts[name] = pl.read_parquet(directory / f"{name}.parquet")
        else:
            parts[name] = np.load(directory / f"{name}.npy")
    return parts


def _resolve_snapshot_target_index(
    *,
    roots: WorkspaceRoots,
    target_system: str,
    country: str,
    settings: Mapping[str, object],
    clustering_strategy: ClusteringStrategy,
    build_settings: TargetIndexBuildSettings,
    target_frame: pl.DataFrame,
    target_id_col: str,
    text_col: str,
) -> tuple[TargetClusteringIndex, bool, Path]:
    candidate_dir = resolve_candidate_dir(
        target_index_store_root(roots),
        TARGET_INDEX_STORE_FACET,
        clustering_strategy.representation,
        target_system,
        country,
        settings=settings,
    )

    if artifact_is_complete(candidate_dir):
        manifest = read_artifact_manifest(candidate_dir) or {}
        kinds = manifest.get("parts", {})
        storable_parts = _read_storable_parts(candidate_dir, kinds)
        index = clustering_strategy.rebuild_target_index(
            storable_parts, build_settings=build_settings
        )
        return index, True, candidate_dir

    index = clustering_strategy.build_target_index(
        target_frame,
        target_id_col=target_id_col,
        text_col=text_col,
        build_settings=build_settings,
    )
    storable_parts = clustering_strategy.target_index_storable_parts(index)
    temporary_dir = begin_artifact_write(candidate_dir)
    kinds = _write_storable_parts(temporary_dir, storable_parts)
    commit_artifact_write(candidate_dir, temporary_dir, manifest={"parts": kinds})
    return index, False, candidate_dir


# --- sbert per-row embedding store -----------------------------------------

_EMBEDDING_DTYPE = "float32"


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _sbert_model_identity(
    build_settings: SbertTargetIndexBuildSettings,
) -> dict[str, object]:
    return {"model_name": resolve_sbert_model_name(build_settings.model_name)}


def _sbert_embedding_store_dir(
    *, roots: WorkspaceRoots, build_settings: SbertTargetIndexBuildSettings
) -> Path:
    return resolve_candidate_dir(
        target_index_store_root(roots),
        SBERT_EMBEDDING_STORE_FACET,
        settings=_sbert_model_identity(build_settings),
    )


def _library_versions() -> str:
    """Best-effort JSON of the encoder libraries' versions, read from their
    installed distribution metadata rather than by importing them: the
    caller below records this alongside every new embedding written, and
    the libraries are installed in this test environment too (see this
    module's tests, which inject a fake encoder precisely to avoid paying
    for that import). A library absent from the installed environment is
    absent from the record, as it was before."""
    versions: dict[str, str] = {}
    for library in ("sentence_transformers", "torch"):
        try:
            versions[library] = importlib.metadata.version(library)
        except importlib.metadata.PackageNotFoundError:
            pass
    return json.dumps(versions, sort_keys=True)


def _describe_encoder_device(vectorizer: Any) -> str:
    """Best-effort device the encoder ran on. `_Encoder` carries no device
    field of its own; the real `_SentenceTransformerEncoder` wraps a loaded
    `SentenceTransformer` on `._model`, whose `.device` this reads. A fake
    encoder (every test here) has no such attribute and reports
    `"unknown"`."""
    model = getattr(vectorizer, "_model", None)
    if model is None:
        return "unknown"
    return str(getattr(model, "device", "unknown"))


def _resolve_sbert_target_index(
    *,
    roots: WorkspaceRoots,
    clustering_strategy: ClusteringStrategy,
    build_settings: SbertTargetIndexBuildSettings,
    target_frame: pl.DataFrame,
    target_id_col: str,
    text_col: str,
) -> tuple[TargetClusteringIndex, bool, Path]:
    embeddings_dir = _sbert_embedding_store_dir(
        roots=roots, build_settings=build_settings
    )

    if target_frame.height == 0:
        index = clustering_strategy.build_target_index(
            target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
            build_settings=build_settings,
        )
        return index, True, embeddings_dir

    ids = target_frame.get_column(target_id_col).cast(pl.Utf8).to_list()
    texts = (
        target_frame.get_column(text_col)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .to_list()
    )
    text_hashes = [_hash_text(text) for text in texts]
    row_keys = [
        f"{system_uri}\x00{text_hash}"
        for system_uri, text_hash in zip(ids, text_hashes, strict=True)
    ]
    population = pl.DataFrame(
        {
            "system_uri": ids,
            "text_hash": text_hashes,
            "row_key": row_keys,
        }
    )

    existing = read_artifact_batches(embeddings_dir, row_key="row_key")
    if existing.height:
        existing_slim = existing.select("row_key", "embedding")
        hits = population.join(existing_slim, on="row_key", how="inner").select(
            "row_key", "embedding"
        )
        missing = population.join(
            existing_slim.select("row_key"), on="row_key", how="anti"
        )
    else:
        hits = pl.DataFrame(
            schema={"row_key": pl.Utf8, "embedding": pl.List(pl.Float32)}
        )
        missing = population

    if missing.height:
        missing_ids = set(missing.get_column("system_uri").to_list())
        missing_frame = target_frame.filter(
            pl.col(target_id_col).cast(pl.Utf8).is_in(missing_ids)
        )
        missing_index = clustering_strategy.build_target_index(
            missing_frame,
            target_id_col=target_id_col,
            text_col=text_col,
            build_settings=build_settings,
        )
        missing_storable = clustering_strategy.target_index_storable_parts(
            missing_index
        )
        matrix = np.asarray(missing_storable["target_matrix"], dtype=np.float32)
        ids_frame = cast("pl.DataFrame", missing_storable["target_ids"])

        if ids_frame.height:
            missing_lookup = missing.select(
                "system_uri", "text_hash", "row_key"
            ).unique(subset=["system_uri"], keep="first")
            new_rows = (
                ids_frame.rename({"target_id": "system_uri"})
                .with_row_index("_position")
                .join(missing_lookup, on="system_uri", how="left")
                .sort("_position")
                .drop("_position")
            )
            checkpoint = resolve_sbert_model_name(build_settings.model_name)
            device = _describe_encoder_device(missing_index.vectorizer)
            library_versions = _library_versions()
            # Straight from the numpy matrix, never through Python lists: one
            # float object per element is several times the matrix itself.
            new_rows = new_rows.with_columns(
                pl.Series("embedding", matrix).cast(pl.List(pl.Float32)),
                pl.lit(checkpoint).alias("checkpoint"),
                pl.lit(device).alias("device"),
                pl.lit(_EMBEDDING_DTYPE).alias("dtype"),
                pl.lit(library_versions).alias("library_versions"),
            )
            batch_key = hashlib.blake2b(
                "\x1e".join(sorted(new_rows.get_column("row_key").to_list())).encode(
                    "utf-8"
                ),
                digest_size=16,
            ).hexdigest()
            write_artifact_batch(embeddings_dir, new_rows, batch_key=batch_key)
            hits = pl.concat(
                [hits, new_rows.select("row_key", "embedding")], how="vertical_relaxed"
            )

    ordered = population.join(hits, on="row_key", how="left")
    present = ordered.filter(pl.col("embedding").is_not_null())
    cache_hit = missing.height == 0

    if present.height == 0:
        index = clustering_strategy.build_target_index(
            target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
            build_settings=build_settings,
        )
        return index, cache_hit, embeddings_dir

    target_ids = present.get_column("system_uri").to_list()
    embeddings = present.get_column("embedding")
    embedding_width = cast("int", embeddings.list.len().max())
    target_matrix = embeddings.cast(pl.Array(pl.Float32, embedding_width)).to_numpy(
        writable=True
    )
    storable_parts: TargetIndexStorableParts = {
        "target_ids": pl.DataFrame({"target_id": target_ids}),
        "target_matrix": target_matrix,
    }
    index = clustering_strategy.rebuild_target_index(
        storable_parts, build_settings=build_settings
    )
    return index, cache_hit, embeddings_dir
