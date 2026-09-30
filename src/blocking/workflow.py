"""One blocking run: a source population scored against a target population, country by country.

A run is one pairing, and each pairing has its own entry point, so which rows a run
indexes is a property of the dataset it was handed, never a flag:

- `execute_blocking_run()`: a source system's records against a target system's.
- `execute_name_variant_recovery_run()`: a system's recorded name rows against its own
  records, truth being each row's `source_uri`.
- `execute_name_variant_cross_system_run()`: a system's name rows against another
  system's records, truth walked back through the source's `matched/` layer.

The target index is never expanded with recorded names: that would make "did a
previous name reach the company" true by construction.

Per country, a run:

1. Loads both sides and raises `contracts.EmptySourceLoadError` when the source loads
   no rows, before anything is written, so a broken load is never read as a result.
2. Derives both sides' name forms (`name_transform`).
3. With `exact_name_filter` on (the default), joins sources to targets on the raw name,
   then the basic, cleansed and transform forms, each join taking what the ones
   before left, and gives each match a `similarity=1.0` edge. Every source is still
   scanned, since an identical name says nothing about a renamed twin; where the scan
   also scores a joined pair the exact edge is kept. With target-neighbour linking on,
   a matched source is left out of the scan, since its target's neighbour edges reach
   the twin. Per-form counts are the `exact_match_summary` artefact.
4. Tokenizes a lowercased copy of the scored column for a token representation, since
   every tokenizer was trained on lowercased text; the column itself keeps its case.
5. Resolves the target index through the shared cache (`validation.target_index_cache`),
   keyed on the index key, and scores the source in chunks against that one index. A
   dense target is refused on `sklearn` when its estimated footprint exceeds available
   memory.
6. Prunes: `candidate_similarity_ratio` drops a source's candidates below a ratio of its
   best, then `max_candidates_per_target` caps the sources claiming one target.
   `matched_edges` and `pair_truth_eval` are the pruned set, the `raw_` frames the
   unpruned one, and `pruning_summary` counts the drops.
7. Scores the result against ground truth (`validation.runner.compute_pair_truth_eval`)
   when the source carries it.

`target_neighbor_min_similarity` and `target_neighbor_max_per_target`, set together, link
each target to its near neighbours before any source is scanned, through a probe of
the target index sized to the cap plus one and cached beside it, so a cluster holds a
name family. `clusters` and `cluster_shape` are then the clustering after that union
and the `_before_union` frames the one before it. With both unset, every output equals
the plain clustering.

Clustering is the connected components of the pruned edges, with cluster shape,
directional coverage and a similarity histogram beside it. `emit_diagnostics` adds one
row per scored row on each side, joinable on `cluster_id`. Every phase of every
country is timed with its memory and CPU (`timings`) and reported as progress events.

A run is identified by four keys (`workspace.identity`): the settings key digests the
whole resolved configuration; each side's population key covers the raw `system_uri`
and `name` read, so it is the same for every representation and profile over one
target; the truth key covers the resolved truth map; and the index key covers the
target population and everything that shapes its index, the tokenizer's content
included, so promoting a tokenizer in place gives a new run. `resolve_blocking_run_keys()`
computes them without deriving or scoring anything, which is how a finished run is
found and reused.

A finished run can be re-read without re-running it: `remeasure_pair_truth_eval()`
re-resolves its truth and rewrites only the three truth frames, and
`compute_recall_curve_for_run()` reads its recall against comparisons spent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import psutil
from company_cleanse import get_effective_noise_words
from company_tokenize.name_preprocessing import (
    DEFAULT_NAME_PREPROCESSING_PROFILE,
    NAME_PREPROCESSED_COLUMN,
    name_preprocessing,
    parse_name_preprocessing_profile,
)
from company_tokenize.paths import tokenizer_directory_files, tokenizer_id
from company_tokenize.tokenization import tokenize_name_dataframe
from company_vectorize.clustering_contract import TargetIndexBuildSettings
from company_vectorize.clustering_factory import (
    resolve_clustering_strategy,
    resolve_target_index_build_settings,
)
from company_vectorize.clustering_metrics import (
    compute_cluster_shape_metrics,
    compute_directional_coverage,
)
from company_vectorize.clustering_policy import resolve_sbert_model_for_jurisdictions
from company_vectorize.dense_target_memory_gate import (
    ensure_sklearn_dense_target_index_memory_supported,
)
from company_vectorize.dense_vocabulary_gate import (
    ensure_dense_vocabulary_scale_supported,
)
from company_vectorize.graph_cluster import build_connected_components
from company_vectorize.hnsw_similarity import hnsw_build_settings
from company_vectorize.sparse_similarity import build_target_nearest_neighbors
from company_vectorize.tfidf_cluster import build_clustering_text_view

from validation.config import resolve_text_view_for_representation
from validation.contracts import ARTIFACT_SCHEMAS
from validation.recall_curve import compute_recall_curve, summarize_recall_curve
from validation.runner import (
    NAME_EQUALITY_BASIC,
    NAME_EQUALITY_CLEANSED,
    NAME_EQUALITY_RAW,
    PAIR_TRUTH_EVAL_DETAIL_COLUMNS,
    ProgressCallback,
    build_name_form_map,
    compute_pair_truth_eval,
    compute_pair_truth_eval_detail,
)
from validation.target_index_cache import (
    resolve_hnsw_index_dir,
    resolve_target_index,
    resolve_target_neighbor_edges,
)
from workspace.identity import (
    RunKeys,
    combine_part_keys,
    digest_file,
    digest_settings,
    neighbour_edges_key,
    rows_digest,
)
from workspace.identity import index_key as build_index_key
from workspace.name_forms_store import resolve_name_forms
from workspace.roots import WorkspaceRoots
from workspace.telemetry import PhaseHandle, Telemetry
from workspace.tokenizer_store import promoted_tokenizer_directory, scope_system

from .comparison import _is_exact_backend
from .contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
    EmptySourceLoadError,
    blocking_settings_key,
    validate_blocking_run_config,
)
from .loader import load_country_frame, load_dataset_descriptor, load_name_variant_frame
from .name_transform import (
    DEFAULT_CLEANSE_PROFILE,
    NAME_FORM_COLUMNS,
    RAW_NAME_COLUMN,
    NameTransform,
    derive_name_forms,
    name_forms_ruleset_digest,
    resolve_name_transform,
)
from .reporting import build_country_blocks, write_recall_curve_report
from .run_layout import BlockingRunLocation, RunArtefact
from .truth import ColumnTruth, MatchedLayerTruth

_TOKEN_COL = "blocking_tokens"
_EMPTY_MATCHED_EDGES_SCHEMA = {
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "similarity": pl.Float64,
    "rank": pl.Int64,
    "country": pl.Utf8,
}
# Blocking writes this artifact against validation's own schema
# (`reporting.py` validates it with `validate_artifact_schema`), so the empty
# frame is taken from that contract rather than restated -- a second literal
# is how the truth-bucket columns drifted between the two areas.
_EMPTY_PAIR_TRUTH_EVAL_SCHEMA = dict(ARTIFACT_SCHEMAS["pair_truth_eval"])
# pair_truth_eval_detail (renamed from residual_pairs) is
# blocking's own artifact (see contracts.py), so its empty frame is taken
# from validation.runner's PAIR_TRUTH_EVAL_DETAIL_COLUMNS -- the same module
# that computes the real rows -- rather than restated here.
_EMPTY_PAIR_TRUTH_EVAL_DETAIL_SCHEMA = dict(PAIR_TRUTH_EVAL_DETAIL_COLUMNS)
_PRUNING_SUMMARY_SCHEMA = {
    "country": pl.Utf8,
    "raw_edge_count": pl.Int64,
    "dropped_by_similarity_ratio": pl.Int64,
    "dropped_by_target_cap": pl.Int64,
    "pruned_edge_count": pl.Int64,
}
# `unmatched_source_row_count` is a source-*row* count, the sources
# no form made equal to a target -- never conflate it with
# `pair_truth_eval_detail`'s per-*pair* population, and never read it as the
# still-unequal remainder (that figure is the `never` population's
# `truth_pairs` in `pair_truth_eval`). Every source is scanned whatever these
# counts say. `exact_match_count` is the sources some form made equal to a
# target, and the `resolved_by_*` columns split it by the first form that did,
# in cascade order; `multi_target_source_count` is how many of those got more
# than one target from their form.
_EXACT_MATCH_SUMMARY_SCHEMA = {
    "country": pl.Utf8,
    "exact_match_count": pl.Int64,
    "resolved_by_raw": pl.Int64,
    "resolved_by_basic": pl.Int64,
    "resolved_by_cleansed": pl.Int64,
    "resolved_by_transform": pl.Int64,
    "multi_target_source_count": pl.Int64,
    "unmatched_source_row_count": pl.Int64,
}
_SIMILARITY_BUCKET_COUNT = 20
_DEFAULT_SOURCE_CHUNK_SIZE = 10_000

# Backends that build and scan their own index (`build_backend_index()`
# below), so an eager whole-target `sklearn.neighbors.NearestNeighbors` fit
# in this module would be built, never used, and -- the reason either
# backend exists at all -- reintroduce the unbounded-memory fit at scale
# each one is meant to avoid.
_BACKENDS_WITHOUT_EAGER_SKLEARN_NN = frozenset({"dense_brute", "hnsw"})
_TARGET_NEIGHBOR_EDGES_SCHEMA = {
    "target_id_a": pl.Utf8,
    "target_id_b": pl.Utf8,
    "similarity": pl.Float64,
    "country": pl.Utf8,
}
_TARGET_NEIGHBOR_SUMMARY_SCHEMA = {
    "country": pl.Utf8,
    "min_similarity": pl.Float64,
    "max_per_target": pl.Int64,
    "edge_count": pl.Int64,
    "targets_at_cap": pl.Int64,
    "cache_hit": pl.Boolean,
}


def _truth_part_key(truth_map: pl.DataFrame) -> str:
    """One partition's truth digest: `truth_map`'s `source_id`/`source_match_uri`
    rows. Carries no cleanse profile -- the map is read straight off the loaded
    frame's own `match_uri` (or walked cross-system), untouched by which
    profile derived the name forms alongside it.
    """
    return rows_digest(truth_map, id_col="source_id", value_col="source_match_uri")


def _can_evaluate_truth(
    config: BlockingRunConfig, full_source_frame: pl.DataFrame
) -> bool:
    return config.truth.can_evaluate(
        full_source_frame, source_has_ground_truth=config.source.has_ground_truth
    )


def _resolve_country_truth_map(
    config: BlockingRunConfig, full_source_frame: pl.DataFrame
) -> pl.DataFrame:
    """One country's truth map over the full source population (exact-matched
    and unmatched alike), the map both the key pass and scoring digest.

    The resolved map itself keys a run, not the rule that produced it: two
    rules resolving the same pairs share a truth key, and one rule over a
    regenerated matched layer does not. `config.roots` is the right root for
    `MatchedLayerTruth`'s cross-system walk on every real run; a
    `ColumnTruth` reads the frame and never touches it.
    """
    return config.truth.resolve(
        full_source_frame,
        roots=config.roots,
        source_system=config.source.system,
        target_system=config.target.system,
    )


def _digest_noise_words(level: str) -> str:
    """Digest of the noise words a run's preprocessing removes at `level`
    (`company_cleanse.get_effective_noise_words`), one of the index key's
    build settings. No file backs it, so it is hashed from the values.
    """
    words = get_effective_noise_words(noise_words_profile=level)
    return digest_settings(
        {"noise_words": sorted(json.dumps(word, sort_keys=True) for word in words)}
    )


def _index_part_key(
    config: BlockingRunConfig,
    *,
    target_population_key: str,
    effective_text_view: str,
    build_settings: TargetIndexBuildSettings,
) -> str:
    """One partition's target index key: its target population plus every
    setting that changes what gets embedded from it -- the transform and
    cleanse profile its names pass through among them -- and the tokenizer
    content a run scoring tokens reads.

    `build_settings` already carries model identity for a checkpoint-backed
    representation (`SbertTargetIndexBuildSettings.model_name`), so it is not
    named again.
    """
    strategy = config.strategy
    settings: dict[str, object] = {
        # The population key covers raw names, so what turns them into the
        # text embedded belongs to the index's own key.
        "name_transform": strategy.name_transform,
        "cleanse_profile": strategy.cleanse_profile,
        "preprocess_profile": strategy.preprocess_profile,
        "representation": strategy.representation,
        "text_view": effective_text_view,
        "build_settings": asdict(build_settings),
    }
    noise_words_level = parse_name_preprocessing_profile(
        strategy.preprocess_profile
    ).noise_words_level
    if noise_words_level is not None:
        settings["noise_resource_digest"] = _digest_noise_words(noise_words_level)
    tokenizer_digest = (
        digest_file(_resolve_tokenizer_path(config))
        if effective_text_view == "tokens"
        else None
    )
    return build_index_key(
        population_key=target_population_key,
        build_settings=settings,
        tokenizer_digest=tokenizer_digest,
    )


def _compute_similarity_distribution(
    edges: pl.DataFrame, *, min_similarity: float
) -> pl.DataFrame:
    """Bucketed histogram of `edges['similarity']` across `_SIMILARITY_BUCKET_COUNT`
    equal-width buckets spanning `[min_similarity, 1.0]`.

    Always emits all buckets, zero-filled when `edges` is empty, so the shape
    is notebook-renderable even for an empty run.
    """

    floor = min(min_similarity, 1.0)
    span = max(1.0 - floor, 1e-9)
    bucket_width = span / _SIMILARITY_BUCKET_COUNT
    bucket_starts = [floor + i * bucket_width for i in range(_SIMILARITY_BUCKET_COUNT)]
    bucket_ends = [start + bucket_width for start in bucket_starts]

    counts = [0] * _SIMILARITY_BUCKET_COUNT
    if edges.height > 0:
        bucket_index_expr = (
            ((pl.col("similarity").clip(floor, 1.0) - floor) / bucket_width)
            .floor()
            .cast(pl.Int64)
            .clip(0, _SIMILARITY_BUCKET_COUNT - 1)
            .alias("bucket")
        )
        counted = (
            edges.select(bucket_index_expr).group_by("bucket").len(name="edge_count")
        )
        count_by_bucket = dict(
            zip(
                counted.get_column("bucket").to_list(),
                counted.get_column("edge_count").to_list(),
            )
        )
        counts = [count_by_bucket.get(i, 0) for i in range(_SIMILARITY_BUCKET_COUNT)]

    return pl.DataFrame(
        {
            "bucket_start": bucket_starts,
            "bucket_end": bucket_ends,
            "edge_count": counts,
        },
        schema={
            "bucket_start": pl.Float64,
            "bucket_end": pl.Float64,
            "edge_count": pl.Int64,
        },
    )


def resolve_blocking_countries(
    *,
    source: BlockingDatasetDescriptor,
    target: BlockingDatasetDescriptor,
    configured_countries: tuple[str, ...] | None,
) -> list[str]:
    """Intersect source/target available countries, then filter by `configured_countries`."""

    countries = sorted(
        set(source.available_countries) & set(target.available_countries)
    )
    if configured_countries:
        allowed = set(configured_countries)
        countries = [country for country in countries if country in allowed]
    return countries


_CASEFOLD_COL = "_blocking_casefold"
_JURISDICTION_COLUMN = "jurisdiction_code"


def _name_forms(
    config: BlockingRunConfig,
    frame: pl.DataFrame,
    *,
    system: str,
    country: str,
    name_transform: NameTransform,
    scored_col: str,
    population_key: str,
) -> pl.DataFrame:
    """`frame` with its cleanse-derived forms and its preprocessed name, read
    from the name forms store or derived and stored there."""
    strategy = config.strategy

    def derive(rows: pl.DataFrame) -> pl.DataFrame:
        forms = name_transform.apply(
            derive_name_forms(rows, profile=strategy.cleanse_profile)
        )
        return name_preprocessing(
            forms,
            name_col=scored_col,
            profile=strategy.preprocess_profile,
            jurisdiction_col=_JURISDICTION_COLUMN,
        )

    forms, _ = resolve_name_forms(
        config.roots,
        frame,
        system=system,
        country=country,
        columns=(*NAME_FORM_COLUMNS, NAME_PREPROCESSED_COLUMN),
        cleanse_profile=strategy.cleanse_profile,
        preprocess_profile=strategy.preprocess_profile,
        scored_col=scored_col,
        deriver_version=name_forms_ruleset_digest(),
        derive=derive,
        population_key=population_key,
    )
    return forms


def _resolve_tokenizer_path(config: BlockingRunConfig) -> Path:
    """The one tokenizer file the run's token-list path reads, the target
    system's under the run's scope, profile and trainer.

    Both sides are tokenized with it: source tokens split by another system's
    vocabulary are not comparable with the target's. A `tokenizer_path` names
    a model file outright, a candidate no profile names, and is used as it is.
    """
    strategy = config.strategy
    if strategy.tokenizer_path is not None:
        supplied = Path(strategy.tokenizer_path)
        if not supplied.is_file():
            raise FileNotFoundError(
                f"strategy.tokenizer_path names no file: {supplied}"
            )
        return supplied
    directory = promoted_tokenizer_directory(
        config.roots,
        system=scope_system(strategy.tokenizer_scope, config.target.system),
        tokenizer_id=tokenizer_id(strategy.tokenizer, strategy.tokenizer_encoding),
        profile=strategy.tokenizer_profile,
    )
    return tokenizer_directory_files(directory, trainer=strategy.tokenizer).model


def _tokenize(
    frame: pl.DataFrame,
    config: BlockingRunConfig,
    *,
    name_col: str = "name",
) -> pl.DataFrame:
    """`frame` with `_TOKEN_COL` holding each row's `name_col` tokenized by
    the run's one tokenizer, whichever side `frame` is.

    The tokenizer sees a lowercased copy of the column, never the column
    itself. Every tokenizer here was trained on the cleanse chain's
    lowercased output and carries no normalizer of its own, so a capital
    letter is outside its vocabulary and came back as an unknown token: the
    raw `name` the identity transform scores is mixed case, and a wordpiece
    run over it was mostly UNKs. `str.to_lowercase` is the cleanse chain's
    own operation, so the tokenizer sees exactly the case it was trained on.
    The copy is dropped afterwards and `name_col` is left as it was, so the
    raw name-equality level, the exact-name fast path and the diagnostics
    keep seeing the original case.
    """
    strategy = config.strategy
    lowered = frame.with_columns(
        pl.col(name_col)
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .alias(_CASEFOLD_COL)
    )
    tokenized = tokenize_name_dataframe(
        lowered,
        tokenizer_path=_resolve_tokenizer_path(config),
        trainer=strategy.tokenizer,
        name_col=_CASEFOLD_COL,
        token_col=_TOKEN_COL,
        # Words are removed in one place, the preprocessing step that wrote
        # `name_col`, so the tokenizer's own removal is given nothing to remove.
        noise_words=(),
    )
    return tokenized.drop(_CASEFOLD_COL)


def _build_source_truth_map(
    source_frame: pl.DataFrame,
    *,
    roots: WorkspaceRoots,
    system: str,
    target_system: str = "",
) -> pl.DataFrame:
    """`MatchedLayerTruth`'s map: each source row's recorded cross-system
    match, read off the loaded frame's own `match_uri` for an entity row and
    walked back through the URI chain for a derived one (see `truth.py`).

    Kept under this name for `remeasure_pair_truth_eval`, which re-scores an
    existing run and has no run config to carry a resolver on: a run
    directory records no truth rule, so a re-measure always reads the
    matched layer's own match, the only rule a persisted run has ever been
    scored under.
    """
    return MatchedLayerTruth().resolve(
        source_frame,
        roots=roots,
        source_system=system,
        target_system=target_system,
    )


def _apply_similarity_ratio_pruning(
    edges: pl.DataFrame, *, ratio: float
) -> pl.DataFrame:
    if edges.height == 0:
        return edges
    return edges.filter(
        pl.col("similarity") >= ratio * pl.col("similarity").max().over("source_id")
    )


def _apply_target_cap(edges: pl.DataFrame, *, max_per_target: int) -> pl.DataFrame:
    if edges.height == 0:
        return edges
    ranked = edges.sort(
        ["target_id", "similarity", "source_id"], descending=[False, True, False]
    )
    return (
        ranked.with_columns(pl.int_range(pl.len()).over("target_id").alias("_rank"))
        .filter(pl.col("_rank") < max_per_target)
        .drop("_rank")
    )


def _probe_target_neighbors(
    target_text: pl.DataFrame,
    *,
    clustering_strategy,
    target_index,
    feature_col: str,
    min_similarity: float,
    max_per_target: int,
    backend: str,
    backend_options: dict[str, object] | None,
    chunk_size: int,
    country: str,
) -> pl.DataFrame:
    """Probe every row of `target_text` against `target_index` on `backend`,
    one target row treated as `score_source_chunk`'s "source" -- the
    sub-linear backend a run already scans real sources with also answers a
    target's own near neighbours, rather than a second exhaustive self-join
    nothing could afford (817,761 target rows on `gleif -> ie`).

    Builds its *own* `NearestNeighbors`/backend-index state, sized to
    `max_per_target + 1` (see below), rather than reusing `_score_country`'s
    already-built `target_nn_index`/`target_backend_index`: those are fit
    for the main scan's own `top_k`, and a backend that fits a fixed
    neighbour count at build time (`sklearn`, `svd_rerank`) hands back
    exactly that many candidates regardless of what `top_k` a later call
    site asks for -- confirmed directly (a shared `top_k=2` scan index
    silently capped every family in this module's fixture down to one real
    neighbour each, verified against a raw cosine similarity matrix over
    the same target rows before this was corrected). The rebuild is a
    second one-time backend-index cost on top of the scan's own, proportional
    to target rows the same way that first build already is.

    Returns `target_id_a`/`target_id_b`/`similarity`/`country` edges,
    self-loops removed and each target's own kept list re-capped at
    `max_per_target` afterwards: a target's own row always scores a perfect
    match against itself, and a same-name twin -- a legitimate duplicate
    target row, the exact-join cascade's own edge-for-every-match case --
    ties it there at `similarity == 1.0`, so the probe asks for one more
    than the cap (`max_per_target + 1`) and only re-caps once the self-id is
    gone, rather than let the guaranteed self-match spend one of the cap's
    real slots.

    Batched over `chunk_size`-row slices of `target_text`, the same
    chunking `_score_country`'s own scan uses, since a target index can be
    as large as the source population it also scores.
    """
    probe_top_k = max_per_target + 1
    probe_nn_index = (
        None
        if backend in _BACKENDS_WITHOUT_EAGER_SKLEARN_NN
        else build_target_nearest_neighbors(
            target_index, top_k=probe_top_k, max_candidates_per_source=probe_top_k
        )
    )
    probe_backend_index = clustering_strategy.build_backend_index(
        target_index,
        backend=backend,
        top_k=probe_top_k,
        max_candidates_per_source=probe_top_k,
        backend_options=backend_options or {},
        min_similarity=min_similarity,
    )
    parts: list[pl.DataFrame] = []
    total = target_text.height
    for batch_start in range(0, total, chunk_size):
        batch_end = min(batch_start + chunk_size, total)
        chunk = target_text.slice(batch_start, batch_end - batch_start)
        parts.append(
            clustering_strategy.score_source_chunk(
                chunk,
                target_index=target_index,
                source_id_col="system_uri",
                text_col=feature_col,
                top_k=probe_top_k,
                min_similarity=min_similarity,
                max_candidates_per_source=probe_top_k,
                nn_index=probe_nn_index,
                backend=backend,
                backend_index=probe_backend_index,
                backend_options=backend_options or {},
            )
        )
    scored = (
        pl.concat(parts, how="vertical_relaxed")
        if parts
        else pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
    )
    edges = (
        scored.rename({"source_id": "target_id_a", "target_id": "target_id_b"})
        .filter(pl.col("target_id_a") != pl.col("target_id_b"))
        .select("target_id_a", "target_id_b", "similarity")
    )
    if edges.height == 0:
        return edges.with_columns(pl.lit(country).alias("country")).select(
            list(_TARGET_NEIGHBOR_EDGES_SCHEMA.keys())
        )
    ranked = edges.sort(
        ["target_id_a", "similarity", "target_id_b"], descending=[False, True, False]
    )
    capped = (
        ranked.with_columns(pl.int_range(pl.len()).over("target_id_a").alias("_rank"))
        .filter(pl.col("_rank") < max_per_target)
        .drop("_rank")
    )
    return capped.with_columns(pl.lit(country).alias("country")).select(
        list(_TARGET_NEIGHBOR_EDGES_SCHEMA.keys())
    )


def _split_exact_name_matches(
    source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    *,
    country: str,
    name_col: str = RAW_NAME_COLUMN,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split `source_frame` on an exact `name_col` equi-join against `target_frame`.

    Returns `(unmatched_source_frame, exact_match_edges)`. `unmatched_source_frame`
    is the subset of `source_frame` with no exact match in `target_frame` --
    the population that still needs the full
    tokenize/text-view/`build_target_index`/`score_source_chunk` path.
    `exact_match_edges` is already shaped like `matched_edges`
    (`source_id, target_id, similarity=1.0, rank, country`), one edge per
    matching target row, ranked by `target_id` ascending, with no
    similarity-backend call -- ready to concatenate straight into the scored
    raw edges before pruning runs, so pruning and clustering see one combined
    edge set.

    `name_col` is the same column on both sides: the raw `name` under the
    identity transform, or the column the run's name transform derived, so
    the fast path resolves exactly the pairs the scored comparison would
    find identical.

    A `null`/blank value never counts as a match: Polars excludes null-key
    rows from an inner join by default (verified directly, not just assumed),
    so two rows that both merely lack a name do not spuriously "match".
    """

    target_names = target_frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias("target_id"),
        pl.col(name_col).cast(pl.Utf8, strict=False).alias("name"),
    )
    source_names = source_frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias("source_id"),
        pl.col(name_col).cast(pl.Utf8, strict=False).alias("name"),
    )
    exact_match_edges = (
        source_names.join(target_names, on="name", how="inner")
        .sort(["source_id", "target_id"])
        .with_columns(
            (pl.int_range(pl.len()).over("source_id") + 1).cast(pl.Int64).alias("rank"),
            pl.lit(1.0).alias("similarity"),
            pl.lit(country).alias("country"),
        )
        .select("source_id", "target_id", "similarity", "rank", "country")
    )

    if exact_match_edges.height == 0:
        return source_frame, pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)

    exact_source_ids = exact_match_edges.get_column("source_id").unique().to_list()
    unmatched_source_frame = source_frame.filter(
        ~pl.col("system_uri").cast(pl.Utf8).is_in(exact_source_ids)
    )
    return unmatched_source_frame, exact_match_edges


_EXACT_JOIN_LEVEL_TRANSFORM = "transform"
_BASIC_NAME_COLUMN = "name_cleansed_basic"
_CLEANSED_NAME_COLUMN = "name_cleansed"


def _exact_join_forms(scored_name_col: str) -> tuple[tuple[str, str], ...]:
    """The name forms the exact-join cascade joins on, in order: the raw
    name, then the basic and the cleansed forms both sides derived live,
    then the transform's own column when it is none of those (`short_name`,
    `acronym`). Each level is named as the evaluation names it, so a row
    resolved here is found at the level the evaluation places its pair."""
    forms = [
        (NAME_EQUALITY_RAW, RAW_NAME_COLUMN),
        (NAME_EQUALITY_BASIC, _BASIC_NAME_COLUMN),
        (NAME_EQUALITY_CLEANSED, _CLEANSED_NAME_COLUMN),
    ]
    if scored_name_col not in {column for _, column in forms}:
        forms.append((_EXACT_JOIN_LEVEL_TRANSFORM, scored_name_col))
    return tuple(forms)


def _cascade_exact_name_matches(
    source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    *,
    country: str,
    forms: Sequence[tuple[str, str]],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int], int]:
    """Find the source rows an exact join on some form makes equal to a target.

    Each form's join takes only the rows the joins before it left, so a row
    is counted under the first form that makes it equal to a target. Returns
    the rows no form matched, every matched edge at similarity 1.0, the count
    of rows each level matched, and how many matched rows got more than one
    target from their form, which the join keeps as the raw join always has.
    The caller scans every source regardless; the edges and counts are a
    guarantee and a measurement, not a skip.
    """
    remaining = source_frame
    edge_parts: list[pl.DataFrame] = []
    resolved_by: dict[str, int] = {}
    for level, column in forms:
        if column not in remaining.columns or column not in target_frame.columns:
            resolved_by[level] = 0
            continue
        remaining, edges = _split_exact_name_matches(
            remaining, target_frame, country=country, name_col=column
        )
        resolved_by[level] = (
            int(edges.get_column("source_id").n_unique()) if edges.height else 0
        )
        if edges.height:
            edge_parts.append(edges)
    exact_match_edges = (
        pl.concat(edge_parts, how="vertical_relaxed")
        if edge_parts
        else pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
    )
    multi_target_sources = (
        exact_match_edges.group_by("source_id").len().filter(pl.col("len") > 1).height
        if exact_match_edges.height
        else 0
    )
    return remaining, exact_match_edges, resolved_by, int(multi_target_sources)


def _build_diagnostic_frame(
    text_frame: pl.DataFrame,
    *,
    id_col: str,
    id_alias: str,
    country: str,
    input_name_col: str,
    feature_col: str,
) -> pl.DataFrame:
    """Snapshot `text_frame`'s operator-transformed values for `country`.

    Carries `input_name_col` (the raw text this run actually read:
    `strategy.name_source` on the source side, `"name"` on the target side)
    and `feature_col` (`cluster_text`/`cluster_tokens`, whichever
    `effective_text_view` produced and `score_source_chunk()` actually
    scored on). `name_cleansed` is included when `text_frame` carries it --
    the canonicalization step whose effect this snapshot exists to make
    visible -- and skipped otherwise rather than erroring. `id_alias`
    matches `matched_edges`' own `source_id`/`target_id` column name, so
    downstream joins need no rename.
    """

    columns: list[pl.Expr] = [
        pl.col(id_col).cast(pl.Utf8).alias(id_alias),
        pl.lit(country).alias("country"),
        pl.col(input_name_col).cast(pl.Utf8, strict=False).alias(input_name_col),
    ]
    if "name_cleansed" in text_frame.columns and input_name_col != "name_cleansed":
        columns.append(
            pl.col("name_cleansed").cast(pl.Utf8, strict=False).alias("name_cleansed")
        )
    columns.append(pl.col(feature_col).alias(feature_col))
    return text_frame.select(columns)


def _finalize_diagnostics(
    parts: list[pl.DataFrame],
    *,
    id_alias: str,
    input_name_col: str,
    feature_col: str,
    clusters: pl.DataFrame,
) -> pl.DataFrame:
    """Concatenate the per-country diagnostic parts, then left-join in
    `cluster_id` from the run's own `clusters` frame -- the documented join
    key connecting `source_diagnostics`/`target_diagnostics` (see
    `BlockingRunResult`'s docstring). Null `cluster_id` means that row's
    `id_alias` value never appears in `clusters`, i.e. it was never matched.
    """

    if parts:
        combined = pl.concat(parts, how="vertical_relaxed")
    else:
        feature_dtype = pl.List(pl.Utf8) if feature_col == "cluster_tokens" else pl.Utf8
        combined = pl.DataFrame(
            schema={
                id_alias: pl.Utf8,
                "country": pl.Utf8,
                input_name_col: pl.Utf8,
                feature_col: feature_dtype,
            }
        )
    cluster_lookup = clusters.select(pl.col("node_id").alias(id_alias), "cluster_id")
    return combined.join(cluster_lookup, on=id_alias, how="left")


@dataclass(slots=True)
class _CountryScoreResult:
    raw_edges: pl.DataFrame
    pruned_edges: pl.DataFrame
    raw_truth_eval: pl.DataFrame | None
    pruned_truth_eval: pl.DataFrame | None
    pair_truth_eval_detail: pl.DataFrame | None
    raw_count: int
    dropped_by_similarity_ratio: int
    dropped_by_target_cap: int
    pruned_count: int
    source_nodes: pl.DataFrame
    target_nodes: pl.DataFrame
    exact_match_summary: pl.DataFrame
    source_diagnostics: pl.DataFrame | None
    target_diagnostics: pl.DataFrame | None
    # This country's part keys, combined across the run's countries into its
    # `RunKeys` by `_run_keys_from_parts`.
    source_population_key: str
    target_population_key: str
    index_key: str
    truth_key: str | None
    # This country's own target-neighbour edges and the one-row
    # summary of the probe that produced them; both empty when the
    # strategy leaves target-neighbour linking off.
    target_neighbor_edges: pl.DataFrame
    target_neighbor_summary_row: pl.DataFrame


def _evaluate_country_truth(
    *,
    full_source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    source_truth_map: pl.DataFrame,
    raw_edges: pl.DataFrame,
    pruned_edges: pl.DataFrame,
    source_system: str,
    target_system: str,
    country: str,
    top_k: int | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Score one country's raw and pruned candidates against its truth map.

    Returns the raw and pruned `pair_truth_eval` rows and the per-pair detail.
    Shared by `_score_country()` and `remeasure_pair_truth_eval()`, so a run and a
    re-measure of it classify and count through one sequence.
    """

    # The difficulty-bucket lookups are built from the *full*
    # source population, matching source_truth_map. An earlier version scoped them to
    # the unmatched population instead, meaning a truth pair whose source row the fast path
    # resolved by exact name was simply absent from the lookup, and a left
    # join's null was read as "these names differ" -- filing the easiest
    # pairs in the dataset into the partition that exists to isolate the
    # hardest. The concern that motivated that scoping (exact matches must
    # not inflate the reported figures) is met properly here: such a pair
    # is classified at the `raw` level on its own values and counted
    # there, outside the `never` level's recall rather than relabelled.
    source_name_forms = build_name_form_map(full_source_frame, id_col="source_id")
    target_name_forms = build_name_form_map(target_frame, id_col="target_id")
    # source_rows is the *full* per-country source population
    # (full_source_frame, exact-match rows included), not the post-exact-match
    # residual that actually goes through clustering -- matched_edges/raw_edges
    # already union both, so the exhaustive comparison space
    # candidate_set_size_ratio measures against has to match. top_k is the
    # retrieval window the run used, read back through matched_edges' rank
    # column for recall_at_k.
    raw_truth_eval = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=raw_edges,
        source_system=source_system,
        target_system=target_system,
        country=country,
        target_rows=target_frame.height,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
        source_rows=full_source_frame.height,
        top_k=top_k,
    )
    pruned_truth_eval = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=pruned_edges,
        source_system=source_system,
        target_system=target_system,
        country=country,
        target_rows=target_frame.height,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
        source_rows=full_source_frame.height,
        top_k=top_k,
    )
    # Per-pair companion to pruned_truth_eval's aggregate
    # buckets, scored against the same pruned edge set -- there is no
    # raw variant mirroring raw_truth_eval. Unlike
    # pruned_truth_eval, this has no null-bucket fallback: a name/
    # name_cleansed pair is what every row's identity is built from, so
    # it stays None (rather than a bucket-less frame) when either side's
    # name-form lookup is unavailable (an ad hoc prepared_base_dir input
    # lacking name_cleansed; real matched/cleansed data always has it).
    pair_truth_eval_detail = None
    if source_name_forms is not None and target_name_forms is not None:
        pair_truth_eval_detail = compute_pair_truth_eval_detail(
            source_truth_map=source_truth_map,
            matched_edges=pruned_edges,
            source_system=source_system,
            target_system=target_system,
            country=country,
            source_name_forms=source_name_forms,
            target_name_forms=target_name_forms,
        )
    return raw_truth_eval, pruned_truth_eval, pair_truth_eval_detail


@contextmanager
def _timed_phase(
    telemetry: Telemetry,
    emit_progress: Callable[..., None],
    name: str,
    *,
    country: str,
    rows_in: int | None = None,
) -> Iterator[PhaseHandle]:
    """One phase of a country's scoring, recorded and announced.

    Records the phase through `telemetry` and emits `phase_start` and
    `phase_complete` progress events around it, carrying the rows it works
    on and, on completion, its elapsed seconds and resident set size at its
    end and at its sampled peak, so a
    long pre-scoring phase such as an encode or an index build is readable
    from the console rather than indistinguishable from a hang.
    """
    emit_progress("phase_start", name=name, country=country, rows=rows_in)
    with telemetry.phase(name, label=country, rows_in=rows_in) as handle:
        yield handle
    record = telemetry.records[-1]
    emit_progress(
        "phase_complete",
        name=name,
        country=country,
        rows=rows_in,
        elapsed_seconds=record.elapsed_seconds,
        rss_bytes=record.rss_bytes,
        peak_rss_bytes=record.peak_rss_bytes,
    )


@dataclass(slots=True)
class _PreparedCountry:
    """One country's rows as a run consumes them: both sides loaded, with each
    side's population key over the raw rows read, and, once `derive` is set,
    their name forms derived and the transform applied."""

    full_source_frame: pl.DataFrame
    target_frame: pl.DataFrame
    source_name_col: str
    target_name_col: str
    source_population_key: str
    target_population_key: str


def _prepare_country(
    *,
    config: BlockingRunConfig,
    country: str,
    phase: Callable[..., AbstractContextManager[PhaseHandle]],
    derive: bool = True,
    full_source_frame_override: pl.DataFrame | None = None,
    target_frame_override: pl.DataFrame | None = None,
) -> _PreparedCountry:
    """Load one country's rows and key them, then derive their name forms,
    the part of a run the key pass and scoring share, so both key exactly the
    rows scoring reads.

    Each side's population key covers the raw `system_uri` and `name` read,
    before any form is derived: every scored form is a function of `name`
    under the run's cleanse profile and transform, which the settings key
    already covers, so the key pass skips the derivation (`derive=False`), the
    one expensive step before scoring. `phase` is the caller's timed-phase
    factory, `_timed_phase` bound to its telemetry and progress callback.
    """
    strategy = config.strategy
    with phase("load_frames") as loaded:
        full_source_frame = (
            full_source_frame_override
            if full_source_frame_override is not None
            else load_country_frame(config.source, country=country)
        )
        target_frame = (
            target_frame_override
            if target_frame_override is not None
            else load_country_frame(config.target, country=country)
        )
        loaded.rows_out = full_source_frame.height + target_frame.height
    # A country resolve_blocking_countries() included in this run is
    # a country the source side is supposed to have rows for. Zero here is
    # never legitimate scoring output -- unlike a nonempty source frame that
    # simply matches nothing, which still completes and reports a real,
    # zero-count candidate set (see EmptySourceLoadError's docstring) -- it
    # means the load itself came back empty, most likely because it raced a
    # concurrent write to the source layer (read_country_partition_frame
    # deliberately treats "no files here" as "legitimately empty" so an
    # ordinary empty country round-trips without erroring, which is exactly
    # what makes a load caught mid-swap indistinguishable from one). Failing
    # here, before any artefact is written, is what lets a consumer reading
    # only the output tell the two apart.
    if full_source_frame.height == 0:
        raise EmptySourceLoadError(
            f"source system {config.source.system!r} loaded zero rows for "
            f"country {country!r} from {config.source.system_dir} -- refusing "
            "to score and write a blocking artefact set for a pairing whose "
            "source side loaded no records."
        )

    # Refuse a dense-vocabulary representation on an exhaustive
    # backend here, at the first point in the run where the target-side row
    # count is known and before anything has been built from it. The strategy
    # enforces the same gate itself -- that is where the contract lives, and
    # it holds for callers that never go through this workflow -- but the
    # earliest it can fire is build_backend_index(), by which point
    # tokenization and the TF-IDF fit over the whole target side have already
    # run. On the gb corpus that difference is about five minutes of silent
    # work against a second here, which is the entire point of the early check.
    ensure_dense_vocabulary_scale_supported(
        representation=strategy.representation,
        backend=strategy.similarity_backend,
        target_rows=target_frame.height,
        backend_options=strategy.backend_options,
    )

    # Each side's population digest over the raw rows read. Every form a run
    # scores is derived from `name` under settings the settings key covers, so
    # this is what the run consumed without paying for the derivation.
    with phase(
        "population_keys", rows_in=full_source_frame.height + target_frame.height
    ):
        source_population_key = rows_digest(
            full_source_frame, id_col="system_uri", value_col=RAW_NAME_COLUMN
        )
        target_population_key = rows_digest(
            target_frame, id_col="system_uri", value_col=RAW_NAME_COLUMN
        )
    name_transform = resolve_name_transform(strategy.name_transform)
    source_name_col = name_transform.scored_column(strategy.name_source)
    target_name_col = name_transform.scored_column(RAW_NAME_COLUMN)
    if not derive:
        return _PreparedCountry(
            full_source_frame=full_source_frame,
            target_frame=target_frame,
            source_name_col=source_name_col,
            target_name_col=target_name_col,
            source_population_key=source_population_key,
            target_population_key=target_population_key,
        )

    # Both sides' name forms are derived live here, under the run's own
    # profile, before the fast path, tokenization, the text view or the
    # name-form lookups read either frame (see `name_transform.py`). Nothing
    # below reads a `name_cleansed` the loaded layer carried: what a run
    # scores and what `pair_truth_eval` classifies from are the forms this
    # run derived on both sides. The transform then names which of those
    # forms is compared: the raw `name` under `identity`, each side keeping
    # the column it always read, or the derived column otherwise.
    #
    # The forms, and the preprocessed name both the tokenizer and the
    # vectorizer read, come through the name forms store: derived the first
    # time these rows are met under these profiles, read back every time
    # after, so two runs, and two areas, score the same text.
    with phase("name_forms", rows_in=full_source_frame.height + target_frame.height):
        full_source_frame = _name_forms(
            config,
            full_source_frame,
            system=config.source.system,
            country=country,
            name_transform=name_transform,
            scored_col=source_name_col,
            population_key=source_population_key,
        )
        target_frame = _name_forms(
            config,
            target_frame,
            system=config.target.system,
            country=country,
            name_transform=name_transform,
            scored_col=target_name_col,
            population_key=target_population_key,
        )

    return _PreparedCountry(
        full_source_frame=full_source_frame,
        target_frame=target_frame,
        source_name_col=source_name_col,
        target_name_col=target_name_col,
        source_population_key=source_population_key,
        target_population_key=target_population_key,
    )


def _score_country(
    *,
    config: BlockingRunConfig,
    country: str,
    effective_text_view: str,
    feature_col: str,
    clustering_strategy,
    build_settings,
    source_chunk_size: int,
    emit_progress: Callable[..., None],
    telemetry: Telemetry,
    full_source_frame_override: pl.DataFrame | None = None,
    target_frame_override: pl.DataFrame | None = None,
    source_system_label: str | None = None,
) -> _CountryScoreResult:
    """Score one country of one pairing.

    `full_source_frame_override`/`target_frame_override` let a caller run
    this same scoring path over a source dataset other than `config.source`'s
    own primary records -- the name-variant pairings are the callers that
    do, supplying their own already-loaded frames instead of
    `config.source`'s disk layer. `execute_blocking_run()` itself never
    passes them, so its own behaviour (and the canonical baseline's own
    metrics) is unchanged by their existence. Ground truth is read by
    `config.truth`, whichever frame the source is (see `truth.py`).
    `source_system_label` overrides the `source_system` value stamped onto
    `pair_truth_eval`/`pair_truth_eval_detail` rows for the same callers
    (`config.source.system` is the wrong label for a name-rows pairing,
    which is not `config.source`'s population at all).
    """
    strategy = config.strategy

    def _phase(name: str, *, rows_in: int | None = None):
        return _timed_phase(
            telemetry, emit_progress, name, country=country, rows_in=rows_in
        )

    prepared = _prepare_country(
        config=config,
        country=country,
        phase=_phase,
        full_source_frame_override=full_source_frame_override,
        target_frame_override=target_frame_override,
    )
    full_source_frame = prepared.full_source_frame
    target_frame = prepared.target_frame
    source_name_col = prepared.source_name_col
    target_name_col = prepared.target_name_col
    source_population_key = prepared.source_population_key
    target_population_key = prepared.target_population_key
    source_system = source_system_label or config.source.system

    # The exact-join cascade: one equi-join per derived name form in the order
    # the evaluation's levels name them, raw, basic, cleansed, then the
    # transform's own column when it is another form, each join taking the
    # sources the joins before it left. A source a form makes equal to a
    # target gets that edge at similarity 1.0 and is counted under the form,
    # so a run says how many of its sources were trivially equal and at which
    # form, which nothing could know a priori. It is a measurement and a
    # guaranteed edge, never a skip: every source is still scanned, since an
    # identical name says nothing about a source's other near neighbours, the
    # renamed twin above all, and a source taken out of the scan never reaches
    # them. The name transform still decides what text the scan scores; it
    # does not decide which forms are joined.
    with _phase("exact_name_split", rows_in=full_source_frame.height) as split:
        if strategy.exact_name_filter:
            (
                remaining_after_exact,
                exact_match_edges,
                resolved_by,
                multi_target_sources,
            ) = _cascade_exact_name_matches(
                full_source_frame,
                target_frame,
                country=country,
                forms=_exact_join_forms(target_name_col),
            )
        else:
            remaining_after_exact = full_source_frame
            exact_match_edges = pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
            resolved_by = {}
            multi_target_sources = 0
        exact_match_count = (
            int(exact_match_edges.get_column("source_id").n_unique())
            if exact_match_edges.height
            else 0
        )
        split.rows_out = exact_match_count

    # A matched source is left out of the scan once target-neighbour
    # linking is on -- its other near targets are now reached through the
    # matched target's own neighbour edges (built below) rather than only
    # through this source's scan, restoring the skip the cascade had to give up
    # (see that item's "Follow-on"). Off (the default), every source is
    # still scanned regardless of a match, as the cascade decided.
    source_frame = (
        remaining_after_exact
        if strategy.target_neighbor_min_similarity is not None
        else full_source_frame
    )
    exact_match_summary = pl.DataFrame(
        {
            "country": [country],
            "exact_match_count": [exact_match_count],
            "resolved_by_raw": [resolved_by.get(NAME_EQUALITY_RAW, 0)],
            "resolved_by_basic": [resolved_by.get(NAME_EQUALITY_BASIC, 0)],
            "resolved_by_cleansed": [resolved_by.get(NAME_EQUALITY_CLEANSED, 0)],
            "resolved_by_transform": [resolved_by.get(_EXACT_JOIN_LEVEL_TRANSFORM, 0)],
            "multi_target_source_count": [multi_target_sources],
            "unmatched_source_row_count": [
                full_source_frame.height - exact_match_count
            ],
        },
        schema=_EXACT_MATCH_SUMMARY_SCHEMA,
    )

    # The target index is always built from target_frame's own
    # canonical rows -- a name-variant sidecar is never folded into it.
    # Expanding the target index with recorded names made "did a previous
    # name reach the current company" true by construction, spending the
    # ground truth those rows carry instead of measuring against it. A
    # recorded name variant is scored as its own labelled run instead (see
    # `execute_name_variant_recovery_run`), diffed against this run's own
    # per-pair outcomes rather than folded into them.
    target_index_frame = target_frame

    # Under the identity transform, strategy.name_source selects the
    # *source* side's text-view/tokenization column only ("name", today's
    # default, or "short_name", company_cleanse's existing suffix/noise-word
    # stem) and the target side stays on "name" -- see
    # BlockingStrategyConfig.name_source's docstring for why there is
    # deliberately no target-side equivalent. Under any other transform
    # both columns are the transform's own derived column.
    text_rows = source_frame.height + target_index_frame.height
    # Both paths below read `name_preprocessed`, the one step between the
    # scored name and whatever turns it into tokens or vectors, which came
    # with the name forms. Each side's scored column is left as it was, for
    # the joins and the diagnostics.
    if effective_text_view == "tokens":
        with _phase("tokenize", rows_in=text_rows):
            source_frame = _tokenize(
                source_frame,
                config,
                name_col=NAME_PREPROCESSED_COLUMN,
            )
            target_index_frame = _tokenize(
                target_index_frame,
                config,
                name_col=NAME_PREPROCESSED_COLUMN,
            )

    token_col = _TOKEN_COL if effective_text_view == "tokens" else None
    include_cluster_text = strategy.representation == "tfidf"
    with _phase("text_view", rows_in=text_rows):
        source_text = build_clustering_text_view(
            source_frame,
            name_col=NAME_PREPROCESSED_COLUMN,
            token_col=token_col,
            text_view=effective_text_view,
            include_cluster_text=include_cluster_text,
        )
        target_text = build_clustering_text_view(
            target_index_frame,
            name_col=NAME_PREPROCESSED_COLUMN,
            token_col=token_col,
            text_view=effective_text_view,
            include_cluster_text=include_cluster_text,
        )

    # Opt-in snapshot of the operator-transformed values this
    # country's scoring pass actually produced. source_diagnostics excludes
    # Exact-match source rows -- they never reach source_text at all
    # (see _split_exact_name_matches above), so there is nothing transformed
    # to report for them. target_diagnostics dedupes target_text down to one
    # row per canonical target_id, dropping any extra name-variant index
    # rows target_index_frame may carry -- see _build_diagnostic_frame's
    # docstring.
    source_diagnostics = None
    target_diagnostics = None
    if config.emit_diagnostics:
        source_diagnostics = _build_diagnostic_frame(
            source_text,
            id_col="system_uri",
            id_alias="source_id",
            country=country,
            input_name_col=source_name_col,
            feature_col=feature_col,
        )
        target_diagnostics = _build_diagnostic_frame(
            target_text.unique(subset=["system_uri"], keep="first"),
            id_col="system_uri",
            id_alias="target_id",
            country=country,
            input_name_col=target_name_col,
            feature_col=feature_col,
        )

    # The target index's own key, the same one the key pass computes, since
    # `_index_part_key` reads the same tokenizer file `_tokenize` above did.
    index_key = _index_part_key(
        config,
        target_population_key=target_population_key,
        effective_text_view=effective_text_view,
        build_settings=build_settings,
    )

    # Resolved through the same shared cache `src/validation`'s own
    # harness uses (`target_index_cache.resolve_target_index`), keyed on
    # `index_key`, so a target this run
    # already indexed -- under this same model, representation and text view
    # -- is reused rather than rebuilt. The target index build is the encode
    # for a dense representation and the vectorizer fit for a sparse one: the
    # phase whose cost decides whether a corpus needs remote batches, so it
    # is timed on its own regardless of whether it was a cache hit or miss.
    with _phase("target_index", rows_in=target_text.height) as target_index_phase:
        target_index, target_index_cache_hit, _target_index_cache_dir = (
            resolve_target_index(
                roots=config.roots,
                target_system=config.target.system,
                country=country,
                settings={"index_key": index_key},
                clustering_strategy=clustering_strategy,
                build_settings=build_settings,
                target_frame=target_text,
                target_id_col="system_uri",
                text_col=feature_col,
            )
        )
        target_index_phase.rows_out = (
            0 if target_index_cache_hit else len(target_index.target_ids)
        )
    # Refuse "sklearn" here, once the target index is built and
    # before the backend index, when it cannot fit a dense target in the
    # memory this process actually has -- a no-op for a sparse target or any
    # other backend (dense_target_memory_gate.py's applicability check).
    ensure_sklearn_dense_target_index_memory_supported(
        target_index,
        backend=strategy.similarity_backend,
        available_bytes=psutil.virtual_memory().available,
    )
    # "dense_brute" and "hnsw" each build and scan their own index
    # (build_backend_index() below); eagerly fitting a whole-target sklearn
    # NearestNeighbors here as well -- unused for every backend but
    # "sklearn" itself, since build_backend_index() builds its own when that
    # one applies -- would reintroduce the unbounded-memory fit those two
    # backends exist to avoid.
    # "hnsw" saves its built graph under a directory this run
    # resolves through the shared target-index cache, keyed on the target
    # index's own identity (`index_key`) plus the backend options that
    # change the built graph (`hnsw_build_settings`, never a search-time
    # option like `expansion_search`) -- so a second run against the same
    # target under the same graph settings reopens it memory-mapped instead
    # of rebuilding. `scoring_backend_options` is what every call below
    # passes as `backend_options`, in place of `strategy.backend_options`
    # directly, so the resolved directory reaches the backend build/score
    # calls (including the target-neighbor probe's, which builds the same
    # graph over the same target and benefits from the same reuse).
    scoring_backend_options = strategy.backend_options or {}
    if strategy.similarity_backend == "hnsw":
        scoring_backend_options = dict(scoring_backend_options)
        scoring_backend_options["index_dir"] = resolve_hnsw_index_dir(
            roots=config.roots,
            target_system=config.target.system,
            country=country,
            index_key=index_key,
            hnsw_build_settings=hnsw_build_settings(strategy.backend_options),
        )
    with _phase("backend_index", rows_in=target_text.height):
        target_nn_index = (
            None
            if strategy.similarity_backend in _BACKENDS_WITHOUT_EAGER_SKLEARN_NN
            else build_target_nearest_neighbors(
                target_index,
                top_k=strategy.top_k,
                max_candidates_per_source=strategy.max_candidates_per_source,
            )
        )
        target_backend_index = clustering_strategy.build_backend_index(
            target_index,
            backend=strategy.similarity_backend,
            top_k=strategy.top_k,
            max_candidates_per_source=strategy.max_candidates_per_source,
            backend_options=scoring_backend_options,
            min_similarity=strategy.min_similarity,
        )

    # Link each target to its near neighbours before any source is
    # scanned, so a cluster below can hold a whole name family rather than
    # only what a source happened to reach. Off (both settings None) by
    # default -- see BlockingStrategyConfig's docstring. Cached beside the
    # target index itself (`resolve_target_neighbor_edges`), keyed on the
    # index's own identity plus the backend and this pair of settings, so a
    # second run against the same target under the same backend and
    # threshold/cap probes nothing.
    target_neighbor_edges = pl.DataFrame(schema=_TARGET_NEIGHBOR_EDGES_SCHEMA)
    target_neighbor_summary_row = pl.DataFrame(schema=_TARGET_NEIGHBOR_SUMMARY_SCHEMA)
    neighbor_min_similarity = strategy.target_neighbor_min_similarity
    neighbor_max_per_target = strategy.target_neighbor_max_per_target
    if neighbor_min_similarity is not None and neighbor_max_per_target is not None:
        neighbor_settings = {
            "neighbour_edges_key": neighbour_edges_key(
                index_key=index_key,
                edge_settings={
                    "similarity_backend": strategy.similarity_backend,
                    "backend_options": strategy.backend_options or {},
                    "target_neighbor_min_similarity": neighbor_min_similarity,
                    "target_neighbor_max_per_target": neighbor_max_per_target,
                },
            )
        }
        with _phase("target_neighbors", rows_in=target_text.height) as neighbor_phase:
            target_neighbor_edges, target_neighbor_cache_hit, _ = (
                resolve_target_neighbor_edges(
                    roots=config.roots,
                    target_system=config.target.system,
                    country=country,
                    settings=neighbor_settings,
                    compute=lambda: _probe_target_neighbors(
                        target_text,
                        clustering_strategy=clustering_strategy,
                        target_index=target_index,
                        feature_col=feature_col,
                        min_similarity=neighbor_min_similarity,
                        max_per_target=neighbor_max_per_target,
                        backend=strategy.similarity_backend,
                        backend_options=scoring_backend_options,
                        chunk_size=source_chunk_size,
                        country=country,
                    ),
                )
            )
            neighbor_phase.rows_out = (
                0 if target_neighbor_cache_hit else target_neighbor_edges.height
            )
        targets_at_cap = (
            int(
                target_neighbor_edges.group_by("target_id_a")
                .len()
                .filter(pl.col("len") >= neighbor_max_per_target)
                .height
            )
            if target_neighbor_edges.height
            else 0
        )
        target_neighbor_summary_row = pl.DataFrame(
            {
                "country": [country],
                "min_similarity": [neighbor_min_similarity],
                "max_per_target": [neighbor_max_per_target],
                "edge_count": [target_neighbor_edges.height],
                "targets_at_cap": [targets_at_cap],
                "cache_hit": [bool(target_neighbor_cache_hit)],
            },
            schema=_TARGET_NEIGHBOR_SUMMARY_SCHEMA,
        )

    source_nonempty = (
        source_text.filter(pl.col("cluster_tokens").list.len() > 0)
        if feature_col == "cluster_tokens"
        else source_text.filter(pl.col("cluster_text").str.strip_chars() != "")
    )

    source_rows_total = source_nonempty.height
    emit_progress(
        "country_start",
        country=country,
        source_rows=source_rows_total,
        target_rows=target_frame.height,
    )

    edge_parts: list[pl.DataFrame] = []
    rows_scored = 0
    scoring_steps: dict[str, float] = {}
    with _phase("scoring", rows_in=source_rows_total) as scored:
        for batch_start in range(0, source_rows_total, source_chunk_size):
            batch_end = min(batch_start + source_chunk_size, source_rows_total)
            chunk = source_nonempty.slice(batch_start, batch_end - batch_start)
            chunk_edges = clustering_strategy.score_source_chunk(
                chunk,
                target_index=target_index,
                source_id_col="system_uri",
                text_col=feature_col,
                top_k=strategy.top_k,
                min_similarity=strategy.min_similarity,
                max_candidates_per_source=strategy.max_candidates_per_source,
                nn_index=target_nn_index,
                backend=strategy.similarity_backend,
                backend_index=target_backend_index,
                backend_options=scoring_backend_options,
                timings=scoring_steps,
            )
            edge_parts.append(chunk_edges)
            rows_scored = batch_end
            emit_progress(
                "country_progress",
                country=country,
                rows_scored=rows_scored,
                source_rows=source_rows_total,
            )

        scored_edges = (
            pl.concat(edge_parts, how="vertical_relaxed")
            if edge_parts
            else pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
        ).with_columns(pl.lit(country).alias("country"))
        scored.rows_out = scored_edges.height
    # The scoring phase's two steps, each summed over its chunks: what the
    # representation costs to vectorize the source, and what the backend costs.
    for step, seconds in scoring_steps.items():
        step_record = telemetry.record_total(
            f"scoring_{step}",
            elapsed_seconds=seconds,
            label=country,
            rows_in=source_rows_total,
        )
        emit_progress(
            "phase_complete",
            name=step_record.routine,
            country=country,
            rows=source_rows_total,
            elapsed_seconds=step_record.elapsed_seconds,
            rss_bytes=None,
        )
    # exact_match_edges are unioned in here, before any pruning, so they flow
    # through the pruning and the clustering exactly like any other candidate
    # edge. A pair the scan also scored, an identical name at a cosine of one
    # less rounding, keeps the exact edge and drops the scored duplicate.
    raw_edges = pl.concat(
        [exact_match_edges, scored_edges], how="vertical_relaxed"
    ).unique(subset=["source_id", "target_id"], keep="first", maintain_order=True)
    emit_progress("country_complete", country=country, matched_edges=raw_edges.height)

    with _phase("pruning", rows_in=raw_edges.height) as pruned:
        pruned_edges = raw_edges
        if strategy.candidate_similarity_ratio is not None:
            pruned_edges = _apply_similarity_ratio_pruning(
                pruned_edges, ratio=strategy.candidate_similarity_ratio
            )
        after_ratio_count = pruned_edges.height
        if strategy.max_candidates_per_target is not None:
            pruned_edges = _apply_target_cap(
                pruned_edges, max_per_target=strategy.max_candidates_per_target
            )
        pruned_count = pruned_edges.height
        pruned.rows_out = pruned_count

    raw_truth_eval = None
    pruned_truth_eval = None
    pair_truth_eval_detail = None
    truth_key = None
    if _can_evaluate_truth(config, full_source_frame):
        with _phase("truth_eval", rows_in=pruned_count):
            source_truth_map = _resolve_country_truth_map(config, full_source_frame)
            truth_key = _truth_part_key(source_truth_map)
            raw_truth_eval, pruned_truth_eval, pair_truth_eval_detail = (
                _evaluate_country_truth(
                    full_source_frame=full_source_frame,
                    target_frame=target_frame,
                    source_truth_map=source_truth_map,
                    raw_edges=raw_edges,
                    pruned_edges=pruned_edges,
                    source_system=source_system,
                    target_system=config.target.system,
                    country=country,
                    top_k=strategy.top_k,
                )
            )

    # Full population again: exact-match source rows are still real graph
    # nodes, just resolved without a similarity-backend call.
    source_nodes = full_source_frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias("node_id"),
        pl.col("name").cast(pl.Utf8, strict=False).alias("node_name"),
    )
    target_nodes = target_frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias("node_id"),
        pl.col("name").cast(pl.Utf8, strict=False).alias("node_name"),
    )

    raw_count = raw_edges.height
    return _CountryScoreResult(
        raw_edges=raw_edges,
        pruned_edges=pruned_edges,
        raw_truth_eval=raw_truth_eval,
        pruned_truth_eval=pruned_truth_eval,
        pair_truth_eval_detail=pair_truth_eval_detail,
        raw_count=raw_count,
        dropped_by_similarity_ratio=raw_count - after_ratio_count,
        dropped_by_target_cap=after_ratio_count - pruned_count,
        pruned_count=pruned_count,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        exact_match_summary=exact_match_summary,
        source_diagnostics=source_diagnostics,
        target_diagnostics=target_diagnostics,
        source_population_key=source_population_key,
        target_population_key=target_population_key,
        index_key=index_key,
        truth_key=truth_key,
        target_neighbor_edges=target_neighbor_edges,
        target_neighbor_summary_row=target_neighbor_summary_row,
    )


def _run_keys_from_parts(
    *,
    settings_key: str,
    countries: Sequence[str],
    parts: Sequence[tuple[str, str, str | None, str]],
) -> RunKeys:
    """A run's keys from each country's `(source population, target
    population, truth, index)` part keys, in `countries` order. A run's truth
    key covers the countries that had truth, and is `None` when none did."""
    by_country = dict(zip(countries, parts, strict=True))
    truth_parts = {
        country: truth
        for country, (_, _, truth, _) in by_country.items()
        if truth is not None
    }
    return RunKeys(
        settings=settings_key,
        source_population=combine_part_keys(
            {country: part[0] for country, part in by_country.items()}
        ),
        target_population=combine_part_keys(
            {country: part[1] for country, part in by_country.items()}
        ),
        truth=combine_part_keys(truth_parts) if truth_parts else None,
        index=combine_part_keys(
            {country: part[3] for country, part in by_country.items()}
        ),
    )


def _aggregate_country_results(
    *,
    strategy: BlockingStrategyConfig,
    emit_diagnostics: bool,
    feature_col: str,
    countries: list[str],
    results: list[_CountryScoreResult],
    has_ground_truth: bool,
    settings_key: str,
    source_system: str,
    target_system: str,
    timings: tuple[dict[str, object], ...] = (),
) -> BlockingRunResult:
    """Combine one run's per-country `_score_country()` results into a
    `BlockingRunResult`, recording the name transform `strategy` applied and
    the per-phase `timings` the run's telemetry recorded.

    Shared by `execute_blocking_run()` (one country per element of
    `results`, in the same order as `countries`), and by
    `execute_name_variant_recovery_run()`/`execute_name_variant_cross_system_run()`,
    which each score a different source dataset against the same
    per-country loop shape -- all three hand this function the same
    `_CountryScoreResult` list to reshape, so a pairing's aggregation logic
    exists in exactly one place. `has_ground_truth`
    replaces reading `config.source.has_ground_truth` directly: the recovery
    run's truth comes from a self-referential map, never from
    `BlockingDatasetDescriptor.has_ground_truth`, so the caller states
    whether truth was computed rather than this function re-deriving it from
    a config that may not describe this pairing at all.
    """

    name_transform = resolve_name_transform(strategy.name_transform)
    raw_edge_parts = [result.raw_edges for result in results]
    pruned_edge_parts = [result.pruned_edges for result in results]
    raw_truth_eval_parts = [
        result.raw_truth_eval for result in results if result.raw_truth_eval is not None
    ]
    pruned_truth_eval_parts = [
        result.pruned_truth_eval
        for result in results
        if result.pruned_truth_eval is not None
    ]
    pair_truth_eval_detail_parts = [
        result.pair_truth_eval_detail
        for result in results
        if result.pair_truth_eval_detail is not None
    ]
    pruning_summary_rows = [
        {
            "country": country,
            "raw_edge_count": result.raw_count,
            "dropped_by_similarity_ratio": result.dropped_by_similarity_ratio,
            "dropped_by_target_cap": result.dropped_by_target_cap,
            "pruned_edge_count": result.pruned_count,
        }
        for country, result in zip(countries, results, strict=True)
    ]
    keys = _run_keys_from_parts(
        settings_key=settings_key,
        countries=countries,
        parts=[
            (
                result.source_population_key,
                result.target_population_key,
                result.truth_key,
                result.index_key,
            )
            for result in results
        ],
    )
    exact_match_summary_parts = [result.exact_match_summary for result in results]
    source_node_parts = [result.source_nodes for result in results]
    target_node_parts = [result.target_nodes for result in results]
    source_diagnostics_parts = [
        result.source_diagnostics
        for result in results
        if result.source_diagnostics is not None
    ]
    target_diagnostics_parts = [
        result.target_diagnostics
        for result in results
        if result.target_diagnostics is not None
    ]
    target_neighbor_edge_parts = [
        result.target_neighbor_edges
        for result in results
        if result.target_neighbor_edges.height
    ]
    target_neighbor_summary_parts = [
        result.target_neighbor_summary_row
        for result in results
        if result.target_neighbor_summary_row.height
    ]

    raw_matched_edges = (
        pl.concat(raw_edge_parts, how="vertical_relaxed")
        if raw_edge_parts
        else pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
    )
    matched_edges = (
        pl.concat(pruned_edge_parts, how="vertical_relaxed")
        if pruned_edge_parts
        else pl.DataFrame(schema=_EMPTY_MATCHED_EDGES_SCHEMA)
    )

    def _combine_truth_eval(parts: list[pl.DataFrame]) -> pl.DataFrame | None:
        if not has_ground_truth:
            return None
        if parts:
            return pl.concat(parts, how="vertical_relaxed")
        return pl.DataFrame(schema=_EMPTY_PAIR_TRUTH_EVAL_SCHEMA)

    raw_pair_truth_eval = _combine_truth_eval(raw_truth_eval_parts)
    pair_truth_eval = _combine_truth_eval(pruned_truth_eval_parts)

    # Unlike pair_truth_eval, pair_truth_eval_detail stays None (not an
    # empty-schema frame) when ground truth is available but no country
    # actually produced a part -- that only happens when every scored
    # country's name/name_cleansed lookup was unavailable (see
    # _score_country), which means this run genuinely cannot answer the
    # per-pair question, not that it answered it with zero rows. Zero
    # *countries* scored is still the legitimate empty-schema case, matching
    # pair_truth_eval's own zero-country behaviour.
    pair_truth_eval_detail = None
    if has_ground_truth and (pair_truth_eval_detail_parts or not countries):
        pair_truth_eval_detail = (
            pl.concat(pair_truth_eval_detail_parts, how="vertical_relaxed")
            if pair_truth_eval_detail_parts
            else pl.DataFrame(schema=_EMPTY_PAIR_TRUTH_EVAL_DETAIL_SCHEMA)
        )

    pruning_summary = (
        pl.DataFrame(pruning_summary_rows, schema=_PRUNING_SUMMARY_SCHEMA)
        if pruning_summary_rows
        else pl.DataFrame(schema=_PRUNING_SUMMARY_SCHEMA)
    )
    exact_match_summary = (
        pl.concat(exact_match_summary_parts, how="vertical_relaxed")
        if exact_match_summary_parts
        else pl.DataFrame(schema=_EXACT_MATCH_SUMMARY_SCHEMA)
    )

    node_id_schema = {"node_id": pl.Utf8, "node_name": pl.Utf8}
    all_source_nodes = (
        pl.concat(source_node_parts, how="vertical_relaxed")
        if source_node_parts
        else pl.DataFrame(schema=node_id_schema)
    ).unique(subset=["node_id"], keep="first")
    all_target_nodes = (
        pl.concat(target_node_parts, how="vertical_relaxed")
        if target_node_parts
        else pl.DataFrame(schema=node_id_schema)
    ).unique(subset=["node_id"], keep="first")
    node_info = pl.concat(
        [
            all_source_nodes.with_columns(pl.lit("source").alias("node_role")),
            all_target_nodes.with_columns(pl.lit("target").alias("node_role")),
        ],
        how="vertical_relaxed",
    ).unique(subset=["node_id"], keep="first")

    # Source-to-target edges only, exactly the clustering this run
    # would have produced before target-neighbour linking existed -- kept
    # beside the (possibly unioned) `clusters`/`cluster_shape` below so a run
    # with the feature on can still show what the union changed.
    components_before_union = build_connected_components(
        matched_edges.select("source_id", "target_id"),
        source_id_col="source_id",
        target_id_col="target_id",
    )
    clusters_before_union = (
        components_before_union.join(node_info, on="node_id", how="left").select(
            list(ARTIFACT_SCHEMAS["clusters"].keys())
        )
        if components_before_union.height
        else pl.DataFrame(schema=ARTIFACT_SCHEMAS["clusters"])
    )
    cluster_shape_before_union = compute_cluster_shape_metrics(
        clusters=clusters_before_union
    ).with_columns(pl.lit("__all__").alias("country"))

    target_neighbor_edges = (
        pl.concat(target_neighbor_edge_parts, how="vertical_relaxed")
        if target_neighbor_edge_parts
        else pl.DataFrame(schema=_TARGET_NEIGHBOR_EDGES_SCHEMA)
    )
    target_neighbor_summary = (
        pl.concat(target_neighbor_summary_parts, how="vertical_relaxed")
        if target_neighbor_summary_parts
        else pl.DataFrame(schema=_TARGET_NEIGHBOR_SUMMARY_SCHEMA)
    )
    # The edge set clustering actually runs over -- source-to-target
    # edges as they are, plus every target-neighbour edge this run's own
    # target index(es) were probed for. Reduces to `components_before_union`
    # exactly when target_neighbor_edges is empty (the feature left off, or
    # a genuinely edge-free target), so `clusters`/`cluster_shape` below stay
    # the run's one clustering result whether or not the feature is on.
    target_neighbor_edges_for_union = (
        target_neighbor_edges.select(
            pl.col("target_id_a").alias("source_id"),
            pl.col("target_id_b").alias("target_id"),
        )
        if target_neighbor_edges.height
        else pl.DataFrame(schema={"source_id": pl.Utf8, "target_id": pl.Utf8})
    )
    components = build_connected_components(
        pl.concat(
            [
                matched_edges.select("source_id", "target_id"),
                target_neighbor_edges_for_union,
            ],
            how="vertical_relaxed",
        ),
        source_id_col="source_id",
        target_id_col="target_id",
    )
    clusters = (
        components.join(node_info, on="node_id", how="left").select(
            list(ARTIFACT_SCHEMAS["clusters"].keys())
        )
        if components.height
        else pl.DataFrame(schema=ARTIFACT_SCHEMAS["clusters"])
    )

    # "__all__" marks these as aggregated across every country in this run,
    # not one specific country -- clustering runs once on the combined
    # matched_edges (see the per-country-safety note above), so there is no
    # single country value to attach. Mirrors the same sentinel
    # `_build_pair_truth_eval_output` uses for its cross-country rollup row.
    cluster_shape = compute_cluster_shape_metrics(clusters=clusters).with_columns(
        pl.lit("__all__").alias("country")
    )
    directional_coverage = compute_directional_coverage(
        source_nodes=all_source_nodes,
        target_nodes=all_target_nodes,
        clusters=clusters,
    ).with_columns(pl.lit("__all__").alias("country"))
    similarity_distribution = _compute_similarity_distribution(
        matched_edges, min_similarity=strategy.min_similarity
    )

    source_diagnostics = None
    target_diagnostics = None
    if emit_diagnostics:
        source_diagnostics = _finalize_diagnostics(
            source_diagnostics_parts,
            id_alias="source_id",
            input_name_col=name_transform.scored_column(strategy.name_source),
            feature_col=feature_col,
            clusters=clusters,
        )
        target_diagnostics = _finalize_diagnostics(
            target_diagnostics_parts,
            id_alias="target_id",
            input_name_col=name_transform.scored_column(RAW_NAME_COLUMN),
            feature_col=feature_col,
            clusters=clusters,
        )

    return BlockingRunResult(
        matched_edges=matched_edges,
        raw_matched_edges=raw_matched_edges,
        pair_truth_eval=pair_truth_eval,
        raw_pair_truth_eval=raw_pair_truth_eval,
        pair_truth_eval_detail=pair_truth_eval_detail,
        pruning_summary=pruning_summary,
        exact_match_summary=exact_match_summary,
        clusters=clusters,
        cluster_shape=cluster_shape,
        directional_coverage=directional_coverage,
        similarity_distribution=similarity_distribution,
        candidate_pair_count=matched_edges.height,
        target_neighbor_edges=target_neighbor_edges,
        clusters_before_union=clusters_before_union,
        cluster_shape_before_union=cluster_shape_before_union,
        target_neighbor_summary=target_neighbor_summary,
        source_diagnostics=source_diagnostics,
        target_diagnostics=target_diagnostics,
        keys=keys,
        source_system=source_system,
        target_system=target_system,
        name_transform={
            **name_transform.describe(),
            # The profile every form was derived under, whichever column
            # was scored: it shapes the name-equality levels of an
            # `identity` run as much as the scored text of a `cleanse` one.
            "cleanse_profile": strategy.cleanse_profile,
            # And the profile the scored text was preprocessed under, which
            # the `preprocessed` level is read from.
            "preprocess_profile": strategy.preprocess_profile,
        },
        timings=timings,
    )


@dataclass(slots=True, frozen=True)
class _RunSetup:
    """What a run resolves from its configuration before it reads a row."""

    clustering_strategy: object
    build_settings: TargetIndexBuildSettings
    effective_text_view: str
    feature_col: str
    countries: list[str]


def _resolve_run_setup(config: BlockingRunConfig) -> _RunSetup:
    strategy = config.strategy
    # An explicit strategy.sbert_model_name always wins; otherwise
    # resolve one from the run's own countries (falling back to the
    # pretrained English default, never failing the run). Harmless to
    # compute for a non-sbert representation -- resolve_target_index_build_settings()
    # only reads sbert_model_name when representation == "sbert".
    sbert_model_name = strategy.sbert_model_name
    if sbert_model_name is None:
        sbert_model_name = resolve_sbert_model_for_jurisdictions(config.countries)
    build_settings = resolve_target_index_build_settings(
        strategy.representation,
        tfidf_ngram_min=strategy.tfidf_ngram_min,
        tfidf_ngram_max=strategy.tfidf_ngram_max,
        tfidf_analyzer=strategy.tfidf_analyzer,
        sbert_model_name=sbert_model_name,
    )
    effective_text_view = resolve_text_view_for_representation(
        representation=strategy.representation,
        requested_text_view=strategy.text_view,
    )
    return _RunSetup(
        clustering_strategy=resolve_clustering_strategy(strategy.representation),
        build_settings=build_settings,
        effective_text_view=effective_text_view,
        feature_col=(
            "cluster_tokens" if effective_text_view == "tokens" else "cluster_text"
        ),
        countries=resolve_blocking_countries(
            source=config.source,
            target=config.target,
            configured_countries=config.countries,
        ),
    )


def resolve_blocking_run_keys(
    config: BlockingRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    telemetry: Telemetry | None = None,
) -> RunKeys:
    """The keys a run over `config` is identified by, without scoring it.

    Reads each country the way `execute_blocking_run` does -- the rows loaded
    and the truth resolved, with no name form derived -- keys it and releases
    it before the next, so a caller can find a finished run of `config`
    (`run_layout.resolve_run_location_for`) before paying for scoring. A run
    that then goes ahead reads every country again.
    """
    validate_blocking_run_config(config)
    telemetry = telemetry if telemetry is not None else Telemetry()

    def _emit(phase: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback({"phase": phase, **payload})

    setup = _resolve_run_setup(config)
    parts: list[tuple[str, str, str | None, str]] = []
    for country in setup.countries:

        def _phase(name: str, *, rows_in: int | None = None, _country: str = country):
            return _timed_phase(
                telemetry, _emit, f"keys:{name}", country=_country, rows_in=rows_in
            )

        prepared = _prepare_country(
            config=config, country=country, phase=_phase, derive=False
        )
        truth = (
            _truth_part_key(
                _resolve_country_truth_map(config, prepared.full_source_frame)
            )
            if _can_evaluate_truth(config, prepared.full_source_frame)
            else None
        )
        parts.append(
            (
                prepared.source_population_key,
                prepared.target_population_key,
                truth,
                _index_part_key(
                    config,
                    target_population_key=prepared.target_population_key,
                    effective_text_view=setup.effective_text_view,
                    build_settings=setup.build_settings,
                ),
            )
        )
        del prepared
    return _run_keys_from_parts(
        settings_key=blocking_settings_key(config),
        countries=setup.countries,
        parts=parts,
    )


def execute_blocking_run(
    config: BlockingRunConfig,
    *,
    source_chunk_size: int = _DEFAULT_SOURCE_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
    telemetry: Telemetry | None = None,
    expected_keys: RunKeys | None = None,
) -> BlockingRunResult:
    """Execute the end-to-end blocking workflow for a two-dataset run.

    `source_chunk_size` bounds how many source rows are scored per
    `ClusteringStrategy.score_source_chunk()` call, letting a large source
    dataset be processed in fixed-size batches against the same prebuilt
    target index (see `ClusteringStrategy`'s docstring for why this is safe
    to do generically). `progress_callback`, when given, receives one
    `{"phase": ..., **payload}` dict per `"country_start"`/`"country_progress"`/
    `"country_complete"` event -- mirrors `validation.runner.run_validation_matrix`'s
    `progress_callback` shape.

    `expected_keys`, the keys `resolve_blocking_run_keys` computed before the
    run, are checked against the keys scoring computes, and a mismatch raises
    rather than filing outputs under keys that no longer describe them.
    """

    validate_blocking_run_config(config)

    def _emit(phase: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback({"phase": phase, **payload})

    telemetry = telemetry if telemetry is not None else Telemetry()

    setup = _resolve_run_setup(config)

    results = [
        _score_country(
            config=config,
            country=country,
            effective_text_view=setup.effective_text_view,
            feature_col=setup.feature_col,
            clustering_strategy=setup.clustering_strategy,
            build_settings=setup.build_settings,
            source_chunk_size=source_chunk_size,
            emit_progress=_emit,
            telemetry=telemetry,
        )
        for country in setup.countries
    ]

    result = _aggregate_country_results(
        strategy=config.strategy,
        emit_diagnostics=config.emit_diagnostics,
        feature_col=setup.feature_col,
        countries=setup.countries,
        results=results,
        has_ground_truth=config.source.has_ground_truth,
        settings_key=blocking_settings_key(config),
        source_system=config.source.system,
        target_system=config.target.system,
        timings=tuple(telemetry.to_dicts()),
    )
    if expected_keys is not None and result.keys != expected_keys:
        raise ValueError(
            "the inputs changed between computing this run's keys and scoring "
            "it -- a layer was rewritten mid-run; run it again."
        )
    return result


def _country_primary_and_variant_frames(
    variant_frame: pl.DataFrame,
    descriptor: BlockingDatasetDescriptor,
    *,
    country: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One country's primary frame, and the variant rows belonging to it.

    Both name-variant runs select a country's variant rows the same way, by
    joining `source_uri` back to the primary frame's `system_uri`; they
    differ only in which frame becomes the scoring target and which columns
    ride along to `_score_country()`. The returned variant frame has zero
    height when no variant row belongs to this country, which both callers
    read as "skip this country".
    """

    primary_frame = load_country_frame(descriptor, country=country)
    primary_ids = primary_frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias("system_uri")
    )
    source_frame_for_country = (
        variant_frame.lazy()
        .select(
            pl.col("system_uri").cast(pl.Utf8).alias("system_uri"),
            pl.col("source_uri").cast(pl.Utf8).alias("source_uri"),
            pl.col("name").cast(pl.Utf8, strict=False).alias("name"),
        )
        .join(
            primary_ids.lazy().rename({"system_uri": "_primary_id"}),
            left_on="source_uri",
            right_on="_primary_id",
            how="inner",
        )
        .collect()
    )
    return primary_frame, source_frame_for_country


def execute_name_variant_recovery_run(
    descriptor: BlockingDatasetDescriptor,
    *,
    variant_frame: pl.DataFrame | None = None,
    strategy: BlockingStrategyConfig,
    roots: WorkspaceRoots,
    countries: tuple[str, ...] | None = None,
    source_chunk_size: int = _DEFAULT_SOURCE_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
    telemetry: Telemetry | None = None,
) -> BlockingRunResult:
    """Score a system's own recorded name variants against that same
    system's primary records, with each row's own back-reference as its
    ground truth rather than a cross-system `matched/` walk.

    This is a labelled run in its own right, not a flag on
    `execute_blocking_run()`: `descriptor` supplies *both* the fixed target
    population (its primary records) and, filtered to `variant_frame`'s own
    rows, the source population -- a source-system's name rows scored
    against a *different* target system's primary records is a related but
    distinct pairing, and reuses the exact same `_score_country` machinery
    under the default `MatchedLayerTruth` (a name-variant row's `system_uri`
    scheme differs from its own system's, which that rule walks via
    `workspace.match_resolution.resolve_cross_system_match`). This entry
    point is the one case that walk cannot answer, since a row's
    relationship to its *own* system's primary record is a local
    back-reference, never a cross-system match, so it scores under
    `ColumnTruth("source_uri")` instead (see `truth.py`).

    `variant_frame` defaults to `loader.load_name_variant_frame(descriptor)`
    when omitted, and must carry a `source_uri` back-reference column (the
    per-row identity `acquisition.name_variant_uri` gives a name-variant
    row) -- a sidecar whose rows still only carry `system_uri` shared
    many-to-one with their primary record cannot be scored as their own
    source records at all, and raises `ValueError` rather than silently
    scoring nothing. Likewise raises when there is no sidecar to load at all
    (`descriptor.system` has no `canonical/<date>/*-names-*.parquet`
    family) -- there is no run to express.

    A country with zero variant rows is skipped rather than raising
    `EmptySourceLoadError`: unlike `execute_blocking_run()`'s source load,
    "no recorded name variants for this country" is a legitimate outcome of
    this pairing, not a sign of a load racing a concurrent write.
    """

    if variant_frame is None:
        variant_frame = load_name_variant_frame(descriptor)
    if variant_frame is None:
        raise ValueError(
            f"No name-variant sidecar found for system {descriptor.system!r} -- "
            "there is nothing to run a name-variant recovery pairing over."
        )
    if "source_uri" not in variant_frame.columns:
        raise ValueError(
            "variant_frame has no 'source_uri' back-reference column -- its "
            "sidecar rows still share 'system_uri' many-to-one with their "
            "primary record and cannot be scored as their own source records."
        )

    def _emit(phase: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback({"phase": phase, **payload})

    telemetry = telemetry if telemetry is not None else Telemetry()

    config = BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=descriptor,
        target=descriptor,
        countries=countries,
        strategy=strategy,
        truth=ColumnTruth("source_uri"),
    )

    clustering_strategy = resolve_clustering_strategy(strategy.representation)
    sbert_model_name = strategy.sbert_model_name
    if sbert_model_name is None:
        sbert_model_name = resolve_sbert_model_for_jurisdictions(countries)
    build_settings = resolve_target_index_build_settings(
        strategy.representation,
        tfidf_ngram_min=strategy.tfidf_ngram_min,
        tfidf_ngram_max=strategy.tfidf_ngram_max,
        tfidf_analyzer=strategy.tfidf_analyzer,
        sbert_model_name=sbert_model_name,
    )
    effective_text_view = resolve_text_view_for_representation(
        representation=strategy.representation,
        requested_text_view=strategy.text_view,
    )
    feature_col = (
        "cluster_tokens" if effective_text_view == "tokens" else "cluster_text"
    )

    candidate_countries = (
        countries if countries is not None else descriptor.available_countries
    )

    scored_countries: list[str] = []
    results: list[_CountryScoreResult] = []
    for country in candidate_countries:
        primary_frame, source_frame_for_country = _country_primary_and_variant_frames(
            variant_frame, descriptor, country=country
        )
        if source_frame_for_country.height == 0:
            continue

        scored_countries.append(country)
        results.append(
            _score_country(
                config=config,
                country=country,
                effective_text_view=effective_text_view,
                feature_col=feature_col,
                clustering_strategy=clustering_strategy,
                build_settings=build_settings,
                source_chunk_size=source_chunk_size,
                emit_progress=_emit,
                telemetry=telemetry,
                # `source_uri` rides along so `config.truth` reads it as each
                # row's ground truth.
                full_source_frame_override=source_frame_for_country.select(
                    "system_uri", "source_uri", "name"
                ),
                target_frame_override=primary_frame,
                source_system_label=f"{descriptor.system}-names",
            )
        )

    return _aggregate_country_results(
        strategy=strategy,
        emit_diagnostics=False,
        feature_col=feature_col,
        countries=scored_countries,
        results=results,
        has_ground_truth=True,
        settings_key=blocking_settings_key(config),
        source_system=f"{descriptor.system}-names",
        target_system=descriptor.system,
        timings=tuple(telemetry.to_dicts()),
    )


def execute_name_variant_cross_system_run(
    descriptor: BlockingDatasetDescriptor,
    target_descriptor: BlockingDatasetDescriptor,
    *,
    variant_frame: pl.DataFrame | None = None,
    strategy: BlockingStrategyConfig,
    roots: WorkspaceRoots,
    countries: tuple[str, ...] | None = None,
    source_chunk_size: int = _DEFAULT_SOURCE_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
    telemetry: Telemetry | None = None,
) -> BlockingRunResult:
    """Score a source system's own recorded name variants against a
    *different* target system's primary records.

    This is the pairing `execute_name_variant_recovery_run()`'s own
    docstring names but does not implement: there, `descriptor` supplies
    both the variant rows and the fixed target population (its own primary
    records), and ground truth is each row's local `source_uri`
    back-reference. Here `descriptor` supplies only the variant rows and
    `target_descriptor` a different system's primary records, so a row's
    ground truth is no longer local -- it is `descriptor.system`'s own
    recorded cross-system match, reached by walking the `name://` row back
    to the primary record that carries it, then reading that record's own
    `match_uri` out of `data/<descriptor.system>/matched/`. That walk is
    `MatchedLayerTruth`'s derived-scheme branch (see `truth.py` and
    `workspace.match_resolution.resolve_cross_system_match`), already
    generic over any `name://` source row -- nothing new is needed to answer
    it, only a source population and a target population handed to
    `_score_country` unchanged from its ordinary path under the default
    rule. Unlike `execute_name_variant_recovery_run()`, this pairing keeps
    that default: the walk is exactly what it needs run, not skipped.

    `descriptor` must resolve to the `matched` layer (`has_ground_truth`
    True): the walk terminates in a lookup into
    `data/<descriptor.system>/matched/`, and `MatchedLayerTruth` evaluates a
    frame only when its source carries ground truth -- raises `ValueError`
    before scoring anything rather than silently reporting no truth for
    every row.

    `variant_frame`, the `source_uri`-column and empty-sidecar checks, and
    per-country zero-rows skip otherwise follow
    `execute_name_variant_recovery_run()`'s own contract: `variant_frame`
    defaults to `loader.load_name_variant_frame(descriptor)`, must carry a
    `source_uri` back-reference column, and each country's source frame is
    `variant_frame`'s rows filtered to `descriptor`'s own primary population
    for that country (never `target_descriptor`'s -- a variant row's
    `source_uri` only ever points back into the system that produced it).
    Countries scored are `resolve_blocking_countries()`'s intersection of
    `descriptor`/`target_descriptor`'s own available countries, filtered by
    `countries` when given.
    """

    if variant_frame is None:
        variant_frame = load_name_variant_frame(descriptor)
    if variant_frame is None:
        raise ValueError(
            f"No name-variant sidecar found for system {descriptor.system!r} -- "
            "there is nothing to run a name-variant cross-system pairing over."
        )
    if "source_uri" not in variant_frame.columns:
        raise ValueError(
            "variant_frame has no 'source_uri' back-reference column -- its "
            "sidecar rows still share 'system_uri' many-to-one with their "
            "primary record and cannot be scored as their own source records."
        )
    if not descriptor.has_ground_truth:
        raise ValueError(
            f"descriptor for system {descriptor.system!r} has no 'matched' "
            "layer (has_ground_truth is False) -- a cross-system pairing's "
            "ground truth is descriptor.system's own recorded match, which "
            "only the 'matched' layer carries; load it with "
            "load_dataset_descriptor(..., require_ground_truth=True)."
        )

    def _emit(phase: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback({"phase": phase, **payload})

    telemetry = telemetry if telemetry is not None else Telemetry()

    config = BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=descriptor,
        target=target_descriptor,
        countries=countries,
        strategy=strategy,
    )

    clustering_strategy = resolve_clustering_strategy(strategy.representation)
    sbert_model_name = strategy.sbert_model_name
    if sbert_model_name is None:
        sbert_model_name = resolve_sbert_model_for_jurisdictions(countries)
    build_settings = resolve_target_index_build_settings(
        strategy.representation,
        tfidf_ngram_min=strategy.tfidf_ngram_min,
        tfidf_ngram_max=strategy.tfidf_ngram_max,
        tfidf_analyzer=strategy.tfidf_analyzer,
        sbert_model_name=sbert_model_name,
    )
    effective_text_view = resolve_text_view_for_representation(
        representation=strategy.representation,
        requested_text_view=strategy.text_view,
    )
    feature_col = (
        "cluster_tokens" if effective_text_view == "tokens" else "cluster_text"
    )

    candidate_countries = resolve_blocking_countries(
        source=descriptor, target=target_descriptor, configured_countries=countries
    )

    scored_countries: list[str] = []
    results: list[_CountryScoreResult] = []
    for country in candidate_countries:
        _primary_frame, source_frame_for_country = _country_primary_and_variant_frames(
            variant_frame, descriptor, country=country
        )
        if source_frame_for_country.height == 0:
            continue

        target_frame_for_country = load_country_frame(
            target_descriptor, country=country
        )

        scored_countries.append(country)
        results.append(
            _score_country(
                config=config,
                country=country,
                effective_text_view=effective_text_view,
                feature_col=feature_col,
                clustering_strategy=clustering_strategy,
                build_settings=build_settings,
                source_chunk_size=source_chunk_size,
                emit_progress=_emit,
                telemetry=telemetry,
                full_source_frame_override=source_frame_for_country.select(
                    "system_uri", "name"
                ),
                target_frame_override=target_frame_for_country,
                source_system_label=f"{descriptor.system}-names",
            )
        )

    return _aggregate_country_results(
        strategy=strategy,
        emit_diagnostics=False,
        feature_col=feature_col,
        countries=scored_countries,
        results=results,
        has_ground_truth=True,
        settings_key=blocking_settings_key(config),
        source_system=f"{descriptor.system}-names",
        target_system=target_descriptor.system,
        timings=tuple(telemetry.to_dicts()),
    )


@dataclass(slots=True, frozen=True)
class RemeasureResult:
    """Result of `remeasure_pair_truth_eval()`.

    Mirrors the subset of `BlockingRunResult` that a re-measure actually
    touches -- `pair_truth_eval`/`raw_pair_truth_eval`/`pair_truth_eval_detail`
    -- plus the paths it wrote. Never `None`: a run with no ground truth (no
    pair-truth evaluation in its directory) has nothing to re-measure and
    `remeasure_pair_truth_eval()` raises rather than returning an empty
    result.
    """

    pair_truth_eval: pl.DataFrame
    raw_pair_truth_eval: pl.DataFrame
    pair_truth_eval_detail: pl.DataFrame | None
    written_paths: tuple[Path, ...]
    remeasured_at: str


def _single_value(
    frame: pl.DataFrame, *, column: str, location: BlockingRunLocation
) -> str:
    values = frame.get_column(column).unique().to_list()
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one distinct {column!r} in "
            f"{location.path(RunArtefact.PAIR_TRUTH_EVAL)}, found {values!r} -- "
            "a single blocking run always scores one source system against "
            "one target system."
        )
    return str(values[0])


def remeasure_pair_truth_eval(
    location: BlockingRunLocation, *, roots: WorkspaceRoots, top_k: int | None = None
) -> RemeasureResult:
    """Re-resolve an existing run's ground truth through the shared URI walk
    (`_build_source_truth_map`) and rewrite `pair_truth_eval.parquet` and
    `raw_pair_truth_eval.parquet` (both always present alongside each other,
    the same `has_ground_truth` gate `execute_blocking_run()` writes them
    under), plus `pair_truth_eval_detail.parquet` whenever this run's
    `name`/`name_cleansed` columns support it -- the same condition
    `_score_country` itself uses (both name-form lookups available),
    checked fresh rather than read off whether a file of that exact name
    already exists. A run directory written before this artefact was named
    `pair_truth_eval_detail` still holds its per-pair detail as
    `residual_pairs.parquet`, an older name for the same population; that
    file is left untouched (stale, not a candidate artefact) while a fresh
    `pair_truth_eval_detail.parquet` is written alongside it.

    Every candidate artefact this run already wrote --
    `matched_edges.parquet`, `raw_matched_edges.parquet`,
    `pruning_summary.parquet`, `exact_match_summary.parquet`,
    `clusters.parquet`, `cluster_shape.parquet`, `directional_coverage.parquet`,
    `similarity_distribution.parquet`, and `source_diagnostics.parquet`/
    `target_diagnostics.parquet` when present -- is read back unchanged from
    `matched_edges.parquet`/`raw_matched_edges.parquet` only (the two the
    truth-eval recompute needs) and never rewritten: candidate generation
    never reads ground truth, so re-scoring it is not this function's job,
    and every other artefact on disk is left byte-identical.

    The run must already hold a pair-truth evaluation -- a run made
    with `--ground-truth false` (`config.source.has_ground_truth ==
    False`) never wrote one, so there is no truth to re-measure and this
    raises `FileNotFoundError`. The run's own `source_system`/
    `target_system`/`country` values are read back from that file rather
    than passed separately, so this is a single self-describing argument
    plus the workspace `roots` its source/target systems' real data layers are
    resolved under.

    `top_k` reproduces the original run's `recall_at_k` figure -- pass the
    same `--top-k` the original `run_blocking.py` invocation used. Omit it
    (the default) to leave `recall_at_k` null on every re-measured row: a
    run directory carries no record of the `top_k` it was scored with, so
    there is nothing to infer it from safely. No F score is stored, so no
    beta needs recovering: the row carries `tp`/`fp`/`fn`, and any F measure
    is one division at read time.
    """

    # Read from the run's own summary rather than a one-row parquet: what this
    # needs is the run's identity and the countries it scored, which is
    # summary-shaped, and a summary is what `summary.json` holds.
    summary_path = location.path(RunArtefact.SUMMARY)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No {summary_path} -- not a finished blocking run directory."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    country_blocks = summary.get("countries")
    if not country_blocks:
        raise ValueError(
            f"{summary_path} records no scored countries -- this run has no "
            "ground truth to re-measure (it was made with "
            "--ground-truth false)."
        )

    raw_matched_edges_path = location.path(RunArtefact.RAW_MATCHED_EDGES)
    matched_edges_path = location.path(RunArtefact.MATCHED_EDGES)
    raw_matched_edges = pl.read_parquet(raw_matched_edges_path)
    matched_edges = pl.read_parquet(matched_edges_path)

    source_system = summary.get("source_system")
    target_system = summary.get("target_system")
    if not source_system or not target_system:
        raise ValueError(
            f"{summary_path} does not name both systems -- cannot re-measure."
        )
    countries = sorted(
        {str(block["country"]) for block in country_blocks if block.get("country")}
    )

    source = load_dataset_descriptor(
        roots=roots, system=source_system, require_ground_truth=True
    )
    target = load_dataset_descriptor(
        roots=roots, system=target_system, require_ground_truth=False
    )

    # config.source.has_ground_truth is exactly the condition under which
    # execute_blocking_run() writes pair_truth_eval.parquet at all -- its
    # existence (checked above) already proves it, so raw_pair_truth_eval is
    # always computed here too (mirroring _combine_truth_eval's own
    # has_ground_truth gate), never conditioned on whether
    # raw_pair_truth_eval.parquet happens to exist on disk yet. Likewise
    # pair_truth_eval_detail is gated on name-form availability alone (the
    # real condition _score_country uses), not on whether a file
    # already named `pair_truth_eval_detail.parquet` exists: a run directory
    # written before the rename holds its per-pair detail as
    # `residual_pairs.parquet` instead, which this function leaves untouched
    # (it is stale, not a candidate artefact) while still writing a fresh
    # `pair_truth_eval_detail.parquet` alongside it.
    raw_truth_eval_parts: list[pl.DataFrame] = []
    pruned_truth_eval_parts: list[pl.DataFrame] = []
    pair_truth_eval_detail_parts: list[pl.DataFrame] = []

    # The name forms are derived live under the profile the run recorded,
    # the same way the run itself derived them, so a re-measure classifies
    # from the forms the run scored rather than whatever the layer carries
    # now. A run written before the profile was recorded derived under the
    # default.
    recorded_transform = summary.get("name_transform") or {}
    cleanse_profile = str(
        recorded_transform.get("cleanse_profile") or DEFAULT_CLEANSE_PROFILE
    )

    # The preprocessed name is made the same way, from the column the run
    # scored and under the profile it recorded, so the `preprocessed` level
    # is read from the text the run's algorithm was handed. A run written
    # before that profile was recorded preprocessed under the default.
    preprocess_profile = str(
        recorded_transform.get("preprocess_profile")
        or DEFAULT_NAME_PREPROCESSING_PROFILE
    )
    scored_col = str(recorded_transform.get("column") or _CLEANSED_NAME_COLUMN)

    def _recorded_forms(frame: pl.DataFrame) -> pl.DataFrame:
        forms = derive_name_forms(frame, profile=cleanse_profile)
        if scored_col not in forms.columns:
            return forms
        return name_preprocessing(
            forms,
            name_col=scored_col,
            profile=preprocess_profile,
            jurisdiction_col=_JURISDICTION_COLUMN,
        )

    truth_parts: dict[str, str] = {}
    for country in countries:
        full_source_frame = _recorded_forms(load_country_frame(source, country=country))
        target_frame = _recorded_forms(load_country_frame(target, country=country))
        source_truth_map = _build_source_truth_map(
            full_source_frame, roots=roots, system=source_system
        )
        # The truth key moves with what this re-measure resolved, while the
        # population and index keys stay untouched: nothing about what was
        # scored or embedded changed, only what it is checked against.
        truth_parts[country] = _truth_part_key(source_truth_map)
        raw_truth_eval, pruned_truth_eval, pair_truth_eval_detail = (
            _evaluate_country_truth(
                full_source_frame=full_source_frame,
                target_frame=target_frame,
                source_truth_map=source_truth_map,
                raw_edges=raw_matched_edges.filter(pl.col("country") == country),
                pruned_edges=matched_edges.filter(pl.col("country") == country),
                source_system=source_system,
                target_system=target_system,
                country=country,
                top_k=top_k,
            )
        )
        raw_truth_eval_parts.append(raw_truth_eval)
        pruned_truth_eval_parts.append(pruned_truth_eval)
        if pair_truth_eval_detail is not None:
            pair_truth_eval_detail_parts.append(pair_truth_eval_detail)

    pair_truth_eval = pl.concat(pruned_truth_eval_parts, how="vertical_relaxed")
    raw_pair_truth_eval = pl.concat(raw_truth_eval_parts, how="vertical_relaxed")
    pair_truth_eval_detail = (
        pl.concat(pair_truth_eval_detail_parts, how="vertical_relaxed")
        if pair_truth_eval_detail_parts
        else None
    )

    remeasured_at = datetime.now(UTC).isoformat()
    written_paths: list[Path] = []

    if pair_truth_eval_detail is not None:
        pair_truth_eval_detail_path = location.path(RunArtefact.PAIR_TRUTH_EVAL_DETAIL)
        pair_truth_eval_detail.write_parquet(pair_truth_eval_detail_path)
        written_paths.append(pair_truth_eval_detail_path)

    # The re-measured evaluation replaces the summary's country blocks in the
    # same nested shape `build_blocking_summary` writes, so a re-measured run
    # reads exactly like a fresh one.
    summary["countries"] = build_country_blocks(pair_truth_eval)
    summary["raw_countries"] = build_country_blocks(raw_pair_truth_eval)
    if isinstance(summary.get("keys"), dict):
        summary["keys"]["truth_key"] = combine_part_keys(truth_parts)
    summary["truth_remeasured"] = {
        "remeasured": True,
        "remeasured_at": remeasured_at,
        "note": (
            "pair_truth_eval/pair_truth_eval_detail re-resolved through "
            "workspace.match_resolution.resolve_cross_system_match "
            "rather than the sidecar's original match_uri column; every "
            "candidate artefact in this run directory is unchanged."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    written_paths.append(summary_path)

    return RemeasureResult(
        pair_truth_eval=pair_truth_eval,
        raw_pair_truth_eval=raw_pair_truth_eval,
        pair_truth_eval_detail=pair_truth_eval_detail,
        written_paths=tuple(written_paths),
        remeasured_at=remeasured_at,
    )


@dataclass(frozen=True)
class RecallCurveReportResult:
    """`compute_recall_curve_for_run()`'s own result: the curve and its
    per-run summary (`validation.recall_curve`), plus the paths
    `reporting.write_recall_curve_report()` wrote them to."""

    curve: pl.DataFrame
    summary: pl.DataFrame
    written_paths: tuple[Path, ...]


def _manifest_setting_value(
    configuration: Mapping[str, object], *, name: str
) -> object | None:
    """One `manifest.json` `"configuration"` entry's resolved `"value"`, read
    from `{"name": {"value": ..., "source": ..., "surface": ...}}`
    (`scripts.cli_common.applicable_settings_record()`'s own shape), or
    `None` when the setting is absent -- a run written before that setting
    existed, or one where it did not apply."""
    entry = configuration.get(name)
    if not isinstance(entry, Mapping):
        return None
    return entry.get("value")


def _manifest_float_setting(
    configuration: Mapping[str, object], *, name: str
) -> float | None:
    value = _manifest_setting_value(configuration, name=name)
    return float(value) if isinstance(value, int | float) else None


def _manifest_int_setting(
    configuration: Mapping[str, object], *, name: str
) -> int | None:
    value = _manifest_setting_value(configuration, name=name)
    return int(value) if isinstance(value, int) else None


def rescore_missed_pairs(
    config: BlockingRunConfig,
    *,
    country: str,
    source_rows: pl.DataFrame,
    backend: str = "sklearn",
    top_k: int | None = None,
) -> pl.DataFrame:
    """Exact-rescore `source_rows` -- a subset of one already-scored run's own
    source population for `country` -- against that run's own cached target
    index, through the same scoring seam `execute_blocking_run()` uses.

    `backend` is forced to an exact backend (`"sklearn"` by default --
    brute-force cosine, always available), with `min_similarity=0.0` and no
    per-source cap, so a missed truth pair's real similarity and rank are
    learned rather than only that the run's own backend did not return it.
    `top_k` is how many targets each source keeps, the whole target
    population when `None`. Every source returns that many edges, so the
    whole target suits a handful of sources and not a run's truth pairs:
    `sources x targets` edges is about 15 billion on `gleif -> ie`. A caller
    that needs only whether a pair clears a cut passes that cut, and a pair
    ranked beyond it comes back absent. `exact_name_filter` and target-neighbour linking are turned
    off, since a caller of this function already knows the pair was not
    resolved by the fast path (see `audit.compute_pair_audit`) and wants the
    scan's own score, not a shortcut edge. Every other setting --
    representation, name transform, cleanse/preprocess profile, text view,
    tokenizer -- is left as `config` has it, since changing any of those
    would score something other than what the run being audited scored.
    None of those changed settings are part of the target index's own key
    (`workflow._index_part_key`), so this resolves the *same* cached target
    index the run built (`validation.target_index_cache.resolve_target_index`)
    rather than rebuilding it.

    Returns the rescore's own raw edges (`source_id`, `target_id`,
    `similarity`, `rank`, `country`), unpruned and carrying no ground-truth
    columns -- a caller reads off the specific pairs it asked about.
    """
    target_frame = load_country_frame(config.target, country=country)
    if top_k is None:
        top_k = max(target_frame.height, 1)
    rescore_strategy = replace(
        config.strategy,
        similarity_backend=backend,
        backend_options=None,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
        exact_name_filter=False,
        target_neighbor_min_similarity=None,
        target_neighbor_max_per_target=None,
    )
    rescore_config = replace(config, strategy=rescore_strategy)
    setup = _resolve_run_setup(rescore_config)
    result = _score_country(
        config=rescore_config,
        country=country,
        effective_text_view=setup.effective_text_view,
        feature_col=setup.feature_col,
        clustering_strategy=setup.clustering_strategy,
        build_settings=setup.build_settings,
        source_chunk_size=max(source_rows.height, 1),
        emit_progress=lambda *_args, **_kwargs: None,
        telemetry=Telemetry(),
        full_source_frame_override=source_rows,
        target_frame_override=target_frame,
    )
    return result.raw_edges


def compute_recall_curve_for_run(
    location: BlockingRunLocation,
) -> RecallCurveReportResult:
    """A completed run's recall-against-comparisons-spent curve and its
    normalised area per name-equality level, read from artefacts already on
    disk without re-running candidate generation.

    `matched_edges.parquet` and `pair_truth_eval_detail.parquet` are the
    inputs `validation.recall_curve.compute_recall_curve()` reads; both must
    already be on disk, the same pair `remeasure_pair_truth_eval()` may
    (re)write. `min_similarity`, `top_k`, `max_candidates_per_source` and
    `similarity_backend` are read from `manifest.json`'s own resolved
    configuration (every run directory carries one, `run_blocking.py`'s own
    `write_run_manifest()` call) and stated beside the curve, since each of
    the first three truncates the curve where that run's own candidates
    stop.
    `target_rows` is read from `summary.json`'s scored-country blocks when
    present, the same figure `comparison.build_strategy_comparison()` reads
    as its own scale-assumption column.

    Writes `recall_curve.parquet` and `recall_curve_summary.parquet`
    alongside the run's other artefacts (`reporting.write_recall_curve_report()`)
    and returns both frames plus the paths written.

    Raises `FileNotFoundError` when the run is missing
    `matched_edges.parquet`, `pair_truth_eval_detail.parquet` or
    `manifest.json`: a run without a name-equality classification or a
    recorded configuration is not an input here, it is re-run.
    """
    matched_edges_path = location.path(RunArtefact.MATCHED_EDGES)
    detail_path = location.path(RunArtefact.PAIR_TRUTH_EVAL_DETAIL)
    manifest_path = location.path(RunArtefact.MANIFEST)
    for path in (matched_edges_path, detail_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- not a completed run directory with a "
                "name-equality classification and a recorded configuration."
            )

    matched_edges = pl.read_parquet(matched_edges_path)
    pair_truth_eval_detail = pl.read_parquet(detail_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = manifest.get("configuration") or {}

    similarity_backend = _manifest_setting_value(
        configuration, name="similarity_backend"
    )

    # pair_truth_eval never reaches disk as its own parquet (`reporting.
    # build_blocking_summary()`'s own comment: it is a section of
    # `summary.json` instead) -- target_rows is read back from each scored
    # country's `whole_population` block there, the same figure
    # `comparison.build_strategy_comparison()` reads off the in-memory frame.
    target_rows: int | None = None
    summary_path = location.path(RunArtefact.SUMMARY)
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target_rows_values = {
            block["whole_population"]["target_rows"]
            for block in (run_summary.get("countries") or [])
            if block.get("whole_population") is not None
            and block["whole_population"].get("target_rows") is not None
        }
        if len(target_rows_values) == 1:
            target_rows = next(iter(target_rows_values))

    curve = compute_recall_curve(
        matched_edges=matched_edges, pair_truth_eval_detail=pair_truth_eval_detail
    )
    recall_curve_summary = summarize_recall_curve(
        curve,
        similarity_backend=str(similarity_backend) if similarity_backend else "",
        is_exact_backend=(
            _is_exact_backend(str(similarity_backend)) if similarity_backend else None
        ),
        target_rows=target_rows,
        min_similarity=_manifest_float_setting(configuration, name="min_similarity"),
        top_k=_manifest_int_setting(configuration, name="top_k"),
        max_candidates_per_source=_manifest_int_setting(
            configuration, name="max_candidates_per_source"
        ),
    )
    written = write_recall_curve_report(
        location, curve=curve, summary=recall_curve_summary
    )
    return RecallCurveReportResult(
        curve=curve, summary=recall_curve_summary, written_paths=tuple(written)
    )
