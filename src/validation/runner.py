"""Run the directional source-to-target validation matrix and score what it produced.

`run_validation_matrix` runs each source against each target, country by country. The scoring each function documents:

- `compute_pair_truth_eval`: `tp`, `fp`, `fn`, precision and recall per population per country, with candidate-volume and ranking columns; `pair_truth_eval_row` selects a population by name.
- `compute_pair_truth_eval_detail`: each truth pair judged by its own two names.
- `compute_run_metrics`: latency per 1,000 source rows and candidate duplication.
- `compute_robustness_eval` and `rollup_robustness_eval`: recall retained under perturbation against the unperturbed baseline.

A run's level classification follows its own cleanse profile, so two runs under different profiles hold different sources in `never`. Whether a change there is the representation or the residual is answered by `blocking.comparison.diff_name_equality_levels`, not by these counts.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, TypedDict

import polars as pl
from company_tokenize import (
    to_portable_path_str,
    tokenize_name_dataframe,
    tokenizer_directory_files,
    tokenizer_id,
)
from company_tokenize.name_preprocessing import NAME_PREPROCESSED_COLUMN
from company_vectorize import (
    resolve_clustering_strategy,
    resolve_target_index_build_settings,
)
from company_vectorize.clustering_contract import (
    TargetClusteringIndex,
    TfidfTargetIndexBuildSettings,
)
from company_vectorize.clustering_metrics import (
    compute_cluster_shape_metrics,
    compute_directional_coverage,
)
from company_vectorize.graph_cluster import build_connected_components
from company_vectorize.sparse_similarity import build_target_nearest_neighbors
from company_vectorize.tfidf_cluster import (
    build_clustering_text_view,
)

from workspace.artifact_archive import MANIFEST_FILENAME as _ARTIFACT_MANIFEST_FILENAME
from workspace.artifact_archive import (
    artifact_is_complete,
    begin_artifact_write,
    commit_artifact_write,
    compute_artifact_signature,
    read_artifact_manifest,
    resolve_candidate_dir,
)
from workspace.artifact_layout import artifact_store_root
from workspace.identity import rows_digest
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots
from workspace.telemetry import Telemetry
from workspace.tokenizer_store import promoted_tokenizer_directory, scope_system

from .config import (
    ValidationRunConfig,
    resolve_text_view_for_representation,
)
from .contracts import ARTIFACT_SCHEMAS as _ARTIFACT_SCHEMAS
from .contracts import NAME_EQUALITY_POPULATIONS as _NAME_EQUALITY_POPULATIONS
from .contracts import POPULATION_UNIVERSE as _POPULATION_UNIVERSE
from .prepared_dataset import (
    PreparedSystemStats,
    list_prepared_countries,
    load_prepared_country_frame,
    prepare_system_dataset,
)
from .target_index_cache import (
    build_target_index_cache_settings,
    resolve_target_index,
)

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8())
# interchangeably in schema dicts; pl.DataType alone only covers the latter.
PolarsDType = type[pl.DataType] | pl.DataType

ProgressCallback = Callable[[dict[str, object]], None]
SOURCE_PROGRESS_CHUNK_SIZE = 10_000
_TOKEN_CACHE_STORE_FACET = "validation-token-cache"
_VALIDATION_TOKEN_COL = "validation_tokens"


def _empty_frame(*, schema: dict[str, PolarsDType]) -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in schema}, schema=schema)


def _country_partition_dir(*, system_dir: Path, country: str) -> Path:
    return layer_partition_dir(system_dir, value=country)


def _prepared_country_fingerprint(*, system_dir: Path, country: str) -> str:
    digest = hashlib.sha256()
    metadata_path = system_dir / "_prepared_metadata.json"
    if metadata_path.exists():
        digest.update(metadata_path.read_bytes())

    partition_dir = _country_partition_dir(system_dir=system_dir, country=country)
    digest.update(country.encode("utf-8"))
    if partition_dir.exists():
        for path in sorted(partition_dir.glob("*.parquet")):
            stat = path.stat()
            digest.update(path.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()[:16]


def _token_cache_dir(
    *, roots: WorkspaceRoots, system: str, country: str, manifest: dict[str, object]
) -> Path:
    """Relocated from `data/<system>/_token_cache/` onto the same
    append-only store protocol the target-index cache uses -- one candidate
    directory per manifest, under `artifacts/store`."""
    return resolve_candidate_dir(
        artifact_store_root(roots),
        _TOKEN_CACHE_STORE_FACET,
        system,
        country,
        settings=manifest,
    )


def _token_cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "tokenized.parquet", cache_dir / _ARTIFACT_MANIFEST_FILENAME


def _tokenizer_artifact_path(*, config: ValidationRunConfig, system: str) -> Path:
    """The tokenizer model this run scores names with.

    The candidate `config/tokenizers.json` names for `config.tokenizer_profile`
    by default. When `config.tokenizer_path` is set, that model file instead
    -- the opt-in path that lets one workload be run twice against two
    tokenizer candidates. `system` is the run's target system for both sides:
    source tokens split by the source system's own vocabulary are not
    comparable with the target's.
    """
    if config.tokenizer_path is not None:
        supplied = Path(config.tokenizer_path)
        if not supplied.is_file():
            raise FileNotFoundError(f"tokenizer_path names no file: {supplied}")
        return supplied

    directory = promoted_tokenizer_directory(
        config.roots,
        system=scope_system(config.tokenizer_scope, system),
        tokenizer_id=tokenizer_id(config.tokenizer),
        profile=config.tokenizer_profile,
    )
    return tokenizer_directory_files(directory, trainer=config.tokenizer).model


def _tokenizer_artifact_fingerprint(*, tokenizer_path: Path) -> str:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {tokenizer_path}")

    stat = tokenizer_path.stat()
    digest = hashlib.sha256()
    digest.update(tokenizer_path.name.encode("utf-8"))
    digest.update(str(stat.st_size).encode("utf-8"))
    digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()[:16]


def _token_cache_manifest(
    *,
    system: str,
    country: str,
    prepared_fingerprint: str,
    tokenizer_path: Path,
    tokenizer_fingerprint: str,
    tokenizer: str,
    tokenizer_scope: str,
    tokenizer_profile: str,
    noise_words_profile: str,
    noise_words_set_kind: str,
    roots: WorkspaceRoots | None = None,
) -> dict[str, object]:
    return {
        "system": system,
        "country": country,
        "prepared_fingerprint": prepared_fingerprint,
        "tokenizer_path": to_portable_path_str(
            tokenizer_path, project_root=roots.checkout if roots is not None else None
        ),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "tokenizer": tokenizer,
        "tokenizer_scope": tokenizer_scope,
        "tokenizer_profile": tokenizer_profile,
        "noise_words_profile": noise_words_profile,
        "noise_words_set_kind": noise_words_set_kind,
    }


def _load_cached_token_frame(
    *, cache_dir: Path, expected_manifest: dict[str, object]
) -> tuple[pl.DataFrame | None, bool]:
    """Reads through `artifact_archive`'s candidate protocol -- complete
    only once its manifest carries `"complete": True` -- rather than a
    hand-rolled tmp-then-rename pair."""
    if not artifact_is_complete(cache_dir):
        return None, False

    stored_manifest = read_artifact_manifest(cache_dir)
    if stored_manifest is None:
        return None, False
    stored_manifest = {
        key: value for key, value in stored_manifest.items() if key != "complete"
    }
    if stored_manifest != expected_manifest:
        return None, False

    token_path, _manifest_path = _token_cache_paths(cache_dir)
    try:
        frame = pl.read_parquet(token_path)
    except (OSError, pl.exceptions.PolarsError):
        return None, False

    return frame, True


def _write_token_cache(
    *, cache_dir: Path, manifest: dict[str, object], token_frame: pl.DataFrame
) -> dict[str, Path]:
    token_path, manifest_path = _token_cache_paths(cache_dir)
    temporary_dir = begin_artifact_write(cache_dir)
    token_frame.write_parquet(temporary_dir / token_path.name)
    commit_artifact_write(cache_dir, temporary_dir, manifest=manifest)
    return {"manifest": manifest_path, "tokenized": token_path}


def _build_validation_text_frame(
    *,
    config: ValidationRunConfig,
    frame: pl.DataFrame,
    stats: PreparedSystemStats,
    system: str,
    target_system: str,
    country: str,
) -> tuple[pl.DataFrame, str | None, bool]:
    """`frame` tokenized, through a cache keyed on `system`'s own rows and the
    tokenizer read, which is `target_system`'s whichever side `frame` is."""
    effective_text_view = resolve_text_view_for_representation(
        representation=str(config.representation),
        requested_text_view=config.text_view,
    )
    if effective_text_view == "name":
        return frame, None, False

    tokenizer_path = _tokenizer_artifact_path(config=config, system=target_system)
    tokenizer_fingerprint = _tokenizer_artifact_fingerprint(
        tokenizer_path=tokenizer_path
    )
    prepared_fingerprint = _prepared_country_fingerprint(
        system_dir=stats.system_dir, country=country
    )

    manifest = _token_cache_manifest(
        system=system,
        country=country,
        prepared_fingerprint=prepared_fingerprint,
        tokenizer_path=tokenizer_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        tokenizer=config.tokenizer,
        tokenizer_scope=config.tokenizer_scope,
        tokenizer_profile=config.tokenizer_profile,
        noise_words_profile=config.noise_words_profile,
        noise_words_set_kind=config.noise_words_set_kind,
        roots=config.roots,
    )
    cache_key = compute_artifact_signature(manifest)

    cache_dir = _token_cache_dir(
        roots=config.roots, system=system, country=country, manifest=manifest
    )
    token_frame, cache_hit = _load_cached_token_frame(
        cache_dir=cache_dir, expected_manifest=manifest
    )
    if token_frame is None or not cache_hit:
        token_frame = tokenize_name_dataframe(
            frame,
            tokenizer_path=tokenizer_path,
            trainer=config.tokenizer,
            name_col="name",
            token_col=_VALIDATION_TOKEN_COL,
            noise_words_profile=config.noise_words_profile,
            noise_words_set_kind=config.noise_words_set_kind,
        )
        _write_token_cache(
            cache_dir=cache_dir, manifest=manifest, token_frame=token_frame
        )

    return token_frame, cache_key, cache_hit


def _build_validation_text_frame_with_progress(
    *,
    role: str,
    source_system: str,
    target_system: str,
    country: str,
    effective_text_view: str,
    config: ValidationRunConfig,
    frame: pl.DataFrame,
    stats: PreparedSystemStats,
    system: str,
    emit_progress: Callable[..., None],
) -> tuple[pl.DataFrame, str | None, bool]:
    if effective_text_view == "tokens":
        emit_progress(
            f"{role}_token_cache_check",
            source_system=source_system,
            target_system=target_system,
            country=country,
        )

    started = time.perf_counter()
    token_frame, token_cache_key, token_cache_hit = _build_validation_text_frame(
        config=config,
        frame=frame,
        stats=stats,
        system=system,
        target_system=target_system,
        country=country,
    )

    if effective_text_view == "tokens":
        token_cache_dir = (
            artifact_store_root(config.roots)
            / _TOKEN_CACHE_STORE_FACET
            / system
            / country
            / token_cache_key
            if token_cache_key is not None
            else None
        )
        emit_progress(
            f"{role}_token_cache_hit"
            if token_cache_hit
            else f"{role}_token_cache_miss",
            source_system=source_system,
            target_system=target_system,
            country=country,
            **{
                f"{role}_token_cache_dir": str(token_cache_dir)
                if token_cache_dir
                else None,
                f"{role}_token_cache_key": token_cache_key,
                f"{role}_token_rows": int(token_frame.height),
                "build_elapsed_seconds": time.perf_counter() - started,
            },
        )

    return token_frame, token_cache_key, token_cache_hit


def _write_incremental_artifact_part(
    *,
    incremental_output_dir: Path | None,
    artifact_name: str,
    part_index: int,
    frame: pl.DataFrame,
) -> None:
    if incremental_output_dir is None or frame.height == 0:
        return
    artifact_dir = incremental_output_dir / "_incremental" / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    part_path = artifact_dir / f"part-{part_index:06d}.parquet"
    frame.write_parquet(part_path)


class _CountryValidationContext(TypedDict):
    source_slice: pl.DataFrame
    target_slice: pl.DataFrame
    source_rows: int
    target_rows: int
    source_text: pl.DataFrame
    source_feature_col: str
    target_index: TargetClusteringIndex
    # Backend/index handles are inherently polymorphic across clustering
    # strategies (sklearn NearestNeighbors, duckdb, pgvector, ...); the
    # consumers below treat them as opaque too, so Any is honest here.
    target_nn_index: Any
    target_backend_index: Any


def _prepare_country_validation_context(
    *,
    config: ValidationRunConfig,
    source_system: str,
    target_system: str,
    country: str,
    source_stats: PreparedSystemStats,
    target_stats: PreparedSystemStats,
    runtime_max_rows: int | None,
    effective_text_view: str,
    target_index_build_settings,
    clustering_strategy,
    completed_units: int,
    total_units: int,
    emit_progress: Callable[..., None],
) -> _CountryValidationContext:
    source_slice = load_prepared_country_frame(stats=source_stats, country=country)
    target_slice = load_prepared_country_frame(stats=target_stats, country=country)

    if runtime_max_rows is not None:
        source_slice = source_slice.head(int(runtime_max_rows))

    source_rows = int(source_slice.height)
    target_rows = int(target_slice.height)

    source_token_frame, _source_token_cache_key, _source_token_cache_hit = (
        _build_validation_text_frame_with_progress(
            role="source",
            source_system=source_system,
            target_system=target_system,
            country=country,
            effective_text_view=effective_text_view,
            config=config,
            frame=source_slice,
            stats=source_stats,
            system=source_system,
            emit_progress=emit_progress,
        )
    )

    target_token_frame, target_token_cache_key, _target_token_cache_hit = (
        _build_validation_text_frame_with_progress(
            role="target",
            source_system=source_system,
            target_system=target_system,
            country=country,
            effective_text_view=effective_text_view,
            config=config,
            frame=target_slice,
            stats=target_stats,
            system=target_system,
            emit_progress=emit_progress,
        )
    )

    # Keyed on the population `target_slice` actually holds right
    # now, before tokenization/text-view/noise-removal touch it -- the same
    # digest `blocking.workflow`'s population keys use (`workspace.identity`,
    # not reimplemented). Unlike the file-mtime
    # fingerprint this replaces, a cleanse-layer re-run that leaves "name"
    # untouched leaves this key -- and therefore the cache below -- untouched
    # too.
    target_population_key = rows_digest(
        target_slice, id_col="system_uri", value_col="name"
    )
    cache_settings = build_target_index_cache_settings(
        population_key=target_population_key,
        representation=str(config.representation),
        text_view=effective_text_view,
        tfidf_ngram_min=(
            int(target_index_build_settings.ngram_min)
            if isinstance(target_index_build_settings, TfidfTargetIndexBuildSettings)
            else None
        ),
        tfidf_ngram_max=(
            int(target_index_build_settings.ngram_max)
            if isinstance(target_index_build_settings, TfidfTargetIndexBuildSettings)
            else None
        ),
        tfidf_analyzer=(
            str(target_index_build_settings.analyzer)
            if isinstance(target_index_build_settings, TfidfTargetIndexBuildSettings)
            else None
        ),
        similarity_backend=str(config.similarity_backend),
        backend_options=(config.backend_options or {}),
        token_cache_key=target_token_cache_key,
        tokenizer=str(config.tokenizer),
        tokenizer_scope=str(config.tokenizer_scope),
        strategy=clustering_strategy.__class__.__name__,
    )

    emit_progress(
        "pair_start",
        source_system=source_system,
        target_system=target_system,
        country=country,
        source_rows=source_rows,
        target_rows=target_rows,
    )

    target_token_col = _resolve_token_col(target_token_frame)
    target_text = build_clustering_text_view(
        target_token_frame,
        name_col="name",
        token_col=target_token_col,
        text_view=effective_text_view,
        include_cluster_text=(str(config.representation) == "tfidf"),
    )
    target_feature_col = (
        "cluster_tokens"
        if config.representation in {"wordpiece", "sentencepiece"}
        else "cluster_text"
    )

    emit_progress(
        "target_cache_check",
        source_system=source_system,
        target_system=target_system,
        country=country,
    )

    target_build_started = time.perf_counter()
    target_index, cache_hit, cache_dir = resolve_target_index(
        roots=config.roots,
        target_system=target_system,
        country=country,
        settings=cache_settings,
        clustering_strategy=clustering_strategy,
        build_settings=target_index_build_settings,
        target_frame=target_text,
        target_id_col="system_uri",
        text_col=target_feature_col,
    )
    if cache_hit:
        emit_progress(
            "target_cache_hit",
            source_system=source_system,
            target_system=target_system,
            country=country,
            target_cache_dir=str(cache_dir),
            target_rows=len(target_index.target_ids),
        )
    else:
        emit_progress(
            "target_cache_miss",
            source_system=source_system,
            target_system=target_system,
            country=country,
            target_cache_dir=str(cache_dir),
            target_rows=len(target_index.target_ids),
            build_elapsed_seconds=time.perf_counter() - target_build_started,
        )

    emit_progress(
        "country_start",
        source_system=source_system,
        target_system=target_system,
        country=country,
        source_rows=source_rows,
        target_rows=target_rows,
        completed_units=completed_units,
        total_units=total_units,
    )

    token_col = _resolve_token_col(source_token_frame)
    source_text = build_clustering_text_view(
        source_token_frame,
        name_col="name",
        token_col=token_col,
        text_view=effective_text_view,
        include_cluster_text=(str(config.representation) == "tfidf"),
    )
    source_feature_col = (
        "cluster_tokens"
        if config.representation in {"wordpiece", "sentencepiece"}
        else "cluster_text"
    )
    target_nn_index = build_target_nearest_neighbors(
        target_index,
        top_k=config.top_k,
        max_candidates_per_source=config.max_candidates_per_source,
    )
    target_backend_index = clustering_strategy.build_backend_index(
        target_index,
        backend=str(config.similarity_backend),
        top_k=config.top_k,
        max_candidates_per_source=config.max_candidates_per_source,
        backend_options=(config.backend_options or {}),
        min_similarity=config.min_similarity,
    )

    emit_progress(
        "target_nn_ready",
        source_system=source_system,
        target_system=target_system,
        country=country,
        target_rows=len(target_index.target_ids),
    )

    return {
        "source_slice": source_slice,
        "target_slice": target_slice,
        "source_rows": source_rows,
        "target_rows": target_rows,
        "source_text": source_text,
        "source_feature_col": source_feature_col,
        "target_index": target_index,
        "target_nn_index": target_nn_index,
        "target_backend_index": target_backend_index,
    }


def _emit_pair_progress_event(
    *,
    emit_progress: Callable[..., None],
    started_at: float,
    source_system: str,
    target_system: str,
    countries: list[str],
    pair_source_rows_processed: int,
    pair_target_rows: int,
    pair_source_rows_with_match: int,
    pair_edges_found: int,
    cumulative_source_rows_processed: int,
    cumulative_source_rows_with_match: int,
    cumulative_edges_found: int,
    cumulative_target_rows: int,
    completed_units: int,
    total_units: int,
) -> None:
    elapsed_seconds = time.perf_counter() - started_at
    eta_seconds = None
    if completed_units > 0 and completed_units < total_units:
        avg_per_unit = elapsed_seconds / float(completed_units)
        eta_seconds = max(0.0, avg_per_unit * float(total_units - completed_units))

    emit_progress(
        "pair",
        source_system=source_system,
        target_system=target_system,
        countries_processed=len(countries),
        source_rows_processed=pair_source_rows_processed,
        target_rows=pair_target_rows,
        source_rows_with_match=pair_source_rows_with_match,
        edges_found=pair_edges_found,
        source_rows_processed_cumulative=cumulative_source_rows_processed,
        source_rows_with_match_cumulative=cumulative_source_rows_with_match,
        edges_found_cumulative=cumulative_edges_found,
        target_rows_cumulative=cumulative_target_rows,
        completed_units=completed_units,
        total_units=total_units,
        elapsed_seconds=elapsed_seconds,
        eta_seconds=eta_seconds,
    )


def _build_pair_truth_eval_output(
    *,
    pair_truth_eval_parts: list[pl.DataFrame],
    schemas: dict[str, dict[str, PolarsDType]],
) -> pl.DataFrame:
    pair_truth_eval_country = (
        pl.concat(pair_truth_eval_parts, how="vertical_relaxed")
        if pair_truth_eval_parts
        else _empty_frame(schema=schemas["pair_truth_eval"])
    )

    if pair_truth_eval_country.height == 0:
        return pair_truth_eval_country

    # Rolled up per population, never across them: summing the universe row
    # with its own level rows would count every source several times over.
    pair_truth_eval_global = (
        pair_truth_eval_country.group_by(
            ["source_system", "target_system", "population"]
        )
        .agg(
            pl.col("labelled_sources").sum().alias("labelled_sources"),
            pl.col("target_rows").sum().alias("target_rows"),
            pl.col("pair_universe").sum().alias("pair_universe"),
            pl.col("truth_pairs").sum().alias("truth_pairs"),
            pl.col("predicted_pairs").sum().alias("predicted_pairs"),
            pl.col("tp").sum().alias("tp"),
            pl.col("fp").sum().alias("fp"),
            pl.col("fn").sum().alias("fn"),
            pl.col("candidate_pair_count").sum().alias("candidate_pair_count"),
        )
        .with_columns(
            pl.lit("__all__").alias("country"),
            # This call site's compute_pair_truth_eval() never receives
            # source_rows/top_k, so candidate_set_size_ratio/recall_at_k are
            # already null on every universe row rolled up here; source_rows
            # and the two derived ratios stay null in the "__all__" rows
            # rather than being (mis)computed from values never supplied.
            pl.lit(None, dtype=pl.Int64).alias("source_rows"),
            pl.lit(None, dtype=pl.Float64).alias("candidate_set_size_ratio"),
            pl.lit(None, dtype=pl.Float64).alias("recall_at_k"),
            pl.when((pl.col("tp") + pl.col("fp")) > 0)
            .then(pl.col("tp") / (pl.col("tp") + pl.col("fp")))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("precision"),
            pl.when((pl.col("tp") + pl.col("fn")) > 0)
            .then(pl.col("tp") / (pl.col("tp") + pl.col("fn")))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("recall"),
            pl.when(pl.col("pair_universe") > 0)
            .then(1.0 - (pl.col("predicted_pairs") / pl.col("pair_universe")))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("reduction_ratio"),
        )
        .select(list(schemas["pair_truth_eval"].keys()))
    )

    return pl.concat(
        [pair_truth_eval_country, pair_truth_eval_global], how="vertical_relaxed"
    )


def _resolve_validation_countries(
    *,
    source_stats: PreparedSystemStats,
    target_stats: PreparedSystemStats,
    configured_countries: tuple[str, ...] | None,
) -> list[str]:
    source_countries = list_prepared_countries(source_stats)
    target_countries = list_prepared_countries(target_stats)
    countries = sorted(source_countries & target_countries)
    if configured_countries:
        allowed_countries = set(configured_countries)
        countries = [country for country in countries if country in allowed_countries]
    return countries


def _resolve_token_col(frame: pl.DataFrame) -> str | None:
    for candidate in (
        _VALIDATION_TOKEN_COL,
        "country_tokens",
        "global_tokens",
        "name_tokens",
    ):
        if candidate in frame.columns:
            return candidate
    return None


def _derive_system_expr(node_id_expr: pl.Expr) -> pl.Expr:
    text = node_id_expr.cast(pl.Utf8, strict=False).fill_null("")
    return (
        pl.when(text.str.contains("://"))
        .then(text.str.split_exact("://", 1).struct.field("field_0"))
        .when(text.str.contains(":"))
        .then(text.str.split_exact(":", 1).struct.field("field_0"))
        .otherwise(pl.lit(""))
        .alias("node_system")
    )


def _result_schemas() -> dict[str, dict[str, PolarsDType]]:
    return {
        "source_outcomes": {
            "source_id": pl.Utf8,
            "source_name": pl.Utf8,
            "source_match_uri": pl.Utf8,
            "target_id": pl.Utf8,
            "target_name": pl.Utf8,
            "similarity": pl.Float64,
            "rank": pl.Int32,
            "is_true_match": pl.Boolean,
            "true_match_status": pl.Utf8,
            "match_status": pl.Utf8,
            "reason_code": pl.Utf8,
        },
        "clusters": {
            "cluster_id": pl.Utf8,
            "node_id": pl.Utf8,
            "node_name": pl.Utf8,
            "node_role": pl.Utf8,
        },
        "directional_coverage": {
            "country": pl.Utf8,
            "source_records_total": pl.Int64,
            "source_records_filtered_out": pl.Int64,
            "source_records": pl.Int64,
            "source_records_with_cluster": pl.Int64,
            "source_records_clustered_with_target": pl.Int64,
            "directional_coverage_ratio": pl.Float64,
        },
        "cluster_shape": {
            "country": pl.Utf8,
            "total_clusters": pl.Int64,
            "singleton_clusters": pl.Int64,
            "singleton_ratio": pl.Float64,
            "mean_cluster_size": pl.Float64,
            "p95_cluster_size": pl.Float64,
            "max_cluster_size": pl.Int64,
        },
        # The contract is the one definition; restating it here is how the
        # empty-frame fallback and the artefact check drifted apart before.
        "pair_truth_eval": dict(_ARTIFACT_SCHEMAS["pair_truth_eval"]),
        "exceptions": {
            "run_id": pl.Utf8,
            "source_system": pl.Utf8,
            "target_system": pl.Utf8,
            "country": pl.Utf8,
            "rule_id": pl.Utf8,
            "node_id": pl.Utf8,
            "included_in_exception": pl.Boolean,
            "reason": pl.Utf8,
        },
        "run_metrics": {
            "source_system": pl.Utf8,
            "target_system": pl.Utf8,
            "country": pl.Utf8,
            "source_rows": pl.Int64,
            "elapsed_seconds": pl.Float64,
            "latency_seconds_per_1k_rows": pl.Float64,
            "candidate_edges": pl.Int64,
            "candidate_distinct_targets": pl.Int64,
            "candidate_duplication_ratio": pl.Float64,
        },
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


# `name_equality` answers one question about a truth pair -- at which name
# form do the two sides become the same string? -- and its value is that form,
# or `never`. The column says what is measured and the value says where it
# became true, which the old `raw_identical`/`cleanse_absorbed`/
# `cleansed_different` names did not: two of them described the pair, two
# described what happened to the difference, and the last two differed by a
# letter of tense.
NAME_EQUALITY_RAW = "raw"
NAME_EQUALITY_BASIC = "basic"
NAME_EQUALITY_CLEANSED = "cleansed"
NAME_EQUALITY_PREPROCESSED = "preprocessed"
NAME_EQUALITY_NEVER = "never"
NAME_EQUALITY_UNKNOWN = "unknown"
# The name forms that carry a matrix of their own, in cascade order: each
# holds the pairs the form before it did not make equal. `unknown` is excluded
# deliberately -- a pair with no verdict has nothing to report a matrix
# against, and stays a count.
#
# `basic` splits what `cleansed` used to hold as one lump, so the report says
# how much of the gap basic cleansing closes and how much needs the full
# cleanse. A run whose inputs carry no `name_cleansed_basic` column reports it
# as zero and those pairs stay under `cleansed`, as before the split.
#
# `preprocessed` holds the pairs whose names differ as cleansed and are equal
# as the run's `name_preprocessed`, the text its algorithm was handed: `Acme
# Ltd` and `Acme DAC` reach the scan as one string, so finding them is not the
# algorithm's work and they are kept out of `never`, the residual only the
# algorithm can find. It is a measurement and never an exact join, since
# joining on it would make `Acme Ltd` and `Acme Plc` one company. A run whose
# inputs carry no `name_preprocessed` column reports it as zero and those
# pairs stay under `never`.
NAME_EQUALITY_LEVELS = (
    NAME_EQUALITY_RAW,
    NAME_EQUALITY_BASIC,
    NAME_EQUALITY_CLEANSED,
    NAME_EQUALITY_PREPROCESSED,
    NAME_EQUALITY_NEVER,
)
_NAME_EQUALITY_VALUES = (*NAME_EQUALITY_LEVELS, NAME_EQUALITY_UNKNOWN)


def _present(column: str) -> pl.Expr:
    """A name value counts as present only when it is non-null and not
    blank: two rows that merely both lack a name must never compare equal
    (mirrors `blocking._split_exact_name_matches`, where Polars' join drops
    null keys for the same reason)."""
    return pl.col(column).is_not_null() & (pl.col(column).str.strip_chars() != "")


def build_name_form_map(frame: pl.DataFrame, *, id_col: str) -> pl.DataFrame | None:
    """Build a `compute_pair_truth_eval()` name-form lookup
    (`id_col`/`name`/`name_cleansed`) from a loaded population frame.

    Returns `None` when `frame` doesn't carry both name columns (real
    matched/cleansed data does; ad hoc prepared inputs may not), which leaves
    the bucket columns null rather than failing the run. `frame` must be the
    *full* population for its side: a missing id is reported as
    `unclassified`, never as a difference, so passing a filtered subset
    silently miscounts every pair whose counterpart was filtered out.
    """
    if "name" not in frame.columns or "name_cleansed" not in frame.columns:
        return None
    selected = [
        pl.col("system_uri").cast(pl.Utf8).alias(id_col),
        pl.col("name").cast(pl.Utf8, strict=False).alias("name"),
        pl.col("name_cleansed").cast(pl.Utf8, strict=False).alias("name_cleansed"),
    ]
    # The basic tier is carried when the side has it, so the cascade can say
    # how much of the gap basic cleansing closes. Optional rather than
    # required: not every prepared input has the column, and its absence
    # collapses the split rather than failing the run.
    if "name_cleansed_basic" in frame.columns:
        selected.append(
            pl.col("name_cleansed_basic")
            .cast(pl.Utf8, strict=False)
            .alias("name_cleansed_basic")
        )
    # The text the run's algorithm was handed, carried when the side has it
    # so the cascade can keep pairs equal in it out of `never`.
    if NAME_PREPROCESSED_COLUMN in frame.columns:
        selected.append(
            pl.col(NAME_PREPROCESSED_COLUMN)
            .cast(pl.Utf8, strict=False)
            .alias(NAME_PREPROCESSED_COLUMN)
        )
    return frame.select(selected)


def _resolve_source_truth_column(
    source_slice: pl.DataFrame, *, target_system: str
) -> str:
    """Self-validation robustness scoring: decide which column
    `build_source_truth_map` should read as ground truth for this
    (source_system, target_system) unit.

    A perturbed row's `source_uri` is its own real, pre-perturbation
    `system_uri` -- when every populated `source_uri` in this slice carries
    the SAME scheme as `target_system`, the source is being validated
    against the very system its perturbed rows were materialized from, so
    the "true" target for each row is its own pre-perturbation identity.
    That is self-validation, and `source_uri` is the truth column.

    Every other case reads `match_uri` unchanged, exactly as before this
    item: plain (non-perturbed) data, and a perturbed row validated against
    a genuinely different target system (cross-system mode) -- where
    `match_uri` is the real cross-system ground truth
    `perturbation_materializer.py` passes through untouched, and
    `source_uri` would point at the wrong system entirely.

    A source_slice with `source_uri` present but no populated values (an
    all-null column, or none survived filtering) also falls through to
    `match_uri`: there is nothing to compare schemes against.
    """
    if "source_uri" not in source_slice.columns or source_slice.height == 0:
        return "match_uri"

    schemes = (
        source_slice.select(
            pl.col("source_uri")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.split_exact("://", 1)
            .struct.field("field_0")
            .alias("_scheme")
        )
        .filter(pl.col("_scheme") != "")
        .get_column("_scheme")
        .unique()
        .to_list()
    )
    if schemes and all(scheme == target_system for scheme in schemes):
        return "source_uri"
    return "match_uri"


def build_source_truth_map(
    source_slice: pl.DataFrame, *, target_system: str
) -> pl.DataFrame:
    """`compute_pair_truth_eval()`'s `source_truth_map` argument: one row per
    source id with its ground-truth `target_id` under `source_match_uri`.

    Reads `match_uri` (real cross-system ground truth) by default, exactly
    as before this item. For a perturbed source slice being validated
    against the very system it was materialized from, reads `source_uri`
    instead -- see `_resolve_source_truth_column` for the rule.
    `compute_pair_truth_eval` itself is unchanged either way: it only ever
    sees a `source_match_uri` column and has no notion of perturbation.
    """
    truth_column = _resolve_source_truth_column(
        source_slice, target_system=target_system
    )
    if truth_column in source_slice.columns:
        return source_slice.select(
            pl.col("system_uri").cast(pl.Utf8).alias("source_id"),
            pl.col(truth_column).cast(pl.Utf8, strict=False).alias("source_match_uri"),
        )
    return source_slice.select(
        pl.col("system_uri").cast(pl.Utf8).alias("source_id")
    ).with_columns(pl.lit(None, dtype=pl.Utf8).alias("source_match_uri"))


def _bucketed_pairs_detailed(
    pairs: pl.DataFrame,
    *,
    source_name_forms: pl.DataFrame,
    target_name_forms: pl.DataFrame,
) -> pl.DataFrame:
    """Join `pairs` (source_id, target_id) against per-id name-form lookups
    and assign each pair exactly one `name_equality` level, keeping both
    sides' raw and cleansed name values alongside it (`compute_pair_truth_eval_detail`'s
    per-pair artefact needs the names; `_bucketed_pairs` below is the
    level-only view `compute_pair_truth_eval`'s aggregate counts use, built
    from this same join rather than a second one). The level is the first
    name form at which the two sides are byte-identical:

    - `raw`: the raw `name` values are equal, so the pair is trivially
      resolvable before cleansing is even applied (exactly the population
      `blocking`'s exact-name fast path, `_split_exact_name_matches`,
      splits off).
    - `basic`: raw names differ but `name_cleansed_basic` values are equal;
      only reported when both sides carry that column.
    - `cleansed`: still different after basic cleansing but `name_cleansed`
      values are equal.
    - `preprocessed`: still different as cleansed but the run's
      `name_preprocessed` values are equal, so the scan was handed one string
      twice; only reported when both sides carry that column.
    - `never`: the two sides still differ after cleansing and preprocessing.
      This is the non-trivial residual the algorithm actually has to earn.
    - `unknown`: no verdict is possible because a cleansed value is missing
      on one side (the id is absent from the lookup, or its value is
      null/blank). Reported as its own count rather than folded into any
      level: absence is not evidence of difference.

    Both lookups are deduplicated on their id column so a duplicated id can
    never fan a pair out into several rows and inflate the bucket counts
    past `truth_pairs`.
    """
    has_basic = (
        "name_cleansed_basic" in source_name_forms.columns
        and "name_cleansed_basic" in target_name_forms.columns
    )

    has_preprocessed = (
        NAME_PREPROCESSED_COLUMN in source_name_forms.columns
        and NAME_PREPROCESSED_COLUMN in target_name_forms.columns
    )

    def _side_lookup(forms: pl.DataFrame, *, id_col: str, prefix: str) -> pl.DataFrame:
        selected = [
            pl.col(id_col),
            pl.col("name").cast(pl.Utf8, strict=False).alias(f"_{prefix}_name"),
            pl.col("name_cleansed")
            .cast(pl.Utf8, strict=False)
            .alias(f"_{prefix}_cleansed"),
        ]
        if has_basic:
            selected.append(
                pl.col("name_cleansed_basic")
                .cast(pl.Utf8, strict=False)
                .alias(f"_{prefix}_basic")
            )
        if has_preprocessed:
            selected.append(
                pl.col(NAME_PREPROCESSED_COLUMN)
                .cast(pl.Utf8, strict=False)
                .alias(f"_{prefix}_preprocessed")
            )
        return forms.select(selected).unique(subset=[id_col], keep="first")

    source_lookup = _side_lookup(source_name_forms, id_col="source_id", prefix="source")
    target_lookup = _side_lookup(target_name_forms, id_col="target_id", prefix="target")

    comparable = _present("_source_cleansed") & _present("_target_cleansed")
    raw_identical = (
        _present("_source_name")
        & _present("_target_name")
        & (pl.col("_source_name") == pl.col("_target_name"))
    )
    basic_identical = (
        (
            _present("_source_basic")
            & _present("_target_basic")
            & (pl.col("_source_basic") == pl.col("_target_basic"))
        )
        if has_basic
        # No basic column on one side or the other: the level cannot fire, and
        # its pairs fall through to `cleansed` as they did before the split
        # existed.
        else pl.lit(False)
    )
    cleansed_identical = pl.col("_source_cleansed") == pl.col("_target_cleansed")
    preprocessed_identical = (
        (
            _present("_source_preprocessed")
            & _present("_target_preprocessed")
            & (pl.col("_source_preprocessed") == pl.col("_target_preprocessed"))
        )
        if has_preprocessed
        else pl.lit(False)
    )

    return (
        pairs.select("source_id", "target_id")
        .join(source_lookup, on="source_id", how="left")
        .join(target_lookup, on="target_id", how="left")
        .with_columns(
            pl.when(~comparable)
            .then(pl.lit(NAME_EQUALITY_UNKNOWN))
            .when(raw_identical)
            .then(pl.lit(NAME_EQUALITY_RAW))
            .when(basic_identical)
            .then(pl.lit(NAME_EQUALITY_BASIC))
            .when(cleansed_identical)
            .then(pl.lit(NAME_EQUALITY_CLEANSED))
            .when(preprocessed_identical)
            .then(pl.lit(NAME_EQUALITY_PREPROCESSED))
            .otherwise(pl.lit(NAME_EQUALITY_NEVER))
            .alias("name_equality")
        )
        .select(
            "source_id",
            "target_id",
            "_source_name",
            "_source_cleansed",
            "_target_name",
            "_target_cleansed",
            "name_equality",
        )
    )


def _bucketed_pairs(
    pairs: pl.DataFrame,
    *,
    source_name_forms: pl.DataFrame,
    target_name_forms: pl.DataFrame,
) -> pl.DataFrame:
    """Bucket-only view of `_bucketed_pairs_detailed` -- see that function's
    docstring for the bucket rules; this just drops the name columns
    `compute_pair_truth_eval`'s aggregate counts don't need."""
    return _bucketed_pairs_detailed(
        pairs,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    ).select("source_id", "target_id", "name_equality")


def _bucket_counts(bucketed: pl.DataFrame) -> dict[str, int]:
    counts = dict.fromkeys(_NAME_EQUALITY_VALUES, 0)
    if bucketed.height == 0:
        return counts
    tallied = (
        bucketed.select(pl.col("name_equality").fill_null(NAME_EQUALITY_UNKNOWN))
        .group_by("name_equality")
        .len()
        .select("name_equality", "len")
    )
    for bucket, count in tallied.iter_rows():
        counts[str(bucket)] = int(count)
    return counts


def compute_pair_truth_eval(
    *,
    source_truth_map: pl.DataFrame,
    matched_edges: pl.DataFrame,
    source_system: str,
    target_system: str,
    country: str,
    target_rows: int,
    beta: float = 1.0,
    source_name_forms: pl.DataFrame | None = None,
    target_name_forms: pl.DataFrame | None = None,
    source_rows: int | None = None,
    top_k: int | None = None,
) -> pl.DataFrame:
    """One row per population: the whole confusion matrix over every labelled
    source (`population="universe"`) and then, when `source_name_forms` and
    `target_name_forms` are both supplied, the same matrix over the sources at
    each `name_equality` level (`contracts.NAME_EQUALITY_POPULATIONS`).

    A source's level is the level of its one truth pair: the first name form at
    which the two sides are byte-identical, judged over the **full** population
    on each side, so no source's level depends on which rows survive an
    upstream split such as `blocking`'s exact-name fast path.
    `source_name_forms` carries `source_id`/`name`/`name_cleansed` (and
    `name_cleansed_basic` where the side has it), and `target_name_forms` the
    same for `target_id`.

    Every population goes through `_population_row` over its own sources, the
    universe with the frames whole and each level with them narrowed, so a level
    is never a second computation that can drift from the universe's. Narrowing
    the sources narrows every cell together, an `fp` counted at the level of the
    source that produced it, and the universe's labelled-scoped cells are the
    sum of the levels'. The levels are the point: this corpus is dominated by
    pairs already identical at `raw`, which the fast path takes before any
    representation runs, so the universe largely measures identical strings
    matching identical strings and `never` is what a representation earns.

    No F score is stored: with the matrix present any beta is one division at
    read time, and a stored score fixes a weighting the row cannot state.

    Omitting either name-form frame writes the universe row alone, since no
    source can then be given a level.

    `source_rows` is the full source population this country was
    scored against -- *not* `labelled_sources`, which only counts rows with
    a known truth pairing. It is optional and independent of every other
    argument: omit it (or pass `0`) to leave the universe row's
    `candidate_set_size_ratio` null. A level row's `source_rows` is always its
    own source count, since the level is what defines that population.
    `candidate_pair_count` (`matched_edges.height`, the same count
    `BlockingRunResult.candidate_pair_count` reports at the whole-run level,
    here scoped to this one country/stage) is always populated regardless,
    since it costs nothing beyond `matched_edges` itself.
    `candidate_set_size_ratio` is `candidate_pair_count / (source_rows *
    target_rows)` -- the fraction of the full source-times-target comparison
    space this stage's candidate set actually contains, independent of
    which sources happen to carry a known truth pairing (unlike
    `reduction_ratio`, which is scoped to `pair_universe =
    labelled_sources * target_rows`). Null whenever `source_rows` is omitted
    or either side is `0`.

    `top_k` is the per-source candidate window `recall_at_k` is
    measured over, read from `matched_edges`' own `rank` column (present on
    every matched-edges frame `blocking.workflow` produces). It is a plain
    threshold on that column -- `rank <= top_k` -- not a re-derivation of
    the backend's own top-k retrieval, so passing the strategy's own
    `top_k` after backend retrieval already capped every source's
    candidates there reproduces the blended `recall` above by construction;
    the parameter earns its keep once a caller passes a *smaller* window
    (recall within the top 1 or top 5 of a `top_k=20` run, for example) or
    reads `recall_at_k` off `raw_matched_edges` -- whose ranks a later
    pruning stage may have partly discarded -- to see how much of the
    ranking-quality signal a candidate-cap or similarity-ratio prune costs.
    `recall_at_k` is null whenever `top_k` is omitted or `matched_edges` has
    no `rank` column at all (a caller-built frame that never carried one).

    `company_classify`'s `MetricBundle` has a `recall_at_k` and a
    `candidate_set_size_ratio` of its own over a different population, pair
    classification rather than candidate generation, so a figure from one is
    not comparable with the same-named figure from the other.
    """
    labelled = (
        source_truth_map.select("source_id", "source_match_uri")
        .filter(
            pl.col("source_match_uri").is_not_null()
            & (pl.col("source_match_uri").str.strip_chars() != "")
        )
        .unique(subset=["source_id"], keep="first")
    )

    truth_pairs = labelled.select(
        pl.col("source_id"),
        pl.col("source_match_uri").alias("target_id"),
    ).unique(subset=["source_id", "target_id"], keep="first")

    predicted_pairs = (
        matched_edges.select("source_id", "target_id")
        .filter(
            pl.col("target_id").is_not_null()
            & (pl.col("target_id").str.strip_chars() != "")
        )
        .join(labelled.select("source_id"), on="source_id", how="inner")
        .unique(subset=["source_id", "target_id"], keep="first")
    )

    rows = [
        _population_row(
            population=_POPULATION_UNIVERSE,
            truth_pairs=truth_pairs,
            predicted_pairs=predicted_pairs,
            matched_edges=matched_edges,
            target_rows=int(target_rows),
            source_rows=None if source_rows is None else int(source_rows),
            top_k=top_k,
        )
    ]

    if source_name_forms is not None and target_name_forms is not None:
        # One truth pair per labelled source, so its pair's level is the
        # source's; a pair with no verdict belongs to `unknown`.
        source_levels = _bucketed_pairs(
            truth_pairs,
            source_name_forms=source_name_forms,
            target_name_forms=target_name_forms,
        ).select(
            "source_id",
            pl.col("name_equality").fill_null(NAME_EQUALITY_UNKNOWN).alias("level"),
        )
        for level in _NAME_EQUALITY_POPULATIONS:
            level_sources = source_levels.filter(pl.col("level") == level).select(
                "source_id"
            )
            rows.append(
                _population_row(
                    population=level,
                    truth_pairs=truth_pairs.join(
                        level_sources, on="source_id", how="semi"
                    ),
                    predicted_pairs=predicted_pairs.join(
                        level_sources, on="source_id", how="semi"
                    ),
                    matched_edges=matched_edges.join(
                        level_sources, on="source_id", how="semi"
                    ),
                    target_rows=int(target_rows),
                    source_rows=level_sources.height,
                    top_k=top_k,
                )
            )

    return pl.DataFrame(
        [
            {
                "source_system": source_system,
                "target_system": target_system,
                "country": country,
                **row,
            }
            for row in rows
        ],
        schema=_ARTIFACT_SCHEMAS["pair_truth_eval"],
    )


def pair_truth_eval_row(frame: pl.DataFrame, population: str) -> dict[str, object]:
    """The one `pair_truth_eval` row for `population`, read by name.

    A frame holds one row per population per country, so a reader taking
    whichever row came first would read an arbitrary population without
    failing. This raises unless exactly one row matches, which also catches a
    caller that forgot to narrow a multi-country frame to one country first.
    """
    matching = frame.filter(pl.col("population") == population)
    if matching.height != 1:
        raise ValueError(
            f"pair_truth_eval holds {matching.height} {population!r} rows, "
            "expected exactly one; narrow it to one country first"
        )
    return matching.row(0, named=True)


def _population_row(
    *,
    population: str,
    truth_pairs: pl.DataFrame,
    predicted_pairs: pl.DataFrame,
    matched_edges: pl.DataFrame,
    target_rows: int,
    source_rows: int | None,
    top_k: int | None,
) -> dict[str, object]:
    """One population's matrix and candidate figures, over whichever sources
    the three frames were narrowed to.

    `predicted_pairs` is already restricted to labelled sources, and
    `truth_pairs` holds one pair per labelled source, so every cell here is a
    function of the source set the caller passed.
    """
    pair_key = ["source_id", "target_id"]
    tp_pairs = predicted_pairs.join(truth_pairs, on=pair_key, how="inner")
    fp = int(predicted_pairs.join(truth_pairs, on=pair_key, how="anti").height)
    fn = int(truth_pairs.join(predicted_pairs, on=pair_key, how="anti").height)

    labelled_sources = int(truth_pairs.get_column("source_id").n_unique())
    pair_universe = labelled_sources * target_rows
    truth_pair_count = int(truth_pairs.height)
    predicted_pair_count = int(predicted_pairs.height)
    tp = int(tp_pairs.height)
    # No `tn`. Ground truth records matching pairs and never records a
    # non-match, so an unlisted pair is unknown rather than a known negative
    # and there is no population of true negatives to count.
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)

    reduction_ratio = None
    if pair_universe > 0:
        sample_ratio = _safe_ratio(predicted_pair_count, pair_universe)
        if sample_ratio is not None:
            reduction_ratio = 1.0 - sample_ratio

    # `candidate_pair_count` is every candidate `matched_edges` holds for this
    # population, labelled or not, unlike `predicted_pair_count`, which is
    # restricted to labelled sources to match `pair_universe`.
    candidate_pair_count = int(matched_edges.height)
    candidate_set_size_ratio = None
    if source_rows is not None and source_rows > 0 and target_rows > 0:
        candidate_set_size_ratio = _safe_ratio(
            candidate_pair_count, source_rows * target_rows
        )

    recall_at_k = None
    if top_k is not None and "rank" in matched_edges.columns:
        at_k_hits = (
            matched_edges.filter(pl.col("rank") <= top_k)
            .select(pair_key)
            .join(truth_pairs, on=pair_key, how="inner")
            .unique(subset=pair_key, keep="first")
        )
        recall_at_k = _safe_ratio(int(at_k_hits.height), truth_pair_count)

    return {
        "population": population,
        "labelled_sources": labelled_sources,
        "target_rows": target_rows,
        "pair_universe": pair_universe,
        "truth_pairs": truth_pair_count,
        "predicted_pairs": predicted_pair_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "reduction_ratio": reduction_ratio,
        "candidate_pair_count": candidate_pair_count,
        "source_rows": source_rows,
        "candidate_set_size_ratio": candidate_set_size_ratio,
        "recall_at_k": recall_at_k,
    }


# The per-pair companion to `compute_pair_truth_eval`'s population
# rows. Owned by `blocking.contracts.BLOCKING_ARTIFACT_SCHEMAS` (this
# module is where the columns are actually produced, so the schema is
# defined here and imported there rather than restated) --
# unlike `pair_truth_eval` itself, `pair_truth_eval_detail` has no consumer
# in `validation`'s own schema registry, so it is not added to
# `ARTIFACT_SCHEMAS`.
#
# Named for the population it actually holds -- every truth pair
# plus every predicted-but-untrue pair, across every name-equality level --
# rather than "residual_pairs", which read as if the frame were already
# filtered down to the still-unequal remainder. It is not: the remainder is
# one level within it, reached via `filter_residual_pairs(bucket=
# NAME_EQUALITY_NEVER)` (the default), never the frame's whole population.
PAIR_TRUTH_EVAL_DETAIL_COLUMNS: dict[str, PolarsDType] = {
    "source_system": pl.Utf8,
    "target_system": pl.Utf8,
    "country": pl.Utf8,
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "source_name": pl.Utf8,
    "source_name_cleansed": pl.Utf8,
    "target_name": pl.Utf8,
    "target_name_cleansed": pl.Utf8,
    "name_equality": pl.Utf8,
    "is_truth_pair": pl.Boolean,
    "found": pl.Boolean,
    "similarity": pl.Float64,
    "rank": pl.Int64,
}


def compute_pair_truth_eval_detail(
    *,
    source_truth_map: pl.DataFrame,
    matched_edges: pl.DataFrame,
    source_system: str,
    target_system: str,
    country: str,
    source_name_forms: pl.DataFrame,
    target_name_forms: pl.DataFrame,
) -> pl.DataFrame:
    """Per-pair companion to `compute_pair_truth_eval`: one row per truth
    pair plus one row per predicted-but-untrue pair, sharing the exact
    `_bucketed_pairs_detailed` classification that function's aggregate
    bucket counts are built from, rather than a second reimplementation of
    it.

    Covers every name-equality level, not only the still-unequal remainder --
    a `raw`-level truth pair appears here too, resolved or not, since this
    evaluates the *complete* truth-pair population independent of which
    resolution path (an exact-name fast path, scored candidates, or neither)
    found it. `inspection.filter_residual_pairs` narrows this frame down to
    the remainder a caller actually wants (`bucket=NAME_EQUALITY_NEVER` by
    default).

    Truth-pair rows (`is_truth_pair=True`) carry `found` -- `True` when the
    pair was also predicted (a true positive), `False` when it was missed (a
    false negative, `compute_pair_truth_eval`'s `fn`). Predicted-but-untrue
    rows (`is_truth_pair=False`, `found=None` -- "found" has no meaning for a
    pair that was never a truth pair to begin with) are candidate-set
    pollution: a predicted pair against a labelled source that ground truth
    says is wrong (`compute_pair_truth_eval`'s `fp`). Both populations carry
    each side's raw `name` and `name_cleansed`, plus the difficulty `bucket`
    (see `_bucketed_pairs_detailed`) computed from those same values, so a
    reader never has to rejoin against the source/target layers to see what
    a pair's names actually were. `similarity`/`rank` come from
    `matched_edges` and are
    null for a truth pair that was never predicted at all (there is no edge
    to read them from).

    `source_name_forms`/`target_name_forms` must cover the **full**
    population on each side, exactly as `compute_pair_truth_eval` requires --
    see that function's docstring.
    """
    labelled = (
        source_truth_map.select("source_id", "source_match_uri")
        .filter(_present("source_match_uri"))
        .unique(subset=["source_id"], keep="first")
    )
    truth_pairs = labelled.select(
        pl.col("source_id"),
        pl.col("source_match_uri").alias("target_id"),
    ).unique(subset=["source_id", "target_id"], keep="first")

    predicted_pairs = (
        matched_edges.select("source_id", "target_id", "similarity", "rank")
        .filter(_present("target_id"))
        .join(labelled.select("source_id"), on="source_id", how="inner")
        .unique(subset=["source_id", "target_id"], keep="first")
    )

    truth_detail = _bucketed_pairs_detailed(
        truth_pairs,
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )
    truth_rows = truth_detail.join(
        predicted_pairs, on=["source_id", "target_id"], how="left"
    ).with_columns(
        pl.lit(True).alias("is_truth_pair"),
        pl.col("similarity").is_not_null().alias("found"),
    )

    fp_pairs = predicted_pairs.join(
        truth_pairs, on=["source_id", "target_id"], how="anti"
    )
    fp_detail = _bucketed_pairs_detailed(
        fp_pairs.select("source_id", "target_id"),
        source_name_forms=source_name_forms,
        target_name_forms=target_name_forms,
    )
    fp_rows = fp_detail.join(
        fp_pairs, on=["source_id", "target_id"], how="left"
    ).with_columns(
        pl.lit(False).alias("is_truth_pair"),
        pl.lit(None, dtype=pl.Boolean).alias("found"),
    )

    combined = pl.concat([truth_rows, fp_rows], how="vertical_relaxed")
    return (
        combined.with_columns(
            pl.lit(source_system).alias("source_system"),
            pl.lit(target_system).alias("target_system"),
            pl.lit(country).alias("country"),
        )
        .rename(
            {
                "_source_name": "source_name",
                "_source_cleansed": "source_name_cleansed",
                "_target_name": "target_name",
                "_target_cleansed": "target_name_cleansed",
            }
        )
        .select(list(PAIR_TRUTH_EVAL_DETAIL_COLUMNS.keys()))
        .cast(pl.Schema(PAIR_TRUTH_EVAL_DETAIL_COLUMNS))
    )


# The v1 robustness metric contract -- see `contracts.ARTIFACT_SCHEMAS["robustness_eval"]`/
# `["robustness_eval_rollup"]` for the column-by-column documentation this reuses rather than
# restating.
_ROBUSTNESS_EVAL_COLUMNS: dict[str, PolarsDType] = _ARTIFACT_SCHEMAS["robustness_eval"]
_ROBUSTNESS_EVAL_ROLLUP_COLUMNS: dict[str, PolarsDType] = _ARTIFACT_SCHEMAS[
    "robustness_eval_rollup"
]


def build_source_perturbation_map(
    frame: pl.DataFrame, *, id_col: str = "source_id"
) -> pl.DataFrame | None:
    """(profile_id, profile_version, scenario_id, intensity) lookup keyed on
    `id_col`, built from a loaded perturbed source slice.

    Returns `None` when `frame` doesn't carry all four perturbation columns
    (real, non-perturbed data doesn't), mirroring `build_name_form_map`'s
    "absent -> `None`, not failure" contract -- a caller with no
    perturbation columns to read should skip the robustness eval, not
    crash. Rows with a null `profile_id` (a baseline row mixed into an
    otherwise-perturbed frame) are dropped rather than surfaced as their
    own group.
    """
    required = ("profile_id", "profile_version", "scenario_id", "intensity")
    if not all(column in frame.columns for column in required):
        return None
    return frame.select(
        pl.col("system_uri").cast(pl.Utf8).alias(id_col),
        pl.col("profile_id").cast(pl.Utf8, strict=False).alias("profile_id"),
        pl.col("profile_version").cast(pl.Utf8, strict=False).alias("profile_version"),
        pl.col("scenario_id").cast(pl.Utf8, strict=False).alias("scenario_id"),
        pl.col("intensity").cast(pl.Float64, strict=False).alias("intensity"),
    ).filter(pl.col("profile_id").is_not_null())


def _as_repo_relative_pointer(value: str | None) -> str | None:
    """A provenance pointer as it is persisted: repo-relative, POSIX separators.

    Both path flavours are checked, not just the running platform's: a
    POSIX-absolute `/data/...` has no drive letter, so `PureWindowsPath`
    alone reads it as merely drive-relative and would let it through on
    Windows.
    """
    if value is None:
        return None
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(
            "dataset_snapshot_manifest_path must be repo-relative, not absolute: "
            f"{value!r}. Resolve it with "
            "perturbation_materializer.relative_materialization_manifest_path."
        )
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def compute_robustness_eval(
    *,
    source_truth_map: pl.DataFrame,
    matched_edges: pl.DataFrame,
    source_perturbation_map: pl.DataFrame,
    source_system: str,
    target_system: str,
    country: str,
    baseline_recall: float | None,
    dataset_snapshot_manifest_path: str | None = None,
) -> pl.DataFrame:
    """v1 robustness metric contract.

    One row per distinct (profile_id, profile_version, scenario_id,
    intensity) combination present in `source_perturbation_map`
    (`build_source_perturbation_map`), scored by restricting
    `source_truth_map`/`matched_edges` to that group's own source ids and
    reusing `compute_pair_truth_eval` unchanged for the tp/fp/fn/precision/
    recall arithmetic -- no second scoring implementation.

    `recall_retention_ratio` is the core robustness signal this item
    defines: `recall / baseline_recall`. Higher is better; `1.0` means no
    degradation versus baseline. `baseline_recall` is supplied by the
    caller (recall of the same target system under a genuinely unperturbed
    run -- e.g. the same `country`'s plain, non-perturbed `pair_truth_eval`
    row) rather than computed here: this function has no opinion on how a
    caller establishes its baseline, only on how a perturbed run's recall
    compares against it. `recall_retention_ratio` is null whenever `recall`
    or `baseline_recall` is null, or `baseline_recall` is `<= 0.0`, rather
    than raising or silently reporting `0.0`.

    The pass/fail threshold that turns this ratio into a promotion decision
    is deliberately not decided here: a separate promotion scorecard is
    meant to consume this ratio and apply whatever threshold it chooses;
    this function only guarantees the field exists with its direction
    documented.

    `dataset_snapshot_manifest_path` is a provenance pointer supplied by a
    caller that knows which `(profile_id, source_system)` fed this call:
    resolve it with `perturbation_materializer.relative_materialization_
    manifest_path` (or check it exists first with
    `load_materialization_manifest`, which returns `None` when
    materialization hasn't produced one yet). Stored as a constant column on
    every output row -- one call always scores one `(source_system,
    country)` pair, so the pointer never varies within a call. Left `None`
    (the default) when the caller has no provenance to offer, same "not
    computed" convention the rest of this contract uses.

    It must be repo-relative, and an absolute path raises rather than being
    persisted: `data/` is one shared physical store that every worktree
    reaches through its own roots, so an absolute path recorded by one
    checkout names a location no other checkout has. Backslashes are
    normalised to POSIX separators so an artifact written on Windows reads
    on Linux.

    Roll this frame up across whatever finer-grained dimension feeds each
    (profile_id, profile_version, scenario_id, intensity) group with
    `rollup_robustness_eval` -- today that dimension is country; a future
    per-draw dimension once the perturbed `system_uri` disambiguates
    repeated draws of one record under one scenario, which it does not yet.
    """
    manifest_pointer = _as_repo_relative_pointer(dataset_snapshot_manifest_path)
    groups = (
        source_perturbation_map.select(
            "profile_id", "profile_version", "scenario_id", "intensity"
        )
        .unique()
        .sort(["profile_id", "profile_version", "scenario_id", "intensity"])
    )

    rows: list[dict[str, object]] = []
    for group in groups.iter_rows(named=True):
        group_ids = source_perturbation_map.filter(
            (pl.col("profile_id") == group["profile_id"])
            & (pl.col("profile_version") == group["profile_version"])
            & (pl.col("scenario_id") == group["scenario_id"])
            & (pl.col("intensity") == group["intensity"])
        ).select("source_id")

        group_truth_map = source_truth_map.join(group_ids, on="source_id", how="inner")
        group_matched_edges = matched_edges.join(group_ids, on="source_id", how="inner")

        group_eval = pair_truth_eval_row(
            compute_pair_truth_eval(
                source_truth_map=group_truth_map,
                matched_edges=group_matched_edges,
                source_system=source_system,
                target_system=target_system,
                country=country,
                target_rows=0,
            ),
            _POPULATION_UNIVERSE,
        )

        recall = group_eval["recall"]
        recall_retention_ratio = None
        if (
            isinstance(recall, (int, float))
            and baseline_recall is not None
            and baseline_recall > 0.0
        ):
            recall_retention_ratio = float(recall) / baseline_recall

        rows.append(
            {
                "profile_id": group["profile_id"],
                "profile_version": group["profile_version"],
                "scenario_id": group["scenario_id"],
                "intensity": group["intensity"],
                "country": country,
                "source_system": source_system,
                "target_system": target_system,
                "labelled_sources": group_eval["labelled_sources"],
                "truth_pairs": group_eval["truth_pairs"],
                "predicted_pairs": group_eval["predicted_pairs"],
                "tp": group_eval["tp"],
                "fp": group_eval["fp"],
                "fn": group_eval["fn"],
                "precision": group_eval["precision"],
                "recall": recall,
                "baseline_recall": baseline_recall,
                "recall_retention_ratio": recall_retention_ratio,
                "dataset_snapshot_manifest_path": manifest_pointer,
            }
        )

    if not rows:
        return _empty_frame(schema=_ROBUSTNESS_EVAL_COLUMNS)
    return pl.DataFrame(rows, schema=_ROBUSTNESS_EVAL_COLUMNS)


def rollup_robustness_eval(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate `compute_robustness_eval`'s per-country rows to one row per
    (profile_id, profile_version, scenario_id, intensity), reporting the
    mean and standard deviation of `recall_retention_ratio` across whatever
    finer-grained rows feed the group -- countries today; a future per-draw
    dimension once the perturbed `system_uri` disambiguates repeated draws
    of one record under one scenario, which it does not yet.

    `recall_retention_ratio_std` is null when only one row feeds a group
    (spread is undefined for n=1), never `0.0` -- a single-sample group
    must never read as "measured, zero variance".
    """
    if frame.height == 0:
        return _empty_frame(schema=_ROBUSTNESS_EVAL_ROLLUP_COLUMNS)

    return (
        frame.group_by(["profile_id", "profile_version", "scenario_id", "intensity"])
        .agg(
            pl.len().alias("sample_count"),
            pl.col("recall_retention_ratio")
            .mean()
            .alias("recall_retention_ratio_mean"),
            pl.when(pl.col("recall_retention_ratio").count() >= 2)
            .then(pl.col("recall_retention_ratio").std())
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("recall_retention_ratio_std"),
        )
        .sort(["profile_id", "profile_version", "scenario_id", "intensity"])
        .select(list(_ROBUSTNESS_EVAL_ROLLUP_COLUMNS.keys()))
    )


def compute_run_metrics(
    *,
    candidate_edges: pl.DataFrame,
    source_system: str,
    target_system: str,
    country: str,
    source_rows: int,
    elapsed_seconds: float,
) -> pl.DataFrame:
    """Compute the latency and candidate-duplication metrics for one
    (source_system, target_system, country) validation unit.

    ``latency_seconds_per_1k_rows`` is a runtime-normalized throughput
    metric: wall-clock seconds spent processing this unit, scaled to a
    fixed 1,000-source-row basis so units with different row counts are
    comparable.

    ``candidate_duplication_ratio`` indicates how much the candidate set
    (the source-target edges surviving ``top_k``/``min_similarity``
    filtering, i.e. ``candidate_edges``) overlaps/duplicates across source
    rows: the fraction of candidate edges whose target is already claimed
    by another source row's candidate set. 0.0 means every candidate edge
    points at a distinct target; values approaching 1.0 mean source rows
    are converging on a small, heavily-reused pool of target candidates.
    """
    total_candidate_edges = int(candidate_edges.height)
    distinct_candidate_targets = (
        int(candidate_edges.select("target_id").unique().height)
        if total_candidate_edges > 0
        else 0
    )
    candidate_duplication_ratio = None
    if total_candidate_edges > 0:
        candidate_duplication_ratio = 1.0 - (
            float(distinct_candidate_targets) / float(total_candidate_edges)
        )

    latency_seconds_per_1k_rows = None
    if source_rows > 0:
        latency_seconds_per_1k_rows = (
            float(elapsed_seconds) / float(source_rows)
        ) * 1000.0

    return pl.DataFrame(
        {
            "source_system": [source_system],
            "target_system": [target_system],
            "country": [country],
            "source_rows": [int(source_rows)],
            "elapsed_seconds": [float(elapsed_seconds)],
            "latency_seconds_per_1k_rows": [latency_seconds_per_1k_rows],
            "candidate_edges": [total_candidate_edges],
            "candidate_distinct_targets": [distinct_candidate_targets],
            "candidate_duplication_ratio": [candidate_duplication_ratio],
        }
    )


def _build_run_metrics_output(
    *,
    run_metrics_parts: list[pl.DataFrame],
    schemas: dict[str, dict[str, PolarsDType]],
) -> pl.DataFrame:
    run_metrics_country = (
        pl.concat(run_metrics_parts, how="vertical_relaxed")
        if run_metrics_parts
        else _empty_frame(schema=schemas["run_metrics"])
    )

    if run_metrics_country.height == 0:
        return run_metrics_country

    run_metrics_global = (
        run_metrics_country.group_by(["source_system", "target_system"])
        .agg(
            pl.col("source_rows").sum().alias("source_rows"),
            pl.col("elapsed_seconds").sum().alias("elapsed_seconds"),
            pl.col("candidate_edges").sum().alias("candidate_edges"),
            pl.col("candidate_distinct_targets")
            .sum()
            .alias("candidate_distinct_targets"),
        )
        .with_columns(
            pl.lit("__all__").alias("country"),
            pl.when(pl.col("source_rows") > 0)
            .then((pl.col("elapsed_seconds") / pl.col("source_rows")) * 1000.0)
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("latency_seconds_per_1k_rows"),
            pl.when(pl.col("candidate_edges") > 0)
            .then(
                1.0 - (pl.col("candidate_distinct_targets") / pl.col("candidate_edges"))
            )
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("candidate_duplication_ratio"),
        )
        .select(list(schemas["run_metrics"].keys()))
    )

    return pl.concat([run_metrics_country, run_metrics_global], how="vertical_relaxed")


def _prepare_nonempty_source_rows(
    *,
    source_text: pl.DataFrame,
    source_feature_col: str,
) -> tuple[pl.DataFrame, int]:
    if source_feature_col == "cluster_tokens":
        source_nonempty = source_text.select(
            pl.col("system_uri").cast(pl.Utf8).alias("source_id"),
            pl.col("cluster_tokens")
            .cast(pl.List(pl.Utf8), strict=False)
            .fill_null([])
            .alias("_tokens"),
        ).filter(pl.col("_tokens").list.len() > 0)
    else:
        source_nonempty = source_text.select(
            pl.col("system_uri").cast(pl.Utf8).alias("source_id"),
            pl.col("cluster_text")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars()
            .alias("_text"),
        ).filter(pl.col("_text") != "")
    return source_nonempty, int(source_nonempty.height)


def _score_country_source_chunks(
    *,
    source_nonempty: pl.DataFrame,
    source_nonempty_count: int,
    source_feature_col: str,
    row_progress_step: int,
    target_index: TargetClusteringIndex,
    target_nn_index,
    target_backend_index,
    clustering_strategy,
    config: ValidationRunConfig,
    source_name_map: pl.DataFrame,
    source_truth_map: pl.DataFrame,
    target_name_map: pl.DataFrame,
    source_outcome_parts: list[pl.DataFrame],
    cluster_parts: list[pl.DataFrame],
    node_info: pl.DataFrame,
    run_id: str,
    source_system: str,
    target_system: str,
    country: str,
    source_rows: int,
    completed_units: int,
    total_units: int,
    incremental_output_dir: Path | None,
    incremental_part_index: dict[str, int],
    schemas: dict[str, dict[str, PolarsDType]],
    emit_progress: ProgressCallback | None,
) -> tuple[list[pl.DataFrame], int]:
    country_edge_parts: list[pl.DataFrame] = []
    source_rows_scored = 0
    rows_since_progress = 0
    max_per_source = (
        int(config.top_k)
        if config.max_candidates_per_source is None
        else min(int(config.top_k), int(config.max_candidates_per_source))
    )
    n_targets = len(target_index.target_ids)
    if (
        source_nonempty_count <= 0
        or n_targets <= 0
        or max_per_source <= 0
        or target_nn_index is None
    ):
        return country_edge_parts, source_rows_scored

    for batch_start in range(0, source_nonempty_count, row_progress_step):
        batch_end = min(batch_start + row_progress_step, source_nonempty_count)
        if source_feature_col == "cluster_tokens":
            source_chunk = source_nonempty.slice(
                batch_start, batch_end - batch_start
            ).rename({"source_id": "system_uri", "_tokens": "cluster_tokens"})
        else:
            source_chunk = source_nonempty.slice(
                batch_start, batch_end - batch_start
            ).rename({"source_id": "system_uri", "_text": "cluster_text"})
        chunk_edges = clustering_strategy.score_source_chunk(
            source_chunk,
            target_index=target_index,
            source_id_col="system_uri",
            text_col=source_feature_col,
            top_k=config.top_k,
            min_similarity=config.min_similarity,
            max_candidates_per_source=config.max_candidates_per_source,
            nn_index=target_nn_index,
            backend=str(config.similarity_backend),
            backend_index=target_backend_index,
            backend_options=(config.backend_options or {}),
        )
        chunk_edges_found = int(chunk_edges.height)
        if chunk_edges_found:
            chunk_edges_out = (
                chunk_edges.join(source_name_map, on="source_id", how="left")
                .join(source_truth_map, on="source_id", how="left")
                .join(target_name_map, on="target_id", how="left")
                .with_columns(
                    pl.when(pl.col("source_match_uri").is_not_null())
                    .then(pl.col("target_id") == pl.col("source_match_uri"))
                    .otherwise(pl.lit(None, dtype=pl.Boolean))
                    .alias("is_true_match"),
                    pl.when(pl.col("source_match_uri").is_null())
                    .then(pl.lit("unknown"))
                    .when(pl.col("target_id") == pl.col("source_match_uri"))
                    .then(pl.lit("true_match"))
                    .otherwise(pl.lit("false_match"))
                    .alias("true_match_status"),
                    pl.lit("matched").alias("match_status"),
                    pl.lit(None, dtype=pl.Utf8).alias("reason_code"),
                )
                .select(list(schemas["source_outcomes"].keys()))
            )
            country_edge_parts.append(chunk_edges_out)
            incremental_part_index["source_outcomes"] += 1
            _write_incremental_artifact_part(
                incremental_output_dir=incremental_output_dir,
                artifact_name="source_outcomes",
                part_index=incremental_part_index["source_outcomes"],
                frame=chunk_edges_out,
            )

            edges_snapshot_frames = source_outcome_parts + country_edge_parts
            if edges_snapshot_frames:
                pl.concat(edges_snapshot_frames, how="vertical_relaxed").write_parquet(
                    config.output_dir / "source_outcomes.parquet"
                )

            current_country_edges = pl.concat(
                country_edge_parts, how="vertical_relaxed"
            ).filter(
                (pl.col("match_status") == "matched")
                & pl.col("target_id").is_not_null()
            )
            current_components = build_connected_components(
                current_country_edges.select("source_id", "target_id"),
                source_id_col="source_id",
                target_id_col="target_id",
            )
            current_clusters = (
                current_components.join(node_info, on="node_id", how="left")
                .with_columns(
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(source_system).alias("source_system"),
                    pl.lit(target_system).alias("target_system"),
                    pl.lit(country).alias("country"),
                )
                .select(list(schemas["clusters"].keys()))
            )
            cluster_snapshot_frames = cluster_parts + [current_clusters]
            if cluster_snapshot_frames:
                pl.concat(
                    cluster_snapshot_frames, how="vertical_relaxed"
                ).write_parquet(config.output_dir / "clusters.parquet")

        source_rows_scored += int(source_chunk.height)
        rows_since_progress += int(source_chunk.height)

        if emit_progress is not None and (
            source_rows_scored >= source_rows
            or rows_since_progress >= row_progress_step
        ):
            emit_progress(
                {
                    "phase": "country_chunk",
                    "source_system": source_system,
                    "target_system": target_system,
                    "country": country,
                    "source_rows_scored": source_rows_scored,
                    "source_rows_total": source_rows,
                    "chunk_edges_found": chunk_edges_found,
                    "completed_units": completed_units,
                    "total_units": total_units,
                }
            )
            rows_since_progress = 0

    return country_edge_parts, source_rows_scored


def run_validation_matrix(
    config: ValidationRunConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    source_chunk_size: int = SOURCE_PROGRESS_CHUNK_SIZE,
    prepare_force: bool = False,
    runtime_max_rows: int | None = None,
    incremental_output_dir: Path | None = None,
    telemetry: Telemetry | None = None,
) -> dict[str, pl.DataFrame]:
    run_id = config.run_date or "latest"
    row_progress_step = max(1, int(source_chunk_size))
    effective_text_view = resolve_text_view_for_representation(
        representation=str(config.representation),
        requested_text_view=config.text_view,
    )
    schemas = _result_schemas()
    started_at = time.perf_counter()
    # Every phase's wall-clock, memory and CPU, returned as the `timings`
    # frame beside the artefact frames (`workspace.telemetry`).
    telemetry = telemetry if telemetry is not None else Telemetry()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    source_outcome_parts: list[pl.DataFrame] = []
    cluster_parts: list[pl.DataFrame] = []
    coverage_parts: list[pl.DataFrame] = []
    shape_parts: list[pl.DataFrame] = []
    pair_truth_eval_parts: list[pl.DataFrame] = []
    run_metrics_parts: list[pl.DataFrame] = []
    incremental_part_index = {
        "source_outcomes": 0,
        "clusters": 0,
        "directional_coverage": 0,
        "cluster_shape": 0,
        "pair_truth_eval": 0,
        "run_metrics": 0,
    }

    total_units = len(config.source_systems) * len(config.target_systems)
    completed_units = 0
    cumulative_source_rows_processed = 0
    cumulative_source_rows_with_match = 0
    cumulative_edges_found = 0
    cumulative_target_rows = 0

    prepared_stats_by_system: dict[str, PreparedSystemStats] = {}
    clustering_strategy = resolve_clustering_strategy(str(config.representation))
    target_index_build_settings = resolve_target_index_build_settings(
        str(config.representation),
        tfidf_ngram_min=int(config.tfidf_ngram_min),
        tfidf_ngram_max=int(config.tfidf_ngram_max),
        tfidf_analyzer=str(config.tfidf_analyzer),
    )

    def _emit_progress(phase: str, **payload: object) -> None:
        if progress_callback is None:
            return
        event = {
            "phase": phase,
            "run_id": run_id,
            **payload,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        progress_callback(event)

    def _prepare(system: str) -> None:
        with telemetry.phase("prepare_system", label=system):
            prepared_stats_by_system[system] = prepare_system_dataset(
                config=config,
                system=system,
                force_rebuild=prepare_force,
            )
        _emit_progress(
            "system_prepared",
            system=system,
            countries=len(list_prepared_countries(prepared_stats_by_system[system])),
        )

    for system in sorted(set(config.target_systems)):
        _prepare(system)

    for system in sorted(set(config.source_systems)):
        if system in prepared_stats_by_system:
            continue
        _prepare(system)

    for source_system in config.source_systems:
        for target_system in config.target_systems:
            source_stats = prepared_stats_by_system[source_system]
            target_stats = prepared_stats_by_system[target_system]
            countries = _resolve_validation_countries(
                source_stats=source_stats,
                target_stats=target_stats,
                configured_countries=config.countries,
            )

            pair_source_rows_processed = 0
            pair_source_rows_with_match = 0
            pair_edges_found = 0
            pair_target_rows = 0

            for country in countries:
                country_started = time.perf_counter()
                unit_label = f"{source_system}->{target_system}:{country}"
                with telemetry.phase("prepare_country", label=unit_label):
                    country_context = _prepare_country_validation_context(
                        config=config,
                        source_system=source_system,
                        target_system=target_system,
                        country=country,
                        source_stats=source_stats,
                        target_stats=target_stats,
                        runtime_max_rows=runtime_max_rows,
                        effective_text_view=effective_text_view,
                        target_index_build_settings=target_index_build_settings,
                        clustering_strategy=clustering_strategy,
                        completed_units=completed_units,
                        total_units=total_units,
                        emit_progress=_emit_progress,
                    )
                source_slice = country_context["source_slice"]
                target_slice = country_context["target_slice"]
                source_rows = int(country_context["source_rows"])
                target_rows = int(country_context["target_rows"])
                pair_source_rows_processed += source_rows
                pair_target_rows += target_rows
                source_text = country_context["source_text"]
                source_feature_col = str(country_context["source_feature_col"])
                target_index = country_context["target_index"]
                target_nn_index = country_context["target_nn_index"]
                target_backend_index = country_context["target_backend_index"]

                country_edge_parts: list[pl.DataFrame] = []
                source_rows_scored = 0
                source_nonempty, source_nonempty_count = _prepare_nonempty_source_rows(
                    source_text=source_text,
                    source_feature_col=source_feature_col,
                )

                source_nodes = source_slice.select(
                    pl.col("system_uri").alias("node_id"),
                    pl.col("name").alias("node_name"),
                )
                target_nodes = target_slice.select(
                    pl.col("system_uri").alias("node_id"),
                    pl.col("name").alias("node_name"),
                )
                source_name_map = source_nodes.rename(
                    {"node_id": "source_id", "node_name": "source_name"}
                )
                source_truth_map = build_source_truth_map(
                    source_slice, target_system=target_system
                )
                target_name_map = target_nodes.rename(
                    {"node_id": "target_id", "node_name": "target_name"}
                )
                # Difficulty-bucket inputs: raw `name` plus
                # `name_cleansed` for each side's whole country slice. Only
                # built when the prepared frame actually carries
                # `name_cleansed` (real matched/cleansed data does; ad hoc
                # prepared_base_dir inputs may not), so its absence just
                # leaves compute_pair_truth_eval()'s bucket columns null
                # rather than failing the run.
                source_name_forms = build_name_form_map(
                    source_slice, id_col="source_id"
                )
                target_name_forms = build_name_form_map(
                    target_slice, id_col="target_id"
                )
                node_info = pl.concat(
                    [
                        source_nodes.with_columns(pl.lit("source").alias("node_role")),
                        target_nodes.with_columns(pl.lit("target").alias("node_role")),
                    ],
                    how="vertical_relaxed",
                ).unique(subset=["node_id"], keep="first")
                with telemetry.phase(
                    "scoring", label=unit_label, rows_in=source_rows
                ) as scoring:
                    country_edge_parts, source_rows_scored = (
                        _score_country_source_chunks(
                            source_nonempty=source_nonempty,
                            source_nonempty_count=source_nonempty_count,
                            source_feature_col=source_feature_col,
                            row_progress_step=row_progress_step,
                            target_index=target_index,
                            target_nn_index=target_nn_index,
                            target_backend_index=target_backend_index,
                            clustering_strategy=clustering_strategy,
                            config=config,
                            source_name_map=source_name_map,
                            source_truth_map=source_truth_map,
                            target_name_map=target_name_map,
                            source_outcome_parts=source_outcome_parts,
                            cluster_parts=cluster_parts,
                            node_info=node_info,
                            run_id=run_id,
                            source_system=source_system,
                            target_system=target_system,
                            country=country,
                            source_rows=source_rows,
                            completed_units=completed_units,
                            total_units=total_units,
                            incremental_output_dir=incremental_output_dir,
                            incremental_part_index=incremental_part_index,
                            schemas=schemas,
                            emit_progress=(
                                None
                                if progress_callback is None
                                else lambda payload: _emit_progress(
                                    str(payload.pop("phase")), **payload
                                )
                            ),
                        )
                    )
                    scoring.rows_out = source_rows_scored

                if progress_callback is not None and source_rows_scored < source_rows:
                    _emit_progress(
                        "country_chunk",
                        source_system=source_system,
                        target_system=target_system,
                        country=country,
                        source_rows_scored=source_rows,
                        source_rows_total=source_rows,
                        chunk_edges_found=0,
                        completed_units=completed_units,
                        total_units=total_units,
                    )

                edges = (
                    pl.concat(country_edge_parts, how="vertical_relaxed")
                    if country_edge_parts
                    else _empty_frame(schema=schemas["source_outcomes"])
                )
                matched_edges = edges.filter(
                    (pl.col("match_status") == "matched")
                    & pl.col("target_id").is_not_null()
                )

                matched_source_ids = matched_edges.select(pl.col("source_id")).unique()
                unmatched_edges = (
                    source_name_map.join(
                        matched_source_ids,
                        on="source_id",
                        how="anti",
                    )
                    .join(
                        source_truth_map,
                        on="source_id",
                        how="left",
                    )
                    .with_columns(
                        pl.lit(None, dtype=pl.Utf8).alias("target_id"),
                        pl.lit(None, dtype=pl.Utf8).alias("target_name"),
                        pl.lit(None, dtype=pl.Float64).alias("similarity"),
                        pl.lit(None, dtype=pl.Int32).alias("rank"),
                        pl.when(pl.col("source_match_uri").is_not_null())
                        .then(pl.lit(False, dtype=pl.Boolean))
                        .otherwise(pl.lit(None, dtype=pl.Boolean))
                        .alias("is_true_match"),
                        pl.when(pl.col("source_match_uri").is_not_null())
                        .then(pl.lit("false_match"))
                        .otherwise(pl.lit("unknown"))
                        .alias("true_match_status"),
                        pl.lit("unmatched").alias("match_status"),
                        pl.lit("no_match_above_threshold").alias("reason_code"),
                    )
                    .select(list(schemas["source_outcomes"].keys()))
                )

                if unmatched_edges.height:
                    edges = pl.concat([edges, unmatched_edges], how="vertical_relaxed")
                    incremental_part_index["source_outcomes"] += 1
                    _write_incremental_artifact_part(
                        incremental_output_dir=incremental_output_dir,
                        artifact_name="source_outcomes",
                        part_index=incremental_part_index["source_outcomes"],
                        frame=unmatched_edges,
                    )

                edge_count = int(matched_edges.height)
                source_rows_with_match = (
                    int(matched_source_ids.height) if edge_count else 0
                )
                pair_edges_found += edge_count
                pair_source_rows_with_match += source_rows_with_match
                edges_out = edges.select(list(schemas["source_outcomes"].keys()))
                source_outcome_parts.append(edges_out)

                components = build_connected_components(
                    matched_edges.select("source_id", "target_id"),
                    source_id_col="source_id",
                    target_id_col="target_id",
                )

                clusters = (
                    components.join(node_info, on="node_id", how="left")
                    .with_columns(
                        pl.lit(run_id).alias("run_id"),
                        pl.lit(source_system).alias("source_system"),
                        pl.lit(target_system).alias("target_system"),
                        pl.lit(country).alias("country"),
                    )
                    .select(list(schemas["clusters"].keys()))
                )
                cluster_parts.append(clusters)
                incremental_part_index["clusters"] += 1
                _write_incremental_artifact_part(
                    incremental_output_dir=incremental_output_dir,
                    artifact_name="clusters",
                    part_index=incremental_part_index["clusters"],
                    frame=clusters,
                )

                coverage = compute_directional_coverage(
                    source_nodes=source_nodes,
                    target_nodes=target_nodes,
                    clusters=components,
                    node_id_col="node_id",
                    cluster_id_col="cluster_id",
                    system_col="system",
                ).with_columns(
                    pl.lit(country).alias("country"),
                    pl.lit(
                        int(source_stats.total_rows_by_country.get(country, 0))
                    ).alias("source_records_total"),
                    pl.lit(
                        int(source_stats.filtered_out_rows_by_country.get(country, 0))
                    ).alias("source_records_filtered_out"),
                )
                coverage_out = coverage.select(
                    list(schemas["directional_coverage"].keys())
                )
                coverage_parts.append(coverage_out)
                incremental_part_index["directional_coverage"] += 1
                _write_incremental_artifact_part(
                    incremental_output_dir=incremental_output_dir,
                    artifact_name="directional_coverage",
                    part_index=incremental_part_index["directional_coverage"],
                    frame=coverage_out,
                )

                shape_metrics = compute_cluster_shape_metrics(
                    clusters=components
                ).with_columns(pl.lit(country).alias("country"))
                shape_out = shape_metrics.select(list(schemas["cluster_shape"].keys()))
                shape_parts.append(shape_out)
                incremental_part_index["cluster_shape"] += 1
                _write_incremental_artifact_part(
                    incremental_output_dir=incremental_output_dir,
                    artifact_name="cluster_shape",
                    part_index=incremental_part_index["cluster_shape"],
                    frame=shape_out,
                )

                pair_truth_eval = compute_pair_truth_eval(
                    source_truth_map=source_truth_map,
                    matched_edges=matched_edges,
                    source_system=source_system,
                    target_system=target_system,
                    country=country,
                    target_rows=target_rows,
                    source_name_forms=source_name_forms,
                    target_name_forms=target_name_forms,
                ).select(list(schemas["pair_truth_eval"].keys()))
                pair_truth_eval_parts.append(pair_truth_eval)
                incremental_part_index["pair_truth_eval"] += 1
                _write_incremental_artifact_part(
                    incremental_output_dir=incremental_output_dir,
                    artifact_name="pair_truth_eval",
                    part_index=incremental_part_index["pair_truth_eval"],
                    frame=pair_truth_eval,
                )

                run_metrics = compute_run_metrics(
                    candidate_edges=matched_edges.select("source_id", "target_id"),
                    source_system=source_system,
                    target_system=target_system,
                    country=country,
                    source_rows=source_rows,
                    elapsed_seconds=time.perf_counter() - country_started,
                ).select(list(schemas["run_metrics"].keys()))
                run_metrics_parts.append(run_metrics)
                incremental_part_index["run_metrics"] += 1
                _write_incremental_artifact_part(
                    incremental_output_dir=incremental_output_dir,
                    artifact_name="run_metrics",
                    part_index=incremental_part_index["run_metrics"],
                    frame=run_metrics,
                )

                _emit_progress(
                    "country",
                    source_system=source_system,
                    target_system=target_system,
                    country=country,
                    source_rows=source_rows,
                    target_rows=target_rows,
                    source_rows_with_match=source_rows_with_match,
                    edges_found=edge_count,
                    source_rows_processed_cumulative=cumulative_source_rows_processed
                    + pair_source_rows_processed,
                    source_rows_with_match_cumulative=cumulative_source_rows_with_match
                    + pair_source_rows_with_match,
                    edges_found_cumulative=cumulative_edges_found + pair_edges_found,
                    target_rows_cumulative=cumulative_target_rows + pair_target_rows,
                    completed_units=completed_units,
                    total_units=total_units,
                )

            cumulative_source_rows_processed += pair_source_rows_processed
            cumulative_source_rows_with_match += pair_source_rows_with_match
            cumulative_edges_found += pair_edges_found
            cumulative_target_rows += pair_target_rows
            completed_units += 1

            _emit_pair_progress_event(
                emit_progress=_emit_progress,
                started_at=started_at,
                source_system=source_system,
                target_system=target_system,
                countries=countries,
                pair_source_rows_processed=pair_source_rows_processed,
                pair_target_rows=pair_target_rows,
                pair_source_rows_with_match=pair_source_rows_with_match,
                pair_edges_found=pair_edges_found,
                cumulative_source_rows_processed=cumulative_source_rows_processed,
                cumulative_source_rows_with_match=cumulative_source_rows_with_match,
                cumulative_edges_found=cumulative_edges_found,
                cumulative_target_rows=cumulative_target_rows,
                completed_units=completed_units,
                total_units=total_units,
            )

    pair_truth_eval_out = _build_pair_truth_eval_output(
        pair_truth_eval_parts=pair_truth_eval_parts,
        schemas=schemas,
    )
    run_metrics_out = _build_run_metrics_output(
        run_metrics_parts=run_metrics_parts,
        schemas=schemas,
    )

    return {
        "source_outcomes": (
            pl.concat(source_outcome_parts, how="vertical_relaxed")
            if source_outcome_parts
            else _empty_frame(schema=schemas["source_outcomes"])
        ),
        "clusters": pl.concat(cluster_parts, how="vertical_relaxed")
        if cluster_parts
        else _empty_frame(schema=schemas["clusters"]),
        "directional_coverage": (
            pl.concat(coverage_parts, how="vertical_relaxed")
            if coverage_parts
            else _empty_frame(schema=schemas["directional_coverage"])
        ),
        "cluster_shape": pl.concat(shape_parts, how="vertical_relaxed")
        if shape_parts
        else _empty_frame(schema=schemas["cluster_shape"]),
        "pair_truth_eval": pair_truth_eval_out,
        "run_metrics": run_metrics_out,
        "exceptions": _empty_frame(schema=schemas["exceptions"]),
        "timings": telemetry.frame(),
    }
