"""The Shard stage: split a system's acquired source into parquet chunks.

The orchestration here is generic; anything specific to one source lives in its own `sharding_<system>.py` module or handler, so parsing one source is tested and changed without touching another. After writing, shard schemas are harmonised, with a stable column order and nulls for fields a sparse chunk lacks, so a downstream reader never fails on per-chunk schema drift.
"""

from __future__ import annotations

# isort: skip_file

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from workspace.data_layout import system_layer_dir
from workspace.roots import WorkspaceRoots

from .constants_status import (
    RUNNABLE_SYSTEM_STATUSES,
    STATUS_LIVE,
    SUPPORTED_COUNTRY_SELECTOR_LIVE,
    SUPPORTED_COUNTRY_SELECTOR_LOCAL,
    SUPPORTED_COUNTRY_SELECTOR_SUPPORTED,
    require_runnable_plan,
)
from .downloader_wikidata import extract_wikidata_company_projection_two_pass
from .output_chunking import write_chunked_output_files
from .plan_registry import COUNTRY_REGISTRY, get_system_plan
from .sharding_batch_processor import ShardBatchProcessor
from .sharding_perf import ShardPerformance
from .sharding_system_handlers_impl import resolve_handler

_GENERIC_COUNTRY_COLUMN_CANDIDATES = (
    "jurisdiction_code",
    "country",
    "Country",
    "country_code",
    "CountryCode",
)

# Keep canonical contract columns in main shards when the source already exposes
# them by identity so source->canonical handoff does not depend on explicit
# self-mappings in system metadata.
_CANONICAL_IDENTITY_MAIN_COLUMN_CANDIDATES = (
    "name",
    "company_number",
    "company_type",
    "current_status",
    "incorporation_date",
    "dissolution_date",
    "inactive",
    "branch",
    "branch_status",
    "registered_address_in_full",
    "registry_url",
    "match_uri",
    "alternative_names",
    "previous_names",
)

_SYSTEM_URI_COLUMN = "system_uri"


@dataclass(frozen=True)
class PreparedShardResource:
    resource: Any
    output_dir: Path
    source_path: Path
    source_format_override: str | None = None


@dataclass
class _StreamingParquetWriter:
    """Stages every appended batch immediately to a transient `chunks/`
    scratch directory (write-through, no in-memory accumulation across
    calls), then repacks the staged files into the final chunk_size-bounded
    (or single, if chunk_size is None) output files on finalize(). This gives
    intermediate durability and progress visibility during the run -- a kill
    mid-run leaves recoverable staged chunks and an untouched previous
    output_dir, never a half-written or partially-overwritten final file
    (previously true even when chunk_size was set, since final chunks were
    written straight into output_dir as they filled).
    """

    output_dir: Path
    prefix: str
    chunk_size: int | None
    progress: Callable[[str], None] | None = None
    telemetry: ShardPerformance | None = None
    _staged_paths: list[Path] | None = None
    _staged_index: int = 1

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._staged_paths = []

    @property
    def _chunks_dir(self) -> Path:
        return self.output_dir / "chunks"

    def append(self, frame: pl.DataFrame) -> None:
        if frame.height == 0:
            return

        write_started_at = time.perf_counter()
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        staged_path = (
            self._chunks_dir / f"{self.prefix}-{self._staged_index:03d}.parquet"
        )
        frame.write_parquet(staged_path, compression="snappy")
        elapsed_s = max(time.perf_counter() - write_started_at, 1e-9)

        if self._staged_paths is not None:
            self._staged_paths.append(staged_path)
        if self.telemetry is not None:
            self.telemetry.increment_kept_rows(frame.height)
            self.telemetry.add_write_elapsed(elapsed_s)
            self.telemetry.record_chunk_write(
                frame.height, elapsed_s, self._staged_index
            )
        elif self.progress is not None:
            self.progress(
                f"[{self.prefix}] staged chunk {self._staged_index}: {frame.height:,} rows"
            )
        self._staged_index += 1

    def finalize(self) -> list[Path]:
        staged_paths = list(self._staged_paths or [])
        written_paths = write_chunked_output_files(
            chunk_sources=staged_paths,
            output_dir=self.output_dir,
            prefix=self.prefix,
            chunk_size=self.chunk_size,
            frame_transformer=pl.read_parquet,
        )
        for staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
        try:
            self._chunks_dir.rmdir()
        except OSError:
            pass
        return written_paths


def _is_shard_output_file(*, file_name: str, system_code: str) -> bool:
    return file_name.startswith(f"{system_code}-") and file_name.endswith(".parquet")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _resolve_system_field_candidates(plan: Any) -> dict[str, list[str]]:
    raw_candidates = getattr(plan, "system_field_candidates", None)
    if not raw_candidates:
        return {}

    return {
        str(field_name): [candidate for candidate in candidates]
        for field_name, candidates in raw_candidates
    }


def _preferred_main_columns(
    *,
    frame: pl.DataFrame,
    system_code: str,
    system_field_candidates: dict[str, list[str]],
    canonical_input_columns: tuple[str, ...] | None,
) -> list[str]:
    columns = list(frame.columns)
    field_candidates = system_field_candidates

    preferred: list[str] = []
    for candidates in field_candidates.values():
        for column_name in candidates:
            if column_name in columns and column_name not in preferred:
                preferred.append(column_name)

    resolve_handler(system_code).extend_preferred_main_columns(
        columns=columns,
        preferred=preferred,
    )

    if canonical_input_columns:
        for column_name in canonical_input_columns:
            if column_name in columns and column_name not in preferred:
                preferred.append(column_name)

    for column_name in _GENERIC_COUNTRY_COLUMN_CANDIDATES:
        if column_name in columns and column_name not in preferred:
            preferred.append(column_name)

    for column_name in _CANONICAL_IDENTITY_MAIN_COLUMN_CANDIDATES:
        if column_name in columns and column_name not in preferred:
            preferred.append(column_name)

    if _SYSTEM_URI_COLUMN in columns and _SYSTEM_URI_COLUMN not in preferred:
        preferred.append(_SYSTEM_URI_COLUMN)

    return preferred


def _alias_source_columns_for_main(
    *,
    frame: pl.DataFrame,
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None,
) -> list[str]:
    if not canonical_source_column_aliases:
        return []

    keep_columns: list[str] = []
    for source_name, _target_name in canonical_source_column_aliases:
        if source_name in frame.columns and source_name not in keep_columns:
            keep_columns.append(source_name)
    return keep_columns


def _normalized_text_expr(expr: pl.Expr) -> pl.Expr:
    value = expr.cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(value == "").then(pl.lit(None, dtype=pl.Utf8)).otherwise(value)


def _coalesce_columns_expr(frame: pl.DataFrame, candidates: tuple[str, ...]) -> pl.Expr:
    present = [
        column_name for column_name in candidates if column_name in frame.columns
    ]
    if not present:
        return pl.lit(None, dtype=pl.Utf8)
    return pl.coalesce(
        [_normalized_text_expr(pl.col(column_name)) for column_name in present]
    )


def _row_fingerprint_expr(frame: pl.DataFrame) -> pl.Expr:
    fingerprint_columns = [
        column_name
        for column_name in frame.columns
        if column_name != _SYSTEM_URI_COLUMN
    ]

    if not fingerprint_columns:
        return pl.lit("0", dtype=pl.Utf8)

    return (
        pl.struct([pl.col(column_name) for column_name in fingerprint_columns])
        .hash(seed=20260628)
        .cast(pl.Utf8)
    )


def _with_system_uri(
    *,
    frame: pl.DataFrame,
    system_code: str,
    identifier_candidates: tuple[str, ...] | None,
) -> pl.DataFrame:
    key_expr = _coalesce_columns_expr(frame, identifier_candidates or ())
    # Make key components URI-safe and deterministic across systems.
    safe_key_expr = (
        key_expr.str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9._~-]+", "_")
        .str.strip_chars("_")
    )
    safe_key_expr = (
        pl.when(safe_key_expr == "")
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(safe_key_expr)
    )
    is_source_uri_expr = key_expr.str.contains(r"^[A-Za-z][A-Za-z0-9+.-]*://")
    direct_or_formatted_key_expr = (
        pl.when(is_source_uri_expr).then(key_expr).otherwise(safe_key_expr)
    )

    fingerprint_expr = _row_fingerprint_expr(frame)
    system_prefix = pl.lit(f"{system_code}://")

    id_uri_expr = (
        pl.when(direct_or_formatted_key_expr.is_not_null())
        .then(
            pl.when(is_source_uri_expr)
            .then(direct_or_formatted_key_expr)
            .otherwise(pl.concat_str([system_prefix, direct_or_formatted_key_expr]))
        )
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    fallback_uri_expr = pl.concat_str([system_prefix, pl.lit("fp/"), fingerprint_expr])

    candidate_column = "_candidate_system_uri"
    frame_with_candidate = frame.with_columns(id_uri_expr.alias(candidate_column))
    system_uri_expr = (
        pl.when(pl.col(candidate_column).is_not_null())
        .then(pl.col(candidate_column))
        .otherwise(fallback_uri_expr)
    )

    result = frame_with_candidate.with_columns(
        system_uri_expr.alias(_SYSTEM_URI_COLUMN)
    ).drop(candidate_column)
    ordered_columns = [_SYSTEM_URI_COLUMN] + [
        column_name
        for column_name in result.columns
        if column_name != _SYSTEM_URI_COLUMN
    ]
    return result.select([pl.col(column_name) for column_name in ordered_columns])


def _default_system_uri_materializer(
    frame: pl.DataFrame,
    system_code: str,
    identifier_candidates: tuple[str, ...] | None,
) -> pl.DataFrame:
    return _with_system_uri(
        frame=frame,
        system_code=system_code,
        identifier_candidates=identifier_candidates,
    )


# Sidecar (and some main-shard, e.g. GB's RegAddress.*) columns keep the
# source system's original headers verbatim, dots included (e.g. Companies
# House's "Accounts.AccountRefDay"). DuckDB-backed tools -- including the
# VS Code parquet-explorer extension -- parse an unquoted dotted identifier
# as a table.column reference and fail to resolve it, which can render as a
# blank/missing column rather than an error. The data is present; query the
# column quoted (SELECT "Accounts.AccountRefDay" ...) or via SELECT * to see it.
def _split_main_sidecar_frame(
    *,
    frame: pl.DataFrame,
    system_code: str,
    system_field_candidates: dict[str, list[str]],
    canonical_input_columns: tuple[str, ...] | None,
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    main_columns = _preferred_main_columns(
        frame=frame,
        system_code=system_code,
        system_field_candidates=system_field_candidates,
        canonical_input_columns=canonical_input_columns,
    )
    alias_keep_columns = _alias_source_columns_for_main(
        frame=frame,
        canonical_source_column_aliases=canonical_source_column_aliases,
    )
    for column_name in alias_keep_columns:
        if column_name not in main_columns:
            main_columns.append(column_name)

    if not main_columns:
        return frame, frame

    if _SYSTEM_URI_COLUMN in main_columns:
        main_columns = [_SYSTEM_URI_COLUMN] + [
            column_name
            for column_name in main_columns
            if column_name != _SYSTEM_URI_COLUMN
        ]

    sidecar_columns = [
        column_name for column_name in frame.columns if column_name not in main_columns
    ]
    if (
        _SYSTEM_URI_COLUMN in frame.columns
        and _SYSTEM_URI_COLUMN not in sidecar_columns
    ):
        sidecar_columns.append(_SYSTEM_URI_COLUMN)
    if _SYSTEM_URI_COLUMN in sidecar_columns:
        sidecar_columns = [_SYSTEM_URI_COLUMN] + [
            column_name
            for column_name in sidecar_columns
            if column_name != _SYSTEM_URI_COLUMN
        ]
    if not sidecar_columns:
        return frame, frame

    return frame.select(main_columns), frame.select(sidecar_columns)


def _sidecar_path_for(*, path: Path, system_code: str) -> Path:
    prefix = f"{system_code}-"
    if not path.name.startswith(prefix):
        return path.with_name(f"{system_code}-sidecar-{path.name}")
    suffix = path.name[len(prefix) :]
    return path.with_name(f"{system_code}-sidecar-{suffix}")


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=path.suffix,
        prefix=f"{path.stem}.rewrite.",
        dir=path.parent,
        delete=False,
    ) as tmp_file:
        temp_path = Path(tmp_file.name)

    try:
        frame.write_parquet(temp_path, compression="snappy")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _materialize_sidecar_outputs(
    *,
    main_paths: list[Path],
    system_code: str,
    identifier_candidates: tuple[str, ...] | None,
    system_field_candidates: dict[str, list[str]],
    canonical_input_columns: tuple[str, ...] | None,
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    sidecar_paths: list[Path] = []
    for path in main_paths:
        frame = pl.read_parquet(path)
        main_frame, sidecar_frame = _split_main_sidecar_frame(
            frame=frame,
            system_code=system_code,
            system_field_candidates=system_field_candidates,
            canonical_input_columns=canonical_input_columns,
            canonical_source_column_aliases=canonical_source_column_aliases,
        )
        _write_parquet_atomic(main_frame, path)
        sidecar_path = _sidecar_path_for(path=path, system_code=system_code)
        _write_parquet_atomic(sidecar_frame, sidecar_path)
        sidecar_paths.append(sidecar_path)

    if progress is not None and sidecar_paths:
        progress(
            f"[{system_code}] sidecar split complete: "
            f"main shards={len(main_paths)}, sidecar shards={len(sidecar_paths)}"
        )

    return sidecar_paths


def _materialize_system_uri_outputs(
    *,
    main_paths: list[Path],
    system_code: str,
    identifier_candidates: tuple[str, ...] | None,
    progress: Callable[[str], None] | None = None,
) -> None:
    rewritten = 0
    for path in main_paths:
        frame = _with_system_uri(
            frame=pl.read_parquet(path),
            system_code=system_code,
            identifier_candidates=identifier_candidates,
        )
        _write_parquet_atomic(frame, path)
        rewritten += 1

    if progress is not None and rewritten:
        progress(
            f"[{system_code}] materialized system_uri for {rewritten} shard file(s)"
        )


def _resolve_supported_country_codes(plan) -> set[str] | None:
    policy = getattr(plan, "supported_countries", None)
    if policy is None:
        return None

    if not isinstance(policy, tuple):
        raise TypeError(f"{plan.code}: supported_countries must be a list or null")

    resolved_codes: set[str] = set()
    for item in policy:
        selector = item.strip().lower()

        if selector == SUPPORTED_COUNTRY_SELECTOR_LOCAL:
            if plan.code not in COUNTRY_REGISTRY:
                raise ValueError(
                    f"{plan.code}: supported_countries=['local'] requires a country registry plan"
                )
            resolved_codes.add(plan.code.upper())
            continue

        if selector == SUPPORTED_COUNTRY_SELECTOR_LIVE:
            resolved_codes.update(
                {
                    code.upper()
                    for code, country_plan in COUNTRY_REGISTRY.items()
                    if country_plan.status == STATUS_LIVE
                }
            )
            continue

        if selector == SUPPORTED_COUNTRY_SELECTOR_SUPPORTED:
            resolved_codes.update(
                {
                    code.upper()
                    for code, country_plan in COUNTRY_REGISTRY.items()
                    if country_plan.status in RUNNABLE_SYSTEM_STATUSES
                }
            )
            continue

        if selector:
            resolved_codes.add(selector.upper())

    return resolved_codes


def _filter_frame_by_supported_countries(
    frame: pl.DataFrame,
    supported_country_codes: set[str] | None,
) -> pl.DataFrame:
    if supported_country_codes is None or frame.height == 0:
        return frame

    available_columns = [
        column_name
        for column_name in _GENERIC_COUNTRY_COLUMN_CANDIDATES
        if column_name in frame.columns
    ]
    if not available_columns:
        return frame

    allowed_values = sorted(supported_country_codes)
    filter_expression = None
    for column_name in available_columns:
        expr = (
            pl.col(column_name)
            .cast(pl.Utf8, strict=False)
            .str.to_uppercase()
            .is_in(allowed_values)
        )
        filter_expression = (
            expr if filter_expression is None else (filter_expression | expr)
        )
    if filter_expression is None:
        return frame
    return frame.filter(filter_expression)


def _resolve_source_artifact(
    *,
    source_root: Path,
    resource,
    effective_run_date: str,
    run_date_provided: bool,
) -> tuple[Path, Path]:
    snapshot_date = resource.resolve_snapshot_date(effective_run_date)
    source_dir = source_root / snapshot_date
    source_path = source_dir / resource.resolve_file_name(effective_run_date)
    if source_path.exists():
        return source_dir, source_path

    if run_date_provided:
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if not source_root.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    snapshot_dirs = sorted(path for path in source_root.glob("*") if path.is_dir())
    for candidate_dir in reversed(snapshot_dirs):
        candidate_run_date = candidate_dir.name
        candidate_path = candidate_dir / resource.resolve_file_name(candidate_run_date)
        if candidate_path.exists():
            return candidate_dir, candidate_path

    raise FileNotFoundError(f"Source file not found: {source_path}")


def _parse_positive_int_stage_option(
    stage_options: dict[str, object] | None, key: str
) -> int | None:
    value = (stage_options or {}).get(key)
    if value is None:
        return None
    if not isinstance(value, int | str):
        raise TypeError(f"shard.{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"shard.{key} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"shard.{key} must be greater than 0")
    return parsed


def _prepare_single_resource_shard_input(
    *,
    resource,
    plan,
    acquire_root: Path,
    source_root: Path,
    prepare_root: Path,
    roots: WorkspaceRoots,
    effective_run_date: str,
    run_date_provided: bool,
    max_companies_int: int | None,
    effective_max_lines: int | None,
    prepare_wiring_mode_str: str | None,
    prepare_stream_resume: bool,
    projection_defaults_override: dict[str, object] | None,
    emit: Callable[[str], None],
    force: bool = False,
) -> PreparedShardResource:
    try:
        snapshot_dir, resolved_source_path = _resolve_source_artifact(
            source_root=acquire_root,
            resource=resource,
            effective_run_date=effective_run_date,
            run_date_provided=run_date_provided,
        )
    except FileNotFoundError:
        snapshot_dir, resolved_source_path = _resolve_source_artifact(
            source_root=source_root,
            resource=resource,
            effective_run_date=effective_run_date,
            run_date_provided=run_date_provided,
        )

    output_dir = source_root / snapshot_dir.name
    source_path = resolved_source_path

    snapshot_dir, source_path, source_format_override = resolve_handler(
        plan.code
    ).prepare_and_resolve_resource_source(
        plan_code=plan.code,
        resource=resource,
        snapshot_dir=snapshot_dir,
        source_path=source_path,
        prepare_root=prepare_root,
        effective_run_date=effective_run_date,
        run_date_provided=run_date_provided,
        roots=roots,
        max_companies=max_companies_int,
        max_lines=effective_max_lines,
        prepare_wiring_mode=prepare_wiring_mode_str or "stream",
        prepare_stream_resume=prepare_stream_resume,
        projection_defaults_override=projection_defaults_override,
        emit=emit,
        resolve_source_artifact=_resolve_source_artifact,
        extract_two_pass=extract_wikidata_company_projection_two_pass,
        force=force,
    )

    output_dir = source_root / snapshot_dir.name

    return PreparedShardResource(
        resource=resource,
        output_dir=output_dir,
        source_path=source_path,
        source_format_override=source_format_override,
    )


def prepare_system_shard_inputs(
    *,
    plan,
    roots: WorkspaceRoots,
    effective_run_date: str,
    run_date_provided: bool,
    stage_options: dict[str, object] | None,
    emit: Callable[[str], None],
) -> list[PreparedShardResource]:
    acquire_root = system_layer_dir(roots, plan.code, layer="acquire")
    prepare_root = system_layer_dir(roots, plan.code, layer="prepare")
    source_root = system_layer_dir(roots, plan.code, layer="source")

    prepared_resources: list[PreparedShardResource] = []
    max_companies_int = _parse_positive_int_stage_option(stage_options, "max_companies")
    effective_max_lines = _parse_positive_int_stage_option(stage_options, "max_lines")
    force = _to_bool((stage_options or {}).get("force", False))

    prepare_wiring_mode = (stage_options or {}).get("prepare_wiring_mode")
    prepare_wiring_mode_str: str | None = None
    if prepare_wiring_mode is not None:
        if not isinstance(prepare_wiring_mode, str):
            raise ValueError("shard.prepare_wiring_mode must be a string")
        prepare_wiring_mode_str = prepare_wiring_mode.strip().lower()

    prepare_stream_resume = _to_bool(
        (stage_options or {}).get("prepare_stream_resume", True)
    )

    projection_engine = (stage_options or {}).get("projection_engine")
    projection_defaults_override: dict[str, object] | None = None
    if projection_engine is not None:
        if not isinstance(projection_engine, str):
            raise ValueError("shard.projection_engine must be a string")
        normalized_projection_engine = projection_engine.strip().lower()
        if normalized_projection_engine not in {"python", "wikisieve"}:
            raise ValueError(
                "shard.projection_engine must be one of: python, wikisieve"
            )
        projection_defaults_override = {"engine": normalized_projection_engine}

    for resource in plan.resources:
        prepared_resources.append(
            _prepare_single_resource_shard_input(
                resource=resource,
                plan=plan,
                acquire_root=acquire_root,
                source_root=source_root,
                prepare_root=prepare_root,
                roots=roots,
                effective_run_date=effective_run_date,
                run_date_provided=run_date_provided,
                max_companies_int=max_companies_int,
                effective_max_lines=effective_max_lines,
                prepare_wiring_mode_str=prepare_wiring_mode_str,
                prepare_stream_resume=prepare_stream_resume,
                projection_defaults_override=projection_defaults_override,
                emit=emit,
                force=force,
            )
        )

    return prepared_resources


def _harmonize_shard_schemas(shard_paths: list[Path]) -> int:
    if not shard_paths:
        return 0

    ordered_columns: list[str] = []
    seen_columns: set[str] = set()
    schema_by_path: dict[Path, tuple[str, ...]] = {}

    for path in shard_paths:
        schema_cols = tuple(pl.read_parquet_schema(path).keys())
        schema_by_path[path] = schema_cols
        for col in schema_cols:
            if col not in seen_columns:
                seen_columns.add(col)
                ordered_columns.append(col)

    rewrites = 0
    for path in shard_paths:
        current_cols = schema_by_path[path]
        if tuple(ordered_columns) == current_cols:
            continue

        frame = pl.read_parquet(path)
        missing = [col for col in ordered_columns if col not in frame.columns]
        if missing:
            frame = frame.with_columns([pl.lit(None).alias(col) for col in missing])
        frame = frame.select([pl.col(col) for col in ordered_columns])
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=path.suffix,
            prefix=f"{path.stem}.harmonize.",
            dir=path.parent,
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            frame.write_parquet(temp_path, compression="snappy")
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        rewrites += 1

    return rewrites


def _parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def _resolve_shard_plan_context(
    *,
    system: str,
    run_date: str | None,
    sidecar: bool,
    allow_research: bool,
    stage_options: dict[str, object] | None,
    emit: Callable[[str], None],
) -> tuple[Any, bool, str, bool, set[str] | None]:
    plan = get_system_plan(system)

    sidecar_enabled = sidecar or _to_bool((stage_options or {}).get("sidecar", False))
    effective_run_date = run_date or datetime.now().astimezone().date().isoformat()
    run_date_provided = run_date is not None

    require_runnable_plan(plan, allow_research=allow_research, stage="shard")

    supported_country_codes = _resolve_supported_country_codes(plan)
    return (
        plan,
        sidecar_enabled,
        effective_run_date,
        run_date_provided,
        supported_country_codes,
    )


def _emit_shard_start_message(
    *,
    emit: Callable[[str], None],
    plan_code: str,
    effective_run_date: str,
    chunk_size: int,
    sidecar_enabled: bool,
) -> None:
    emit(
        f"[{plan_code}] start sharding run_date={effective_run_date} "
        f"chunk_size={chunk_size} sidecar={sidecar_enabled}"
    )


def _cleanup_output_stale_shards(
    *,
    output_dirs: set[Path],
    plan_code: str,
    roots: WorkspaceRoots,
    emit: Callable[[str], None],
) -> None:
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        removed_count = 0
        for stale_file in output_dir.glob(f"{plan_code}-*.parquet"):
            stale_file.unlink()
            removed_count += 1
        if removed_count:
            emit(
                f"[{plan_code}] removed {removed_count} stale shard(s) from "
                + output_dir.relative_to(roots.data).as_posix()
            )
        stale_chunks_dir = output_dir / "chunks"
        if stale_chunks_dir.exists():
            shutil.rmtree(stale_chunks_dir, ignore_errors=True)
            emit(
                f"[{plan_code}] removed stale scratch chunks from "
                + stale_chunks_dir.relative_to(roots.data).as_posix()
            )


def _handle_custom_system_sharding(
    *,
    plan,
    resolved_resources: list[tuple[object, Path, Path]],
    chunk_size: int,
    stage_options: dict[str, object] | None,
    emit: Callable[[str], None],
    sidecar_enabled: bool,
) -> list[Path] | None:
    handler = resolve_handler(plan.code)
    custom_system_paths = handler.try_shard_system_with_custom_logic(
        plan_code=plan.code,
        resolved_resources=resolved_resources,
        chunk_size=chunk_size,
        stage_options=stage_options,
        emit=emit,
        sidecar_enabled=sidecar_enabled,
        materialize_system_uri=lambda frame: handler.materialize_system_uri_for_system(
            frame=frame,
            system_code=plan.code,
            identifier_candidates=plan.system_uri_identifier_candidates,
            default_materializer=_default_system_uri_materializer,
        ),
        split_main_sidecar=lambda frame: _split_main_sidecar_frame(
            frame=frame,
            system_code=plan.code,
            system_field_candidates=_resolve_system_field_candidates(plan),
            canonical_input_columns=plan.canonical_input_columns,
            canonical_source_column_aliases=plan.canonical_source_column_aliases,
        ),
    )
    if custom_system_paths is None:
        return None

    written_paths = list(custom_system_paths.main_paths)
    sidecar_paths = list(custom_system_paths.sidecar_paths)

    rewrites = _harmonize_shard_schemas(written_paths)
    if rewrites:
        emit(f"[{plan.code}] harmonized shard schemas across {rewrites} shard file(s)")

    _emit_shard_completion(
        emit=emit,
        plan_code=plan.code,
        written_paths=written_paths,
        sidecar_enabled=sidecar_enabled,
        sidecar_paths=sidecar_paths,
    )
    return written_paths


def _emit_shard_completion(
    *,
    emit: Callable[[str], None],
    plan_code: str,
    written_paths: list[Path],
    sidecar_enabled: bool,
    sidecar_paths: list[Path],
) -> None:
    if sidecar_enabled and sidecar_paths:
        sidecar_rewrites = _harmonize_shard_schemas(sidecar_paths)
        if sidecar_rewrites:
            emit(
                f"[{plan_code}] harmonized sidecar schemas across "
                f"{sidecar_rewrites} shard file(s)"
            )
        emit(
            f"[{plan_code}] sidecar shard naming prefix: "
            f"{plan_code}-sidecar-<index>.parquet"
        )

    emit(f"[{plan_code}] completed sharding: total shards={len(written_paths)}")


def _shard_prepared_resource(
    *,
    plan,
    prepared,
    chunk_size: int,
    supported_country_codes: set[str] | None,
    stage_options: dict[str, object] | None,
    sidecar_enabled: bool,
    system_field_candidates,
    emit: Callable[[str], None],
) -> tuple[list[Path], list[Path]]:
    canonical_input_columns = getattr(plan, "canonical_input_columns", None)
    canonical_source_column_aliases = getattr(
        plan, "canonical_source_column_aliases", None
    )
    resolve_chunk_size = getattr(prepared.resource, "resolve_output_chunk_size", None)
    write_chunk_size = (
        resolve_chunk_size(chunk_size) if callable(resolve_chunk_size) else chunk_size
    )
    if write_chunk_size != chunk_size:
        emit(
            f"[{plan.code}] resource '{prepared.resource.name}' overrides shard "
            f"chunk_size={chunk_size} -> {write_chunk_size}"
        )

    processor = ShardBatchProcessor(
        plan=plan,
        chunk_size=chunk_size,
        write_chunk_size=write_chunk_size,
        supported_country_codes=supported_country_codes,
        stage_options=stage_options,
        sidecar_enabled=sidecar_enabled,
        system_field_candidates=system_field_candidates,
        emit=emit,
        canonical_input_columns=canonical_input_columns,
        canonical_source_column_aliases=canonical_source_column_aliases,
        default_system_uri_materializer=_default_system_uri_materializer,
        split_main_sidecar_frame=_split_main_sidecar_frame,
        filter_frame_by_supported_countries=_filter_frame_by_supported_countries,
        writer_factory=_StreamingParquetWriter,
    )
    return processor.shard_prepared_resource(prepared=prepared)


def shard_system_source(
    system: str,
    *,
    roots: WorkspaceRoots,
    run_date: str | None = None,
    chunk_size: int = 100_000,
    sidecar: bool = False,
    allow_research: bool = False,
    stage_options: dict[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    (
        plan,
        sidecar_enabled,
        effective_run_date,
        run_date_provided,
        supported_country_codes,
    ) = _resolve_shard_plan_context(
        system=system,
        run_date=run_date,
        sidecar=sidecar,
        allow_research=allow_research,
        stage_options=stage_options,
        emit=emit,
    )
    _emit_shard_start_message(
        emit=emit,
        plan_code=plan.code,
        effective_run_date=effective_run_date,
        chunk_size=chunk_size,
        sidecar_enabled=sidecar_enabled,
    )

    if supported_country_codes is not None:
        country_list = ", ".join(sorted(supported_country_codes))
        emit(f"[{plan.code}] applying country filter: {country_list}")

    prepared_resources = prepare_system_shard_inputs(
        plan=plan,
        roots=roots,
        effective_run_date=effective_run_date,
        run_date_provided=run_date_provided,
        stage_options=stage_options,
        emit=emit,
    )
    for prepared in prepared_resources:
        emit(
            f"[{plan.code}] using source file "
            + prepared.source_path.relative_to(roots.data).as_posix()
        )

    resolved_resources = [
        (prepared.resource, prepared.output_dir, prepared.source_path)
        for prepared in prepared_resources
    ]
    output_dirs = {prepared.output_dir for prepared in prepared_resources}
    _cleanup_output_stale_shards(
        output_dirs=output_dirs,
        plan_code=plan.code,
        roots=roots,
        emit=emit,
    )

    custom_written_paths = _handle_custom_system_sharding(
        plan=plan,
        resolved_resources=resolved_resources,
        chunk_size=chunk_size,
        stage_options=stage_options,
        emit=emit,
        sidecar_enabled=sidecar_enabled,
    )
    if custom_written_paths is not None:
        return custom_written_paths

    written_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    system_field_candidates = _resolve_system_field_candidates(plan)
    for prepared in prepared_resources:
        resource_main_paths, resource_sidecar_paths = _shard_prepared_resource(
            plan=plan,
            prepared=prepared,
            chunk_size=chunk_size,
            supported_country_codes=supported_country_codes,
            stage_options=stage_options,
            sidecar_enabled=sidecar_enabled,
            system_field_candidates=system_field_candidates,
            emit=emit,
        )
        written_paths.extend(resource_main_paths)
        sidecar_paths.extend(resource_sidecar_paths)

    rewrites = _harmonize_shard_schemas(written_paths)
    if rewrites:
        emit(f"[{plan.code}] harmonized shard schemas across {rewrites} shard file(s)")

    _emit_shard_completion(
        emit=emit,
        plan_code=plan.code,
        written_paths=written_paths,
        sidecar_enabled=sidecar_enabled,
        sidecar_paths=sidecar_paths,
    )
    return written_paths
