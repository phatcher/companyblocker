from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from workspace.roots import WorkspaceRoots

from .chunking import write_chunked_parquet
from .sharding_dbpedia import DbpediaShardHandler
from .sharding_gleif import write_gleif_xml_zip_chunked
from .sharding_offeneregister import OffeneregisterShardTransformer
from .sharding_perf import ShardPerformance
from .sharding_system_handlers_base import (
    CustomShardOutputs,
    ShardSystemHandler,
    SystemUriMaterializer,
)
from .sharding_wikidata import WikidataShardHandler

_DBPEDIA_SHARDER = DbpediaShardHandler()
_OFFENEREGISTER_TRANSFORMER = OffeneregisterShardTransformer()
_WIKIDATA_SHARDER = WikidataShardHandler()


def _extend_gb_preferred_columns(columns: list[str], preferred: list[str]) -> None:
    for column_name in columns:
        if column_name.startswith("RegAddress.") and column_name not in preferred:
            preferred.append(column_name)


def _extend_gleif_preferred_columns(columns: list[str], preferred: list[str]) -> None:
    for column_name in columns:
        if (
            column_name.startswith(("Entity_", "Registration_"))
        ) and column_name not in preferred:
            preferred.append(column_name)


def _resolve_wikidata_source(
    *,
    resource: Any,
    snapshot_date: str,
    prepare_root: Path,
    source_path: Path,
    roots: WorkspaceRoots,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, str | None]:
    return _WIKIDATA_SHARDER.resolve_shard_source(
        resource_name=resource.name,
        snapshot_date=snapshot_date,
        prepare_root=prepare_root,
        source_path=source_path,
        roots=roots,
        progress=progress,
    )


def _shard_dbpedia_system(
    *,
    plan_code: str,
    resolved_resources: list[tuple[Any, Path, Path]],
    chunk_size: int,
    stage_options: dict[str, object] | None,
    emit: Callable[[str], None],
    sidecar_enabled: bool,
    materialize_system_uri: Callable[[pl.DataFrame], pl.DataFrame],
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]],
) -> CustomShardOutputs | None:
    output_dir = resolved_resources[0][1]
    enable_link_inference = bool((stage_options or {}).get("infer_links", False))
    perf = ShardPerformance(prefix=plan_code, progress=emit)
    if enable_link_inference:
        emit("[dbpedia] link inference enabled via shard.infer_links")

    build_started_at = time.perf_counter()
    frame = _DBPEDIA_SHARDER.build_company_frame(
        resolved_resources,
        progress=emit,
        enable_link_inference=enable_link_inference,
    )
    perf.add_phase_elapsed("build_frame", time.perf_counter() - build_started_at)

    materialize_started_at = time.perf_counter()
    frame = materialize_system_uri(frame)
    perf.add_phase_elapsed(
        "materialize_system_uri", time.perf_counter() - materialize_started_at
    )

    if sidecar_enabled:
        main_frame, sidecar_frame = split_main_sidecar(frame)
        main_paths = write_chunked_parquet(
            main_frame,
            output_dir,
            plan_code,
            chunk_size=chunk_size,
            progress=emit,
            telemetry=perf,
        )
        sidecar_write_started_at = time.perf_counter()
        sidecar_paths = write_chunked_parquet(
            sidecar_frame,
            output_dir,
            f"{plan_code}-sidecar",
            chunk_size=chunk_size,
            progress=emit,
            emit_summary=False,
        )
        perf.add_phase_elapsed("write", time.perf_counter() - sidecar_write_started_at)
        perf.emit_summary()
        emit(
            f"[{plan_code}] wrote {len(main_paths)} shard(s) and {len(sidecar_paths)} sidecar shard(s) from RDF company extraction"
        )
        return CustomShardOutputs(main_paths=main_paths, sidecar_paths=sidecar_paths)

    chunk_paths = write_chunked_parquet(
        frame,
        output_dir,
        plan_code,
        chunk_size=chunk_size,
        progress=emit,
        telemetry=perf,
    )
    perf.emit_summary()
    emit(f"[{plan_code}] wrote {len(chunk_paths)} shard(s) from RDF company extraction")
    return CustomShardOutputs(main_paths=chunk_paths, sidecar_paths=[])


def _shard_gleif_resource(
    *,
    plan_code: str,
    resource: Any,
    source_path: Path,
    source_dir: Path,
    chunk_size: int,
    supported_country_codes: set[str] | None,
    emit: Callable[[str], None],
    sidecar_enabled: bool,
    materialize_system_uri: Callable[[pl.DataFrame], pl.DataFrame],
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]],
    stage_options: dict[str, object] | None = None,
) -> CustomShardOutputs | None:
    if resource.file_format.lower() != "zip":
        return None

    max_rows = None
    if stage_options is not None:
        max_rows_val = stage_options.get("max_rows")
        if isinstance(max_rows_val, (int, float, str)):
            max_rows = int(max_rows_val)

    gleif_outputs = write_gleif_xml_zip_chunked(
        source_path,
        source_dir,
        plan_code,
        chunk_size=chunk_size,
        supported_country_codes=supported_country_codes,
        progress=emit,
        sidecar_enabled=sidecar_enabled,
        materialize_system_uri=materialize_system_uri,
        split_main_sidecar=split_main_sidecar,
        max_rows=max_rows,
    )
    if gleif_outputs is not None:
        xml_shards, sidecar_shards = gleif_outputs
        if sidecar_enabled:
            emit(
                f"[{plan_code}] wrote {len(xml_shards)} XML shard(s) and {len(sidecar_shards)} sidecar shard(s) for '{resource.name}'"
            )
        else:
            emit(
                f"[{plan_code}] wrote {len(xml_shards)} XML shard(s) for '{resource.name}'"
            )
        return CustomShardOutputs(main_paths=xml_shards, sidecar_paths=sidecar_shards)

    return None


_SYSTEM_URI_MATERIALIZERS: dict[str, SystemUriMaterializer] = {}


_DEFAULT_HANDLER = ShardSystemHandler()
_SYSTEM_HANDLERS: dict[str, ShardSystemHandler] = {
    "wikidata": ShardSystemHandler(
        source_resolver=_resolve_wikidata_source,
        resource_preparer=_WIKIDATA_SHARDER.prepare_shard_input,
    ),
    "dbpedia": ShardSystemHandler(system_sharder=_shard_dbpedia_system),
    "gleif": ShardSystemHandler(
        extender=_extend_gleif_preferred_columns,
        resource_sharder=_shard_gleif_resource,
    ),
    "offeneregister": ShardSystemHandler(
        transformer=_OFFENEREGISTER_TRANSFORMER.flatten_frame
    ),
    "gb": ShardSystemHandler(extender=_extend_gb_preferred_columns),
}

for _system_code, _materializer in _SYSTEM_URI_MATERIALIZERS.items():
    _SYSTEM_HANDLERS[_system_code] = ShardSystemHandler(materializer=_materializer)


def resolve_handler(plan_code: str) -> ShardSystemHandler:
    return _SYSTEM_HANDLERS.get(plan_code, _DEFAULT_HANDLER)
