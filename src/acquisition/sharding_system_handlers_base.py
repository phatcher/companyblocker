from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from workspace.data_layout import system_layer_dir
from workspace.roots import WorkspaceRoots


@dataclass(frozen=True)
class CustomShardOutputs:
    main_paths: list[Path]
    sidecar_paths: list[Path]


SourceResolver = Callable[..., tuple[Path, str | None]]
SystemSharder = Callable[..., CustomShardOutputs | None]
ResourceSharder = Callable[..., CustomShardOutputs | None]
ResourcePreparer = Callable[..., tuple[Path, Path] | None]
FrameTransformer = Callable[[pl.DataFrame], pl.DataFrame]
MainColumnExtender = Callable[[list[str], list[str]], None]
SystemUriMaterializer = Callable[
    [pl.DataFrame, str, tuple[str, ...] | None], pl.DataFrame
]


class ShardSystemHandler:
    """Flat per-system sharding handler with safe default behaviour."""

    def __init__(
        self,
        *,
        extender: MainColumnExtender | None = None,
        source_resolver: SourceResolver | None = None,
        system_sharder: SystemSharder | None = None,
        resource_sharder: ResourceSharder | None = None,
        resource_preparer: ResourcePreparer | None = None,
        transformer: FrameTransformer | None = None,
        materializer: SystemUriMaterializer | None = None,
    ) -> None:
        self._extender = extender
        self._source_resolver = source_resolver
        self._system_sharder = system_sharder
        self._resource_sharder = resource_sharder
        self._resource_preparer = resource_preparer
        self._transformer = transformer
        self._materializer = materializer

    def extend_preferred_main_columns(
        self, *, columns: list[str], preferred: list[str]
    ) -> None:
        if self._extender is not None:
            self._extender(columns, preferred)

    def resolve_effective_resource_source(
        self,
        *,
        resource: Any,
        snapshot_date: str,
        prepare_root: Path,
        source_path: Path,
        roots: WorkspaceRoots,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[Path, str | None]:
        if self._source_resolver is not None:
            return self._source_resolver(
                resource=resource,
                snapshot_date=snapshot_date,
                prepare_root=prepare_root,
                source_path=source_path,
                roots=roots,
                progress=progress,
            )
        return source_path, None

    def try_shard_system_with_custom_logic(
        self,
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
        if self._system_sharder is not None:
            return self._system_sharder(
                plan_code=plan_code,
                resolved_resources=resolved_resources,
                chunk_size=chunk_size,
                stage_options=stage_options,
                emit=emit,
                sidecar_enabled=sidecar_enabled,
                materialize_system_uri=materialize_system_uri,
                split_main_sidecar=split_main_sidecar,
            )
        return None

    def try_shard_resource_with_custom_logic(
        self,
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
        if self._resource_sharder is not None:
            return self._resource_sharder(
                plan_code=plan_code,
                resource=resource,
                source_path=source_path,
                source_dir=source_dir,
                chunk_size=chunk_size,
                supported_country_codes=supported_country_codes,
                emit=emit,
                sidecar_enabled=sidecar_enabled,
                materialize_system_uri=materialize_system_uri,
                split_main_sidecar=split_main_sidecar,
                stage_options=stage_options,
            )
        return None

    def try_prepare_resource_with_custom_logic(
        self,
        *,
        resource: Any,
        acquire_root: Path,
        prepare_root: Path,
        effective_run_date: str,
        run_date_provided: bool,
        roots: WorkspaceRoots,
        max_companies: int | None,
        max_lines: int | None,
        prepare_wiring_mode: str,
        prepare_stream_resume: bool,
        projection_defaults_override: dict[str, object] | None,
        emit: Callable[[str], None],
        resolve_source_artifact: Callable[..., tuple[Path, Path]],
        extract_two_pass: Callable[..., int],
        force: bool = False,
    ) -> tuple[Path, Path] | None:
        if self._resource_preparer is not None:
            return self._resource_preparer(
                resource=resource,
                acquire_root=acquire_root,
                prepare_root=prepare_root,
                effective_run_date=effective_run_date,
                run_date_provided=run_date_provided,
                roots=roots,
                max_companies=max_companies,
                max_lines=max_lines,
                prepare_wiring_mode=prepare_wiring_mode,
                prepare_stream_resume=prepare_stream_resume,
                projection_defaults_override=projection_defaults_override,
                emit=emit,
                resolve_source_artifact=resolve_source_artifact,
                extract_two_pass=extract_two_pass,
                force=force,
            )
        return None

    def prepare_and_resolve_resource_source(
        self,
        *,
        plan_code: str,
        resource: Any,
        snapshot_dir: Path,
        source_path: Path,
        prepare_root: Path,
        effective_run_date: str,
        run_date_provided: bool,
        roots: WorkspaceRoots,
        max_companies: int | None,
        max_lines: int | None,
        prepare_wiring_mode: str,
        prepare_stream_resume: bool,
        projection_defaults_override: dict[str, object] | None,
        emit: Callable[[str], None],
        resolve_source_artifact: Callable[..., tuple[Path, Path]],
        extract_two_pass: Callable[..., int],
        force: bool = False,
    ) -> tuple[Path, Path, str | None]:
        prepared_override = self.try_prepare_resource_with_custom_logic(
            resource=resource,
            acquire_root=system_layer_dir(roots, plan_code, layer="acquire"),
            prepare_root=prepare_root,
            effective_run_date=effective_run_date,
            run_date_provided=run_date_provided,
            roots=roots,
            max_companies=max_companies,
            max_lines=max_lines,
            prepare_wiring_mode=prepare_wiring_mode,
            prepare_stream_resume=prepare_stream_resume,
            projection_defaults_override=projection_defaults_override,
            emit=emit,
            resolve_source_artifact=resolve_source_artifact,
            extract_two_pass=extract_two_pass,
            force=force,
        )
        if prepared_override is not None:
            snapshot_dir, source_path = prepared_override

        source_path, source_format_override = self.resolve_effective_resource_source(
            resource=resource,
            snapshot_date=snapshot_dir.name,
            prepare_root=prepare_root,
            source_path=source_path,
            roots=roots,
            progress=emit,
        )
        return snapshot_dir, source_path, source_format_override

    def transform_frame_for_system(self, *, frame: pl.DataFrame) -> pl.DataFrame:
        if self._transformer is not None:
            return self._transformer(frame)
        return frame

    def materialize_and_split_for_write(
        self,
        *,
        frame: pl.DataFrame,
        sidecar_enabled: bool,
        system_code: str,
        identifier_candidates: tuple[str, ...] | None,
        default_materializer: SystemUriMaterializer,
        split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        materialized_frame = self.materialize_system_uri_for_system(
            frame=frame,
            system_code=system_code,
            identifier_candidates=identifier_candidates,
            default_materializer=default_materializer,
        )
        if sidecar_enabled:
            return split_main_sidecar(materialized_frame)
        return materialized_frame, pl.DataFrame()

    def materialize_system_uri_for_system(
        self,
        *,
        frame: pl.DataFrame,
        system_code: str,
        identifier_candidates: tuple[str, ...] | None,
        default_materializer: SystemUriMaterializer,
    ) -> pl.DataFrame:
        if self._materializer is not None:
            return self._materializer(frame, system_code, identifier_candidates)
        return default_materializer(frame, system_code, identifier_candidates)
