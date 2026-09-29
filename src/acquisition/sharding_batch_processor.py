from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import polars as pl

from .sharding_io import read_source_batches
from .sharding_perf import ShardPerformance
from .sharding_system_handlers_impl import resolve_handler


class ShardBatchProcessor:
    """Processes one prepared resource through transform/filter/materialize/write."""

    def __init__(
        self,
        *,
        plan,
        chunk_size: int,
        write_chunk_size: int | None,
        supported_country_codes: set[str] | None,
        stage_options: dict[str, object] | None,
        sidecar_enabled: bool,
        system_field_candidates: dict[str, list[str]],
        emit: Callable[[str], None],
        canonical_input_columns: tuple[str, ...] | None,
        canonical_source_column_aliases: tuple[tuple[str, str], ...] | None,
        default_system_uri_materializer: Callable[
            [pl.DataFrame, str, tuple[str, ...] | None], pl.DataFrame
        ],
        split_main_sidecar_frame: Callable[..., tuple[pl.DataFrame, pl.DataFrame]],
        filter_frame_by_supported_countries: Callable[
            [pl.DataFrame, set[str] | None], pl.DataFrame
        ],
        writer_factory: Callable[..., Any],
    ) -> None:
        self.plan = plan
        self.chunk_size = chunk_size
        self.write_chunk_size = write_chunk_size
        self.supported_country_codes = supported_country_codes
        self.stage_options = stage_options
        self.sidecar_enabled = sidecar_enabled
        self.system_field_candidates = system_field_candidates
        self.emit = emit
        self.canonical_input_columns = canonical_input_columns
        self.canonical_source_column_aliases = canonical_source_column_aliases
        self.default_system_uri_materializer = default_system_uri_materializer
        self.split_main_sidecar_frame = split_main_sidecar_frame
        self.filter_frame_by_supported_countries = filter_frame_by_supported_countries
        self.writer_factory = writer_factory

    def shard_prepared_resource(self, *, prepared) -> tuple[list, list]:
        resource = prepared.resource
        source_dir = prepared.output_dir
        source_path = prepared.source_path
        handler = resolve_handler(self.plan.code)
        self.emit(f"[{self.plan.code}] sharding resource '{resource.name}'")
        perf = ShardPerformance(prefix=self.plan.code, progress=self.emit)

        custom_resource_paths = handler.try_shard_resource_with_custom_logic(
            plan_code=self.plan.code,
            resource=resource,
            source_path=source_path,
            source_dir=source_dir,
            chunk_size=self.chunk_size,
            supported_country_codes=self.supported_country_codes,
            emit=self.emit,
            sidecar_enabled=self.sidecar_enabled,
            materialize_system_uri=lambda frame: (
                handler.materialize_system_uri_for_system(
                    frame=frame,
                    system_code=self.plan.code,
                    identifier_candidates=self.plan.system_uri_identifier_candidates,
                    default_materializer=self.default_system_uri_materializer,
                )
            ),
            split_main_sidecar=lambda frame: self.split_main_sidecar_frame(
                frame=frame,
                system_code=self.plan.code,
                system_field_candidates=self.system_field_candidates,
                canonical_input_columns=self.canonical_input_columns,
                canonical_source_column_aliases=self.canonical_source_column_aliases,
            ),
            stage_options=self.stage_options,
        )
        if custom_resource_paths is not None:
            return list(custom_resource_paths.main_paths), list(
                custom_resource_paths.sidecar_paths
            )

        source_format = prepared.source_format_override or resource.file_format
        main_writer = self.writer_factory(
            output_dir=source_dir,
            prefix=self.plan.code,
            chunk_size=self.write_chunk_size,
            progress=self.emit,
            telemetry=perf,
        )
        sidecar_writer = (
            self.writer_factory(
                output_dir=source_dir,
                prefix=f"{self.plan.code}-sidecar",
                chunk_size=self.write_chunk_size,
                progress=self.emit,
            )
            if self.sidecar_enabled
            else None
        )

        total_input_rows = 0
        read_options_resolver = getattr(resource, "resolve_read_options_for_path", None)
        resolved_read_options = (
            read_options_resolver(source_path)
            if callable(read_options_resolver)
            else {}
        )
        for batch_index, frame in enumerate(
            read_source_batches(
                source_path,
                source_format,
                batch_size=self.chunk_size,
                resolved_read_options=resolved_read_options,
            ),
            start=1,
        ):
            total_input_rows += frame.height
            perf.increment_input_rows(frame.height)
            self.emit(
                f"[{self.plan.code}] processing batch {batch_index}: {frame.height:,} row(s)"
            )

            transform_started_at = time.perf_counter()
            frame = handler.transform_frame_for_system(frame=frame)
            perf.add_phase_elapsed(
                "transform", time.perf_counter() - transform_started_at
            )

            pre_filter_rows = frame.height
            frame = self.filter_frame_by_supported_countries(
                frame, self.supported_country_codes
            )
            filtered_rows = pre_filter_rows - frame.height

            if filtered_rows:
                perf.increment_filtered_rows(filtered_rows)
                self.emit(
                    f"[{self.plan.code}] filtered {filtered_rows:,} row(s) by supported country policy"
                )

            if frame.height == 0:
                continue

            materialize_and_split_started_at = time.perf_counter()
            main_frame, sidecar_frame = handler.materialize_and_split_for_write(
                frame=frame,
                sidecar_enabled=self.sidecar_enabled,
                system_code=self.plan.code,
                identifier_candidates=self.plan.system_uri_identifier_candidates,
                default_materializer=self.default_system_uri_materializer,
                split_main_sidecar=lambda frame_to_split: self.split_main_sidecar_frame(
                    frame=frame_to_split,
                    system_code=self.plan.code,
                    system_field_candidates=self.system_field_candidates,
                    canonical_input_columns=self.canonical_input_columns,
                    canonical_source_column_aliases=self.canonical_source_column_aliases,
                ),
            )
            perf.add_phase_elapsed(
                "materialize_and_split",
                time.perf_counter() - materialize_and_split_started_at,
            )

            main_writer.append(main_frame)
            if sidecar_writer is not None and sidecar_frame.height:
                sidecar_writer.append(sidecar_frame)

        if total_input_rows == 0:
            self.emit(
                f"[{self.plan.code}] no rows read from resource '{resource.name}'"
            )

        main_paths = main_writer.finalize()
        sidecar_paths = sidecar_writer.finalize() if sidecar_writer is not None else []

        perf.emit_summary()
        return main_paths, sidecar_paths
