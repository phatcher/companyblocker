from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from acquisition import sharding_batch_processor
from acquisition.sharding_batch_processor import ShardBatchProcessor
from acquisition.sharding_system_handlers_base import CustomShardOutputs


class _FakeWriter:
    """Records every appended frame; `finalize` returns one path per append."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.appended: list[pl.DataFrame] = []
        self.finalized = False

    def append(self, frame: pl.DataFrame) -> None:
        self.appended.append(frame)

    def finalize(self) -> list[Path]:
        self.finalized = True
        return [
            Path(f"{self.kwargs['prefix']}-{i}.parquet")
            for i in range(len(self.appended))
        ]


def _make_processor(
    *,
    writers: dict[str, _FakeWriter],
    materialize_calls: list[pl.DataFrame],
    split_calls: list[pl.DataFrame],
    filter_calls: list[tuple[pl.DataFrame, object]],
    emit: list[str],
    sidecar_enabled: bool = True,
    chunk_size: int = 2,
    supported_country_codes: set[str] | None = None,
) -> ShardBatchProcessor:
    def writer_factory(**kwargs: object) -> _FakeWriter:
        writer = _FakeWriter(**kwargs)
        prefix = kwargs["prefix"]
        assert isinstance(prefix, str)
        writers[prefix] = writer
        return writer

    def default_system_uri_materializer(frame, system_code, identifier_candidates):
        materialize_calls.append(frame)
        return frame.with_columns(pl.lit(True).alias("materialized"))

    def split_main_sidecar_frame(*, frame, **kwargs):
        split_calls.append(frame)
        if frame.height == 0:
            return frame, pl.DataFrame()
        return frame, frame.select(pl.col("id").alias("sidecar_marker"))

    def filter_frame_by_supported_countries(frame, supported):
        filter_calls.append((frame, supported))
        if supported is None:
            return frame
        return frame.filter(pl.col("country_code").is_in(sorted(supported)))

    plan = SimpleNamespace(code="stub-system", system_uri_identifier_candidates=None)

    return ShardBatchProcessor(
        plan=plan,
        chunk_size=chunk_size,
        write_chunk_size=None,
        supported_country_codes=supported_country_codes,
        stage_options=None,
        sidecar_enabled=sidecar_enabled,
        system_field_candidates={},
        emit=emit.append,
        canonical_input_columns=None,
        canonical_source_column_aliases=None,
        default_system_uri_materializer=default_system_uri_materializer,
        split_main_sidecar_frame=split_main_sidecar_frame,
        filter_frame_by_supported_countries=filter_frame_by_supported_countries,
        writer_factory=writer_factory,
    )


def _make_prepared(source_path: Path, *, resource_name: str = "test-resource"):
    return SimpleNamespace(
        resource=SimpleNamespace(name=resource_name, file_format="parquet"),
        output_dir=source_path.parent,
        source_path=source_path,
        source_format_override=None,
    )


def test_shard_prepared_resource_batches_filters_materializes_and_writes(
    tmp_path: Path,
):
    source_path = tmp_path / "input.parquet"
    pl.DataFrame({"id": [1, 2, 3], "country_code": ["gb", "fr", "gb"]}).write_parquet(
        source_path
    )

    writers: dict[str, _FakeWriter] = {}
    materialize_calls: list[pl.DataFrame] = []
    split_calls: list[pl.DataFrame] = []
    filter_calls: list[tuple[pl.DataFrame, object]] = []
    emit: list[str] = []

    processor = _make_processor(
        writers=writers,
        materialize_calls=materialize_calls,
        split_calls=split_calls,
        filter_calls=filter_calls,
        emit=emit,
        supported_country_codes={"gb"},
        chunk_size=2,
    )

    main_paths, sidecar_paths = processor.shard_prepared_resource(
        prepared=_make_prepared(source_path)
    )

    # Two batches of chunk_size=2: rows [id1(gb), id2(fr)] then [id3(gb)].
    assert len(filter_calls) == 2
    assert filter_calls[0][1] == {"gb"}

    main_writer = writers["stub-system"]
    sidecar_writer = writers["stub-system-sidecar"]
    # Both batches keep exactly one gb row after filtering, so both are
    # appended to main (materialized) and sidecar (split produced rows).
    assert len(main_writer.appended) == 2
    assert len(sidecar_writer.appended) == 2
    assert all(frame.height == 1 for frame in main_writer.appended)
    assert all("materialized" in frame.columns for frame in main_writer.appended)

    assert main_paths == [Path("stub-system-0.parquet"), Path("stub-system-1.parquet")]
    assert sidecar_paths == [
        Path("stub-system-sidecar-0.parquet"),
        Path("stub-system-sidecar-1.parquet"),
    ]
    assert main_writer.finalized
    assert sidecar_writer.finalized

    assert any("processing batch 1" in line for line in emit)
    assert any("processing batch 2" in line for line in emit)
    assert any("filtered 1 row(s)" in line for line in emit)


def test_shard_prepared_resource_skips_write_for_a_batch_filtered_to_zero_rows(
    tmp_path: Path,
):
    source_path = tmp_path / "input.parquet"
    pl.DataFrame({"id": [1], "country_code": ["de"]}).write_parquet(source_path)

    writers: dict[str, _FakeWriter] = {}
    materialize_calls: list[pl.DataFrame] = []
    split_calls: list[pl.DataFrame] = []
    filter_calls: list[tuple[pl.DataFrame, object]] = []
    emit: list[str] = []

    processor = _make_processor(
        writers=writers,
        materialize_calls=materialize_calls,
        split_calls=split_calls,
        filter_calls=filter_calls,
        emit=emit,
        supported_country_codes={"gb"},
        sidecar_enabled=False,
    )

    main_paths, sidecar_paths = processor.shard_prepared_resource(
        prepared=_make_prepared(source_path)
    )

    assert materialize_calls == []  # never reached: the only batch is fully filtered
    assert writers["stub-system"].appended == []
    assert main_paths == []
    assert sidecar_paths == []


def test_shard_prepared_resource_reports_when_no_rows_are_read(tmp_path: Path):
    source_path = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"id": pl.Int64, "country_code": pl.Utf8}).write_parquet(
        source_path
    )

    writers: dict[str, _FakeWriter] = {}
    emit: list[str] = []
    processor = _make_processor(
        writers=writers,
        materialize_calls=[],
        split_calls=[],
        filter_calls=[],
        emit=emit,
        sidecar_enabled=False,
    )

    main_paths, sidecar_paths = processor.shard_prepared_resource(
        prepared=_make_prepared(source_path, resource_name="empty-resource")
    )

    assert any("no rows read from resource 'empty-resource'" in line for line in emit)
    assert main_paths == []
    assert sidecar_paths == []


def test_shard_prepared_resource_returns_early_with_custom_shard_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    custom_outputs = CustomShardOutputs(
        main_paths=[tmp_path / "custom-main.parquet"],
        sidecar_paths=[tmp_path / "custom-sidecar.parquet"],
    )

    class _CustomHandler:
        def try_shard_resource_with_custom_logic(self, **kwargs: object):
            return custom_outputs

    monkeypatch.setattr(
        sharding_batch_processor, "resolve_handler", lambda plan_code: _CustomHandler()
    )

    def _fail_writer_factory(**kwargs: object):
        raise AssertionError("writer_factory should not be called on the custom path")

    plan = SimpleNamespace(code="custom-system", system_uri_identifier_candidates=None)
    processor = ShardBatchProcessor(
        plan=plan,
        chunk_size=100,
        write_chunk_size=None,
        supported_country_codes=None,
        stage_options=None,
        sidecar_enabled=True,
        system_field_candidates={},
        emit=lambda _line: None,
        canonical_input_columns=None,
        canonical_source_column_aliases=None,
        default_system_uri_materializer=lambda frame, system_code, ids: frame,
        split_main_sidecar_frame=lambda **kwargs: (pl.DataFrame(), pl.DataFrame()),
        filter_frame_by_supported_countries=lambda frame, supported: frame,
        writer_factory=_fail_writer_factory,
    )

    prepared = _make_prepared(
        tmp_path / "unused.parquet", resource_name="custom-resource"
    )

    main_paths, sidecar_paths = processor.shard_prepared_resource(prepared=prepared)

    assert main_paths == custom_outputs.main_paths
    assert sidecar_paths == custom_outputs.sidecar_paths
