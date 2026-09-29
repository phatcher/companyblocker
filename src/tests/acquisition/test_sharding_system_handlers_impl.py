from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from acquisition import sharding_system_handlers_impl as handlers_impl
from acquisition.sharding_system_handlers_base import CustomShardOutputs


def test_extend_gb_preferred_columns_appends_regaddress_columns_once():
    preferred: list[str] = ["existing"]
    handlers_impl._extend_gb_preferred_columns(
        ["RegAddress.PostCode", "CompanyName", "RegAddress.PostCode"], preferred
    )
    assert preferred == ["existing", "RegAddress.PostCode"]


def test_extend_gleif_preferred_columns_appends_entity_and_registration_columns():
    preferred: list[str] = []
    handlers_impl._extend_gleif_preferred_columns(
        ["Entity_LegalName", "Registration_Status", "Other"], preferred
    )
    assert preferred == ["Entity_LegalName", "Registration_Status"]


def test_shard_dbpedia_system_writes_single_output_without_sidecar(
    tmp_path: Path, mocker
):
    frame = pl.DataFrame({"a": [1, 2]})
    mocker.patch.object(
        handlers_impl._DBPEDIA_SHARDER, "build_company_frame", return_value=frame
    )
    write_chunked = mocker.patch.object(
        handlers_impl,
        "write_chunked_parquet",
        return_value=[tmp_path / "out-1.parquet"],
    )
    messages: list[str] = []

    result = handlers_impl._shard_dbpedia_system(
        plan_code="dbpedia",
        resolved_resources=[(SimpleNamespace(name="r1"), tmp_path, tmp_path / "src")],
        chunk_size=100,
        stage_options={"infer_links": True},
        emit=messages.append,
        sidecar_enabled=False,
        materialize_system_uri=lambda f: f,
        split_main_sidecar=lambda f: (f, f),
    )

    assert isinstance(result, CustomShardOutputs)
    assert result.main_paths == [tmp_path / "out-1.parquet"]
    assert result.sidecar_paths == []
    write_chunked.assert_called_once()
    assert any("link inference enabled" in message for message in messages)


def test_shard_dbpedia_system_writes_sidecar_when_enabled(tmp_path: Path, mocker):
    frame = pl.DataFrame({"a": [1, 2]})
    mocker.patch.object(
        handlers_impl._DBPEDIA_SHARDER, "build_company_frame", return_value=frame
    )
    write_chunked = mocker.patch.object(
        handlers_impl,
        "write_chunked_parquet",
        side_effect=[
            [tmp_path / "main-1.parquet"],
            [tmp_path / "sidecar-1.parquet"],
        ],
    )

    result = handlers_impl._shard_dbpedia_system(
        plan_code="dbpedia",
        resolved_resources=[(SimpleNamespace(name="r1"), tmp_path, tmp_path / "src")],
        chunk_size=100,
        stage_options=None,
        emit=lambda message: None,
        sidecar_enabled=True,
        materialize_system_uri=lambda f: f,
        split_main_sidecar=lambda f: (f, f),
    )

    assert result is not None
    assert result.main_paths == [tmp_path / "main-1.parquet"]
    assert result.sidecar_paths == [tmp_path / "sidecar-1.parquet"]
    assert write_chunked.call_count == 2


def test_shard_gleif_resource_returns_none_for_non_zip_format():
    result = handlers_impl._shard_gleif_resource(
        plan_code="gleif",
        resource=SimpleNamespace(file_format="csv", name="r1"),
        source_path=Path("source.csv"),
        source_dir=Path("dir"),
        chunk_size=100,
        supported_country_codes=None,
        emit=lambda message: None,
        sidecar_enabled=False,
        materialize_system_uri=lambda f: f,
        split_main_sidecar=lambda f: (f, f),
    )
    assert result is None


def test_shard_gleif_resource_parses_max_rows_and_reports_sidecar(mocker):
    messages: list[str] = []
    gleif = mocker.patch.object(
        handlers_impl,
        "write_gleif_xml_zip_chunked",
        return_value=(["xml1.parquet"], ["sidecar1.parquet"]),
    )

    result = handlers_impl._shard_gleif_resource(
        plan_code="gleif",
        resource=SimpleNamespace(file_format="ZIP", name="r1"),
        source_path=Path("source.zip"),
        source_dir=Path("dir"),
        chunk_size=100,
        supported_country_codes={"fr"},
        emit=messages.append,
        sidecar_enabled=True,
        materialize_system_uri=lambda f: f,
        split_main_sidecar=lambda f: (f, f),
        stage_options={"max_rows": "50"},
    )

    assert result is not None
    assert result.main_paths == ["xml1.parquet"]
    assert result.sidecar_paths == ["sidecar1.parquet"]
    _, kwargs = gleif.call_args
    assert kwargs["max_rows"] == 50
    assert any("sidecar shard" in message for message in messages)


def test_shard_gleif_resource_returns_none_when_writer_returns_none(mocker):
    mocker.patch.object(handlers_impl, "write_gleif_xml_zip_chunked", return_value=None)

    result = handlers_impl._shard_gleif_resource(
        plan_code="gleif",
        resource=SimpleNamespace(file_format="zip", name="r1"),
        source_path=Path("source.zip"),
        source_dir=Path("dir"),
        chunk_size=100,
        supported_country_codes=None,
        emit=lambda message: None,
        sidecar_enabled=False,
        materialize_system_uri=lambda f: f,
        split_main_sidecar=lambda f: (f, f),
    )

    assert result is None


def test_resolve_handler_returns_registered_and_default_handlers():
    assert (
        handlers_impl.resolve_handler("wikidata")
        is handlers_impl._SYSTEM_HANDLERS["wikidata"]
    )
    assert (
        handlers_impl.resolve_handler("unknown-system")
        is handlers_impl._DEFAULT_HANDLER
    )
