from __future__ import annotations

import bz2
import gzip
import json
import zipfile
from pathlib import Path

import polars as pl

from acquisition.sharding_io import read_source_frame


def test_read_source_frame_reads_zip_with_jsonl_member(tmp_path: Path):
    source = tmp_path / "records.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "records.jsonl",
            '{"CompanyName":"Alpha","value":1}\n{"CompanyName":"Beta","value":2}\n',
        )

    frame = read_source_frame(source, "zip")

    assert frame.height == 2
    assert frame["CompanyName"].to_list() == ["Alpha", "Beta"]


def test_read_source_frame_reads_zip_with_spaced_csv_headers(tmp_path: Path):
    source = tmp_path / "records.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "records.csv",
            "CompanyName, CompanyNumber, value\nAlpha, 123, 1\nBeta, 456, 2\n",
        )

    frame = read_source_frame(source, "zip")

    assert frame.height == 2
    assert "CompanyNumber" in frame.columns
    assert " CompanyNumber" not in frame.columns
    assert frame["CompanyNumber"].to_list() == [" 123", " 456"]


def test_read_source_frame_reads_zip_with_parquet_member(tmp_path: Path):
    parquet_path = tmp_path / "records.parquet"
    pl.DataFrame({"CompanyName": ["Alpha", "Beta"], "value": [1, 2]}).write_parquet(
        parquet_path
    )

    source = tmp_path / "records.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.write(parquet_path, arcname="records.parquet")

    frame = read_source_frame(source, "zip")

    assert frame.height == 2
    assert frame["CompanyName"].to_list() == ["Alpha", "Beta"]


def test_read_source_frame_preserves_sparse_jsonl_columns_beyond_early_rows(
    tmp_path: Path,
):
    source = tmp_path / "wikidata.jsonl.bz2"

    with bz2.open(source, "wt", encoding="utf-8") as handle:
        for idx in range(150):
            payload: dict[str, object] = {"id": f"Q{idx}", "label_en": f"Name {idx}"}
            if idx == 120:
                payload["official_name_en"] = ["Official Name"]
            handle.write(json.dumps(payload) + "\n")

    frame = read_source_frame(source, "jsonl")

    assert "official_name_en" in frame.columns
    extracted = frame.filter(pl.col("id") == "Q120").row(0, named=True)[
        "official_name_en"
    ]
    assert extracted == ["Official Name"]


def test_read_source_frame_reads_gzipped_jsonl(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl.gz"

    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write('{"id":"Q1","label_en":"Alpha"}\n')
        handle.write('{"id":"Q2","label_en":"Beta"}\n')

    frame = read_source_frame(source, "jsonl")

    assert frame.height == 2
    assert frame["id"].to_list() == ["Q1", "Q2"]
