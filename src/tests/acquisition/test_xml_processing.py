from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from acquisition import xml_processing
from acquisition.xml_processing import (
    iter_zip_xml_records,
    xml_local_name,
    xml_namespace_uri,
)


def test_iter_zip_xml_records_handles_namespaced_records(tmp_path: Path) -> None:
    source_zip = tmp_path / "sample.zip"
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<lei:LEIData xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
    <lei:LEIRecords>
        <lei:LEIRecord><lei:LEI>ABC123</lei:LEI></lei:LEIRecord>
        <lei:LEIRecord><lei:LEI>DEF456</lei:LEI></lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.xml", xml_payload)

    records = list(
        iter_zip_xml_records(
            source_zip,
            record_local_name="LEIRecord",
            expected_root_namespace="http://www.gleif.org/data/schema/leidata/2016",
            expected_root_local_name="LEIData",
        )
    )

    assert len(records) == 2


def test_iter_zip_xml_records_handles_non_namespaced_records(tmp_path: Path) -> None:
    source_zip = tmp_path / "sample.zip"
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<data>
    <records>
        <record><id>1</id></record>
        <record><id>2</id></record>
        <record><id>3</id></record>
    </records>
</data>
"""
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.xml", xml_payload)

    records = list(
        iter_zip_xml_records(
            source_zip,
            record_local_name="record",
            expected_root_local_name="data",
        )
    )

    assert len(records) == 3


def test_xml_local_name_and_namespace_uri_helpers() -> None:
    assert xml_local_name("{urn:test}Record") == "Record"
    assert xml_local_name("Record") == "Record"

    assert xml_namespace_uri("{urn:test}Record") == "urn:test"
    assert xml_namespace_uri("Record") is None
    assert xml_namespace_uri("{invalid") is None


def test_iter_zip_xml_records_returns_empty_when_no_xml_member(tmp_path: Path) -> None:
    source_zip = tmp_path / "sample.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.txt", "not xml")

    records = list(iter_zip_xml_records(source_zip, record_local_name="record"))
    assert records == []


def test_iter_zip_xml_records_uses_explicit_member_name(tmp_path: Path) -> None:
    source_zip = tmp_path / "sample.zip"
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<data><record><id>1</id></record></data>
"""
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("nested/custom.xml", xml_payload)

    records = list(
        iter_zip_xml_records(
            source_zip,
            record_local_name="record",
            xml_member_name="nested/custom.xml",
        )
    )
    assert len(records) == 1


def test_iter_zip_xml_records_raises_on_unexpected_root_local_name(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "sample.zip"
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<root><record /></root>
"""
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.xml", xml_payload)

    with pytest.raises(ValueError, match="Unexpected XML root local name"):
        list(
            iter_zip_xml_records(
                source_zip,
                record_local_name="record",
                expected_root_local_name="LEIData",
            )
        )


def test_iter_zip_xml_records_raises_on_unexpected_root_namespace(
    tmp_path: Path,
) -> None:
    source_zip = tmp_path / "sample.zip"
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<x:data xmlns:x="urn:wrong"><x:record /></x:data>
"""
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.xml", xml_payload)

    with pytest.raises(ValueError, match="Unexpected XML root namespace"):
        list(
            iter_zip_xml_records(
                source_zip,
                record_local_name="record",
                expected_root_namespace="urn:expected",
            )
        )


def test_iter_zip_xml_records_raises_when_first_event_not_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_zip = tmp_path / "sample.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("sample.xml", "<root />")

    original_iterparse = ET.iterparse

    def fake_iterparse(_stream, events=("start", "end")):
        _ = events
        yield ("end", ET.Element("root"))

    monkeypatch.setattr(xml_processing.ET, "iterparse", fake_iterparse)
    try:
        with pytest.raises(ValueError, match="Unexpected first XML parse event"):
            list(iter_zip_xml_records(source_zip, record_local_name="record"))
    finally:
        monkeypatch.setattr(xml_processing.ET, "iterparse", original_iterparse)
