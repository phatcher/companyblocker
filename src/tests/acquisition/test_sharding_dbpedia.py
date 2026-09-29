from __future__ import annotations

import bz2
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from acquisition.sharding_dbpedia import build_dbpedia_company_frame


def _write_bz2(path: Path, text: str) -> None:
    with bz2.open(path, mode="wt", encoding="utf-8") as handle:
        handle.write(text)


def _resource(name: str):
    return SimpleNamespace(name=name)


def _resolved_resources(source_dir: Path) -> list[tuple[object, Path, Path]]:
    return [
        (
            _resource("dbpedia_mappings_instance_types_en_latest"),
            source_dir,
            source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        ),
        (
            _resource("dbpedia_generic_labels_en_latest"),
            source_dir,
            source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        ),
        (
            _resource("dbpedia_mappings_mappingbased_literals_en_latest"),
            source_dir,
            source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2",
        ),
    ]


def test_dbpedia_build_frame_filters_to_org_types(tmp_path: Path, layer_fixture_dir):
    source_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    _write_bz2(
        source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        "<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Company> .\n<http://dbpedia.org/resource/John_Doe> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Person> ."
        + "\n",
    )
    _write_bz2(
        source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/2000/01/rdf-schema#label> "Acme Corp"@en .\n<http://dbpedia.org/resource/John_Doe> <http://www.w3.org/2000/01/rdf-schema#label> "John Doe"@en .'
        + "\n",
    )
    _write_bz2(source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2", "")

    frame = build_dbpedia_company_frame(_resolved_resources(source_dir))

    assert frame.height == 1
    assert frame["CompanyName"].to_list() == ["Acme Corp"]


def test_dbpedia_build_frame_uses_english_labels_only(
    tmp_path: Path, layer_fixture_dir
):
    source_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    _write_bz2(
        source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        "<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Company> .\n",
    )
    _write_bz2(
        source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/2000/01/rdf-schema#label> "Acme GmbH"@de .\n<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/2000/01/rdf-schema#label> "Acme Corp"@en .'
        + "\n",
    )
    _write_bz2(source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2", "")

    frame = build_dbpedia_company_frame(_resolved_resources(source_dir))

    assert frame.height == 1
    assert frame["CompanyName"].to_list() == ["Acme Corp"]


def test_dbpedia_build_frame_derives_company_number_and_jurisdiction(
    tmp_path: Path, layer_fixture_dir
):
    source_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    _write_bz2(
        source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        "<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Company> .\n",
    )
    _write_bz2(
        source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/2000/01/rdf-schema#label> "Acme Corp"@en .\n',
    )
    _write_bz2(
        source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://dbpedia.org/property/companyNumber> "01234567"@en .\n<http://dbpedia.org/resource/Acme_Corp> <http://dbpedia.org/property/countryCode> "gb"@en .'
        + "\n",
    )

    frame = build_dbpedia_company_frame(_resolved_resources(source_dir))

    row = frame.row(0, named=True)
    assert row["company_number"] == "01234567"
    assert row["jurisdiction_code"] == "GB"


def test_dbpedia_build_frame_leaves_unknown_fields_null(
    tmp_path: Path, layer_fixture_dir
):
    source_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    _write_bz2(
        source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        "<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Company> .\n",
    )
    _write_bz2(
        source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://www.w3.org/2000/01/rdf-schema#label> "Acme Corp"@en .\n',
    )
    _write_bz2(
        source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Acme_Corp> <http://dbpedia.org/property/foundingYear> "1980"^^<http://www.w3.org/2001/XMLSchema#integer> .\n',
    )

    frame = build_dbpedia_company_frame(_resolved_resources(source_dir))

    row = frame.row(0, named=True)
    assert row["company_number"] is None
    assert row["jurisdiction_code"] is None


def test_dbpedia_build_frame_can_infer_linked_subject_from_alias(
    tmp_path: Path, layer_fixture_dir
):
    source_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    _write_bz2(
        source_dir / "dbpedia-instance-types-en-2026-06-01.ttl.bz2",
        "<http://dbpedia.org/resource/Cisco> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Company> .\n",
    )
    _write_bz2(
        source_dir / "dbpedia-labels-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Cisco> <http://www.w3.org/2000/01/rdf-schema#label> "Cisco"@en .\n<http://dbpedia.org/resource/Cisco_Systems> <http://www.w3.org/2000/01/rdf-schema#label> "Cisco Systems"@en .'
        + "\n",
    )
    _write_bz2(
        source_dir / "dbpedia-mappingbased-literals-en-2026-06-01.ttl.bz2",
        '<http://dbpedia.org/resource/Cisco> <http://dbpedia.org/ontology/alias> "Cisco Systems"@en .\n<http://dbpedia.org/resource/Cisco> <http://dbpedia.org/property/companyNumber> "12345678"@en .\n<http://dbpedia.org/resource/Cisco> <http://dbpedia.org/property/countryCode> "us"@en .'
        + "\n",
    )

    base_frame = build_dbpedia_company_frame(_resolved_resources(source_dir))
    inferred_frame = build_dbpedia_company_frame(
        _resolved_resources(source_dir),
        enable_link_inference=True,
    )

    assert base_frame.height == 1
    assert inferred_frame.height == 2

    inferred_row = inferred_frame.filter(
        pl.col("dbpedia_uri") == "http://dbpedia.org/resource/Cisco_Systems"
    ).row(0, named=True)
    assert inferred_row["CompanyName"] == "Cisco Systems"
    assert inferred_row["inclusion_basis"] == "link_inferred"
    assert inferred_row["inclusion_flags"] == "ontology_type,alias"
    assert inferred_row["canonical_dbpedia_uri"] == "http://dbpedia.org/resource/Cisco"
    assert inferred_row["inference_method"] == "alias_literal_exact"
    assert inferred_row["evidence_confidence"] == "medium"
    assert inferred_row["company_number"] == "12345678"
    assert inferred_row["jurisdiction_code"] == "US"
