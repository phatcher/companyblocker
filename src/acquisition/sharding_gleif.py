from __future__ import annotations

import time
import zipfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import polars as pl
from defusedxml import ElementTree as ET

from .sharding_perf import ShardPerformance, write_shard_chunk
from .xml_processing import iter_zip_xml_records

_GLEIF_NS = {"lei": "http://www.gleif.org/data/schema/leidata/2016"}
_XSD_NS = {"xs": "http://www.w3.org/2001/XMLSchema"}
_GLEIF_LEI_CDF_XSD = Path(__file__).resolve().parent / "schema" / "lei-cdf-v3-1.xsd"
_GLEIF_COUNTRY_FIELD_CANDIDATES = (
    "Entity_LegalAddress_Country",
    "Entity_HeadquartersAddress_Country",
    "Entity_LegalJurisdiction",
)


def _local_name(name: str) -> str:
    sep = name.rfind("}")
    return name[sep + 1 :] if sep >= 0 else name


def _clean_xml_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _join_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    if len(element) == 0:
        return _clean_xml_text(element.text)
    return _clean_xml_text("".join(element.itertext()))


def _qname_local(name: str | None) -> str | None:
    if not name:
        return None
    if ":" in name:
        return name.split(":", 1)[1]
    return name


@lru_cache(maxsize=128)
def _segment_is_excluded(segment: str) -> bool:
    return "officer" in segment.lower()


def _append_value(target: dict[str, str | None], key: str, value: str) -> None:
    existing = target.get(key)
    if existing is None:
        target[key] = value
        return
    if value in existing.split(" | "):
        return
    target[key] = f"{existing} | {value}"


def _sanitize_gleif_column_name(name: str) -> str:
    return name.replace(".", "_").replace("@", "_")


def _build_gleif_schema_type_indexes(
    schema_root: ET.Element,
) -> tuple[dict[str, ET.Element], set[str]]:
    complex_types = {
        item.attrib["name"]: item
        for item in schema_root.findall("xs:complexType", _XSD_NS)
        if "name" in item.attrib
    }
    simple_types = {
        item.attrib["name"]
        for item in schema_root.findall("xs:simpleType", _XSD_NS)
        if "name" in item.attrib
    }
    return complex_types, simple_types


def _schema_collect_attributes(
    *,
    container: ET.Element,
    prefix: str,
    paths: set[str],
) -> None:
    for attribute in container.findall("xs:attribute", _XSD_NS):
        attr_name = _qname_local(
            attribute.attrib.get("name") or attribute.attrib.get("ref")
        )
        if attr_name:
            paths.add(f"{prefix}.@{attr_name}")


def _schema_collect_model(
    *,
    container: ET.Element,
    prefix: str,
    paths: set[str],
    simple_types: set[str],
    complex_types: dict[str, ET.Element],
    visiting: set[str],
) -> bool:
    found = False
    for model in ("xs:sequence", "xs:choice", "xs:all"):
        for model_node in container.findall(model, _XSD_NS):
            for element in model_node.findall("xs:element", _XSD_NS):
                child_name = _qname_local(
                    element.attrib.get("name") or element.attrib.get("ref")
                )
                if not child_name:
                    continue
                found = True
                child_prefix = f"{prefix}.{child_name}"

                inline_complex = element.find("xs:complexType", _XSD_NS)
                if inline_complex is not None:
                    _schema_collect_from_complex(
                        complex_node=inline_complex,
                        prefix=child_prefix,
                        paths=paths,
                        simple_types=simple_types,
                        complex_types=complex_types,
                        visiting=visiting,
                    )
                    continue

                child_type = element.attrib.get("type")
                if child_type:
                    _schema_collect_from_type(
                        type_name=child_type,
                        prefix=child_prefix,
                        paths=paths,
                        simple_types=simple_types,
                        complex_types=complex_types,
                        visiting=visiting,
                    )
                else:
                    paths.add(child_prefix)
    return found


def _resolve_extension_base_type(
    *,
    extension: ET.Element,
    prefix: str,
    paths: set[str],
    simple_types: set[str],
    complex_types: dict[str, ET.Element],
    visiting: set[str],
) -> str | None:
    base_type = extension.attrib.get("base")
    if base_type:
        _schema_collect_from_type(
            type_name=base_type,
            prefix=prefix,
            paths=paths,
            simple_types=simple_types,
            complex_types=complex_types,
            visiting=visiting,
        )
    return base_type


def _collect_complex_content_extension(
    *,
    complex_node: ET.Element,
    prefix: str,
    paths: set[str],
    simple_types: set[str],
    complex_types: dict[str, ET.Element],
    visiting: set[str],
    has_model: bool,
) -> bool:
    complex_content = complex_node.find("xs:complexContent", _XSD_NS)
    if complex_content is None:
        return has_model
    extension = complex_content.find("xs:extension", _XSD_NS)
    if extension is None:
        return has_model

    _resolve_extension_base_type(
        extension=extension,
        prefix=prefix,
        paths=paths,
        simple_types=simple_types,
        complex_types=complex_types,
        visiting=visiting,
    )
    has_model = (
        _schema_collect_model(
            container=extension,
            prefix=prefix,
            paths=paths,
            simple_types=simple_types,
            complex_types=complex_types,
            visiting=visiting,
        )
        or has_model
    )
    _schema_collect_attributes(container=extension, prefix=prefix, paths=paths)
    return has_model


def _schema_collect_from_complex(
    *,
    complex_node: ET.Element,
    prefix: str,
    paths: set[str],
    simple_types: set[str],
    complex_types: dict[str, ET.Element],
    visiting: set[str],
) -> None:
    node_id = complex_node.attrib.get("name")
    if node_id:
        if node_id in visiting:
            return
        visiting.add(node_id)

    try:
        handled_simple_content = False

        simple_content = complex_node.find("xs:simpleContent", _XSD_NS)
        if simple_content is not None:
            extension = simple_content.find("xs:extension", _XSD_NS)
            if extension is not None:
                base_type = _resolve_extension_base_type(
                    extension=extension,
                    prefix=prefix,
                    paths=paths,
                    simple_types=simple_types,
                    complex_types=complex_types,
                    visiting=visiting,
                )
                if base_type is None:
                    paths.add(prefix)
                _schema_collect_attributes(
                    container=extension, prefix=prefix, paths=paths
                )
                handled_simple_content = True

        if not handled_simple_content:
            has_model = _schema_collect_model(
                container=complex_node,
                prefix=prefix,
                paths=paths,
                simple_types=simple_types,
                complex_types=complex_types,
                visiting=visiting,
            )

            has_model = _collect_complex_content_extension(
                complex_node=complex_node,
                prefix=prefix,
                paths=paths,
                simple_types=simple_types,
                complex_types=complex_types,
                visiting=visiting,
                has_model=has_model,
            )

            _schema_collect_attributes(
                container=complex_node, prefix=prefix, paths=paths
            )

            if not has_model and not list(
                complex_node.findall("xs:attribute", _XSD_NS)
            ):
                paths.add(prefix)
    finally:
        if node_id:
            visiting.discard(node_id)


def _schema_collect_from_type(
    *,
    type_name: str,
    prefix: str,
    paths: set[str],
    simple_types: set[str],
    complex_types: dict[str, ET.Element],
    visiting: set[str],
) -> None:
    local_type = _qname_local(type_name)
    if local_type is None:
        return

    if local_type in simple_types:
        paths.add(prefix)
        return

    complex_type = complex_types.get(local_type)
    if complex_type is None:
        paths.add(prefix)
        return

    _schema_collect_from_complex(
        complex_node=complex_type,
        prefix=prefix,
        paths=paths,
        simple_types=simple_types,
        complex_types=complex_types,
        visiting=visiting,
    )


@lru_cache(maxsize=1)
def _gleif_company_schema_paths() -> frozenset[str]:
    if not _GLEIF_LEI_CDF_XSD.exists():
        raise FileNotFoundError(f"Missing LEI-CDF schema file: {_GLEIF_LEI_CDF_XSD}")

    schema_root = ET.parse(_GLEIF_LEI_CDF_XSD).getroot()
    complex_types, simple_types = _build_gleif_schema_type_indexes(schema_root)

    paths: set[str] = set()
    visiting: set[str] = set()
    _schema_collect_from_type(
        type_name="lei:LEIRecordType",
        prefix="",
        paths=paths,
        simple_types=simple_types,
        complex_types=complex_types,
        visiting=visiting,
    )
    normalized = {path.lstrip(".") for path in paths if path}
    return frozenset(normalized)


@lru_cache(maxsize=1)
def _gleif_company_schema_indexes() -> tuple[
    frozenset[str], frozenset[str], dict[str, str]
]:
    allowed_paths = _gleif_company_schema_paths()
    allowed_prefixes: set[str] = set()
    sanitized_by_path: dict[str, str] = {}

    for path in allowed_paths:
        sanitized_by_path[path] = _sanitize_gleif_column_name(path)
        segments = path.split(".")
        for idx in range(1, len(segments)):
            allowed_prefixes.add(".".join(segments[:idx]))

    return allowed_paths, frozenset(allowed_prefixes), sanitized_by_path


def _iter_gleif_relevant_children(
    element: ET.Element,
    *,
    base_path: str | None,
    allowed_paths: frozenset[str],
    allowed_prefixes: frozenset[str],
) -> tuple[tuple[ET.Element, str], ...]:
    children: list[tuple[ET.Element, str]] = []
    for child in element:
        child_name = _local_name(child.tag)
        if _segment_is_excluded(child_name):
            continue

        child_path = child_name if base_path is None else f"{base_path}.{child_name}"
        if child_path not in allowed_paths and child_path not in allowed_prefixes:
            continue
        children.append((child, child_path))
    return tuple(children)


def _flatten_xml_company_fields(
    element: ET.Element,
    path: str,
    output: dict[str, str | None],
    *,
    allowed_paths: frozenset[str],
    allowed_prefixes: frozenset[str],
) -> None:
    for attr_name, attr_value in element.attrib.items():
        attr_local = _local_name(attr_name)
        attr_path = f"{path}.@{attr_local}"
        if attr_path not in allowed_paths:
            continue
        cleaned = _clean_xml_text(attr_value)
        if cleaned is not None:
            _append_value(output, attr_path, cleaned)

    if len(element) == 0:
        if path not in allowed_paths:
            return
        text = _join_text(element)
        if text is not None:
            _append_value(output, path, text)
        return

    for child, child_path in _iter_gleif_relevant_children(
        element,
        base_path=path,
        allowed_paths=allowed_paths,
        allowed_prefixes=allowed_prefixes,
    ):
        _flatten_xml_company_fields(
            child,
            child_path,
            output,
            allowed_paths=allowed_paths,
            allowed_prefixes=allowed_prefixes,
        )


def _extract_gleif_record(lei_record: ET.Element) -> dict[str, str | None]:
    allowed_paths, allowed_prefixes, sanitized_by_path = _gleif_company_schema_indexes()

    flat: dict[str, str | None] = {}

    for child, child_name in _iter_gleif_relevant_children(
        lei_record,
        base_path=None,
        allowed_paths=allowed_paths,
        allowed_prefixes=allowed_prefixes,
    ):
        _flatten_xml_company_fields(
            child,
            child_name,
            flat,
            allowed_paths=allowed_paths,
            allowed_prefixes=allowed_prefixes,
        )

    if "LEI" not in flat:
        flat["LEI"] = _clean_xml_text(
            lei_record.findtext("./lei:LEI", default=None, namespaces=_GLEIF_NS)
        )

    sanitized: dict[str, str | None] = {}
    for key, value in flat.items():
        if value is None:
            continue
        sanitized_key = sanitized_by_path.get(key)
        if sanitized_key is None:
            sanitized_key = _sanitize_gleif_column_name(key)
        _append_value(sanitized, sanitized_key, value)

    if "LEI" not in sanitized:
        sanitized["LEI"] = flat.get("LEI")

    return sanitized


_XML_LANG_ATTR = "{http://www.w3.org/XML/1998/namespace}lang"


def _append_gleif_name_row(
    rows: list[dict[str, str | None]],
    name_element: ET.Element | None,
    *,
    lei: str | None,
    source_type: str | None,
    derivation_note: str,
) -> None:
    if name_element is None:
        return
    text = _join_text(name_element)
    if text is None:
        return
    rows.append(
        {
            "LEI": lei,
            "name": text,
            "source_type": source_type,
            "language_code": name_element.get(_XML_LANG_ATTR),
            "derivation_note": derivation_note,
        }
    )


def _extract_gleif_name_rows(
    lei_record: ET.Element,
    *,
    lei: str | None,
) -> list[dict[str, str | None]]:
    """Mechanically extract every native name form carried on one LEIRecord.

    Reads directly from the parsed XML tree rather than the generic
    flattened/pipe-joined columns produced by `_flatten_xml_company_fields`.
    That generic path dedups each column independently in `_append_value`,
    which desyncs the positional pairing between a name and its `@type` when
    multiple `OtherEntityName` entries share a type -- the common case for an
    entity with more than one previous name, not an edge case. Reading each
    element directly keeps every name's text and type as the naturally
    paired unit the XML already provides, so no pairing/dedup logic is
    needed here at all.
    """
    rows: list[dict[str, str | None]] = []

    legal_name = lei_record.find("./lei:Entity/lei:LegalName", _GLEIF_NS)
    _append_gleif_name_row(
        rows,
        legal_name,
        lei=lei,
        source_type="LEGAL_NAME",
        derivation_note="native LegalName",
    )

    for other_name in lei_record.findall(
        "./lei:Entity/lei:OtherEntityNames/lei:OtherEntityName", _GLEIF_NS
    ):
        _append_gleif_name_row(
            rows,
            other_name,
            lei=lei,
            source_type=other_name.get("type"),
            derivation_note="native OtherEntityName",
        )

    for transliterated_name in lei_record.findall(
        "./lei:Entity/lei:TransliteratedOtherEntityNames/lei:TransliteratedOtherEntityName",
        _GLEIF_NS,
    ):
        _append_gleif_name_row(
            rows,
            transliterated_name,
            lei=lei,
            source_type=transliterated_name.get("type"),
            derivation_note="native TransliteratedOtherEntityName",
        )

    return rows


def _extract_gleif_successor_rows(
    lei_record: ET.Element,
    *,
    lei: str | None,
) -> list[dict[str, str | None]]:
    """Mechanically extract every SuccessorEntity carried on one LEIRecord.

    Reads directly from the parsed XML tree for the same reason
    `_extract_gleif_name_rows` does: `SuccessorEntity` is `maxOccurs`
    unbounded (an entity can split into several successors), so the generic
    flattened/pipe-joined columns would desync pairing across multiple
    entries. `SuccessorEntity` is also a `choice` between `SuccessorLEI` and
    `SuccessorEntityName` -- a successor can be identified by name only, with
    no LEI -- so both are captured here even though only LEI-identified
    successors are walked downstream, so nothing from the XML is silently
    dropped at this layer.

    Also captures this record's own `Entity/LegalName` as `predecessor_name`.
    This is deliberate, not redundant with the names sidecar: canonical
    entity output already drops any row whose own SuccessorLEI is populated
    (see `canonical_frame_transform.py` -- "the successor's own row... always
    exists independently... to represent the entity"), so a predecessor's
    name is not otherwise recoverable from canonical entity files by the time
    a cross-file successor-chain pass could look it up. Capturing it here,
    at the one point the full record is still in hand, avoids depending on
    a canonical row that this same GLEIF-specific rule guarantees won't
    exist.
    """
    rows: list[dict[str, str | None]] = []

    successor_entities = lei_record.findall(
        "./lei:Entity/lei:SuccessorEntity", _GLEIF_NS
    )
    if not successor_entities:
        return rows

    predecessor_name = _join_text(
        lei_record.find("./lei:Entity/lei:LegalName", _GLEIF_NS)
    )

    for successor in successor_entities:
        successor_lei = _join_text(successor.find("./lei:SuccessorLEI", _GLEIF_NS))
        successor_name = _join_text(
            successor.find("./lei:SuccessorEntityName", _GLEIF_NS)
        )
        if successor_lei is None and successor_name is None:
            continue
        rows.append(
            {
                "LEI": lei,
                "predecessor_name": predecessor_name,
                "successor_lei": successor_lei,
                "successor_name": successor_name,
            }
        )

    return rows


def _record_matches_supported_countries(
    record: dict[str, str | None],
    supported_country_codes: frozenset[str] | None,
) -> bool:
    if supported_country_codes is None:
        return True

    has_country_value = False
    for field_name in _GLEIF_COUNTRY_FIELD_CANDIDATES:
        raw_value = record.get(field_name)
        if raw_value is None:
            continue
        has_country_value = True
        for part in raw_value.split(" | "):
            if part.strip().upper() in supported_country_codes:
                return True

    return not has_country_value


def _resolve_gleif_xml_member_name(source_path: Path) -> str | None:
    with zipfile.ZipFile(source_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        csv_members = [
            member for member in members if member.filename.lower().endswith(".csv")
        ]
        if csv_members:
            return None

        xml_members = [
            member for member in members if member.filename.lower().endswith(".xml")
        ]
        if not xml_members:
            return None

    return xml_members[0].filename


def _write_gleif_xml_zip_chunked_impl(
    source_path: Path,
    output_dir: Path,
    prefix: str,
    *,
    chunk_size: int,
    supported_country_codes: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
    sidecar_enabled: bool = False,
    materialize_system_uri: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
    | None = None,
    max_rows: int | None = None,
) -> tuple[list[Path], list[Path]] | None:
    # NOTE: DataFrame schema hint optimization was attempted but rejected (2026-07-08).
    # Pre-defining schema for all ~200+ XSD fields caused 21% regression vs baseline
    # because Polars' dynamic type inference is faster for sparse/dynamic records.
    # See docs/optimizations.md for details.

    # Optimization #6: Convert country codes to frozenset for O(1) lookup
    supported_country_codes_frozen: frozenset[str] | None = None
    if supported_country_codes is not None:
        supported_country_codes_frozen = frozenset(
            c.upper() for c in supported_country_codes
        )
    xml_member_name = _resolve_gleif_xml_member_name(source_path)
    if xml_member_name is None:
        return None

    rows: list[dict[str, str | None]] = []
    name_rows: list[dict[str, str | None]] = []
    successor_rows: list[dict[str, str | None]] = []
    chunk_index = 1
    written_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    name_written_paths: list[Path] = []
    name_sidecar_paths: list[Path] = []
    successor_written_paths: list[Path] = []
    successor_sidecar_paths: list[Path] = []
    chunk_started_at: float | None = None
    perf = ShardPerformance(prefix=prefix, progress=progress)
    name_perf = ShardPerformance(prefix=f"{prefix}-names", progress=progress)
    successor_perf = ShardPerformance(prefix=f"{prefix}-successors", progress=progress)

    def flush_current_rows() -> None:
        if rows:
            write_shard_chunk(
                rows=rows,
                output_dir=output_dir,
                prefix=prefix,
                chunk_index=chunk_index,
                sidecar_enabled=sidecar_enabled,
                split_main_sidecar=split_main_sidecar,
                materialize_system_uri=materialize_system_uri,
                written_paths=written_paths,
                sidecar_paths=sidecar_paths,
                perf=perf,
                chunk_started_at=chunk_started_at,
            )
        if name_rows:
            write_shard_chunk(
                rows=name_rows,
                output_dir=output_dir,
                prefix=f"{prefix}-names",
                chunk_index=chunk_index,
                sidecar_enabled=False,
                split_main_sidecar=None,
                materialize_system_uri=materialize_system_uri,
                written_paths=name_written_paths,
                sidecar_paths=name_sidecar_paths,
                perf=name_perf,
                chunk_started_at=chunk_started_at,
            )
        if successor_rows:
            write_shard_chunk(
                rows=successor_rows,
                output_dir=output_dir,
                prefix=f"{prefix}-successors",
                chunk_index=chunk_index,
                sidecar_enabled=False,
                split_main_sidecar=None,
                materialize_system_uri=materialize_system_uri,
                written_paths=successor_written_paths,
                sidecar_paths=successor_sidecar_paths,
                perf=successor_perf,
                chunk_started_at=chunk_started_at,
            )

    for element in iter_zip_xml_records(
        source_path,
        xml_member_name=xml_member_name,
        record_local_name="LEIRecord",
        expected_root_namespace="http://www.gleif.org/data/schema/leidata/2016",
        expected_root_local_name="LEIData",
    ):
        if not rows:
            chunk_started_at = time.perf_counter()
        extract_started_at = time.perf_counter()
        record = _extract_gleif_record(element)
        perf.add_extract_elapsed(time.perf_counter() - extract_started_at)
        if not _record_matches_supported_countries(
            record, supported_country_codes_frozen
        ):
            perf.increment_filtered_rows()
            continue
        rows.append(record)
        name_rows.extend(_extract_gleif_name_rows(element, lei=record.get("LEI")))
        successor_rows.extend(
            _extract_gleif_successor_rows(element, lei=record.get("LEI"))
        )
        perf.increment_total_rows()
        if max_rows is not None and perf.total_rows >= max_rows:
            # Flush remaining rows and stop processing
            flush_current_rows()
            rows = []
            name_rows = []
            successor_rows = []
            break
        if len(rows) >= chunk_size:
            flush_current_rows()
            rows = []
            name_rows = []
            successor_rows = []
            chunk_started_at = None
            chunk_index += 1

    flush_current_rows()

    perf.emit_summary()
    name_perf.emit_summary()
    successor_perf.emit_summary()

    return written_paths, sidecar_paths


def write_gleif_xml_zip_chunked(
    source_path: Path,
    output_dir: Path,
    prefix: str,
    *,
    chunk_size: int,
    supported_country_codes: set[str] | None = None,
    progress: Callable[[str], None] | None = None,
    sidecar_enabled: bool = False,
    materialize_system_uri: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
    | None = None,
    max_rows: int | None = None,
) -> tuple[list[Path], list[Path]] | None:
    return _write_gleif_xml_zip_chunked_impl(
        source_path,
        output_dir,
        prefix,
        chunk_size=chunk_size,
        supported_country_codes=supported_country_codes,
        progress=progress,
        sidecar_enabled=sidecar_enabled,
        materialize_system_uri=materialize_system_uri,
        split_main_sidecar=split_main_sidecar,
        max_rows=max_rows,
    )
