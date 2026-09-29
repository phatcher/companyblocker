from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

from defusedxml import ElementTree as ET


def xml_local_name(tag: str) -> str:
    brace_idx = tag.rfind("}")
    if brace_idx >= 0:
        return tag[brace_idx + 1 :]
    return tag


def xml_namespace_uri(tag: str) -> str | None:
    if not tag.startswith("{"):
        return None
    end = tag.find("}")
    if end < 0:
        return None
    return tag[1:end]


def _resolve_xml_member_name(
    archive: zipfile.ZipFile,
    *,
    xml_member_name: str | None,
) -> str | None:
    if xml_member_name is not None:
        return xml_member_name

    for member in archive.infolist():
        if member.filename.lower().endswith(".xml"):
            return member.filename
    return None


def _validate_root_element(
    *,
    source_path: Path,
    first_event: str,
    first_element: ET.Element,
    expected_root_namespace: str | None,
    expected_root_local_name: str | None,
) -> str | None:
    if first_event != "start":
        raise ValueError(
            f"Unexpected first XML parse event '{first_event}' in {source_path}"
        )

    root_local_name = xml_local_name(first_element.tag)
    root_namespace = xml_namespace_uri(first_element.tag)
    if (
        expected_root_local_name is not None
        and root_local_name != expected_root_local_name
    ):
        raise ValueError(
            f"Unexpected XML root local name '{root_local_name}' in {source_path}; "
            f"expected '{expected_root_local_name}'"
        )
    if (
        expected_root_namespace is not None
        and root_namespace != expected_root_namespace
    ):
        raise ValueError(
            f"Unexpected XML root namespace '{root_namespace}' in {source_path}; "
            f"expected '{expected_root_namespace}'"
        )
    return root_namespace


def _resolve_record_tag_matchers(
    *,
    record_local_name: str,
    root_namespace: str | None,
) -> tuple[str, str | None]:
    record_tag_exact = record_local_name
    record_namespaced_suffix = None
    if root_namespace is not None:
        record_tag_exact = f"{{{root_namespace}}}{record_local_name}"
    else:
        record_namespaced_suffix = f"}}{record_local_name}"
    return record_tag_exact, record_namespaced_suffix


def _is_matching_record_tag(
    *,
    tag: str,
    record_tag_exact: str,
    record_namespaced_suffix: str | None,
) -> bool:
    if tag == record_tag_exact:
        return True
    if record_namespaced_suffix is None:
        return False
    return tag.endswith(record_namespaced_suffix)


def iter_zip_xml_records(
    source_path: Path,
    *,
    record_local_name: str,
    xml_member_name: str | None = None,
    expected_root_namespace: str | None = None,
    expected_root_local_name: str | None = None,
) -> Iterator[ET.Element]:
    with zipfile.ZipFile(source_path) as archive:
        member_name = _resolve_xml_member_name(archive, xml_member_name=xml_member_name)
        if member_name is None:
            return

        with archive.open(member_name) as xml_stream:
            parser = ET.iterparse(xml_stream, events=("start", "end"))
            first_event, first_element = next(parser)
            root_namespace = _validate_root_element(
                source_path=source_path,
                first_event=first_event,
                first_element=first_element,
                expected_root_namespace=expected_root_namespace,
                expected_root_local_name=expected_root_local_name,
            )
            record_tag_exact, record_namespaced_suffix = _resolve_record_tag_matchers(
                record_local_name=record_local_name,
                root_namespace=root_namespace,
            )

            for event, element in parser:
                if event != "end":
                    continue
                if not _is_matching_record_tag(
                    tag=element.tag,
                    record_tag_exact=record_tag_exact,
                    record_namespaced_suffix=record_namespaced_suffix,
                ):
                    continue
                yield element
                element.clear()
