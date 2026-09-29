"""DBpedia sharding implementation.

Design rationale and parser tradeoffs are documented in
`src/acquisition/sharding_dbpedia.README.md`.
"""

from __future__ import annotations

import bz2
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

_DBPEDIA_TRIPLE_RE = re.compile(
    r'^\s*<(?P<subject>[^>]+)>\s+<(?P<predicate>[^>]+)>\s+(?:<(?P<object_uri>[^>]+)>|"(?P<literal>(?:\\.|[^"\\])*)"(?:@(?P<lang>[a-zA-Z-]+))?(?:\^\^<[^>]+>)?)\s*\.\s*$'
)
_DBPEDIA_ORG_TYPE_URIS = frozenset(
    {
        "http://dbpedia.org/ontology/Company",
        "http://dbpedia.org/ontology/Organisation",
        "http://dbpedia.org/ontology/Organization",
        "http://dbpedia.org/ontology/Business",
    }
)
_DBPEDIA_LABEL_PREDICATES = frozenset(
    {
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://xmlns.com/foaf/0.1/name",
    }
)
_DBPEDIA_COMPANY_NUMBER_PREDICATES = frozenset(
    {
        "http://dbpedia.org/property/companyNumber",
        "http://dbpedia.org/property/companyNo",
        "http://dbpedia.org/property/registrationNumber",
        "http://dbpedia.org/property/companyRegistrationNumber",
        "http://dbpedia.org/ontology/registrationNumber",
        "http://dbpedia.org/property/lei",
    }
)
_DBPEDIA_JURISDICTION_PREDICATES = frozenset(
    {
        "http://dbpedia.org/property/jurisdiction",
        "http://dbpedia.org/property/countryCode",
        "http://dbpedia.org/ontology/countryCode",
        "http://dbpedia.org/property/isoCode",
        "http://dbpedia.org/property/locationCountry",
    }
)
_DBPEDIA_ALIAS_PREDICATES = frozenset(
    {
        "http://dbpedia.org/ontology/alias",
        "http://dbpedia.org/property/alias",
    }
)
_DBPEDIA_LABEL_KEY_RE = re.compile(r"[^a-z0-9]+")


def _normalize_dbpedia_jurisdiction_code(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().upper()
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned
    return None


def _normalize_dbpedia_label_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = _DBPEDIA_LABEL_KEY_RE.sub(" ", value.strip().lower()).strip()
    if not cleaned:
        return None
    return cleaned


def _iter_dbpedia_triples(source_path: Path):
    stream_factory = bz2.open if source_path.suffix.lower() == ".bz2" else open
    with stream_factory(
        source_path, mode="rt", encoding="utf-8", errors="replace"
    ) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _DBPEDIA_TRIPLE_RE.match(line)
            if not match:
                continue
            literal_value = match.group("literal")
            if literal_value is not None:
                literal_value = bytes(literal_value, "utf-8").decode("unicode_escape")
            yield {
                "subject": match.group("subject"),
                "predicate": match.group("predicate"),
                "object_uri": match.group("object_uri"),
                "literal": literal_value,
                "lang": (match.group("lang") or "").lower(),
            }


def _resolve_dbpedia_resource_paths(
    resolved_resources: list[tuple[Any, Path, Path]],
) -> tuple[Path, Path, Path | None]:
    resource_by_name = {
        resource.name: source_path for resource, _, source_path in resolved_resources
    }
    instance_types_path = resource_by_name.get(
        "dbpedia_mappings_instance_types_en_latest"
    )
    labels_path = resource_by_name.get("dbpedia_generic_labels_en_latest")
    literals_path = resource_by_name.get(
        "dbpedia_mappings_mappingbased_literals_en_latest"
    )
    if instance_types_path is None or labels_path is None:
        raise RuntimeError(
            "DBpedia sharding requires instance-types and labels resources to be present."
        )
    return instance_types_path, labels_path, literals_path


def _collect_dbpedia_org_subjects(instance_types_path: Path) -> set[str]:
    org_subjects: set[str] = set()
    for triple in _iter_dbpedia_triples(instance_types_path):
        object_uri = triple["object_uri"]
        if object_uri in _DBPEDIA_ORG_TYPE_URIS:
            org_subjects.add(str(triple["subject"]))
    return org_subjects


def _collect_dbpedia_english_labels(labels_path: Path) -> dict[str, str]:
    labels_by_subject: dict[str, str] = {}
    for triple in _iter_dbpedia_triples(labels_path):
        if triple["predicate"] not in _DBPEDIA_LABEL_PREDICATES:
            continue
        if triple["lang"] != "en":
            continue
        subject = str(triple["subject"])
        label = str(triple["literal"] or "").strip()
        if not label or subject in labels_by_subject:
            continue
        labels_by_subject[subject] = label
    return labels_by_subject


def _collect_dbpedia_literals_metadata(
    *,
    literals_path: Path | None,
    org_subjects: set[str],
    enable_link_inference: bool,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    company_number_by_subject: dict[str, str] = {}
    jurisdiction_by_subject: dict[str, str] = {}
    alias_subjects_by_key: dict[str, set[str]] = {}
    if literals_path is None:
        return company_number_by_subject, jurisdiction_by_subject, alias_subjects_by_key

    for triple in _iter_dbpedia_triples(literals_path):
        subject = str(triple["subject"])
        if subject not in org_subjects:
            continue

        predicate = str(triple["predicate"])
        literal = str(triple["literal"] or "").strip()
        if not literal:
            continue

        lang = str(triple["lang"] or "")
        if lang and lang != "en":
            continue

        if (
            predicate in _DBPEDIA_COMPANY_NUMBER_PREDICATES
            and subject not in company_number_by_subject
        ):
            company_number_by_subject[subject] = literal
            continue

        if (
            predicate in _DBPEDIA_JURISDICTION_PREDICATES
            and subject not in jurisdiction_by_subject
        ):
            normalized_code = _normalize_dbpedia_jurisdiction_code(literal)
            if normalized_code is not None:
                jurisdiction_by_subject[subject] = normalized_code

        if enable_link_inference and predicate in _DBPEDIA_ALIAS_PREDICATES:
            label_key = _normalize_dbpedia_label_key(literal)
            if label_key is not None:
                alias_subjects_by_key.setdefault(label_key, set()).add(subject)

    return company_number_by_subject, jurisdiction_by_subject, alias_subjects_by_key


def _build_alias_unique_subject_map(
    alias_subjects_by_key: dict[str, set[str]],
) -> dict[str, str]:
    alias_unique_subject_by_key: dict[str, str] = {}
    for label_key, subjects in alias_subjects_by_key.items():
        if len(subjects) == 1:
            alias_unique_subject_by_key[label_key] = min(subjects)
    return alias_unique_subject_by_key


def _build_dbpedia_company_rows(
    *,
    org_subjects: set[str],
    labels_by_subject: dict[str, str],
    jurisdiction_by_subject: dict[str, str],
    company_number_by_subject: dict[str, str],
) -> list[dict[str, str | None]]:
    def build_company_row(
        *,
        company_name: str,
        subject: str,
        metadata_subject: str,
        inclusion_basis: str,
        inclusion_flags: str,
        canonical_dbpedia_uri: str,
        inference_method: str | None,
        evidence_confidence: str,
    ) -> dict[str, str | None]:
        return {
            "CompanyName": company_name,
            "dbpedia_uri": subject,
            "jurisdiction_code": jurisdiction_by_subject.get(metadata_subject),
            "company_number": company_number_by_subject.get(metadata_subject),
            "inclusion_basis": inclusion_basis,
            "inclusion_flags": inclusion_flags,
            "canonical_dbpedia_uri": canonical_dbpedia_uri,
            "inference_method": inference_method,
            "evidence_confidence": evidence_confidence,
        }

    rows: list[dict[str, str | None]] = []
    for subject in sorted(org_subjects):
        label = labels_by_subject.get(subject)
        if label is None:
            continue
        rows.append(
            build_company_row(
                company_name=label,
                subject=subject,
                metadata_subject=subject,
                inclusion_basis="ontology_type",
                inclusion_flags="ontology_type",
                canonical_dbpedia_uri=subject,
                inference_method=None,
                evidence_confidence="high",
            )
        )
    return rows


def _append_dbpedia_inferred_rows(
    *,
    rows: list[dict[str, str | None]],
    labels_by_subject: dict[str, str],
    org_subjects: set[str],
    alias_unique_subject_by_key: dict[str, str],
    company_number_by_subject: dict[str, str],
    jurisdiction_by_subject: dict[str, str],
) -> int:
    inferred_rows = 0
    for subject, label in sorted(labels_by_subject.items()):
        if subject in org_subjects:
            continue

        label_key = _normalize_dbpedia_label_key(label)
        if label_key is None:
            continue

        canonical_subject = alias_unique_subject_by_key.get(label_key)
        if canonical_subject is None:
            continue

        rows.append(
            {
                "CompanyName": label,
                "dbpedia_uri": subject,
                "jurisdiction_code": jurisdiction_by_subject.get(canonical_subject),
                "company_number": company_number_by_subject.get(canonical_subject),
                "inclusion_basis": "link_inferred",
                "inclusion_flags": "ontology_type,alias",
                "canonical_dbpedia_uri": canonical_subject,
                "inference_method": "alias_literal_exact",
                "evidence_confidence": "medium",
            }
        )
        inferred_rows += 1
    return inferred_rows


class DbpediaShardHandler:
    def build_company_frame(
        self,
        resolved_resources: list[tuple[Any, Path, Path]],
        *,
        enable_link_inference: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> pl.DataFrame:
        instance_types_path, labels_path, literals_path = (
            _resolve_dbpedia_resource_paths(resolved_resources)
        )
        org_subjects = _collect_dbpedia_org_subjects(instance_types_path)
        labels_by_subject = _collect_dbpedia_english_labels(labels_path)
        company_number_by_subject, jurisdiction_by_subject, alias_subjects_by_key = (
            _collect_dbpedia_literals_metadata(
                literals_path=literals_path,
                org_subjects=org_subjects,
                enable_link_inference=enable_link_inference,
            )
        )
        alias_unique_subject_by_key = _build_alias_unique_subject_map(
            alias_subjects_by_key
        )

        rows = _build_dbpedia_company_rows(
            org_subjects=org_subjects,
            labels_by_subject=labels_by_subject,
            jurisdiction_by_subject=jurisdiction_by_subject,
            company_number_by_subject=company_number_by_subject,
        )

        inferred_rows = 0
        if enable_link_inference:
            inferred_rows = _append_dbpedia_inferred_rows(
                rows=rows,
                labels_by_subject=labels_by_subject,
                org_subjects=org_subjects,
                alias_unique_subject_by_key=alias_unique_subject_by_key,
                company_number_by_subject=company_number_by_subject,
                jurisdiction_by_subject=jurisdiction_by_subject,
            )

        if progress is not None:
            if enable_link_inference:
                progress(
                    f"[dbpedia] org subjects={len(org_subjects):,}, english labels={len(labels_by_subject):,}, alias keys={len(alias_unique_subject_by_key):,}, inferred rows={inferred_rows:,}, matched companies={len(rows):,}"
                )
            else:
                progress(
                    f"[dbpedia] org subjects={len(org_subjects):,}, english labels={len(labels_by_subject):,}, matched companies={len(rows):,}"
                )

        return pl.DataFrame(
            rows,
            schema={
                "CompanyName": pl.Utf8,
                "dbpedia_uri": pl.Utf8,
                "jurisdiction_code": pl.Utf8,
                "company_number": pl.Utf8,
                "inclusion_basis": pl.Utf8,
                "inclusion_flags": pl.Utf8,
                "canonical_dbpedia_uri": pl.Utf8,
                "inference_method": pl.Utf8,
                "evidence_confidence": pl.Utf8,
            },
        )


_DBPEDIA_SHARD_HANDLER = DbpediaShardHandler()


def build_dbpedia_company_frame(
    resolved_resources: list[tuple[Any, Path, Path]],
    *,
    enable_link_inference: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pl.DataFrame:
    return _DBPEDIA_SHARD_HANDLER.build_company_frame(
        resolved_resources,
        enable_link_inference=enable_link_inference,
        progress=progress,
    )
