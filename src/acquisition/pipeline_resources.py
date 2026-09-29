from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from workspace.roots import WorkspaceRoots

from .downloader import sha256_file
from .extractors import extract_zip_archive
from .models import ArtifactRecord

_WIKIDATA_COMPANY_CHUNK_DIR_NAME = "wikidata-companies.chunks"
_WIKIDATA_PROJECTED_JSONL_NAME = "wikidata-companies.jsonl"


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _latest_acquired_source_file(
    acquire_root: Path,
    *,
    resource,
    parse_iso_date_fn: Callable[[str | None], date | None],
) -> tuple[date | None, Path | None]:
    latest_date: date | None = None
    latest_path: Path | None = None

    if not acquire_root.exists():
        return None, None

    for candidate_dir in sorted(
        (path for path in acquire_root.glob("*") if path.is_dir()),
        key=lambda path: path.name,
    ):
        candidate_date = parse_iso_date_fn(candidate_dir.name)
        if candidate_date is None:
            continue

        candidate_path = candidate_dir / resource.resolve_file_name(candidate_dir.name)
        if not candidate_path.exists():
            continue

        if latest_date is None or candidate_date > latest_date:
            latest_date = candidate_date
            latest_path = candidate_path

    return latest_date, latest_path


def _resolve_adapter_handler(
    adapter: str,
    *,
    adapter_handlers: Mapping[str, Callable[..., str]],
) -> Callable[..., str]:
    handler = adapter_handlers.get(adapter)
    if handler is None:
        raise KeyError(adapter)
    return handler


def _build_wikidata_projection_artifact(
    *,
    roots: WorkspaceRoots,
    system_code: str,
    resource,
    source_destination: Path,
    prepare_dir_for_resource: Path,
    snapshot_date: str,
    effective_run_date: str,
    now_utc: str,
    last_acquired_at: str | None,
    days_since_last: int | None,
    freshness_applied: bool,
    freshness_passed: bool,
    freshness_reason: str,
    emit: Callable[[str], None],
    extract_wikidata_projection_fn: Callable[..., int],
) -> ArtifactRecord | None:
    if system_code != "wikidata" or resource.name != "wikidata_entities_latest_all":
        return None

    source_snapshot_date = snapshot_date
    source_snapshot_candidate = _parse_iso_date(source_destination.parent.name)
    if source_snapshot_candidate is not None:
        source_snapshot_date = source_snapshot_candidate.isoformat()

    prepare_dir_for_resource = prepare_dir_for_resource.parent / source_snapshot_date
    prepare_dir_for_resource.mkdir(parents=True, exist_ok=True)
    projected_path = prepare_dir_for_resource / _WIKIDATA_PROJECTED_JSONL_NAME
    projected_chunk_dir = prepare_dir_for_resource / _WIKIDATA_COMPANY_CHUNK_DIR_NAME
    has_chunked_output = projected_chunk_dir.exists() and any(
        chunk_path.is_file() for chunk_path in projected_chunk_dir.rglob("*.jsonl")
    )
    has_non_empty_projected_file = (
        projected_path.exists() and projected_path.stat().st_size > 0
    )

    if has_non_empty_projected_file:
        emit(
            f"[{system_code}] reusing existing projected source artifact: "
            f"{projected_path.relative_to(roots.data).as_posix()}"
        )
        try:
            projected_row_count = sum(
                1
                for line in projected_path.open("r", encoding="utf-8", errors="replace")
                if line.strip()
            )
        except OSError:
            projected_row_count = None
    elif has_chunked_output:
        emit(
            f"[{system_code}] reusing existing projected chunk directory: "
            f"{projected_chunk_dir.relative_to(roots.data).as_posix()}"
        )
        projected_row_count = None
    else:
        emit(
            f"[{system_code}] projecting company-like entities for sharding: "
            f"{projected_path.relative_to(roots.data).as_posix()}"
        )
        read_options_resolver = getattr(resource, "resolve_read_options_for_path", None)
        resolved_read_options = (
            read_options_resolver(source_destination)
            if callable(read_options_resolver)
            else {}
        )
        projection_kwargs: dict[str, object] = {"progress": emit}
        if resolved_read_options:
            projection_kwargs["resolved_read_options"] = resolved_read_options
        if getattr(resource, "projection_defaults", None) is not None:
            projection_kwargs["projection_defaults"] = resource.projection_defaults

        extract_signature = inspect.signature(extract_wikidata_projection_fn)
        if "projection_defaults" not in extract_signature.parameters:
            projection_kwargs.pop("projection_defaults", None)
        if "stream_resume" in extract_signature.parameters:
            projection_kwargs["stream_resume"] = True

        projected_row_count = extract_wikidata_projection_fn(
            source_destination,
            projected_path,
            **projection_kwargs,
        )

    return ArtifactRecord(
        country=system_code,
        run_date=effective_run_date,
        artifact_type="source_derived_file",
        source_name=f"{resource.name}_companies",
        source_url=resource.url,
        format="jsonl",
        file_path=projected_path.relative_to(roots.data).as_posix(),
        content_hash=None,
        row_count=projected_row_count,
        note="Projected company-like Wikidata entities with selected fields",
        source_snapshot_date=source_snapshot_date,
        publication_frequency=resource.publication_frequency,
        refresh_if_older_than_days=resource.refresh_if_older_than_days,
        acquired_at_utc=now_utc,
        last_acquired_at_utc=last_acquired_at,
        days_since_last_acquisition=days_since_last,
        freshness_gate_applied=freshness_applied,
        freshness_gate_passed=freshness_passed,
        freshness_gate_reason=freshness_reason,
    )


def _resolve_resource_freshness_state(
    *,
    effective_run_date: str,
    expected_source_path: Path,
    acquire_dir: Path,
    resource,
    parse_iso_date_fn: Callable[[str | None], date | None],
) -> tuple[Path | None, date | None, str | None, int | None, bool, bool, str]:
    current_date = date.fromisoformat(effective_run_date)
    if expected_source_path.exists():
        last_date: date | None = current_date
        last_acquired_at: str | None = None
        days_since_last: int | None = 0
        freshness_applied = resource.refresh_if_older_than_days is not None
        freshness_passed = True
        freshness_reason = "Current acquisition artifact already exists on disk"
        return (
            expected_source_path,
            last_date,
            last_acquired_at,
            days_since_last,
            freshness_applied,
            freshness_passed,
            freshness_reason,
        )

    latest_date, latest_existing_source_path = _latest_acquired_source_file(
        acquire_dir,
        resource=resource,
        parse_iso_date_fn=parse_iso_date_fn,
    )
    last_date = latest_date
    last_acquired_at = None
    days_since_last = (current_date - last_date).days if last_date else None
    freshness_applied = resource.refresh_if_older_than_days is not None
    freshness_passed = True
    freshness_reason = "No freshness gate configured"

    if freshness_applied:
        threshold = resource.refresh_if_older_than_days
        if days_since_last is None:
            freshness_passed = True
            freshness_reason = "No prior acquisition found on disk"
        elif days_since_last >= threshold:
            freshness_passed = True
            freshness_reason = f"Last acquisition is {days_since_last} day(s) old, meets threshold {threshold}"
        else:
            freshness_passed = False
            freshness_reason = f"Last acquisition is {days_since_last} day(s) old, below threshold {threshold}"

    return (
        latest_existing_source_path,
        last_date,
        last_acquired_at,
        days_since_last,
        freshness_applied,
        freshness_passed,
        freshness_reason,
    )


def _append_freshness_skip_artifacts(
    *,
    artifact_rows: list[ArtifactRecord],
    roots: WorkspaceRoots,
    system_code: str,
    resource,
    expected_source_path: Path,
    latest_existing_source_path: Path | None,
    prepare_dir_for_resource: Path,
    snapshot_date: str,
    effective_run_date: str,
    now_utc: str,
    last_acquired_at: str | None,
    days_since_last: int | None,
    freshness_applied: bool,
    freshness_passed: bool,
    freshness_reason: str,
    emit: Callable[[str], None],
    extract_wikidata_projection_fn: Callable[..., int],
) -> None:
    emit(f"[{system_code}] freshness skip for '{resource.name}': {freshness_reason}")
    emit(
        f"[{system_code}] To force re-download, delete: "
        f"{(latest_existing_source_path or expected_source_path).relative_to(roots.data).as_posix()}"
    )

    destination = (
        expected_source_path
        if expected_source_path.exists()
        else latest_existing_source_path
    )
    if destination is not None:
        projection_artifact = _build_wikidata_projection_artifact(
            roots=roots,
            system_code=system_code,
            resource=resource,
            source_destination=destination,
            prepare_dir_for_resource=prepare_dir_for_resource,
            snapshot_date=snapshot_date,
            effective_run_date=effective_run_date,
            now_utc=now_utc,
            last_acquired_at=last_acquired_at,
            days_since_last=days_since_last,
            freshness_applied=freshness_applied,
            freshness_passed=freshness_passed,
            freshness_reason=freshness_reason,
            emit=emit,
            extract_wikidata_projection_fn=extract_wikidata_projection_fn,
        )
        if projection_artifact is not None:
            artifact_rows.append(projection_artifact)

    artifact_rows.append(
        ArtifactRecord(
            country=system_code,
            run_date=effective_run_date,
            artifact_type="freshness_skip",
            source_name=resource.name,
            source_url=resource.url,
            format=resource.file_format,
            file_path="",
            note=resource.notes,
            source_snapshot_date=snapshot_date,
            publication_frequency=resource.publication_frequency,
            refresh_if_older_than_days=resource.refresh_if_older_than_days,
            acquired_at_utc=now_utc,
            last_acquired_at_utc=last_acquired_at,
            days_since_last_acquisition=days_since_last,
            freshness_gate_applied=freshness_applied,
            freshness_gate_passed=freshness_passed,
            freshness_gate_reason=freshness_reason,
        )
    )


def _acquire_or_reuse_source_artifact(
    *,
    roots: WorkspaceRoots,
    system_code: str,
    resource,
    effective_run_date: str,
    destination: Path,
    adapter_handlers: Mapping[str, Callable[..., str]],
    emit: Callable[[str], None],
    sha256_file_fn: Callable[[Path], str],
) -> str:
    download_source = resource.resolve_download_url(effective_run_date)
    effective_resource = replace(resource, url=download_source)
    if destination.exists():
        emit(
            f"[{system_code}] reusing existing source artifact for '{resource.name}': "
            f"{destination.relative_to(roots.data).as_posix()}"
        )
        return sha256_file_fn(destination)

    emit(f"[{system_code}] downloading '{resource.name}' from {download_source}")
    try:
        adapter_handler = _resolve_adapter_handler(
            effective_resource.adapter,
            adapter_handlers=adapter_handlers,
        )
    except KeyError:
        raise RuntimeError(
            f"Unsupported adapter '{effective_resource.adapter}' for source '{effective_resource.name}'."
        )
    return adapter_handler(effective_resource, destination)


def _append_source_file_artifact(
    *,
    artifact_rows: list[ArtifactRecord],
    roots: WorkspaceRoots,
    system_code: str,
    resource,
    effective_run_date: str,
    destination: Path,
    content_hash: str,
    snapshot_date: str,
    now_utc: str,
    last_acquired_at: str | None,
    days_since_last: int | None,
    freshness_applied: bool,
    freshness_passed: bool,
    freshness_reason: str,
) -> None:
    artifact_rows.append(
        ArtifactRecord(
            country=system_code,
            run_date=effective_run_date,
            artifact_type="source_file",
            source_name=resource.name,
            source_url=resource.url,
            format=resource.file_format,
            file_path=destination.relative_to(roots.data).as_posix(),
            content_hash=content_hash,
            note=resource.notes,
            source_snapshot_date=snapshot_date,
            publication_frequency=resource.publication_frequency,
            refresh_if_older_than_days=resource.refresh_if_older_than_days,
            acquired_at_utc=now_utc,
            last_acquired_at_utc=last_acquired_at,
            days_since_last_acquisition=days_since_last,
            freshness_gate_applied=freshness_applied,
            freshness_gate_passed=freshness_passed,
            freshness_gate_reason=freshness_reason,
        )
    )


def _append_zip_extracted_artifacts(
    *,
    artifact_rows: list[ArtifactRecord],
    roots: WorkspaceRoots,
    system_code: str,
    resource,
    effective_run_date: str,
    destination: Path,
    prepare_dir_for_resource: Path,
    snapshot_date: str,
    now_utc: str,
    last_acquired_at: str | None,
    days_since_last: int | None,
    freshness_applied: bool,
    freshness_passed: bool,
    freshness_reason: str,
    emit: Callable[[str], None],
    extract_zip_archive_fn: Callable[[Path, Path], list[Path]],
    sha256_file_fn: Callable[[Path], str],
) -> None:
    if not resource.extract or resource.file_format.lower() != "zip":
        return

    extracted_dir = prepare_dir_for_resource / "extracted"
    emit(f"[{system_code}] extracting zip for '{resource.name}'")
    extracted_paths = extract_zip_archive_fn(destination, extracted_dir)
    for extracted_path in extracted_paths:
        artifact_rows.append(
            ArtifactRecord(
                country=system_code,
                run_date=effective_run_date,
                artifact_type="extracted_file",
                source_name=resource.name,
                source_url=resource.url,
                format=extracted_path.suffix.lstrip(".") or "file",
                file_path=extracted_path.relative_to(roots.data).as_posix(),
                content_hash=sha256_file_fn(extracted_path),
                source_snapshot_date=snapshot_date,
                publication_frequency=resource.publication_frequency,
                refresh_if_older_than_days=resource.refresh_if_older_than_days,
                acquired_at_utc=now_utc,
                last_acquired_at_utc=last_acquired_at,
                days_since_last_acquisition=days_since_last,
                freshness_gate_applied=freshness_applied,
                freshness_gate_passed=freshness_passed,
                freshness_gate_reason=freshness_reason,
            )
        )

    emit(
        f"[{system_code}] extracted {len(extracted_paths)} file(s) for '{resource.name}'"
    )


def collect_resource_artifacts(
    *,
    roots: WorkspaceRoots,
    plan,
    paths,
    effective_run_date: str,
    adapter_handlers: Mapping[str, Callable[..., str]],
    emit: Callable[[str], None],
    extract_wikidata_projection_fn: Callable[..., int],
    extract_zip_archive_fn: Callable[[Path, Path], list[Path]] = extract_zip_archive,
    sha256_file_fn: Callable[[Path], str] = sha256_file,
    parse_iso_date_fn=_parse_iso_date,
) -> tuple[list[ArtifactRecord], bool, int]:
    system_code = plan.code
    artifact_rows: list[ArtifactRecord] = []
    downloaded_any = False
    skipped_for_freshness = 0

    for resource in plan.resources:
        emit(f"[{system_code}] processing resource '{resource.name}'")
        now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshot_date = resource.resolve_snapshot_date(effective_run_date)
        acquire_dir_for_resource = paths.acquire_dir / snapshot_date
        prepare_dir_for_resource = paths.prepare_dir / snapshot_date
        expected_source_path = acquire_dir_for_resource / resource.resolve_file_name(
            effective_run_date
        )
        (
            latest_existing_source_path,
            _last_date,
            last_acquired_at,
            days_since_last,
            freshness_applied,
            freshness_passed,
            freshness_reason,
        ) = _resolve_resource_freshness_state(
            effective_run_date=effective_run_date,
            expected_source_path=expected_source_path,
            acquire_dir=paths.acquire_dir,
            resource=resource,
            parse_iso_date_fn=parse_iso_date_fn,
        )

        if not freshness_passed:
            skipped_for_freshness += 1
            _append_freshness_skip_artifacts(
                artifact_rows=artifact_rows,
                roots=roots,
                system_code=system_code,
                resource=resource,
                expected_source_path=expected_source_path,
                latest_existing_source_path=latest_existing_source_path,
                prepare_dir_for_resource=prepare_dir_for_resource,
                snapshot_date=snapshot_date,
                effective_run_date=effective_run_date,
                now_utc=now_utc,
                last_acquired_at=last_acquired_at,
                days_since_last=days_since_last,
                freshness_applied=freshness_applied,
                freshness_passed=freshness_passed,
                freshness_reason=freshness_reason,
                emit=emit,
                extract_wikidata_projection_fn=extract_wikidata_projection_fn,
            )
            continue

        acquire_dir_for_resource.mkdir(parents=True, exist_ok=True)
        destination = expected_source_path
        content_hash = _acquire_or_reuse_source_artifact(
            roots=roots,
            system_code=system_code,
            resource=resource,
            effective_run_date=effective_run_date,
            destination=destination,
            adapter_handlers=adapter_handlers,
            emit=emit,
            sha256_file_fn=sha256_file_fn,
        )
        downloaded_any = True
        emit(
            f"[{system_code}] wrote source artifact to {destination.relative_to(roots.data).as_posix()}"
        )
        _append_source_file_artifact(
            artifact_rows=artifact_rows,
            roots=roots,
            system_code=system_code,
            resource=resource,
            effective_run_date=effective_run_date,
            destination=destination,
            content_hash=content_hash,
            snapshot_date=snapshot_date,
            now_utc=now_utc,
            last_acquired_at=last_acquired_at,
            days_since_last=days_since_last,
            freshness_applied=freshness_applied,
            freshness_passed=freshness_passed,
            freshness_reason=freshness_reason,
        )

        projection_artifact = _build_wikidata_projection_artifact(
            roots=roots,
            system_code=system_code,
            resource=resource,
            source_destination=destination,
            prepare_dir_for_resource=prepare_dir_for_resource,
            snapshot_date=snapshot_date,
            effective_run_date=effective_run_date,
            now_utc=now_utc,
            last_acquired_at=last_acquired_at,
            days_since_last=days_since_last,
            freshness_applied=freshness_applied,
            freshness_passed=freshness_passed,
            freshness_reason=freshness_reason,
            emit=emit,
            extract_wikidata_projection_fn=extract_wikidata_projection_fn,
        )
        if projection_artifact is not None:
            artifact_rows.append(projection_artifact)

        _append_zip_extracted_artifacts(
            artifact_rows=artifact_rows,
            roots=roots,
            system_code=system_code,
            resource=resource,
            effective_run_date=effective_run_date,
            destination=destination,
            prepare_dir_for_resource=prepare_dir_for_resource,
            snapshot_date=snapshot_date,
            now_utc=now_utc,
            last_acquired_at=last_acquired_at,
            days_since_last=days_since_last,
            freshness_applied=freshness_applied,
            freshness_passed=freshness_passed,
            freshness_reason=freshness_reason,
            emit=emit,
            extract_zip_archive_fn=extract_zip_archive_fn,
            sha256_file_fn=sha256_file_fn,
        )

    return artifact_rows, downloaded_any, skipped_for_freshness
