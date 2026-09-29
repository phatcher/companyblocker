"""Wikidata's Shard stage, over the company extract `downloader_wikidata` projects from the dump.

An existing non-empty extract is reused rather than rescanning the dump unless `force` asks for a fresh extraction, and explicit shard limits are checked against it first: `shard.max_companies` reuses the extract if it already holds that many rows and re-extracts otherwise, and `shard.max_lines` caps the dump lines a fresh extraction scans. A chunk-directory extract (`wikidata-companies.chunks/`) is still read where one exists, though neither engine writes that layout.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from workspace.roots import WorkspaceRoots

WIKIDATA_COMPANY_DUMP_RESOURCE = "wikidata_entities_latest_all"
WIKIDATA_COMPANY_CHUNK_DIR_NAME = "wikidata-companies.chunks"
WIKIDATA_PROJECTED_JSONL_NAME = "wikidata-companies.jsonl"
WIKIDATA_PROJECTED_SPEC_SIDECAR_SUFFIX = ".spec.json"


def _wikidata_projected_spec_sidecar_path(projected_path: Path) -> Path:
    return projected_path.with_name(
        projected_path.name + WIKIDATA_PROJECTED_SPEC_SIDECAR_SUFFIX
    )


def _write_wikidata_projected_spec_sidecar(
    *, projected_path: Path, projection_defaults: dict[str, object]
) -> None:
    """Records the effective `projection_defaults` (engine, spec_path,
    binary_path, output_mode -- whatever keys the caller configured) that
    produced `projected_path`, so a later run can tell whether reusing this
    artifact still matches the spec it would otherwise re-extract with.
    """
    sidecar_path = _wikidata_projected_spec_sidecar_path(projected_path)
    sidecar_path.write_text(
        json.dumps(projection_defaults, sort_keys=True), encoding="utf-8"
    )


def _read_wikidata_projected_spec_sidecar(
    projected_path: Path,
) -> dict[str, object] | None:
    """Returns the spec recorded for `projected_path` by a prior extraction,
    or None when no sidecar exists (for example: an artifact that predates
    this recording, or one produced outside this pipeline) or it cannot be
    parsed. None means "unknown", not "matches" -- callers must not treat it
    as a positive match.
    """
    sidecar_path = _wikidata_projected_spec_sidecar_path(projected_path)
    if not sidecar_path.exists():
        return None
    try:
        loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _resolve_effective_wikidata_projection_defaults(
    *,
    resource: Any,
    projection_defaults_override: dict[str, object] | None,
) -> dict[str, object]:
    projection_defaults = dict(getattr(resource, "projection_defaults", None) or {})
    if projection_defaults_override is not None:
        projection_defaults.update(projection_defaults_override)
    return projection_defaults


def _maybe_trigger_extraction(
    *,
    condition: bool,
    message: str,
    selected_prepare_output_path: Path,
    roots: WorkspaceRoots,
    emit: Callable[[str], None],
) -> bool:
    if not condition:
        return False
    emit(message + selected_prepare_output_path.relative_to(roots.data).as_posix())
    return True


class WikidataShardHandler:
    def resolve_shard_source(
        self,
        *,
        resource_name: str,
        snapshot_date: str,
        prepare_root: Path,
        source_path: Path,
        roots: WorkspaceRoots,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[Path, str | None]:
        """Resolve the prepared Wikidata shard input.

        Wikidata sharding requires the prepare stage to have produced the projected
        JSONL extract. If that file is missing, sharding must fail rather than fall
        back to the raw dump.
        """
        if resource_name != WIKIDATA_COMPANY_DUMP_RESOURCE:
            return source_path, None

        if source_path.exists():
            if source_path.is_file() and source_path.suffix.lower() == ".jsonl":
                if progress is not None:
                    progress(
                        "[wikidata] using projected source artifact "
                        + source_path.relative_to(roots.data).as_posix()
                    )
                return source_path, "jsonl"

            if source_path.is_dir() and _has_jsonl_files(source_path):
                if progress is not None:
                    progress(
                        "[wikidata] using projected chunk directory "
                        + source_path.relative_to(roots.data).as_posix()
                    )
                return source_path, "jsonl"

        projected_chunk_dir = (
            prepare_root / snapshot_date / WIKIDATA_COMPANY_CHUNK_DIR_NAME
        )
        if projected_chunk_dir.exists() and _has_jsonl_files(projected_chunk_dir):
            if progress is not None:
                progress(
                    "[wikidata] using projected chunk directory "
                    + projected_chunk_dir.relative_to(roots.data).as_posix()
                )
            return projected_chunk_dir, "jsonl"

        projected_path = prepare_root / snapshot_date / WIKIDATA_PROJECTED_JSONL_NAME
        if projected_path.exists():
            if progress is not None:
                progress(
                    "[wikidata] using projected source artifact "
                    + projected_path.relative_to(roots.data).as_posix()
                )
            return projected_path, "jsonl"

        raise FileNotFoundError(f"Wikidata prepare output not found: {projected_path}")

    def prepare_shard_input(
        self,
        *,
        resource: Any,
        acquire_root: Path,
        prepare_root: Path,
        effective_run_date: str,
        run_date_provided: bool,
        roots: WorkspaceRoots,
        max_companies: int | None,
        max_lines: int | None,
        prepare_wiring_mode: str,
        prepare_stream_resume: bool,
        projection_defaults_override: dict[str, object] | None,
        emit: Callable[[str], None],
        resolve_source_artifact: Callable[..., tuple[Path, Path]],
        extract_two_pass: Callable[..., int],
        force: bool = False,
    ) -> tuple[Path, Path] | None:
        if resource.name != WIKIDATA_COMPANY_DUMP_RESOURCE:
            return None

        acquire_dir, acquire_path = resolve_source_artifact(
            source_root=acquire_root,
            resource=resource,
            effective_run_date=effective_run_date,
            run_date_provided=run_date_provided,
        )

        prepare_dir = prepare_root / acquire_dir.name
        prepare_dir.mkdir(parents=True, exist_ok=True)
        projected_path = prepare_dir / WIKIDATA_PROJECTED_JSONL_NAME
        projected_chunk_dir = prepare_dir / WIKIDATA_COMPANY_CHUNK_DIR_NAME
        has_projected_file = (
            projected_path.exists() and projected_path.stat().st_size > 0
        )
        has_chunked_output = projected_chunk_dir.exists() and _has_jsonl_files(
            projected_chunk_dir
        )
        limit_overrides_requested = max_companies is not None or max_lines is not None

        prefer_chunked_prepare_output = prepare_wiring_mode == "stream"
        selected_prepare_output_path = (
            projected_chunk_dir if prefer_chunked_prepare_output else projected_path
        )

        if limit_overrides_requested:
            emit(
                "[wikidata] stage prepare: applying explicit limits using resumable projected artifact "
                + selected_prepare_output_path.relative_to(roots.data).as_posix()
            )

        effective_projection_defaults = _resolve_effective_wikidata_projection_defaults(
            resource=resource,
            projection_defaults_override=projection_defaults_override,
        )

        should_extract = force or (not has_projected_file and not has_chunked_output)

        if force and (has_projected_file or has_chunked_output):
            emit(
                "[wikidata] stage prepare: force requested; re-extracting despite existing projected artifact "
                + selected_prepare_output_path.relative_to(roots.data).as_posix()
            )

        if _maybe_trigger_extraction(
            condition=(
                prefer_chunked_prepare_output
                and prepare_stream_resume
                and has_chunked_output
                and not limit_overrides_requested
                and not force
            ),
            message="[wikidata] stage prepare: found projected chunks; invoking resumable stream projection ",
            selected_prepare_output_path=selected_prepare_output_path,
            roots=roots,
            emit=emit,
        ):
            should_extract = True

        if has_projected_file and not has_chunked_output and not should_extract:
            recorded_spec = _read_wikidata_projected_spec_sidecar(projected_path)
            reuse_message = (
                "[wikidata] stage prepare: reusing existing projected artifact "
                + projected_path.relative_to(roots.data).as_posix()
            )
            if (
                recorded_spec is not None
                and recorded_spec != effective_projection_defaults
            ):
                # A silent no-op here is exactly the stale-artifact failure mode: the
                # artifact looks reused but was built under a different spec
                # (engine, spec_path, ...), so it can carry a stale field set.
                # print(), not emit(), so this survives the "[info] " prefix
                # every progress callback wraps emit() messages in.
                print(
                    "[warn] "
                    + reuse_message
                    + " -- but its recorded projection spec no longer matches "
                    "the configured one; pass --force to re-extract"
                )
            else:
                emit(reuse_message)

        if limit_overrides_requested and not should_extract:
            if max_companies is not None:
                existing_rows = (
                    _count_non_empty_jsonl_rows_in_dir(projected_chunk_dir)
                    if has_chunked_output
                    else _count_non_empty_jsonl_rows(projected_path)
                )
                if existing_rows >= max_companies:
                    emit(
                        f"[wikidata] stage prepare: existing projected rows={existing_rows:,} "
                        f"already satisfy max_companies={max_companies:,}; reusing existing data"
                    )
                else:
                    should_extract = True
            else:
                should_extract = True

        if should_extract:
            # The live projected file stays in place: the extractor replaces it
            # only with a finished, row-count-verified extract, so a failed run
            # leaves it as it was.
            emit(
                "[wikidata] stage prepare: building/resuming projected source artifact "
                + selected_prepare_output_path.relative_to(roots.data).as_posix()
            )
            extract_two_pass(
                acquire_path,
                projected_path,
                progress=emit,
                max_companies=max_companies,
                max_lines=max_lines,
                phase1_wiring_mode=prepare_wiring_mode,
                stream_resume=prepare_stream_resume,
                projection_defaults=effective_projection_defaults or None,
            )
            _write_wikidata_projected_spec_sidecar(
                projected_path=projected_path,
                projection_defaults=effective_projection_defaults,
            )

        if projected_chunk_dir.exists() and _has_jsonl_files(projected_chunk_dir):
            return acquire_dir, projected_chunk_dir

        return acquire_dir, projected_path


def _count_non_empty_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _count_non_empty_jsonl_rows_in_dir(path: Path) -> int:
    total = 0
    for chunk_path in sorted(path.rglob("*.jsonl")):
        if not chunk_path.is_file():
            continue
        total += _count_non_empty_jsonl_rows(chunk_path)
    return total


def _has_jsonl_files(path: Path) -> bool:
    return any(chunk_path.is_file() for chunk_path in path.rglob("*.jsonl"))
