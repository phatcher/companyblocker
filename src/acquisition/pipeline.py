"""The acquisition pipeline: acquire, shard, canonical, match, cleanse and tokenize, per system.

`run_acquisition` is the Acquire stage for one or more systems, writing a dated run folder under `data/<system>/acquire/`. A system whose catalog status is `research_required` runs only when research is explicitly enabled, and then only for the stages it allows (`constants_status`), so a partially trusted source never enters a normal run by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from workspace.data_layout import CANONICAL_LAYER_NAME, system_layer_dir
from workspace.data_layout import data_root as workspace_data_root
from workspace.roots import WorkspaceRoots

from .constants_status import (
    STATUS_RESEARCH_REQUIRED,
    ProgressReporter,
    is_research_stage_allowed,
    resolve_runnable_statuses,
)
from .downloader import (
    download_api_json_snapshot,
    download_dbpedia_databus_latest,
    download_resource,
    sha256_file,
)
from .downloader_gleif import download_gleif_latest_concatenated
from .downloader_wikidata import extract_wikidata_company_projection_two_pass
from .extractors import extract_zip_archive
from .models import AcquisitionPaths, AcquisitionResult
from .pipeline_resources import _parse_iso_date as _resources_parse_iso_date
from .pipeline_resources import (
    collect_resource_artifacts,
)
from .pipeline_selection import (
    SystemSelection,
)
from .pipeline_selection import _normalize_codes as _selection_normalize_codes
from .pipeline_selection import (
    _supported_system_codes as _selection_supported_system_codes,
)
from .pipeline_selection import normalize_systems as _selection_normalize_systems
from .pipeline_selection import (
    resolve_system_selection as _selection_resolve_system_selection,
)
from .pipeline_selection import (
    validate_requested_systems as _selection_validate_requested_systems,
)
from .plan_registry import get_system_plan

ADAPTER_HANDLERS: dict[str, Callable[..., str]] = {
    "direct_download": download_resource,
    "api_json_snapshot": download_api_json_snapshot,
    "dbpedia_databus_latest": download_dbpedia_databus_latest,
    "gleif_latest_concatenated": download_gleif_latest_concatenated,
}

# Keep adapter callables as module attributes for compatibility and explicit dispatch wiring.
_ADAPTER_EXPORTS = (
    download_resource,
    download_api_json_snapshot,
    download_dbpedia_databus_latest,
    download_gleif_latest_concatenated,
)


def build_paths(roots: WorkspaceRoots, system: str) -> AcquisitionPaths:
    data_dir = workspace_data_root(roots)
    return AcquisitionPaths(
        roots=roots,
        data_root=data_dir,
        acquire_dir=system_layer_dir(roots, system, layer="acquire"),
        prepare_dir=system_layer_dir(roots, system, layer="prepare"),
        source_dir=system_layer_dir(roots, system, layer="source"),
        canonical_dir=system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME),
    )


# Compatibility wrappers kept for callers/tests importing private helpers from this module.
def _normalize_codes(
    codes: str | list[str] | tuple[str, ...], *, label: str
) -> list[str]:
    return _selection_normalize_codes(codes, label=label)


def _supported_system_codes() -> list[str]:
    return _selection_supported_system_codes()


def normalize_systems(
    systems: str | list[str] | tuple[str, ...],
    *,
    expand_all: bool = True,
) -> list[str]:
    return _selection_normalize_systems(
        systems,
        expand_all=expand_all,
        supported_codes_resolver=_supported_system_codes,
    )


def resolve_system_selection(
    systems: str | list[str] | tuple[str, ...],
    *,
    add_global_for_all: bool = False,
    global_expands_to_all: bool = False,
) -> SystemSelection:
    return _selection_resolve_system_selection(
        systems,
        add_global_for_all=add_global_for_all,
        global_expands_to_all=global_expands_to_all,
        normalize_systems_fn=normalize_systems,
    )


def validate_requested_systems(requested: list[str]) -> None:
    _selection_validate_requested_systems(requested, get_system_plan_fn=get_system_plan)


def _parse_iso_date(value: str | None):
    return _resources_parse_iso_date(value)


def _resolve_adapter_handler(
    adapter: str,
    *,
    adapter_handlers: Mapping[str, Callable[..., str]] | None = None,
) -> Callable[..., str]:
    handlers = ADAPTER_HANDLERS if adapter_handlers is None else adapter_handlers
    handler = handlers.get(adapter)
    if handler is None:
        raise KeyError(adapter)
    return handler


def _collect_resource_artifacts(
    *,
    roots: WorkspaceRoots,
    plan,
    paths,
    effective_run_date: str,
    adapter_handlers: Mapping[str, Callable[..., str]] | None,
    emit: Callable[[str], None],
):
    handlers = ADAPTER_HANDLERS if adapter_handlers is None else adapter_handlers
    return collect_resource_artifacts(
        roots=roots,
        plan=plan,
        paths=paths,
        effective_run_date=effective_run_date,
        adapter_handlers=handlers,
        emit=emit,
        extract_wikidata_projection_fn=extract_wikidata_company_projection_two_pass,
        extract_zip_archive_fn=extract_zip_archive,
        sha256_file_fn=sha256_file,
        parse_iso_date_fn=_parse_iso_date,
    )


def run_acquisition(
    systems: str | list[str] | tuple[str, ...],
    *,
    roots: WorkspaceRoots,
    run_date: str | None = None,
    dry_run: bool = False,
    skip_unsupported: bool = True,
    allow_research: bool = False,
    progress: ProgressReporter | None = None,
    adapter_handlers: Mapping[str, Callable[..., str]] | None = None,
) -> list[AcquisitionResult]:
    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    effective_run_date = run_date or datetime.now(UTC).date().isoformat()

    targets = normalize_systems(systems)
    emit(
        f"Starting acquisition for {len(targets)} system(s) on run date {effective_run_date}"
    )

    results: list[AcquisitionResult] = []
    for target in targets:
        emit(f"Resolving plan for system '{target}'")
        plan = get_system_plan(target)
        system_code = plan.code
        paths = build_paths(roots, system_code)
        emit(f"[{system_code}] status={plan.status}, resources={len(plan.resources)}")

        runnable_statuses = resolve_runnable_statuses(allow_research)

        if (
            plan.status == STATUS_RESEARCH_REQUIRED
            and allow_research
            and not is_research_stage_allowed(plan, stage="acquire")
        ):
            message = (
                f"{system_code}: research execution policy denies stage 'acquire'. "
                "Set research.allow_research_runtime=true and include 'acquire' in research.allowed_stages."
            )
            if skip_unsupported:
                emit(
                    f"[{system_code}] skipping research system due to execution policy"
                )
                results.append(
                    AcquisitionResult(
                        country=system_code, status=plan.status, message=message
                    )
                )
                continue
            raise RuntimeError(message)

        if plan.status not in runnable_statuses:
            message = f"{system_code}: {plan.status} - {plan.notes}"
            if skip_unsupported:
                emit(f"[{system_code}] skipping unsupported system: {plan.status}")
                results.append(
                    AcquisitionResult(
                        country=system_code, status=plan.status, message=message
                    )
                )
                continue
            raise RuntimeError(message)

        if dry_run:
            target_dirs = [
                (
                    paths.acquire_dir
                    / resource.resolve_snapshot_date(effective_run_date)
                ).as_posix()
                for resource in plan.resources
            ]
            unique_dirs = sorted(set(target_dirs))
            message = (
                f"{system_code}: would download {len(plan.resources)} resource(s) into "
                + ", ".join(unique_dirs)
            )
            emit(f"[{system_code}] dry run: {message}")
            results.append(
                AcquisitionResult(
                    country=system_code, status="dry_run", message=message
                )
            )
            continue

        artifact_rows, downloaded_any, skipped_for_freshness = (
            _collect_resource_artifacts(
                roots=roots,
                plan=plan,
                paths=paths,
                effective_run_date=effective_run_date,
                adapter_handlers=adapter_handlers,
                emit=emit,
            )
        )

        if downloaded_any:
            status = "downloaded"
            message = f"{system_code}: downloaded {len(plan.resources)} resource(s)"
        else:
            status = "freshness_skipped"
            message = f"{system_code}: skipped {skipped_for_freshness} resource(s) due to freshness policy"

        emit(f"[{system_code}] completed with status '{status}'")

        results.append(
            AcquisitionResult(
                country=system_code,
                status=status,
                message=message,
                artifact_count=len(artifact_rows),
            )
        )

    emit(f"Acquisition completed for {len(results)} system(s)")
    return results
