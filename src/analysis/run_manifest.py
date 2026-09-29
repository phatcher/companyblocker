"""Shared run-manifest schema for `token_performance` phase runs.

Each phase (`io_contract.py`'s Phase 0 schema profile, `token_metrics.py`'s
Phase 1 token metrics, and any phase added later under
`token_performance_plan.md`) writes one entry into a single
`run_manifest.json` at the run roots
(`artifacts/analysis/token_performance/runs/<run_date>/run_manifest.json`),
keyed by phase name. A shared entry schema is what lets one run be compared
against another on stable ground -- the same field names under the same
`phase` key across `run_date`s -- rather than every phase inventing its own
ad hoc manifest shape. Mirrors the `company_tokenize.manifest` convention
(`RUN_MANIFEST_MANDATORY_FIELDS`/`build_run_manifest`/`validate_run_manifest`),
which formalized a validated mandatory/optional-field schema and a
build-then-validate helper pair for tokenizer promotion manifests; this
module adapts that shape for analysis phase runs rather than tokenizer
promotion: a `phase` key instead of `mode`, no tokenizer-specific fields
(`vocab_size`, `corpus_content_hash`, ...), and one shared file per run date
instead of one file per artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

# Fields every phase entry must carry.
PHASE_ENTRY_MANDATORY_FIELDS: tuple[str, ...] = (
    "created_utc",
    "phase",
    "run_date",
    "systems",
    "active_systems",
    "parameters",
    "outputs",
)

# Fields only some phases populate. `row_counts` is omitted by a phase with
# no tabular output; `runtime_seconds` is omitted when a caller does not
# time the phase; `notes` is omitted when there is nothing to say.
PHASE_ENTRY_OPTIONAL_FIELDS: tuple[str, ...] = (
    "row_counts",
    "runtime_seconds",
    "notes",
)

_STR_FIELDS: tuple[str, ...] = ("created_utc", "phase", "run_date")
_DICT_FIELDS: tuple[str, ...] = ("parameters", "outputs")


def build_phase_entry(
    *,
    created_utc: str,
    phase: str,
    run_date: str,
    systems: list[str],
    active_systems: int,
    parameters: dict[str, Any],
    outputs: dict[str, str],
    row_counts: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build one phase's manifest entry payload.

    Validates the built payload against `validate_phase_entry` before
    returning it, so a caller can never build an entry that fails its own
    schema.
    """
    entry: dict[str, Any] = {
        "created_utc": created_utc,
        "phase": phase,
        "run_date": run_date,
        "systems": list(systems),
        "active_systems": active_systems,
        "parameters": parameters,
        "outputs": outputs,
    }
    if row_counts is not None:
        entry["row_counts"] = row_counts
    if runtime_seconds is not None:
        entry["runtime_seconds"] = runtime_seconds
    if notes is not None:
        entry["notes"] = notes

    validate_phase_entry(entry)
    return entry


def validate_phase_entry(payload: dict[str, Any]) -> None:
    """Validate a phase-entry payload against the shared phase-run schema.

    Raises on the first problem found, naming the offending field, so a
    schema drift between phases fails loudly at write time instead of
    silently shipping a `run_manifest.json` entry that a later comparison
    can't rely on.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Phase entry must be a dict, got {type(payload).__name__}.")

    missing = [field for field in PHASE_ENTRY_MANDATORY_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            "Phase entry missing mandatory field(s): " + ", ".join(missing)
        )

    allowed = set(PHASE_ENTRY_MANDATORY_FIELDS) | set(PHASE_ENTRY_OPTIONAL_FIELDS)
    unknown = [key for key in payload if key not in allowed]
    if unknown:
        raise ValueError(
            "Phase entry has unrecognized field(s): " + ", ".join(sorted(unknown))
        )

    for field in _STR_FIELDS:
        if not isinstance(payload[field], str):
            raise TypeError(f"Phase entry '{field}' must be a str.")

    systems = payload["systems"]
    if not isinstance(systems, list) or not all(
        isinstance(item, str) for item in systems
    ):
        raise TypeError("Phase entry 'systems' must be a list of strings.")

    active_systems = payload["active_systems"]
    if not isinstance(active_systems, int) or isinstance(active_systems, bool):
        raise TypeError("Phase entry 'active_systems' must be an int.")

    for field in _DICT_FIELDS:
        if not isinstance(payload[field], dict):
            raise TypeError(f"Phase entry '{field}' must be a dict.")

    if "row_counts" in payload and not isinstance(payload["row_counts"], dict):
        raise TypeError("Phase entry 'row_counts' must be a dict when present.")

    if "runtime_seconds" in payload and not isinstance(
        payload["runtime_seconds"], (int, float)
    ):
        raise TypeError("Phase entry 'runtime_seconds' must be a number when present.")

    if "notes" in payload:
        notes = payload["notes"]
        if not isinstance(notes, list) or not all(
            isinstance(item, str) for item in notes
        ):
            raise TypeError(
                "Phase entry 'notes' must be a list of strings when present."
            )


def resolve_run_manifest_path(roots: WorkspaceRoots, run_date: str) -> Path:
    """Resolve the shared `run_manifest.json` path for one run date."""
    return (
        analysis_report_run_dir(roots, "token_performance", run_date)
        / "run_manifest.json"
    )


def load_run_manifest(roots: WorkspaceRoots, run_date: str) -> dict[str, Any]:
    """Load the shared run manifest for `run_date`, or an empty skeleton.

    Absence is a normal case (no phase has run for this `run_date` yet), not
    an error -- callers get back `{"run_date": ..., "phases": {}}` rather
    than an exception.
    """
    manifest_path = resolve_run_manifest_path(roots, run_date)
    if not manifest_path.exists():
        return {"run_date": run_date, "phases": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def record_phase_run(
    roots: WorkspaceRoots,
    run_date: str,
    entry: dict[str, Any],
) -> Path:
    """Merge one phase's manifest entry into the run's shared manifest file.

    Read-modify-write keyed by `entry["phase"]`: an entry for a phase that
    already ran for this `run_date` is replaced (a rerun reflects the latest
    execution), but every other phase's entry in the same file is left
    untouched -- this is what lets Phase 0 and Phase 1 (and any later phase)
    accumulate into one comparable `run_manifest.json` per run date instead
    of each phase clobbering the others' record of the same run.
    """
    validate_phase_entry(entry)
    manifest_path = resolve_run_manifest_path(roots, run_date)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_run_manifest(roots, run_date)
    manifest.setdefault("run_date", run_date)
    manifest.setdefault("phases", {})
    manifest["phases"][entry["phase"]] = entry

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path
