"""Profile a system's parquet files against the columns analysis expects.

Reports required and optional column presence, null rates and dtype consistency across the files, writing schema-profile artefacts, and records the run as the schema-profiling phase in `run_manifest`. Run through `scripts/profile_analysis_schema.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from acquisition.tokenizer_ops import TOKENIZED_OUTPUT_COLUMNS
from analysis.report_layout import reports_dir as _reports_dir
from analysis.run_manifest import build_phase_entry, record_phase_run
from analysis.system_discovery import discover_systems_with_tokenized_parquet
from workspace.artifact_layout import analysis_report_run_dir
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots

DEFAULT_OPTIONAL_COLUMNS: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractSpec:
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]


def build_contract(
    required_columns: Iterable[str] | None = None,
    optional_columns: Iterable[str] | None = None,
) -> ContractSpec:
    required = tuple(_normalize_columns(required_columns or TOKENIZED_OUTPUT_COLUMNS))
    optional = tuple(_normalize_columns(optional_columns or DEFAULT_OPTIONAL_COLUMNS))
    return ContractSpec(required_columns=required, optional_columns=optional)


def _normalize_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for col in columns:
        key = str(col).strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _iter_system_parquet_files(
    roots: WorkspaceRoots, system: str, input_file: str | None = None
) -> list[Path]:
    system_dir = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    if not system_dir.exists():
        return []
    if input_file:
        selected = system_dir / input_file
        return [selected] if selected.exists() else []
    return sorted(system_dir.glob("*.parquet"))


def _safe_null_count(lazyframe: pl.LazyFrame, column: str) -> int:
    try:
        return int(lazyframe.select(pl.col(column).is_null().sum()).collect().item())
    except Exception:  # noqa: BLE001 -- "safe" by name/contract: any failure returns the -1 sentinel, never raises
        return -1


def _dtype_name_for_column(
    schema_map: dict[str, pl.DataType], column: str
) -> str | None:
    dtype = schema_map.get(column)
    return str(dtype) if dtype is not None else None


def _discover_profile_systems(
    *, roots: WorkspaceRoots, input_file: str | None
) -> list[str]:
    return discover_systems_with_tokenized_parquet(roots=roots, input_file=input_file)


def _missing_dataset_note(*, system: str, input_file: str | None) -> str:
    if input_file:
        return f"{system}: parquet file '{input_file}' not found in data/{system}/tokenized."
    return f"{system}: no parquet files found in data/{system}/tokenized."


def _profile_system_parquet_files(
    *,
    system: str,
    parquet_files: list[Path],
) -> tuple[int, dict[str, str], dict[str, int], list[str]]:
    system_row_count = 0
    system_schema: dict[str, str] = {}
    system_null_counts: dict[str, int] = {}
    schema_conflicts: list[str] = []

    for parquet_file in parquet_files:
        lf = pl.scan_parquet(parquet_file)
        schema_map = lf.collect_schema()
        file_row_count = int(lf.select(pl.len()).collect().item())
        system_row_count += file_row_count

        for column_name in schema_map.names():
            dtype_name = _dtype_name_for_column(schema_map, column_name)
            if dtype_name is None:
                continue
            existing = system_schema.get(column_name)
            if existing is None:
                system_schema[column_name] = dtype_name
            elif existing != dtype_name:
                schema_conflicts.append(
                    f"{system}: column '{column_name}' has mixed dtypes ({existing}, {dtype_name}) in {parquet_file.name}."
                )

            null_count = _safe_null_count(lf, column_name)
            if null_count >= 0:
                system_null_counts[column_name] = (
                    system_null_counts.get(column_name, 0) + null_count
                )

    return system_row_count, system_schema, system_null_counts, schema_conflicts


def _build_contract_column_row(
    *,
    system: str,
    system_row_count: int,
    system_schema: dict[str, str],
    system_null_counts: dict[str, int],
    column_name: str,
    required: bool,
) -> dict[str, object]:
    present = column_name in system_schema
    null_count = system_null_counts.get(column_name) if present else None
    null_rate = (
        (float(null_count) / float(system_row_count))
        if present and null_count is not None and system_row_count > 0
        else None
    )
    return {
        "system": system,
        "file_name": "*",
        "row_count": system_row_count,
        "column_name": column_name,
        "dtype": system_schema.get(column_name),
        "null_count": null_count,
        "null_rate": null_rate,
        "required": required,
        "present": present,
        "status": "ok"
        if present
        else ("missing_required" if required else "missing_optional"),
    }


def profile_data_contract(
    roots: WorkspaceRoots,
    *,
    systems: Iterable[str] | None = None,
    contract: ContractSpec | None = None,
    input_file: str | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    active_contract = contract or build_contract()

    if systems is None:
        systems = _discover_profile_systems(roots=roots, input_file=input_file)

    rows: list[dict[str, object]] = []
    drift_notes: list[str] = []

    for system in systems:
        parquet_files = _iter_system_parquet_files(roots, system, input_file)
        if not parquet_files:
            drift_notes.append(
                _missing_dataset_note(system=system, input_file=input_file)
            )
            rows.append(
                {
                    "system": system,
                    "file_name": None,
                    "row_count": 0,
                    "column_name": "__dataset__",
                    "dtype": None,
                    "null_count": None,
                    "null_rate": None,
                    "required": True,
                    "present": False,
                    "status": "missing_dataset",
                }
            )
            continue

        system_row_count, system_schema, system_null_counts, schema_conflicts = (
            _profile_system_parquet_files(
                system=system,
                parquet_files=parquet_files,
            )
        )

        if schema_conflicts:
            drift_notes.extend(schema_conflicts)

        for required in active_contract.required_columns:
            row = _build_contract_column_row(
                system=system,
                system_row_count=system_row_count,
                system_schema=system_schema,
                system_null_counts=system_null_counts,
                column_name=required,
                required=True,
            )
            if not row["present"]:
                drift_notes.append(f"{system}: missing required column '{required}'.")
            rows.append(row)

        for optional in active_contract.optional_columns:
            rows.append(
                _build_contract_column_row(
                    system=system,
                    system_row_count=system_row_count,
                    system_schema=system_schema,
                    system_null_counts=system_null_counts,
                    column_name=optional,
                    required=False,
                )
            )

    profile_df = pl.DataFrame(rows)
    if profile_df.height > 0:
        profile_df = profile_df.sort(
            ["system", "required", "column_name"], descending=[False, True, False]
        )
    return profile_df, drift_notes


def write_schema_profile_artifacts(
    roots: WorkspaceRoots,
    *,
    run_date: str,
    profile_df: pl.DataFrame,
    drift_notes: list[str],
    contract: ContractSpec | None = None,
) -> tuple[Path, Path, Path]:
    run_root = analysis_report_run_dir(roots, "token_performance", run_date)
    profile_dir = run_root / "profile"
    reports_dir = _reports_dir(run_root)
    profile_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = profile_dir / "schema_profile.parquet"
    markdown_path = reports_dir / "schema_profile.md"

    profile_df.write_parquet(parquet_path)

    required_failures = (
        profile_df.filter(pl.col("required") & ~pl.col("present")).height
        if profile_df.height > 0
        else 0
    )
    optional_missing = (
        profile_df.filter(~pl.col("required") & ~pl.col("present")).height
        if profile_df.height > 0
        else 0
    )

    lines = [
        "# Schema Profile",
        "",
        f"- Run date: {run_date}",
        f"- Rows: {profile_df.height}",
        f"- Required column failures: {required_failures}",
        f"- Missing optional columns: {optional_missing}",
        "",
        "## Drift Notes",
    ]

    if drift_notes:
        lines.extend([f"- {note}" for note in drift_notes])
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Required Columns By System",
            "",
            "| System | Column | Present | DType | Null Rate |",
            "|---|---|---|---|---|",
        ]
    )

    if profile_df.height > 0:
        required_rows = profile_df.filter(pl.col("required")).to_dicts()
        for row in required_rows:
            null_rate = row.get("null_rate")
            null_rate_text = "" if null_rate is None else f"{float(null_rate):.4f}"
            lines.append(
                "| "
                + str(row.get("system", ""))
                + " | "
                + str(row.get("column_name", ""))
                + " | "
                + ("yes" if row.get("present") else "no")
                + " | "
                + str(row.get("dtype") or "")
                + " | "
                + null_rate_text
                + " |"
            )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    systems = (
        sorted(profile_df.get_column("system").unique().to_list())
        if profile_df.height > 0
        else []
    )
    active_systems = (
        profile_df.filter(pl.col("status") != "missing_dataset")
        .get_column("system")
        .n_unique()
        if profile_df.height > 0
        else 0
    )
    manifest_entry = build_phase_entry(
        created_utc=datetime.now(UTC).isoformat(),
        phase="phase0_schema_profile",
        run_date=run_date,
        systems=systems,
        active_systems=active_systems,
        parameters={
            "required_columns": list(contract.required_columns) if contract else [],
            "optional_columns": list(contract.optional_columns) if contract else [],
        },
        outputs={
            "schema_profile_parquet": parquet_path.as_posix(),
            "schema_profile_markdown": markdown_path.as_posix(),
        },
        row_counts={
            "profile_rows": profile_df.height,
            "required_failures": required_failures,
            "optional_missing": optional_missing,
        },
    )
    manifest_path = record_phase_run(roots, run_date, manifest_entry)

    return parquet_path, markdown_path, manifest_path
