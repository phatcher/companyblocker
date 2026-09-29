"""Load a validation snapshot and split it into per-country source and target frames.

`load_validation_snapshot_from_files` backfills the perturbation lineage columns and holds every row to `input_contract.VALIDATION_INPUT_SCHEMA`; `build_directional_frames` assembles the source and target sides.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from workspace.data_file_naming import is_primary_data_file
from workspace.roots import WorkspaceRoots

from .config import ValidationRunConfig
from .input_contract import (
    ensure_perturbation_columns,
    validate_required_input_columns,
)
from .prepared_dataset import resolve_prepared_system_dir


def _resolve_validation_input_dir(
    *,
    roots: WorkspaceRoots,
    system: str,
    run_date: str | None,
) -> Path:
    return resolve_prepared_system_dir(
        roots=roots, system=system, prepared_base_dir=None
    )


def list_validation_input_files(
    *,
    roots: WorkspaceRoots,
    system: str,
    run_date: str | None,
) -> list[Path]:
    input_dir = _resolve_validation_input_dir(
        roots=roots, system=system, run_date=run_date
    )
    parquet_files = sorted(
        path
        for path in input_dir.glob(f"{system}-*.parquet")
        if is_primary_data_file(file_name=path.name, system_code=system)
    )
    if not parquet_files:
        parquet_files = sorted(
            path
            for path in input_dir.glob("*.parquet")
            if path.name not in {"sample.parquet", "empty_cleanser.parquet"}
        )
    if not parquet_files:
        parquet_files = sorted(
            path
            for path in input_dir.glob("**/*.parquet")
            if path.name not in {"sample.parquet", "empty_cleanser.parquet"}
        )
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {input_dir} for system '{system}'."
        )
    return parquet_files


def load_validation_snapshot_from_files(
    *,
    files: list[Path],
    system: str,
    name_col: str,
) -> pl.DataFrame:
    frames = [pl.read_parquet(path) for path in files]
    frame = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]

    validate_required_input_columns(
        frame, system=system, name_col=name_col, country_col="jurisdiction_code"
    )

    if "match_uri" in frame.columns:
        match_uri_expr = (
            pl.col("match_uri").cast(pl.Utf8, strict=False).alias("match_uri")
        )
    else:
        match_uri_expr = pl.lit(None, dtype=pl.Utf8).alias("match_uri")

    normalized = (
        frame.with_columns(
            pl.col("system_uri").cast(pl.Utf8).alias("system_uri"),
            pl.col(name_col).cast(pl.Utf8, strict=False).fill_null("").alias("name"),
            pl.col("jurisdiction_code")
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.to_lowercase()
            .alias("country"),
            match_uri_expr,
        )
        .filter(pl.col("system_uri").is_not_null() & (pl.col("system_uri") != ""))
        .unique(subset=["system_uri"], keep="first")
    )
    normalized = ensure_perturbation_columns(normalized)
    return normalized


def load_validation_snapshot(
    *,
    roots: WorkspaceRoots,
    system: str,
    run_date: str | None,
    name_col: str,
) -> pl.DataFrame:
    files = list_validation_input_files(
        roots=roots,
        system=system,
        run_date=run_date,
    )
    return load_validation_snapshot_from_files(
        files=files, system=system, name_col=name_col
    )


def build_directional_frames(
    *,
    config: ValidationRunConfig,
    source_system: str,
    target_system: str,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, int]]]:
    source_name_col = str(config.source_name_col or config.name_col).strip()
    source_frame = load_validation_snapshot(
        roots=config.roots,
        system=source_system,
        run_date=config.run_date,
        name_col=source_name_col,
    )
    target_frame = load_validation_snapshot(
        roots=config.roots,
        system=target_system,
        run_date=config.run_date,
        name_col=config.name_col,
    )

    if config.countries:
        allowed = set(config.countries)
        source_frame = source_frame.filter(pl.col("country").is_in(sorted(allowed)))
        target_frame = target_frame.filter(pl.col("country").is_in(sorted(allowed)))

    # Restrict source workload to jurisdictions that exist in target data.
    target_countries = (
        sorted(set(target_frame.get_column("country").to_list()))
        if target_frame.height
        else []
    )
    if target_countries:
        source_frame = source_frame.filter(pl.col("country").is_in(target_countries))
    else:
        source_frame = source_frame.clear()

    total_by_country = {
        row["country"]: int(row["count"])
        for row in source_frame.group_by("country").len(name="count").to_dicts()
    }

    source_frame = source_frame.filter(pl.col("name").str.strip_chars() != "")
    kept_by_country = {
        row["country"]: int(row["count"])
        for row in source_frame.group_by("country").len(name="count").to_dicts()
    }
    source_filter_stats = {
        country: {
            "source_total_rows": total_by_country.get(country, 0),
            "source_filtered_out_rows": max(
                total_by_country.get(country, 0) - kept_by_country.get(country, 0), 0
            ),
        }
        for country in set(total_by_country) | set(kept_by_country)
    }

    return source_frame, target_frame, source_filter_stats
