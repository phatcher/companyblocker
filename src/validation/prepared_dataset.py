"""The prepared, jurisdiction-partitioned copy of a system a validation run reads, and loading one country of it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from acquisition.country_counts import coerce_country_counts
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    data_root,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.layer_layout import (
    PARTITION_COLUMN,
    layer_partition_dir,
    partition_values,
    resolve_partition_dir,
)
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import is_perturbed_source, perturbed_parts

from .config import ValidationRunConfig


@dataclass(frozen=True)
class PreparedSystemStats:
    system: str
    system_dir: Path
    total_rows_by_country: dict[str, int]
    filtered_out_rows_by_country: dict[str, int]
    kept_rows_by_country: dict[str, int]


def _country_partition_dir(*, system_dir: Path, country: str) -> Path:
    return layer_partition_dir(system_dir, value=country)


def _country_from_partition_dir_name(name: str) -> str | None:
    prefix = f"{PARTITION_COLUMN}="
    if not name.startswith(prefix):
        return None
    country = name[len(prefix) :].strip().lower()
    return country or None


def _existing_partition_dirs(*, system_dir: Path) -> list[Path]:
    return [
        partition_dir
        for country in partition_values(system_dir)
        if (partition_dir := resolve_partition_dir(system_dir, value=country))
        is not None
    ]


def _has_any_parquet_files(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    if any(directory.glob("*.parquet")):
        return True
    return any(directory.glob("**/*.parquet"))


def _has_materialized_cleansed_view(directory: Path) -> bool:
    """True when `directory` (a system's cleansed/ dir) holds the merged/
    deduped view at its own top level -- partitioned jurisdiction_code=*/
    subdirectories, or flat {code}-NNN.parquet files -- as opposed to only
    raw, not-yet-merged chunks/ content."""
    if not directory.exists() or not directory.is_dir():
        return False
    if any(partition_values(directory)):
        return True
    return any(directory.glob("*.parquet"))


def resolve_prepared_system_dir(
    *,
    roots: WorkspaceRoots,
    system: str,
    prepared_base_dir: Path | None,
) -> Path:
    base_dir = (
        Path(prepared_base_dir).resolve()
        if prepared_base_dir is not None
        else data_root(roots).resolve()
    )

    # A "perturbed://<system>/<profile>/<version>/<seed>" source resolves through
    # `workspace.kind_layout`, which owns the tree perturbed data lives in, so it
    # ignores `base_dir`: a perturbed dataset is not one system among the systems
    # under `data/`, and an override naming a `data/`-shaped tree cannot host it.
    if is_perturbed_source(system):
        source_system, profile_id, version, seed = perturbed_parts(system)
        system_dir = perturbed_dataset_dir(
            roots=roots,
            source_system=source_system,
            profile_id=profile_id,
            version=version,
            seed=seed,
        )
    else:
        system_dir = base_dir / system

    if prepared_base_dir is None:
        matched_dir = layer_directory(system_dir, MATCHED_LAYER_NAME)
        if _has_any_parquet_files(matched_dir):
            return matched_dir

    cleansed_dir = layer_directory(system_dir, CLEANSED_LAYER_NAME)
    if _has_materialized_cleansed_view(cleansed_dir):
        return cleansed_dir

    if prepared_base_dir is None:
        raise FileNotFoundError(
            f"No prepared parquet files found for system '{system}' in {system_dir}. "
            "Expected matched or cleansed outputs."
        )

    raise FileNotFoundError(
        f"No prepared parquet files found for system '{system}' under configured prepared base directory {base_dir}."
    )


def _coerce_country_counts(raw: object) -> dict[str, int]:
    return coerce_country_counts(raw)


def _load_existing_stats(
    *, system: str, system_dir: Path, partition_dirs: list[Path]
) -> PreparedSystemStats:
    metadata_path = system_dir / "_prepared_metadata.json"
    total_rows_by_country: dict[str, int] = {}
    filtered_out_rows_by_country: dict[str, int] = {}
    kept_rows_by_country: dict[str, int] = {}

    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            total_rows_by_country = _coerce_country_counts(
                payload.get("total_rows_by_country")
            )
            kept_rows_by_country = _coerce_country_counts(
                payload.get("shard_rows_by_country")
            )
            if not kept_rows_by_country:
                kept_rows_by_country = _coerce_country_counts(
                    payload.get("kept_rows_by_country")
                )
            filtered_out_rows_by_country = _coerce_country_counts(
                payload.get("filtered_out_rows_by_country")
            )
        except (json.JSONDecodeError, OSError):
            total_rows_by_country = {}
            filtered_out_rows_by_country = {}
            kept_rows_by_country = {}

    if not kept_rows_by_country and total_rows_by_country:
        # Current sharding keeps all rows; when no explicit kept map exists, treat kept as total.
        kept_rows_by_country = dict(total_rows_by_country)
    if not kept_rows_by_country:
        for partition_dir in partition_dirs:
            country = _country_from_partition_dir_name(partition_dir.name)
            if country is None:
                continue
            kept_rows_by_country[country] = 0
    if not total_rows_by_country:
        total_rows_by_country = dict(kept_rows_by_country)
    if not filtered_out_rows_by_country:
        filtered_out_rows_by_country = {
            country: max(
                total_rows_by_country.get(country, 0)
                - kept_rows_by_country.get(country, 0),
                0,
            )
            for country in set(total_rows_by_country) | set(kept_rows_by_country)
        }

    return PreparedSystemStats(
        system=system,
        system_dir=system_dir,
        total_rows_by_country=total_rows_by_country,
        filtered_out_rows_by_country=filtered_out_rows_by_country,
        kept_rows_by_country=kept_rows_by_country,
    )


def prepare_system_dataset(
    *,
    config: ValidationRunConfig,
    system: str,
    force_rebuild: bool = False,
) -> PreparedSystemStats:
    system_dir = resolve_prepared_system_dir(
        roots=config.roots,
        system=system,
        prepared_base_dir=config.prepared_base_dir,
    )
    existing_partitions = _existing_partition_dirs(system_dir=system_dir)

    return _load_existing_stats(
        system=system, system_dir=system_dir, partition_dirs=existing_partitions
    )


def list_prepared_countries(stats: PreparedSystemStats) -> set[str]:
    return set(partition_values(stats.system_dir))


def read_country_partition_frame(partition_dir: Path) -> pl.DataFrame:
    """Read every parquet file in one country partition directory as one frame.

    An absent directory and an empty one both yield the same empty
    `system_uri`/`name` frame, so a country with no rows concatenates and
    round-trips identically to a populated one rather than raising.
    """
    if not partition_dir.exists():
        return pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})

    files = sorted(partition_dir.glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})

    frames = [pl.read_parquet(path) for path in files]
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def load_prepared_country_frame(
    *,
    stats: PreparedSystemStats,
    country: str,
) -> pl.DataFrame:
    return read_country_partition_frame(
        _country_partition_dir(system_dir=stats.system_dir, country=country)
    )
