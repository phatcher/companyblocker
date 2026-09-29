"""Formal input-row contract for `loader.load_validation_snapshot_from_files`.

A **baseline** row (real acquired/cleansed data) always carries three required, always-
non-null columns -- `system_uri`, `name`, `country`. A **perturbed** row (materialized by
`perturbation_materializer.py`, loadable through the SAME unmodified loader via the
`perturbed://<system>/<profile>/<version>/<seed>` dataset -- see `workspace.run_inputs`'s docstring) additionally carries
`source_uri`/`profile_id`/`profile_version`/`scenario_id`/`original_name`/`changed`. `match_uri`
sits between the two: always present on the loaded frame, but only ever non-null on a row
(baseline or perturbed) that already carries real cross-system ground truth.

One schema formalizes both shapes rather than two: `load_validation_snapshot_from_files`'s
returned frame always carries every perturbation column (backfilled as null when the source
parquet didn't have it), so a caller never has to branch on `"source_uri" in frame.columns`
to know whether the column exists -- only on whether it's populated. The one place that still
legitimately branches on a perturbation column's presence is `runner.py`'s
`_resolve_source_truth_column`, which decides which truth column to trust, not whether the
schema is satisfied.

`validate_perturbed_rows_are_complete` is the "nullable except on baseline rows" half of the
contract: a row whose `system_uri` scheme is the perturbed-system code must have every
perturbation column populated, matching what `perturbation_materializer.py` itself already
guarantees at materialization time (`_run_quality_checks`'s null `source_uri` check, and
`PERTURBATION_DATASET_REQUIRED_COLUMNS` in `perturbation_dataset_contracts.py`) -- this is the
same guarantee re-checked at load time, for a frame that didn't necessarily come from that
pipeline (e.g. a hand-built fixture, or a future second producer of perturbed rows).
"""

from __future__ import annotations

import polars as pl

from workspace.derived_uri import PERTURBED_SCHEME

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8())
# interchangeably in schema dicts; pl.DataType alone only covers the latter (matches
# contracts.py's own alias, duplicated locally rather than importing a leading-underscore
# name across modules).
PolarsDType = type[pl.DataType] | pl.DataType

# Always required, always non-null: enforced by `load_validation_snapshot_from_files`
# regardless of whether the loaded frame is baseline or perturbed data.
REQUIRED_BASELINE_COLUMNS: tuple[str, ...] = ("system_uri", "name", "country")

# Post-normalization contract for what `load_validation_snapshot_from_files` returns: the
# three required baseline columns, `match_uri` (pre-existing, present but nullable), and
# the six perturbation-lineage columns (present but nullable on a baseline row, required
# non-null on a perturbed one -- see `validate_perturbed_rows_are_complete`).
VALIDATION_INPUT_SCHEMA: dict[str, PolarsDType] = {
    "system_uri": pl.Utf8,
    "name": pl.Utf8,
    "country": pl.Utf8,
    "match_uri": pl.Utf8,
    "source_uri": pl.Utf8,
    "profile_id": pl.Utf8,
    "profile_version": pl.Utf8,
    "scenario_id": pl.Utf8,
    "original_name": pl.Utf8,
    "changed": pl.Boolean,
}

# The subset of VALIDATION_INPUT_SCHEMA that only a perturbed row populates.
PERTURBATION_INPUT_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in VALIDATION_INPUT_SCHEMA
    if column not in REQUIRED_BASELINE_COLUMNS and column != "match_uri"
)


def validate_required_input_columns(
    frame: pl.DataFrame, *, system: str, name_col: str, country_col: str
) -> None:
    """Raw-frame precondition check, replacing `load_validation_snapshot_from_files`'s three
    separate ad hoc `ValueError`s with one aggregated check: every missing column is reported
    together rather than one failure at a time across repeated runs. `name_col`/`country_col`
    are the pre-normalization column names (a caller-configurable `name_col`, and the fixed
    `"jurisdiction_code"` acquisition/cleanse convention) -- distinct from the normalized
    `"name"`/`"country"` output columns `VALIDATION_INPUT_SCHEMA` describes.
    """
    required = {
        "system_uri": "system_uri",
        name_col: name_col,
        country_col: country_col,
    }
    missing = [raw_name for raw_name in required if raw_name not in frame.columns]
    if missing:
        raise ValueError(
            f"Validation frame for system {system!r} is missing required column(s): "
            f"{', '.join(sorted(set(missing)))}."
        )


def ensure_perturbation_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Backfill every `PERTURBATION_INPUT_COLUMNS` entry absent from `frame` as a null literal
    of its declared dtype, and cast any that ARE present -- so the frame this function returns
    always carries the full `VALIDATION_INPUT_SCHEMA` column set, whether the source parquet
    was plain baseline data or already-materialized perturbed output.
    """
    exprs = []
    for column in PERTURBATION_INPUT_COLUMNS:
        dtype = VALIDATION_INPUT_SCHEMA[column]
        if column in frame.columns:
            exprs.append(pl.col(column).cast(dtype, strict=False).alias(column))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(column))
    return frame.with_columns(exprs)


def validate_perturbed_rows_are_complete(frame: pl.DataFrame, *, system: str) -> None:
    """Called by whatever consumes the perturbation columns, not on every load.

    Clustering needs `system_uri`, `name`, `country` and a truth column; the
    perturbation columns are provenance that only robustness scoring groups by. A
    loader that enforced them refused frames every other consumer would have been
    happy with, which is a requirement asserted where it is convenient rather than
    where it is needed.
    """
    """Hard-fail if any row whose `system_uri` identifies it as a perturbed row (scheme ==
    `PERTURBED_SCHEME`) has a null value in any `PERTURBATION_INPUT_COLUMNS` column --
    the "nullable except on baseline rows" half of the contract. A frame with no perturbed
    rows at all (the overwhelming majority of loads) is always accepted without inspecting
    the perturbation columns further.
    """
    if frame.height == 0:
        return

    is_perturbed_row = (
        pl.col("system_uri")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.starts_with(f"{PERTURBED_SCHEME}://")
    )
    incomplete = frame.filter(
        is_perturbed_row
        & pl.any_horizontal(
            [pl.col(column).is_null() for column in PERTURBATION_INPUT_COLUMNS]
        )
    )
    if incomplete.height:
        raise ValueError(
            f"Validation frame for system {system!r} has {incomplete.height} row(s) "
            "identified as perturbed (system_uri scheme "
            f"{PERTURBED_SCHEME!r}) with a null value in one of "
            f"{PERTURBATION_INPUT_COLUMNS}; every perturbation column must be non-null "
            "on a perturbed row."
        )
