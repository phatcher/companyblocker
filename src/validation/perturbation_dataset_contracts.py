"""Required-column contract for materialized perturbation datasets.

Mirrors `acquisition.cleansed_contracts.CLEANSED_MERGE_REQUIRED_COLUMNS`'s pattern (a plain
required-columns tuple) rather than `contracts.py`'s `ARTIFACT_SCHEMAS`, which is for
`artifacts/validation/` run outputs, not dataset-layer parquet, per its own docstring.

There is no `seed` column. The seed a run was asked for is in every row's `system_uri`
(`perturbed://<source>/<family>/<profile>/<version>/<seed>/<fingerprint>`) and in the directory the
dataset lives at, so a column repeating one value on every row would carry nothing the
identity does not already say.

`original_name` and `changed` exist because every source record emits a row, including one
whose drawn scenario landed no change. That keeps the perturbed set joinable one-to-one
against its source, and `changed` says which case a row is without comparing two strings.

`source_uri` and `match_uri` are two distinct, always-present columns, not one overloaded
field: `source_uri` (required, non-null) is the record's own real `system_uri` before
perturbation -- "this perturbed row is a variant of that row." `match_uri` (required column,
value may be null) is whatever cross-system ground truth the original row already carried,
passed through unchanged -- perturbing a matched-set row (e.g. gleif's real match in gb) must
not destroy that truth to make room for the lineage link. See
`perturbation_materializer.py`'s module docstring for the full reasoning.
"""

from __future__ import annotations

import polars as pl

PERTURBATION_DATASET_REQUIRED_COLUMNS: tuple[str, ...] = (
    "system_uri",
    "source_uri",
    "match_uri",
    "name",
    "name_cleansed",
    "name_cleansed_basic",
    "jurisdiction_code",
    "profile_id",
    "profile_version",
    "scenario_id",
    "original_name",
    "changed",
)


def validate_perturbation_dataset_schema(
    frame: pl.DataFrame, *, profile_id: str
) -> None:
    present = set(frame.columns)
    missing = [
        column
        for column in PERTURBATION_DATASET_REQUIRED_COLUMNS
        if column not in present
    ]
    if missing:
        raise ValueError(
            f"Materialized perturbation dataset for profile {profile_id!r} is missing "
            f"required column(s): {', '.join(missing)}"
        )
