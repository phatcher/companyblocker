"""Which canonical snapshot the Cleanse stage reads: the one a run names, or the latest on disk."""

from __future__ import annotations

from pathlib import Path

from .data_layout import CANONICAL_LAYER_NAME, canonical_snapshot_dir, system_layer_dir
from .layer_layout import resolve_primary_files
from .roots import WorkspaceRoots


def _pick_latest_canonical_run(canonical_root: Path) -> Path | None:
    if not canonical_root.exists() or not canonical_root.is_dir():
        return None

    run_dirs = sorted(path for path in canonical_root.iterdir() if path.is_dir())
    if not run_dirs:
        return None
    return run_dirs[-1]


def resolve_input_dir(
    *,
    roots: WorkspaceRoots,
    run_date: str | None,
    system: str,
) -> Path:
    scope = system.strip().lower()
    if not scope:
        raise ValueError("System must be provided.")

    canonical_root = system_layer_dir(roots, scope, layer=CANONICAL_LAYER_NAME)

    if run_date:
        canonical_dir: Path | None = canonical_snapshot_dir(
            roots, system=scope, run_date=run_date
        )
    else:
        canonical_dir = _pick_latest_canonical_run(canonical_root)

    # Entity files specifically, not any parquet: a snapshot whose entity
    # view is partitioned has nothing but `sample.parquet` and the
    # name-variant sidecar at its top level, so a blind `*.parquet` glob
    # would accept a snapshot that Cleanse then finds no input in.
    if canonical_dir is not None and resolve_primary_files(
        canonical_dir, system_code=scope
    ):
        return canonical_dir

    if run_date:
        raise FileNotFoundError(
            f"No canonical parquet shards found in {canonical_root / run_date}. "
            "Run canonicalization before cleansing."
        )
    raise FileNotFoundError(
        f"No canonical parquet shards found under {canonical_root}. "
        "Run canonicalization before cleansing."
    )
