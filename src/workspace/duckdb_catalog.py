"""Prototype: resolve a layer through DuckDB instead of pathlib.

`layer_layout` answers "which files hold this layer's data" by walking the
filesystem with `Path.glob`/`Path.iterdir`. This module answers the same
questions -- same priority rules, same return types -- through a DuckDB
connection's `glob()` table function instead, to test the claim in
`README.md`'s Boundaries section: that a change of backing is a change
inside this package and invisible to every caller.

**What stayed identical.** `resolve_primary_files`, `resolve_name_files` and
`partition_values` each have a `_via_duckdb` twin here reproducing their
exact priority order -- `primary/`/`names/` preferred over the pre-family-split
top level, a partitioned primary family preferred over a flat one -- and
`primary_data_file_glob`/`is_primary_data_file`/`companion_data_file_glob`
from `data_file_naming` are reused unchanged, so the *rule* for what counts as
a layer's data is asserted once, not duplicated per backing.
`src/tests/workspace/test_duckdb_catalog.py` asserts every twin returns
exactly what its pathlib original returns, across every layout shape
`layer_layout`'s own module docstring describes (partitioned, flat, the
pre-split fallback, a `names/` sidecar, and `chunks/` staging excluded).

**Why this is a second backing and not just a second glob library.** DuckDB
does the file enumeration through its own query engine and connection
object rather than Python's `pathlib`, which is the seam an actual warehouse
or catalog backing would sit behind: swap what answers `glob()` and every
caller here is unaffected, because a caller only ever receives the same
`list[Path]` / `Path | None` / `list[str]` it always did.

**What this does not prototype.** The files themselves are still real
parquet on disk; DuckDB is answering "where are they", not holding the rows.
Every caller of these resolvers was audited (grep across `src/`) and each one does exactly one of three things with what comes
back: hands a `Path` list to `pl.scan_parquet`/`pl.read_parquet`, checks it
for truthiness, or builds a further path from it to write into. None of that
is satisfiable by anything other than a real filesystem path a caller can
open itself -- a backing that keeps rows only inside DuckDB (or any store
with no on-disk parquet to point at) would need every one of those call
sites to change from "read this path" to "read from this connection", which
is the seam `resolve_primary_files` et al. do not have today. That is the
line between what a second backing can change here for free (the query
engine finding the files) and what it cannot (there being files at all).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .data_file_naming import (
    COMPANION_NAMES,
    companion_data_file_glob,
    is_primary_data_file,
    primary_data_file_glob,
)
from .layer_layout import (
    PARTITION_COLUMN,
    names_family_dir,
    primary_family_dir,
)

_PARTITION_DIR_GLOB = f"{PARTITION_COLUMN}=*"


def _glob_paths(con: duckdb.DuckDBPyConnection, pattern: Path) -> list[Path]:
    """Every file DuckDB's `glob()` matches against `pattern`, sorted.

    Mirrors `sorted(root.glob(...))`'s ordering so a `_via_duckdb` twin's
    output is byte-for-byte comparable to its pathlib original, not just
    equal as a set.
    """
    rows = con.execute("SELECT file FROM glob(?)", [str(pattern)]).fetchall()
    return sorted(Path(row[0]) for row in rows)


def resolve_primary_files_via_duckdb(
    con: duckdb.DuckDBPyConnection, layer_dir: Path, *, system_code: str
) -> list[Path]:
    """`layer_layout.resolve_primary_files`'s twin, resolved through `con`
    instead of `pathlib`. Same priority order, same return type."""
    if not layer_dir.exists():
        return []
    for root in (primary_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        partitioned = _glob_paths(con, root / _PARTITION_DIR_GLOB / "*.parquet")
        if partitioned:
            return partitioned
        flat = sorted(
            path
            for path in _glob_paths(con, root / primary_data_file_glob(system_code))
            if is_primary_data_file(file_name=path.name, system_code=system_code)
        )
        if flat:
            return flat
    return []


def resolve_name_files_via_duckdb(
    con: duckdb.DuckDBPyConnection, layer_dir: Path, *, system_code: str
) -> list[Path]:
    """`layer_layout.resolve_name_files`'s twin, resolved through `con`
    instead of `pathlib`. Same priority order, same return type."""
    if not layer_dir.exists():
        return []
    names_glob = companion_data_file_glob(
        system_code=system_code, family=COMPANION_NAMES
    )
    for root in (names_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        found = _glob_paths(con, root / names_glob)
        if found:
            return found
    return []


def partition_values_via_duckdb(
    con: duckdb.DuckDBPyConnection, layer_dir: Path
) -> list[str]:
    """`layer_layout.partition_values`'s twin, resolved through `con`
    instead of `pathlib`. Same lowercasing, same sorted order.

    One documented divergence: DuckDB's `glob()` matches files, never
    directories (confirmed empirically -- a pattern ending in the partition
    directory itself returns nothing, one ending in `/*` returns the files
    inside it), so this derives a partition's value from a file one level
    inside it rather than from the directory entry `partition_values`
    reads. An *empty* partition directory therefore reads as absent here
    though `partition_values` would still report it. No caller in `src/`
    creates a partition directory without also writing into it in the same
    operation (`layer_partition_dir` names where to write, a stage always
    writes before anything reads the value back), so this divergence never
    fires against real data; it is recorded because it is exactly the kind
    of gap a backing swap can introduce silently.
    """
    prefix = f"{PARTITION_COLUMN}="
    for root in (primary_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        files_inside_partitions = _glob_paths(con, root / _PARTITION_DIR_GLOB / "*")
        values = sorted(
            {
                path.parent.name[len(prefix) :].strip().lower()
                for path in files_inside_partitions
                if path.parent.name[len(prefix) :].strip()
            }
        )
        if values:
            return values
    return []
