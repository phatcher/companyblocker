"""Sole owner of how a system's shard/canonical data files are named.

A system's data files come in families. The **primary** family holds its
per-entity rows and is the unsuffixed one (`gleif-001.parquet`); every
**companion** family is named by the suffix in its filename
(`gleif-names-001.parquet`), and each carries a different row shape keyed off
the same `system_uri` rather than a company record.

Both the family list and the glob patterns live here so a caller asks rather
than restates: a hand-written `f"{code}-names-*.parquet"` in one module and a
family-list check in another drift the moment a family is added, which is how
a companion file gets read back in as primary rows.

Note that "primary" here names a *file family*, unrelated to the
`name_type="primary"` value inside a name-variant row, which means that
entity's own legal name.
"""

from __future__ import annotations

COMPANION_SIDECAR = "sidecar"
"""Column-split overflow file from Shard (see sharding.py's
_split_main_sidecar_frame) -- same row count as its primary file, subset of
columns."""

COMPANION_NAMES = "names"
"""One row per name-variant, not per entity (see
canonicalize_system_name_rows in canonical.py)."""

COMPANION_SUCCESSORS = "successors"
"""One row per SuccessorEntity edge, not per entity (see
canonicalize_system_successor_chain in canonical.py). GLEIF-only today."""

COMPANION_DUPLICATES = "duplicates"
"""Rows the Canonical stage's dedupe pass dropped, captured for inspection
with an added `dropped_reason` column (see canonical_dedupe.py). Shaped like
primary rows, but deliberately *not* part of the canonical view -- reading
them back in would undo the dedupe."""

COMPANION_FAMILIES: tuple[str, ...] = (
    COMPANION_SIDECAR,
    COMPANION_NAMES,
    COMPANION_SUCCESSORS,
    COMPANION_DUPLICATES,
)
"""Every companion family written alongside a system's primary data files.
Any code that globs a shard or canonical directory for primary rows must
exclude these, or it will silently feed the wrong row shape into whatever
comes next.
"""


def primary_data_file_glob(system_code: str) -> str:
    """Glob matching a system's primary data files -- and, unavoidably, its
    companions.

    A parquet glob cannot express "not one of these suffixes", so this
    over-matches by design and `is_primary_data_file` is what narrows it. The
    two are always used together; every family is glob-then-filter, and this
    is the one family where the filter is load-bearing.
    """
    return f"{system_code}-*.parquet"


def is_primary_data_file(*, file_name: str, system_code: str) -> bool:
    """Whether `file_name` holds the system's per-entity rows rather than any
    companion family's."""
    if not file_name.startswith(f"{system_code}-"):
        return False
    return not any(
        is_companion_data_file(
            file_name=file_name, system_code=system_code, family=family
        )
        for family in COMPANION_FAMILIES
    )


def companion_data_file_glob(*, system_code: str, family: str) -> str:
    """Glob matching one companion family, e.g. `gleif-names-*.parquet`."""
    return f"{_companion_file_prefix(system_code=system_code, family=family)}*.parquet"


def is_companion_data_file(*, file_name: str, system_code: str, family: str) -> bool:
    """Whether `file_name` belongs to one specific companion family."""
    return file_name.startswith(
        _companion_file_prefix(system_code=system_code, family=family)
    )


def _companion_file_prefix(*, system_code: str, family: str) -> str:
    return f"{system_code}-{family}-"
