"""Direct tests for `workspace.data_file_naming`.

Pure string arithmetic over file names -- no filesystem needed. Each
assertion pins the family-glob/filter contract that `layer_layout` and every
other caller in `src/` relies on to tell a system's per-entity rows apart
from a companion family's, since a parquet glob cannot express "not one of
these suffixes" and this module's filter is what narrows it.
"""

from __future__ import annotations

import pytest

from workspace.data_file_naming import (
    COMPANION_DUPLICATES,
    COMPANION_FAMILIES,
    COMPANION_NAMES,
    COMPANION_SIDECAR,
    COMPANION_SUCCESSORS,
    companion_data_file_glob,
    is_companion_data_file,
    is_primary_data_file,
    primary_data_file_glob,
)


def test_primary_data_file_glob_is_the_system_prefixed_wildcard():
    assert primary_data_file_glob("gleif") == "gleif-*.parquet"


def test_companion_data_file_glob_inserts_the_family_segment():
    assert (
        companion_data_file_glob(system_code="gleif", family=COMPANION_NAMES)
        == "gleif-names-*.parquet"
    )


@pytest.mark.parametrize(
    "file_name,expected",
    [
        ("gleif-001.parquet", True),
        ("gleif-names-001.parquet", False),
        ("gleif-sidecar-001.parquet", False),
        ("gleif-successors-001.parquet", False),
        ("gleif-duplicates-001.parquet", False),
        ("gb-001.parquet", False),  # different system code entirely
        ("gleifx-001.parquet", False),  # prefix must be followed by "-"
    ],
)
def test_is_primary_data_file(file_name: str, expected: bool):
    assert is_primary_data_file(file_name=file_name, system_code="gleif") is expected


@pytest.mark.parametrize(
    "family,matching_file,other_file",
    [
        (COMPANION_SIDECAR, "gleif-sidecar-001.parquet", "gleif-names-001.parquet"),
        (COMPANION_NAMES, "gleif-names-001.parquet", "gleif-sidecar-001.parquet"),
        (
            COMPANION_SUCCESSORS,
            "gleif-successors-001.parquet",
            "gleif-duplicates-001.parquet",
        ),
        (
            COMPANION_DUPLICATES,
            "gleif-duplicates-001.parquet",
            "gleif-successors-001.parquet",
        ),
    ],
)
def test_is_companion_data_file_matches_only_its_own_family(
    family: str, matching_file: str, other_file: str
):
    assert (
        is_companion_data_file(
            file_name=matching_file, system_code="gleif", family=family
        )
        is True
    )
    assert (
        is_companion_data_file(file_name=other_file, system_code="gleif", family=family)
        is False
    )


def test_every_companion_family_is_excluded_from_primary():
    """`COMPANION_FAMILIES` is what `is_primary_data_file` iterates to narrow
    the over-matching glob; a family added there but missing here would
    silently leak that family's rows in as primary ones."""
    for family in COMPANION_FAMILIES:
        file_name = f"gleif-{family}-001.parquet"
        assert is_primary_data_file(file_name=file_name, system_code="gleif") is False
