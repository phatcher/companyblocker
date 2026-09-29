"""Direct tests for `workspace.parquet_discovery`.

`discover_child_dirs_with_parquet` is the one function this module owns:
which of `parent_dir`'s children hold parquet data, either at the child's
own top level or in a named subdirectory, and either "any parquet at all" or
one specific required file. One case round-trips a real (if tiny) parquet
file through polars rather than an empty placeholder, proving the function's
`glob("*.parquet")` branch matches genuine written data, not just a name.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from workspace.parquet_discovery import discover_child_dirs_with_parquet


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_missing_parent_dir_is_empty(tmp_path: Path):
    assert discover_child_dirs_with_parquet(parent_dir=tmp_path / "absent") == []


def test_parent_dir_that_is_a_file_is_empty(tmp_path: Path):
    parent_as_file = tmp_path / "not-a-dir"
    parent_as_file.write_bytes(b"")

    assert discover_child_dirs_with_parquet(parent_dir=parent_as_file) == []


def test_child_that_is_a_file_is_skipped(tmp_path: Path):
    (tmp_path / "some-file").write_bytes(b"")

    assert discover_child_dirs_with_parquet(parent_dir=tmp_path) == []


def test_subdir_name_missing_under_child_is_skipped(tmp_path: Path):
    child = tmp_path / "gb"
    child.mkdir()
    _touch(child / "cleansed" / "gb-001.parquet")

    assert (
        discover_child_dirs_with_parquet(parent_dir=tmp_path, subdir_name="tokenized")
        == []
    )


def test_subdir_name_present_with_no_required_file_uses_any_parquet(tmp_path: Path):
    child = tmp_path / "gb"
    _touch(child / "tokenized" / "gb-001.parquet")

    assert discover_child_dirs_with_parquet(
        parent_dir=tmp_path, subdir_name="tokenized"
    ) == [child]


def test_no_subdir_name_checks_the_child_itself(tmp_path: Path):
    child = tmp_path / "gb"
    _touch(child / "gb-001.parquet")

    assert discover_child_dirs_with_parquet(parent_dir=tmp_path) == [child]


def test_no_parquet_present_is_excluded(tmp_path: Path):
    child = tmp_path / "gb"
    child.mkdir()
    (child / "notes.txt").write_bytes(b"")

    assert discover_child_dirs_with_parquet(parent_dir=tmp_path) == []


def test_required_file_present_is_included_even_without_a_bare_parquet_glob_match(
    tmp_path: Path,
):
    child = tmp_path / "gb"
    _touch(child / "tokenized" / "sample.parquet")

    assert discover_child_dirs_with_parquet(
        parent_dir=tmp_path, subdir_name="tokenized", required_file="sample.parquet"
    ) == [child]


def test_required_file_absent_excludes_the_child_even_with_other_parquet_present(
    tmp_path: Path,
):
    child = tmp_path / "gb"
    _touch(child / "tokenized" / "gb-001.parquet")

    assert (
        discover_child_dirs_with_parquet(
            parent_dir=tmp_path,
            subdir_name="tokenized",
            required_file="sample.parquet",
        )
        == []
    )


def test_discovers_a_real_parquet_round_trip(tmp_path: Path):
    """A genuine write-then-read, at tens of rows, rather than an empty
    placeholder file: proves the format and this module's discovery rule
    agree, which an empty `*.parquet` name alone cannot."""
    child = tmp_path / "gb"
    data_dir = child / "tokenized"
    data_dir.mkdir(parents=True)
    frame = pl.DataFrame(
        {
            "system_uri": [f"gb:{i}" for i in range(40)],
            "name": [f"COMPANY {i}" for i in range(40)],
        }
    )
    frame.write_parquet(data_dir / "gb-001.parquet")

    discovered = discover_child_dirs_with_parquet(
        parent_dir=tmp_path, subdir_name="tokenized"
    )

    assert discovered == [child]
    round_tripped = pl.read_parquet(data_dir / "gb-001.parquet")
    assert round_tripped.height == 40
    assert round_tripped["name"].to_list()[0] == "COMPANY 0"
