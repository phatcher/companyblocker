from __future__ import annotations

from pathlib import Path

import polars as pl

from acquisition.canonical_sampling import (
    compute_z_sample_size,
    write_canonical_sample_file,
)

_Z_95 = 1.96


def test_compute_z_sample_size_returns_zero_for_empty_population():
    assert (
        compute_z_sample_size(
            population_size=0, z_score=_Z_95, margin_of_error=0.05, proportion=0.5
        )
        == 0
    )


def test_compute_z_sample_size_never_exceeds_population_size():
    size = compute_z_sample_size(
        population_size=10, z_score=_Z_95, margin_of_error=0.01, proportion=0.5
    )

    assert size == 10


def test_compute_z_sample_size_is_at_least_one_for_a_nonempty_population():
    size = compute_z_sample_size(
        population_size=1, z_score=_Z_95, margin_of_error=0.05, proportion=0.5
    )

    assert size == 1


def test_compute_z_sample_size_grows_with_population_and_shrinks_with_margin():
    small_population = compute_z_sample_size(
        population_size=1_000, z_score=_Z_95, margin_of_error=0.05, proportion=0.5
    )
    large_population = compute_z_sample_size(
        population_size=1_000_000, z_score=_Z_95, margin_of_error=0.05, proportion=0.5
    )
    tighter_margin = compute_z_sample_size(
        population_size=1_000_000, z_score=_Z_95, margin_of_error=0.01, proportion=0.5
    )

    assert small_population < large_population
    assert large_population < tighter_margin


def test_write_canonical_sample_file_is_a_no_op_for_no_written_paths(tmp_path: Path):
    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[],
        sample_file_name="sample.parquet",
        seed=1,
        z_score=_Z_95,
        margin_of_error=0.05,
        proportion=0.5,
    )

    assert list(tmp_path.iterdir()) == []


def test_write_canonical_sample_file_is_a_no_op_when_every_input_file_is_empty(
    tmp_path: Path,
):
    empty_path = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"id": pl.Int64}).write_parquet(empty_path)

    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[empty_path],
        sample_file_name="sample.parquet",
        seed=1,
        z_score=_Z_95,
        margin_of_error=0.05,
        proportion=0.5,
    )

    assert not (tmp_path / "sample.parquet").exists()


def test_write_canonical_sample_file_keeps_every_row_when_quota_covers_it(
    tmp_path: Path,
):
    source_path = tmp_path / "source.parquet"
    pl.DataFrame({"id": [1, 2]}).write_parquet(source_path)

    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[source_path],
        sample_file_name="sample.parquet",
        seed=1,
        z_score=_Z_95,
        margin_of_error=0.5,  # loose margin on a tiny population: quota == population
        proportion=0.5,
    )

    sample = pl.read_parquet(tmp_path / "sample.parquet")
    assert sorted(sample["id"].to_list()) == [1, 2]


def test_write_canonical_sample_file_combines_rows_across_multiple_source_files(
    tmp_path: Path,
):
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    pl.DataFrame({"id": list(range(100))}).write_parquet(first_path)
    pl.DataFrame({"id": list(range(100, 200))}).write_parquet(second_path)

    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[first_path, second_path],
        sample_file_name="sample.parquet",
        seed=7,
        z_score=_Z_95,
        margin_of_error=0.05,
        proportion=0.5,
    )

    sample = pl.read_parquet(tmp_path / "sample.parquet")
    ids = sample["id"].to_list()
    assert any(value < 100 for value in ids)
    assert any(value >= 100 for value in ids)
    assert len(ids) == len(set(ids))


def test_write_canonical_sample_file_is_deterministic_for_a_given_seed(
    tmp_path: Path,
):
    source_path = tmp_path / "source.parquet"
    pl.DataFrame({"id": list(range(500))}).write_parquet(source_path)

    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[source_path],
        sample_file_name="sample_a.parquet",
        seed=42,
        z_score=_Z_95,
        margin_of_error=0.05,
        proportion=0.5,
    )
    write_canonical_sample_file(
        output_dir=tmp_path,
        written_paths=[source_path],
        sample_file_name="sample_b.parquet",
        seed=42,
        z_score=_Z_95,
        margin_of_error=0.05,
        proportion=0.5,
    )

    sample_a = pl.read_parquet(tmp_path / "sample_a.parquet")
    sample_b = pl.read_parquet(tmp_path / "sample_b.parquet")
    assert sample_a["id"].to_list() == sample_b["id"].to_list()
