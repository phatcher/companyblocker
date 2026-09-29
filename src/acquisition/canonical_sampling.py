from __future__ import annotations

import math
from pathlib import Path

import polars as pl


def compute_z_sample_size(
    *,
    population_size: int,
    z_score: float,
    margin_of_error: float,
    proportion: float,
) -> int:
    if population_size <= 0:
        return 0

    initial_size = (z_score**2) * proportion * (1 - proportion) / (margin_of_error**2)
    finite_corrected = initial_size / (1 + ((initial_size - 1) / population_size))
    return max(1, min(population_size, math.ceil(finite_corrected)))


def _allocate_sample_quotas(row_counts: list[int], sample_size: int) -> list[int]:
    total_rows = sum(row_counts)
    raw_quotas = [sample_size * (count / total_rows) for count in row_counts]
    quotas = [math.floor(value) for value in raw_quotas]
    remaining = sample_size - sum(quotas)

    if remaining > 0:
        fractional_order = sorted(
            range(len(raw_quotas)),
            key=lambda idx: (raw_quotas[idx] - quotas[idx], row_counts[idx]),
            reverse=True,
        )
        for idx in fractional_order:
            if remaining == 0:
                break
            if quotas[idx] < row_counts[idx]:
                quotas[idx] += 1
                remaining -= 1

    return quotas


def write_canonical_sample_file(
    *,
    output_dir: Path,
    written_paths: list[Path],
    sample_file_name: str,
    seed: int,
    z_score: float,
    margin_of_error: float,
    proportion: float,
) -> None:
    if not written_paths:
        return

    row_counts: list[int] = []
    for path in written_paths:
        row_count = pl.read_parquet(path).height
        if row_count > 0:
            row_counts.append(row_count)
        else:
            row_counts.append(0)

    total_rows = sum(row_counts)
    if total_rows <= 0:
        return

    sample_size = compute_z_sample_size(
        population_size=total_rows,
        z_score=z_score,
        margin_of_error=margin_of_error,
        proportion=proportion,
    )

    quotas = _allocate_sample_quotas(row_counts, sample_size)

    sample_frames: list[pl.DataFrame] = []
    for idx, (path, quota, row_count) in enumerate(
        zip(written_paths, quotas, row_counts)
    ):
        if quota <= 0 or row_count <= 0:
            continue

        frame = pl.read_parquet(path)
        if quota >= row_count:
            sample_frames.append(frame)
            continue

        sampled = frame.sample(
            n=quota, with_replacement=False, shuffle=True, seed=seed + idx
        )
        sample_frames.append(sampled)

    if not sample_frames:
        return

    combined_sample = pl.concat(sample_frames, how="diagonal_relaxed")
    if combined_sample.height > sample_size:
        combined_sample = combined_sample.sample(
            n=sample_size, with_replacement=False, shuffle=True, seed=seed
        )

    sample_path = output_dir / sample_file_name
    combined_sample.write_parquet(sample_path, compression="snappy")
