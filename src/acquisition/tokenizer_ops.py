"""The Tokenize stage: read `data/<system>/cleansed/`, write `data/<system>/tokenized/`.

Whether the cleansed layer is the merged, `jurisdiction_code=`-partitioned view is decided by its shape, not a directory name, and that partitioning is kept in `tokenized/` when present.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

import polars as pl
from company_tokenize import tokenize_name as _tokenize_name_runtime
from company_tokenize import tokenize_name_dual as _tokenize_name_dual_runtime
from company_tokenize import train_wordpiece as _train_wordpiece_runtime
from company_tokenize.name_preprocessing import (
    TRAINING_PREPROCESS_PROFILE,
    name_preprocessing,
)

from workspace.data_layout import layer_system
from workspace.layer_layout import resolve_primary_files


class SystemSamplingInfo(TypedDict):
    system: str
    files: list[dict[str, int | str]]
    available_rows: int
    skipped_files: int
    target_rows: int


TOKENIZED_OUTPUT_COLUMNS = (
    "country_tokens",
    "global_tokens",
)
DEFAULT_SINGLE_TOKEN_COLUMN = "name_tokens"


def _normalised_training_names(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame` with its `name` column normalised, dropping names left empty."""
    return name_preprocessing(
        frame, name_col="name", profile=TRAINING_PREPROCESS_PROFILE, out_col="name"
    ).filter(pl.col("name").is_not_null() & (pl.col("name") != ""))


def _corpus_content_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimize_sweeps_at_risk(scope_directory: Path) -> list[tuple[str, int]]:
    """[(tokenizer id, candidate_count), ...] for every non-empty run-log under
    `scope_directory`, across every tokenizer -- corpus_path is shared across
    tokenizers for a system, so a wordpiece sweep is just as much at risk from
    a sentencepiece resample as from a wordpiece one.

    Sweep leaves live nested under `scope_directory/<tokenizer id>/optimize/
    <corpus_hash>/<grid_hash>/` (see `company_tokenize.resolve_optimize_sweep_paths`),
    hence the glob depth. A resample that changes this
    system's corpus content produces a *new* `<corpus_hash>` folder rather
    than destroying an existing sweep in place -- old sweep leaves survive on
    disk untouched, they just become unreachable via a new sweep's hash. The
    refusal below still applies: an --optimize-wipe-acknowledged resample
    means "I know the old sweep(s) become unreachable from here on," not
    "nothing is preserved."
    """
    if not scope_directory.exists():
        return []
    at_risk: list[tuple[str, int]] = []
    for run_log_path in sorted(
        scope_directory.glob("*/optimize/*/*/optimize_runs.parquet")
    ):
        try:
            row_count = pl.scan_parquet(run_log_path).select(pl.len()).collect().item()
        except (OSError, pl.exceptions.PolarsError):
            continue
        if row_count == 0:
            continue
        at_risk.append((run_log_path.parents[3].name, int(row_count)))
    return at_risk


def _write_corpus_guarding_optimize_sweeps(
    corpus_df: pl.DataFrame, corpus_path: Path, *, allow_optimize_wipe: bool
) -> bool:
    """Write `corpus_df` to `corpus_path`, refusing to silently orphan an
    optimize sweep that depends on this exact file's bytes.

    Returns True if the file was actually (re)written, False if the existing
    file's content already matched and was left untouched (its mtime isn't
    bumped for a no-op rewrite -- see `is_final_result_fresh` in
    `training/optimize_execution.py`, which compares mtimes against this
    corpus and would otherwise be fooled into thinking a promoted result had
    gone stale when nothing actually changed).
    """
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    if not corpus_path.exists():
        corpus_df.write_parquet(corpus_path)
        return True

    fd, tmp_name = tempfile.mkstemp(
        suffix=".parquet", dir=corpus_path.parent, prefix=f".{corpus_path.stem}-"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        corpus_df.write_parquet(tmp_path)
        if _corpus_content_hash(tmp_path) == _corpus_content_hash(corpus_path):
            return False

        at_risk = _optimize_sweeps_at_risk(corpus_path.parent)
        if at_risk and not allow_optimize_wipe:
            details = ", ".join(
                f"{trainer} ({count} candidates)" for trainer, count in at_risk
            )
            raise RuntimeError(
                f"Resampling would change {corpus_path}'s content, and an optimize "
                f"sweep for this system depends on its current bytes: {details}. "
                "Its sweep directory is keyed by corpus content, so it won't be "
                "deleted -- but a resample makes it unreachable from any *new* "
                "sweep (which will hash to a different, empty directory and start "
                "over from scratch instead of extending it). Re-run with "
                "--optimize-wipe if you intend this, or archive the candidates you "
                "want to keep first (scripts/archive_optimize_candidate.py)."
            )
        if at_risk:
            details = ", ".join(
                f"{trainer} ({count} candidates)" for trainer, count in at_risk
            )
            print(
                f"[tokenizer] --optimize-wipe acknowledged: proceeding despite "
                f"at-risk sweep(s) for {corpus_path}: {details}"
            )
        os.replace(tmp_path, corpus_path)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def sample_training_corpus(
    cleansed_dir: Path,
    corpus_path: Path,
    sample_fraction: float = 1.0,
    name_col: str = "name_cleansed",
    allow_optimize_wipe: bool = False,
) -> int:
    if not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be in the interval (0, 1].")

    cleansed_files = [
        str(path)
        for path in resolve_primary_files(
            cleansed_dir, system_code=layer_system(cleansed_dir)
        )
    ]
    if not cleansed_files:
        raise FileNotFoundError(
            f"No cleansed parquet files under {cleansed_dir}/jurisdiction_code=*/. Run Pass 1 first."
        )

    required_columns = {"system_uri", name_col}
    eligible_files: list[str] = []
    skipped_files = 0
    for file_path in cleansed_files:
        schema_names = set(pl.read_parquet_schema(file_path).keys())
        if required_columns.issubset(schema_names):
            eligible_files.append(file_path)
        else:
            skipped_files += 1

    if not eligible_files:
        raise ValueError(
            f"No cleansed files in {cleansed_dir} contain required columns 'system_uri' and '{name_col}'."
        )

    selected = list(eligible_files)
    if sample_fraction >= 1.0:
        print(f"Using all rows from all {len(selected)} eligible files...\n")
    else:
        print(
            f"Sampling {sample_fraction * 100:.0f}% of rows from all {len(selected)} eligible files...\n"
        )

    corpus_frames: list[pl.DataFrame] = []
    for file_path in selected:
        df = (
            pl.read_parquet(file_path, columns=["system_uri", name_col])
            .sample(fraction=sample_fraction, seed=42)
            .with_columns(
                [
                    pl.col("system_uri").cast(pl.Utf8),
                    pl.col(name_col)
                    .cast(pl.Utf8)
                    .fill_null("")
                    .str.strip_chars()
                    .alias("name"),
                ]
            )
            .select(["system_uri", "name"])
        )
        corpus_frames.append(_normalised_training_names(df))
        del df

    if corpus_frames:
        corpus_df = pl.concat(corpus_frames, how="vertical", rechunk=True)
    else:
        corpus_df = pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})
    written = _write_corpus_guarding_optimize_sweeps(
        corpus_df, corpus_path, allow_optimize_wipe=allow_optimize_wipe
    )

    print(
        f"Training corpus parquet: {corpus_df.height:,} rows -> {corpus_path} "
        f"(columns=system_uri,name; skipped {skipped_files} file(s) without required columns"
        f"{'' if written else '; unchanged, left in place'})"
    )
    return corpus_df.height


def _validate_global_sampling_args(
    *,
    sample_fraction: float,
    balance_mode: str,
    system_target_rows: int,
    system_min_rows: int,
    system_max_rows: int | None,
    file_min_rows: int,
) -> None:
    if not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be in the interval (0, 1].")
    if balance_mode not in {"equal", "sqrt", "proportional"}:
        raise ValueError("balance_mode must be one of: equal, sqrt, proportional.")
    if system_target_rows <= 0:
        raise ValueError("system_target_rows must be greater than zero.")
    if system_min_rows < 0:
        raise ValueError("system_min_rows must be >= 0.")
    if system_max_rows is not None and system_max_rows <= 0:
        raise ValueError("system_max_rows must be greater than zero when provided.")
    if system_max_rows is not None and system_min_rows > system_max_rows:
        raise ValueError("system_min_rows cannot exceed system_max_rows.")
    if file_min_rows < 0:
        raise ValueError("file_min_rows must be >= 0.")


def _collect_system_sampling_info(
    *,
    cleansed_dirs: Iterable[Path],
    name_col: str,
) -> list[SystemSamplingInfo]:
    system_infos: list[SystemSamplingInfo] = []
    for cleansed_dir in cleansed_dirs:
        system = layer_system(cleansed_dir)
        cleansed_files = [
            str(path)
            for path in resolve_primary_files(cleansed_dir, system_code=system)
        ]
        if not cleansed_files:
            raise FileNotFoundError(
                f"No cleansed parquet files under {cleansed_dir}/jurisdiction_code=*/. Run Pass 1 first."
            )

        file_infos: list[dict[str, int | str]] = []
        skipped_files = 0
        for file_path in cleansed_files:
            schema_names = set(pl.read_parquet_schema(file_path).keys())
            if "system_uri" not in schema_names or name_col not in schema_names:
                skipped_files += 1
                continue

            non_empty_rows = (
                pl.scan_parquet(file_path)
                .select(
                    pl.col(name_col)
                    .cast(pl.Utf8)
                    .fill_null("")
                    .str.strip_chars()
                    .ne("")
                    .sum()
                    .alias("non_empty_rows")
                )
                .collect()
                .item()
            )
            file_infos.append({"path": file_path, "rows": int(non_empty_rows)})

        if not file_infos:
            raise ValueError(
                f"No cleansed files in {cleansed_dir} contain required columns 'system_uri' and '{name_col}'."
            )

        available_rows = sum(int(info["rows"]) for info in file_infos)
        system_infos.append(
            {
                "system": system,
                "files": file_infos,
                "available_rows": available_rows,
                "skipped_files": skipped_files,
                "target_rows": 0,
            }
        )
    return system_infos


def _weighted_rows_for_systems(
    *, system_infos: list[SystemSamplingInfo], balance_mode: str
) -> list[float]:
    weighted_rows: list[float] = []
    for info in system_infos:
        available_rows = info["available_rows"]
        if available_rows <= 0:
            weighted_rows.append(0.0)
        elif balance_mode == "equal":
            weighted_rows.append(1.0)
        elif balance_mode == "sqrt":
            weighted_rows.append(math.sqrt(float(available_rows)))
        else:
            weighted_rows.append(float(available_rows))
    return weighted_rows


def _assign_system_target_rows(
    *,
    system_infos: list[SystemSamplingInfo],
    weighted_rows: list[float],
    balance_mode: str,
    system_target_rows: int,
    system_min_rows: int,
    system_max_rows: int | None,
) -> None:
    positive_weight_sum = sum(weighted_rows)
    systems_count = len(system_infos)
    total_budget = system_target_rows * systems_count

    for idx, info in enumerate(system_infos):
        available_rows = info["available_rows"]
        if available_rows <= 0:
            info["target_rows"] = 0
            continue

        if balance_mode == "equal" or positive_weight_sum <= 0:
            raw_target = float(system_target_rows)
        else:
            raw_target = total_budget * (weighted_rows[idx] / positive_weight_sum)

        target_rows = round(raw_target)
        target_rows = max(target_rows, system_min_rows)
        if system_max_rows is not None:
            target_rows = min(target_rows, system_max_rows)
        target_rows = min(target_rows, available_rows)
        info["target_rows"] = max(target_rows, 0)


def _redistribute_diff(
    system_counts: list[int], headroom: list[int], diff: int
) -> None:
    increasing = diff > 0
    for idx in range(len(system_counts)):
        if diff == 0:
            break
        available = headroom[idx]
        if available <= 0:
            continue
        amount = min(available, abs(diff))
        if increasing:
            system_counts[idx] += amount
            diff -= amount
        else:
            system_counts[idx] -= amount
            diff += amount


def _rebalance_counts_to_target(
    desired_counts: list[int], capacities: list[int], target_rows: int
) -> list[int]:
    if sum(desired_counts) > target_rows:
        scale = target_rows / sum(desired_counts)
        system_counts = [
            min(capacity, max(0, round(desired * scale)))
            for desired, capacity in zip(desired_counts, capacities, strict=True)
        ]
    else:
        system_counts = list(desired_counts)

    diff = target_rows - sum(system_counts)
    if diff > 0:
        headroom = [
            capacities[idx] - system_counts[idx] for idx in range(len(system_counts))
        ]
        _redistribute_diff(system_counts, headroom, diff)
    elif diff < 0:
        _redistribute_diff(system_counts, list(system_counts), diff)

    return system_counts


def _compute_system_counts(
    *,
    file_infos: list[dict[str, int | str]],
    target_rows: int,
    sample_fraction: float,
    file_min_rows: int,
) -> list[int]:
    desired_counts: list[int] = []
    file_rows: list[int] = []
    for file_info in file_infos:
        row_count = int(file_info["rows"])
        file_rows.append(row_count)
        min_from_file = min(row_count, file_min_rows)
        sample_based = round(row_count * sample_fraction)
        desired = max(min_from_file, sample_based)
        if row_count > 0:
            desired = max(desired, 1)
        desired_counts.append(min(desired, row_count))

    desired_total = sum(desired_counts)
    if target_rows <= 0 or desired_total <= 0:
        return [0 for _ in desired_counts]
    if desired_total == target_rows:
        return list(desired_counts)

    return _rebalance_counts_to_target(desired_counts, file_rows, target_rows)


def _sample_system_frames(
    *,
    system: str,
    file_infos: list[dict[str, int | str]],
    system_counts: list[int],
    target_rows: int,
    balance_seed: int,
    name_col: str,
    sample_fraction: float,
    available_rows: int,
) -> pl.DataFrame:
    system_frames: list[pl.DataFrame] = []
    print(
        f"Sampling {sample_fraction * 100:.0f}% base rows for {system}: "
        f"target={target_rows:,}, available={available_rows:,}, files={len(file_infos)}"
    )

    for file_idx, (file_info, sample_count) in enumerate(
        zip(file_infos, system_counts, strict=True), start=1
    ):
        file_path = str(file_info["path"])
        row_count = int(file_info["rows"])
        if sample_count <= 0 or row_count <= 0:
            continue

        seeded = (balance_seed + zlib.crc32(f"{system}:{file_path}".encode())) % (
            2**32 - 1
        )
        frame = (
            pl.read_parquet(file_path, columns=["system_uri", name_col])
            .with_columns(
                [
                    pl.col("system_uri").cast(pl.Utf8),
                    pl.col(name_col)
                    .cast(pl.Utf8)
                    .fill_null("")
                    .str.strip_chars()
                    .alias("name"),
                ]
            )
            .select(["system_uri", "name"])
            .filter(pl.col("name") != "")
        )
        if frame.height <= sample_count:
            sampled = frame
        else:
            sampled = frame.sample(n=sample_count, seed=seeded, shuffle=True)
        sampled = _normalised_training_names(sampled)

        system_frames.append(sampled)
        del frame
        print(
            f"  [{system}] file {file_idx}/{len(file_infos)} -> sampled {sampled.height:,}/{row_count:,}"
        )

    if system_frames:
        system_df = pl.concat(system_frames, how="vertical", rechunk=True)
    else:
        system_df = pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})

    if system_df.height > target_rows > 0:
        seeded_system = (balance_seed + zlib.crc32(system.encode("utf-8"))) % (
            2**32 - 1
        )
        system_df = system_df.sample(n=target_rows, seed=seeded_system, shuffle=True)

    print(f"  [{system}] contributed {system_df.height:,} row(s)")
    return system_df


def sample_training_corpus_for_global(
    *,
    cleansed_dirs: Iterable[Path],
    corpus_path: Path,
    sample_fraction: float = 1.0,
    balance_mode: str = "equal",
    system_target_rows: int = 300000,
    system_min_rows: int = 50000,
    system_max_rows: int | None = None,
    file_min_rows: int = 500,
    balance_seed: int = 42,
    name_col: str = "name_cleansed",
    allow_optimize_wipe: bool = False,
) -> int:
    corpus_frames: list[pl.DataFrame] = []
    total_skipped_files = 0

    _validate_global_sampling_args(
        sample_fraction=sample_fraction,
        balance_mode=balance_mode,
        system_target_rows=system_target_rows,
        system_min_rows=system_min_rows,
        system_max_rows=system_max_rows,
        file_min_rows=file_min_rows,
    )

    system_infos = _collect_system_sampling_info(
        cleansed_dirs=cleansed_dirs, name_col=name_col
    )
    weighted_rows = _weighted_rows_for_systems(
        system_infos=system_infos, balance_mode=balance_mode
    )
    _assign_system_target_rows(
        system_infos=system_infos,
        weighted_rows=weighted_rows,
        balance_mode=balance_mode,
        system_target_rows=system_target_rows,
        system_min_rows=system_min_rows,
        system_max_rows=system_max_rows,
    )

    for info in system_infos:
        system = info["system"]
        file_infos = info["files"]
        skipped_files = info["skipped_files"]
        target_rows = info["target_rows"]
        total_skipped_files += skipped_files

        system_counts = _compute_system_counts(
            file_infos=file_infos,
            target_rows=target_rows,
            sample_fraction=sample_fraction,
            file_min_rows=file_min_rows,
        )
        system_df = _sample_system_frames(
            system=system,
            file_infos=file_infos,
            system_counts=system_counts,
            target_rows=target_rows,
            balance_seed=balance_seed,
            name_col=name_col,
            sample_fraction=sample_fraction,
            available_rows=int(info["available_rows"]),
        )
        corpus_frames.append(system_df)

    if corpus_frames:
        corpus_df = pl.concat(corpus_frames, how="vertical", rechunk=True)
    else:
        corpus_df = pl.DataFrame(schema={"system_uri": pl.Utf8, "name": pl.Utf8})
    written = _write_corpus_guarding_optimize_sweeps(
        corpus_df, corpus_path, allow_optimize_wipe=allow_optimize_wipe
    )
    print(
        f"Global training corpus parquet: {corpus_df.height:,} rows -> {corpus_path} "
        f"(columns=system_uri,name; mode={balance_mode}, skipped {total_skipped_files} file(s) without required columns"
        f"{'' if written else '; unchanged, left in place'})"
    )
    return corpus_df.height


def train_wordpiece(
    corpus_path: Path,
    tokenizer_path: Path,
    vocab_size: int | None = None,
    show_progress: bool = True,
) -> int:
    return _train_wordpiece_runtime(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        show_progress=show_progress,
    )


def tokenize_name(
    cleansed_dir: Path,
    tokenized_dir: Path,
    tokenizer_path: Path | None = None,
    name_col: str = "name_cleansed",
    token_col: str = DEFAULT_SINGLE_TOKEN_COLUMN,
    input_file: str | None = None,
) -> int:
    return _tokenize_name_runtime(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        tokenizer_path=tokenizer_path,
        name_col=name_col,
        token_col=token_col,
        input_file=input_file,
    )


def tokenize_name_dual(
    cleansed_dir: Path,
    tokenized_dir: Path,
    *,
    country_tokenizer_path: Path,
    global_tokenizer_path: Path,
    name_col: str = "name_cleansed",
    country_token_col: str = "country_tokens",
    global_token_col: str = "global_tokens",
    input_file: str | None = None,
) -> int:
    return _tokenize_name_dual_runtime(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        country_tokenizer_path=country_tokenizer_path,
        global_tokenizer_path=global_tokenizer_path,
        name_col=name_col,
        country_token_col=country_token_col,
        global_token_col=global_token_col,
        input_file=input_file,
    )
