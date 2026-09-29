"""The Cleanse stage: canonical rows through `company_cleanse`, written to `data/<system>/cleansed/`.

Per-chunk output is staged under `cleansed/chunks/`; `materialize_cleansed_merge` then dedupes by `system_uri` and writes the merged view partitioned by `jurisdiction_code=<country>/` at `cleansed/`'s top level. Readers use that top level, never `chunks/`.
"""

from __future__ import annotations

import gc
import glob
import json
import os
import shutil
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlflow
import polars as pl
from company_cleanse import CleanseConfig, cleanse_lazyframe

from workspace.data_layout import CLEANSED_LAYER_NAME, data_root, system_layer_dir


def _write_cleanse_output(
    lf_cleansed: pl.LazyFrame,
    output_path: str,
    *,
    reporter: RowProgress | None = None,
    source_rows: int | None = None,
) -> int:
    """Collect a cleansed lazyframe and write it, reporting as it goes.

    Collected in row batches rather than in one call so a run reports while
    a file is in flight. A partition file holds 1,000,000 rows and takes over
    a minute, and `company_cleanse` exposes no hook inside a single collect,
    so the batching is what makes progress observable at all. Safe because
    cleansing is row-wise throughout: no grouping, window, sort or
    cumulative operation appears anywhere in `company_cleanse`, so slicing
    the scan cannot change a row's result.

    Peak memory is unchanged: the whole frame is concatenated before writing,
    exactly as a single collect would have held it.
    """
    if reporter is None or not source_rows:
        frame = lf_cleansed.collect()
        frame.write_parquet(output_path, compression="snappy")
        return frame.height

    batches: list[pl.DataFrame] = []
    for offset in range(0, source_rows, PROGRESS_ROW_INTERVAL):
        batch = lf_cleansed.slice(offset, PROGRESS_ROW_INTERVAL).collect()
        if batch.height == 0:
            continue
        batches.append(batch)
        reporter.advance(batch.height)

    if not batches:
        frame = lf_cleansed.collect()
    else:
        frame = batches[0] if len(batches) == 1 else pl.concat(batches)
    frame.write_parquet(output_path, compression="snappy")
    return frame.height


from workspace.data_file_naming import is_primary_data_file, primary_data_file_glob
from workspace.data_layout import layer_system
from workspace.layer_layout import (
    PARTITION_COLUMN,
    PARTITION_ROWS_PER_FILE,
    LayerSwapError,
    fresh_layer_staging_dir,
    is_partitioned_layer,
    mirror_relative_path,
    partition_value_of,
    primary_family_dir,
    resolve_primary_files,
    swap_layer_into_place,
)
from workspace.roots import WorkspaceRoots, default_workspace_roots

_LEGACY_PARTITION_DIR_GLOB = f"{PARTITION_COLUMN}=*"
"""Glob for a partition directory at a layer's own top level -- the
pre-family-split location `primary_family_dir` superseded. Built from the
shared `PARTITION_COLUMN` rather than a literal string so this can't drift
from `layer_layout`'s own definition of the column name."""

from .cleansed_contracts import CLEANSED_MERGE_REQUIRED_COLUMNS
from .country_counts import coerce_country_counts
from .output_chunking import (
    NULL_PARTITION_SENTINEL,
    collate_chunk_name,
    write_chunked_output_files_from_frame,
    write_partitioned_output_files_from_frame,
)
from .row_progress import PROGRESS_ROW_INTERVAL, RowProgress


def _coerce_country_counts(raw: object) -> dict[str, int]:
    return coerce_country_counts(raw)


def _country_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.height == 0:
        return {}
    return {
        str(row["country"]): int(row["count"])
        for row in frame.group_by("country").len(name="count").to_dicts()
    }


def _sorted_country_counts(values: dict[str, int]) -> dict[str, int]:
    return {key: int(values[key]) for key in sorted(values)}


def _clear_stale_cleansed_view(*, cleansed_dir: Path) -> None:
    """Clear a prior merged/deduped view (partitioned jurisdiction_code=*/
    directories, or flat {code}-NNN.parquet files) from cleansed_dir's own
    top level -- but never touch cleansed_dir/chunks (the raw per-input-file
    output materialize_cleansed_merge reads from) or a scoped sample.parquet
    (a distinct, separately-managed artifact a general run never produces or
    consumes -- see the input_file is not None handling below).
    """
    shutil.rmtree(primary_family_dir(cleansed_dir), ignore_errors=True)
    for partition_dir in cleansed_dir.glob(_LEGACY_PARTITION_DIR_GLOB):
        if partition_dir.is_dir():
            shutil.rmtree(partition_dir)
    for stale_file in cleansed_dir.glob("*.parquet"):
        if stale_file.name == "sample.parquet":
            continue
        stale_file.unlink()
    metadata_path = cleansed_dir / "_prepared_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()


def _migrate_legacy_flat_cleanse_chunks(*, cleansed_dir: Path) -> None:
    """One-time, automatic migration for a system's cleansed/ directory
    still in the legacy flat layout (raw per-input-file chunks sitting
    directly at cleansed_dir's top level, no chunks/ subdirectory yet). Any
    flat *.parquet found there while neither chunks/ nor a partitioned
    jurisdiction_code=*/ view exists yet is unambiguously the old raw-chunk
    layout (a new merge always produces a jurisdiction_code=*/ view by
    default, or lands its own raw chunks under chunks/ first), so this can
    move them into cleansed_dir/chunks in place with no ambiguity. A no-op
    once chunks/ or a partitioned view already exists -- checked via
    `is_partitioned_layer`, which covers both the current `primary/`-family
    shape and the pre-family-split top-level one, so a system already fully
    migrated to `primary/` doesn't get a stray top-level flat file swept up
    as though it predated any merge.
    """
    chunks_dir = cleansed_dir / "chunks"
    if chunks_dir.exists() or is_partitioned_layer(cleansed_dir):
        return
    legacy_files = [
        path
        for path in cleansed_dir.glob("*.parquet")
        if path.name not in {"empty_cleanser.parquet", "sample.parquet"}
    ]
    if not legacy_files:
        return
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for legacy_file in legacy_files:
        legacy_file.rename(chunks_dir / legacy_file.name)


def _read_valid_cleansed_frames(files: list[Path]) -> list[pl.DataFrame]:
    """Read `files` into frames, skipping any that are unreadable or missing
    a required merge column -- a scoped/sample run's `chunks/` directory can
    contain partial or stale files that shouldn't abort the whole merge.
    """
    frames: list[pl.DataFrame] = []
    for path in files:
        try:
            schema_names = set(pl.read_parquet_schema(path).keys())
        except (OSError, pl.exceptions.PolarsError):
            continue
        if any(
            column_name not in schema_names
            for column_name in CLEANSED_MERGE_REQUIRED_COLUMNS
        ):
            continue
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError):
            continue
        frames.append(frame)
    return frames


def _write_cleansed_metadata(
    *,
    cleansed_dir: Path,
    system: str,
    rows_per_file: int,
    rows_by_country: dict[str, int],
) -> Path:
    """Write `_prepared_metadata.json`, which `validation.prepared_dataset`
    and `validation.runner` read to size and cache a prepared dataset.

    The three row-count fields are kept identical to each other, as they have
    been since this file was introduced: nothing between counting and writing
    filters rows, so a distinct "filtered out" count would be a fabricated
    number rather than a measured one.
    """
    metadata_path = cleansed_dir / "_prepared_metadata.json"
    sorted_counts = _sorted_country_counts(rows_by_country)
    metadata_path.write_text(
        json.dumps(
            {
                "system": system,
                "rows_per_file": rows_per_file,
                "total_rows_by_country": sorted_counts,
                "shard_rows_by_country": sorted_counts,
                "kept_rows_by_country": sorted_counts,
                "filtered_out_rows_by_country": {
                    country: 0 for country in sorted_counts
                },
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata_path


def materialize_cleansed_merge(
    *,
    source_dir: Path,
    output_dir: Path,
    system: str,
    name_col: str,
    rows_per_file: int,
    input_file: str | None = None,
    force_rebuild: bool = False,
    partition_by: str | None = "jurisdiction_code",
) -> Path:
    """Reads every staged chunk under `source_dir` (cleansed_dir/chunks),
    validates and concatenates them into one snapshot, and writes the merged
    view directly to `output_dir` (cleansed_dir itself) -- partitioned by
    jurisdiction_code (the default) or flat if `partition_by` is None.

    Not part of the acquisition Cleanse stage any more. Dedupe and
    jurisdiction partitioning belong to Canonical, so Cleanse maps each
    canonical partition file straight to its cleansed counterpart
    (`cleanse_canonical_view`) and stages nothing. What still needs a merge
    is `validation.perturbation_materializer`, which accumulates several
    source systems into one perturbed system's cleansed view via a staged
    chunk slot each, and so has many chunks and no canonical partitions to
    map from.

    **Identity is the caller's responsibility.** This pass no longer dedupes
    by `system_uri`: the acquisition path's dedupe now happens at
    Canonical, and the perturbation path's uniqueness is enforced ahead of
    any disk write by `build_perturbed_frame`'s in-batch collision hard-fail
    and `_check_no_cross_run_collision`'s cross-call one. Rows with a null or
    empty `system_uri` are still dropped, since a row with no identity cannot
    be a record. A duplicate `system_uri` reaching here is now written, not
    silently collapsed.
    """
    cleansed_dir = Path(output_dir).resolve()
    already_materialized = bool(
        resolve_primary_files(cleansed_dir, system_code=system)
    ) or any(
        path
        for path in cleansed_dir.glob("*.parquet")
        if path.name not in {"empty_cleanser.parquet", "sample.parquet"}
    )
    if not force_rebuild and already_materialized:
        return cleansed_dir

    cleansed_dir.mkdir(parents=True, exist_ok=True)
    if input_file is not None:
        # Scoped/sample run -- only clear a stale copy of this exact file,
        # never the rest of cleansed_dir's top level (which may hold other
        # unrelated, already-materialized content this run shouldn't touch).
        stale_scoped_output = cleansed_dir / input_file
        if stale_scoped_output.exists():
            stale_scoped_output.unlink()
    else:
        _clear_stale_cleansed_view(cleansed_dir=cleansed_dir)

    if not source_dir.exists():
        raise FileNotFoundError(
            f"No cleansed parquet files found in {source_dir} for system '{system}'. Run cleanse before validation."
        )

    files = sorted(
        path
        for path in source_dir.glob("*.parquet")
        if path.is_file() and (input_file is not None or path.name != "sample.parquet")
    )
    frames = _read_valid_cleansed_frames(files)

    if not frames:
        raise FileNotFoundError(
            f"No cleansed parquet files found in {source_dir} for system '{system}'. Run cleanse before validation."
        )

    snapshot = (
        pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]
    )

    missing_required = [
        column_name
        for column_name in CLEANSED_MERGE_REQUIRED_COLUMNS
        if column_name not in snapshot.columns
    ]
    if missing_required:
        missing_display = ", ".join(missing_required)
        raise ValueError(
            f"Cleansed frame for system '{system}' is missing required column(s): {missing_display}."
        )

    if name_col not in snapshot.columns:
        raise ValueError(
            f"Cleansed frame for system '{system}' is missing selected name column '{name_col}'."
        )

    snapshot = snapshot.with_columns(
        pl.col("system_uri").cast(pl.Utf8).alias("system_uri"),
        pl.col("name").cast(pl.Utf8, strict=False).fill_null("").alias("name"),
        pl.col("jurisdiction_code")
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .fill_null(NULL_PARTITION_SENTINEL)
        .alias("country"),
    ).filter(pl.col("system_uri").is_not_null() & (pl.col("system_uri") != ""))

    total_rows_by_country = _country_counts(snapshot)

    if input_file is not None:
        # Single-file/sample-scoped run: preserve the requested filename
        # exactly (e.g. "sample.parquet") rather than the general
        # chunk_size/partition-bounded numbering scheme, so downstream
        # consumers that ask for this exact file by name can still find it.
        snapshot.drop("country").write_parquet(
            cleansed_dir / input_file, compression="snappy"
        )
    elif partition_by:
        write_partitioned_output_files_from_frame(
            frame=snapshot,
            output_dir=primary_family_dir(cleansed_dir),
            partition_by="country",
            label=partition_by,
            chunk_size=rows_per_file,
            include_key=False,
        )
    else:
        write_chunked_output_files_from_frame(
            frame=snapshot.drop("country"),
            output_dir=primary_family_dir(cleansed_dir),
            prefix=system,
            chunk_size=rows_per_file,
        )

    _write_cleansed_metadata(
        cleansed_dir=cleansed_dir,
        system=system,
        rows_per_file=rows_per_file,
        rows_by_country=total_rows_by_country,
    )

    return cleansed_dir


@dataclass
class PipelineConfig:
    country: str
    is_azure_ml: bool
    input_dir: Path
    cleansed_dir: Path


def configure(
    country: str = "gb",
    roots: WorkspaceRoots | None = None,
) -> PipelineConfig:
    """
    Resolve all paths, create directories, and start an MLflow run.
    Detects Azure ML automatically via AZUREML_RUN_ID, where the roots are
    taken from BLOB_ROOT; locally the caller's resolved roots are required.
    Returns a PipelineConfig with all resolved paths.
    """
    is_azure_ml = "AZUREML_RUN_ID" in os.environ

    if is_azure_ml:
        print(f"Running on Azure ML  (system={country})")
        _roots = default_workspace_roots(Path(os.environ["BLOB_ROOT"]))
    else:
        print(f"Running locally  (system={country})")
        if roots is None:
            raise ValueError("Resolved workspace roots are required outside Azure ML.")
        _roots = roots

    cfg = PipelineConfig(
        country=country,
        is_azure_ml=is_azure_ml,
        input_dir=data_root(_roots) / country,
        cleansed_dir=system_layer_dir(_roots, country, layer=CLEANSED_LAYER_NAME),
    )

    cfg.cleansed_dir.mkdir(parents=True, exist_ok=True)

    active_run = mlflow.active_run()
    started_run = False
    if active_run is None:
        mlflow.start_run()
        started_run = True
    else:
        print(f"Reusing active MLflow run: {active_run.info.run_id}")

    if started_run:
        mlflow.log_params(
            {
                "country": country,
                "environment": "azure_ml" if is_azure_ml else "local",
            }
        )
    else:
        print("Skipping MLflow param logging on reused run.")

    print(f"INPUT_DIR:      {cfg.input_dir}")
    print(f"CLEANSED_DIR:   {cfg.cleansed_dir}")
    print("Ready.")
    return cfg


def _resolve_name_cleanse_input_files(
    input_dir: Path, input_file: str | None
) -> list[str]:
    if input_file is not None:
        selected_path = input_dir / input_file
        if not selected_path.exists():
            raise FileNotFoundError(f"Input parquet file not found: {selected_path}")
        return [str(selected_path)]

    system_prefix = (
        input_dir.parent.parent.name if input_dir.parent.parent is not None else ""
    )
    prefixed_files = (
        sorted(
            path
            for path in glob.glob(
                str(input_dir / primary_data_file_glob(system_prefix))
            )
            if is_primary_data_file(
                file_name=Path(path).name, system_code=system_prefix
            )
        )
        if system_prefix
        else []
    )
    return prefixed_files or sorted(glob.glob(str(input_dir / "*.parquet")))


def _validate_cleansed_output_schema(written_outputs: list[str]) -> None:
    baseline_schema: tuple[tuple[str, str], ...] | None = None
    for path in written_outputs:
        schema = pl.read_parquet_schema(path)
        if "name_cleansed" not in schema:
            raise ValueError(
                f"Cleansed output schema violation: 'name_cleansed' missing in {path}"
            )

        schema_signature = tuple((name, str(dtype)) for name, dtype in schema.items())
        if baseline_schema is None:
            baseline_schema = schema_signature
        elif schema_signature != baseline_schema:
            raise ValueError(
                "Cleansed output schema mismatch detected across written chunks; "
                f"file with mismatch: {path}"
            )


def _audit_empty_cleansed_rows(
    written_outputs: list[str], empty_output_path: Path
) -> None:
    empty_rows = (
        pl.scan_parquet(written_outputs)
        .filter(
            pl.col("name_cleansed").is_null()
            | pl.col("name_cleansed")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .eq("")
        )
        .collect()
    )
    if empty_rows.height > 0:
        empty_rows.write_parquet(empty_output_path, compression="snappy")
        print(
            f"[warn] Wrote {empty_rows.height:,} row(s) with empty cleansed names to {empty_output_path}"
        )
    elif empty_output_path.exists():
        empty_output_path.unlink()
        print(f"[info] Removed stale empty cleanser output: {empty_output_path}")


def _cleanse_single_chunk(
    *,
    file_path: str,
    output_path: str,
    cleanse_config: CleanseConfig,
    shared_file_columns: list[str],
    query_plan_output_dir: Path | None,
    write_output: bool,
    written_outputs: list[str],
    post_cleanse_transform: Callable[[pl.LazyFrame], pl.LazyFrame] | None = None,
    reporter: RowProgress | None = None,
    source_rows: int | None = None,
    display_name: str | None = None,
) -> tuple[int, float]:
    t_chunk_start = time.perf_counter()
    base_name = display_name or os.path.basename(file_path)

    lf = pl.scan_parquet(file_path)
    lf_cleansed = cleanse_lazyframe(
        lf, cleanse_config, file_columns=shared_file_columns
    )
    if post_cleanse_transform is not None:
        lf_cleansed = post_cleanse_transform(lf_cleansed)

    if query_plan_output_dir is not None:
        query_plan_output_dir.mkdir(parents=True, exist_ok=True)
        plan_stem = Path(base_name).stem
        optimized_plan = lf_cleansed.explain(optimized=True)
        unoptimized_plan = lf_cleansed.explain(optimized=False)
        (query_plan_output_dir / f"{plan_stem}.optimized.plan.txt").write_text(
            optimized_plan, encoding="utf-8"
        )
        (query_plan_output_dir / f"{plan_stem}.unoptimized.plan.txt").write_text(
            unoptimized_plan, encoding="utf-8"
        )

    if write_output:
        chunk_rows = _write_cleanse_output(
            lf_cleansed, output_path, reporter=reporter, source_rows=source_rows
        )
        written_outputs.append(output_path)
    else:
        # Compute-only mode for benchmarking CPU/query execution without parquet write cost.
        chunk_rows = lf_cleansed.collect().height
        if reporter is not None:
            reporter.advance(chunk_rows)
    elapsed_s = max(time.perf_counter() - t_chunk_start, 1e-9)
    rows_per_sec = chunk_rows / elapsed_s
    mode_suffix = " [compute-only]" if not write_output else ""
    print(
        f"  {base_name}: {chunk_rows:,} rows in {elapsed_s:.2f}s "
        f"({rows_per_sec:,.0f} rows/s){mode_suffix}"
    )

    gc.collect()
    return chunk_rows, rows_per_sec


def name_cleanse(
    input_dir: Path,
    output_dir: Path,
    input_file: str | None = None,
    company_type_regex: str | None = None,
    company_col: str = "CompanyName",
    and_tokens: tuple[str, ...] | list[str] | None = None,
    personal_owner_markers: tuple[str, ...] | list[str] | None = None,
    char_whitelist: str = r"[^a-z0-9\s!&]",
    company_type_mapping: dict[str, str] | None = None,
    company_type_matcher: str = "trie",
    source_company_type_col: str | None = None,
    source_company_type_mapping: dict[str, str] | None = None,
    query_plan_output_dir: Path | None = None,
    step_engines: dict[str, str] | None = None,
    write_output: bool = True,
    chunk_name_prefix: str | None = None,
) -> int:
    """
    Deterministic name cleaning across selected parquet chunks.

    Args:
        input_dir: Directory with input parquet files
        output_dir: The system's cleansed directory (e.g. data/<code>/cleansed) --
            per-input-file chunk output is written to output_dir/chunks/, not
            output_dir itself; output_dir's own top level is reserved for the
            merged/deduped view materialize_cleansed_view writes separately.
        input_file: Optional single parquet file name to process from input_dir
        company_type_regex: Regex to extract company type suffix
        company_col: Name of company name column
        and_tokens: List of AND connector tokens to normalize to "&"
        personal_owner_markers: Optional owner markers used to split owner text into personal_owner
        char_whitelist: Regex of chars to keep (others become spaces)
        company_type_mapping: Dict mapping source company type values to canonical forms (e.g. {"LIMITED": "LTD", "SOCIEDAD ANONIMA": "SA"})
        company_type_matcher: Matching strategy for company types: "trie" (default) or "regex"
        source_company_type_col: Optional source column containing explicit company type/category values
        source_company_type_mapping: Optional mapping for source_company_type_col values
        query_plan_output_dir: Optional directory to dump optimized/unoptimized Polars query plans per input file
        step_engines: Optional per-step engine overrides for cleanse pipeline steps
        write_output: Whether to write cleansed parquet output. When False, executes collect-only for compute benchmarking.
        chunk_name_prefix: Prefix for per-chunk output filenames. Defaults to the name of
            output_dir's parent, which is the system code for an acquired layer; pass it
            explicitly when your layout puts something else there.

    Returns total number of rows processed.
    """
    cleanse_config = CleanseConfig(
        company_col=company_col,
        and_tokens=tuple(and_tokens) if and_tokens is not None else None,
        personal_owner_markers=(
            tuple(personal_owner_markers)
            if personal_owner_markers is not None
            else None
        ),
        char_whitelist=char_whitelist,
        company_type_regex=company_type_regex,
        company_type_mapping=company_type_mapping,
        company_type_matcher=company_type_matcher,
        source_company_type_col=source_company_type_col,
        source_company_type_mapping=source_company_type_mapping,
        step_engines=step_engines,
    )

    input_files = _resolve_name_cleanse_input_files(input_dir, input_file)

    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    # Canonical inputs are expected to be schema-homogeneous within a run snapshot.
    # Resolve columns once from the first file and reuse across all files.
    shared_file_columns = list(pl.read_parquet_schema(input_files[0]).keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # Defaults to the directory above `cleansed/`, which for an acquired system is
    # its system code. That is a positional assumption, and it is wrong for any
    # caller whose layout puts something else there -- `validation`'s perturbed
    # datasets sit at `<source>/<profile>/<version>/<seed>/cleansed/`, where it silently named
    # every chunk after the seed. Callers outside the acquired-layer shape pass their
    # own prefix rather than relying on where they happen to be.
    naming_prefix = chunk_name_prefix or layer_system(output_dir)
    empty_output_path = output_dir / "empty_cleanser.parquet"
    total_rows = 0
    chunk_throughputs: list[float] = []
    written_outputs: list[str] = []

    for file_path in input_files:
        chunk_rows, rows_per_sec = _cleanse_single_chunk(
            file_path=file_path,
            output_path=str(
                chunks_dir
                / collate_chunk_name(os.path.basename(file_path), naming_prefix)
            ),
            cleanse_config=cleanse_config,
            shared_file_columns=shared_file_columns,
            query_plan_output_dir=query_plan_output_dir,
            write_output=write_output,
            written_outputs=written_outputs,
        )
        chunk_throughputs.append(rows_per_sec)
        total_rows += chunk_rows

    if chunk_throughputs:
        print(
            "Pass 1 throughput summary — "
            f"min: {min(chunk_throughputs):,.0f} rows/s, "
            f"median: {statistics.median(chunk_throughputs):,.0f} rows/s, "
            f"max: {max(chunk_throughputs):,.0f} rows/s"
        )

    if written_outputs:
        _validate_cleansed_output_schema(written_outputs)
        _audit_empty_cleansed_rows(written_outputs, empty_output_path)

    if write_output:
        print(f"\nPass 1 complete — {len(input_files)} file(s) written")
    else:
        print(
            f"\nPass 1 complete — {len(input_files)} file(s) processed in compute-only mode"
        )
    return total_rows


def _cleanse_family(
    *,
    input_files: list[Path],
    input_layer_dir: Path,
    output_layer_dir: Path,
    cleanse_config: CleanseConfig,
    query_plan_output_dir: Path | None,
    write_output: bool,
    written_outputs: list[str],
    scoped_output_name: str | None = None,
    progress_label: str = "cleanse",
) -> tuple[int, list[float], dict[str, int]]:
    """Cleanse one family's files, mirroring each input's own relative path.

    The whole of this stage's shape handling. It reads no schema of its own,
    derives no partition column, and constructs no path beyond
    `mirror_relative_path` -- so `names/` needs no code here to work, only a
    different `input_files`/`input_layer_dir` pair, and a family that is or
    is not Hive-partitioned is the same to it either way.
    """
    total_rows = 0
    throughputs: list[float] = []
    rows_by_partition: dict[str, int] = {}
    reporter = RowProgress(label=progress_label, progress=print)

    for index, input_file in enumerate(input_files, start=1):
        relative_path = (
            Path(scoped_output_name)
            if scoped_output_name is not None
            else mirror_relative_path(input_file, input_layer_dir=input_layer_dir)
        )
        # Announced before the work, not only after it: a partition file can
        # hold 1,000,000 rows and take over a minute, so a run that only
        # spoke on completion looked stalled for the whole of it. Only for
        # files large enough to wait on, though -- GLEIF has 308 partitions
        # and most hold a few dozen rows, where an announcement is noise
        # rather than reassurance.
        source_rows = int(pl.scan_parquet(input_file).select(pl.len()).collect().item())
        if source_rows >= PROGRESS_ROW_INTERVAL:
            print(
                f"  [{index}/{len(input_files)}] cleansing "
                f"{relative_path.as_posix()} ({source_rows:,} rows) ..."
            )
        output_path = output_layer_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows, rows_per_sec = _cleanse_single_chunk(
            file_path=str(input_file),
            output_path=str(output_path),
            cleanse_config=cleanse_config,
            # The schema of the file being cleansed, not one re-derived once
            # from a sibling: a mapping stage has no business deciding that
            # its inputs agree with each other.
            shared_file_columns=list(pl.read_parquet_schema(input_file).keys()),
            query_plan_output_dir=query_plan_output_dir,
            write_output=write_output,
            written_outputs=written_outputs,
            reporter=reporter,
            source_rows=source_rows,
            display_name=relative_path.as_posix(),
        )
        throughputs.append(rows_per_sec)
        total_rows += rows

        partition_value = partition_value_of(relative_path)
        if partition_value is not None:
            rows_by_partition[partition_value] = (
                rows_by_partition.get(partition_value, 0) + rows
            )

    reporter.finish()
    return total_rows, throughputs, rows_by_partition


def cleanse_canonical_view(
    input_dir: Path,
    output_dir: Path,
    system: str,
    *,
    input_file: str | None = None,
    partition_by: str | None = PARTITION_COLUMN,
    query_plan_output_dir: Path | None = None,
    write_output: bool = True,
    **cleanse_config_kwargs: object,
) -> int:
    """Cleanse a canonical snapshot into the cleansed layer, one output file
    per input file.

    **Mirrors, never decides.** Canonical writes a deduped view, split into
    families and partitioned or not; this reproduces each input's own
    relative path under `output_dir` and cleanses its rows. So
    `primary/jurisdiction_code=gb/part-00001.parquet` becomes the same
    relative path under `cleansed/`, and `cleansed/` has a `primary/`
    because `canonical/` does -- not because this stage chose to. It does no
    merging, dedupe, partition-column derivation, or schema normalisation:
    every one of those was a place a downstream stage could disagree with
    the layer it reads about what that layer's shape is.

    Adding a second family is therefore a different `input_files` argument
    and no new code -- see `_cleanse_family`. Only `primary` is wired today,
    since nothing consumes a cleansed `names/` yet.

    `partition_by` is the system's own declared partition column (None for a
    resource that opts out). Because this mirrors, a system that should
    partition but whose canonical snapshot is flat -- written before
    partitioning moved into Canonical -- would quietly produce a *flat*
    cleansed view where a partitioned one existed. That case raises instead.

    **Writes are all-or-nothing.** A full run builds the new layer beside
    the live one and swaps it in at the end, so an interrupted run leaves
    the previous view intact rather than a half-written mixture of both --
    the durability the old `chunks/`-then-merge flow had by accident, and
    which writing partition files in place for seven minutes did not.

    Returns the total number of rows written.
    """
    cleansed_dir = Path(output_dir)
    canonical_dir = Path(input_dir)

    if input_file is not None:
        selected = canonical_dir / input_file
        if not selected.exists():
            raise FileNotFoundError(f"Input parquet file not found: {selected}")
        primary_files = [selected]
    else:
        if partition_by and not is_partitioned_layer(canonical_dir):
            raise ValueError(
                f"Canonical output for system '{system}' at {canonical_dir} is flat, "
                f"but the system partitions by '{partition_by}'. It was written before "
                "the Canonical stage partitioned its output; cleansing it as-is "
                "would replace the partitioned cleansed view with a flat one. Re-run "
                "'--processes canonical --force' for this system first."
            )
        primary_files = resolve_primary_files(canonical_dir, system_code=system)
    if not primary_files:
        raise FileNotFoundError(
            f"No canonical entity parquet files found in {canonical_dir} for system "
            f"'{system}'. Run canonicalization before cleansing."
        )

    cleanse_config = CleanseConfig(**cleanse_config_kwargs)  # type: ignore[arg-type]
    written_outputs: list[str] = []

    if input_file is not None:
        # Scoped run: it owns one named file, so it writes in place and must
        # not disturb the rest of a view it is not rebuilding.
        cleansed_dir.mkdir(parents=True, exist_ok=True)
        stale_scoped_output = cleansed_dir / input_file
        if stale_scoped_output.exists():
            stale_scoped_output.unlink()
        staging_dir = cleansed_dir
    else:
        staging_dir = fresh_layer_staging_dir(cleansed_dir)

    try:
        total_rows, throughputs, rows_by_partition = _cleanse_family(
            input_files=primary_files,
            input_layer_dir=canonical_dir,
            output_layer_dir=staging_dir,
            cleanse_config=cleanse_config,
            query_plan_output_dir=query_plan_output_dir,
            write_output=write_output,
            written_outputs=written_outputs,
            scoped_output_name=input_file,
            progress_label=f"{system} cleanse",
        )

        if throughputs:
            print(
                "Cleanse throughput summary — "
                f"min: {min(throughputs):,.0f} rows/s, "
                f"median: {statistics.median(throughputs):,.0f} rows/s, "
                f"max: {max(throughputs):,.0f} rows/s"
            )

        if written_outputs:
            _validate_cleansed_output_schema(written_outputs)
            _audit_empty_cleansed_rows(
                written_outputs, staging_dir / "empty_cleanser.parquet"
            )

        if write_output and input_file is None:
            _write_cleansed_metadata(
                cleansed_dir=staging_dir,
                system=system,
                rows_per_file=PARTITION_ROWS_PER_FILE,
                rows_by_country=rows_by_partition,
            )
            swap_layer_into_place(staging=staging_dir, live=cleansed_dir)
    except LayerSwapError:
        # The layer itself is finished and on disk; only moving it into
        # place failed. Discarding it here would throw away the whole run
        # over a transient reader.
        raise
    except BaseException:
        if staging_dir != cleansed_dir:
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"\nCleanse complete — {len(primary_files)} file(s) written")
    return total_rows
