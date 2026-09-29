"""Streaming identity dedupe for the Canonical stage's final write.

The Canonical stage normalizes each Shard-stage file onto one schema and
stages the result, then writes the deduped, `jurisdiction_code=`-partitioned
view downstream stages read. This module owns that final pass.

Streaming rather than full-materialize: concatenating every staged chunk and
calling `.unique(subset=["system_uri"]).partition_by()` -- the shape
`materialize_cleansed_merge` used at Cleanse -- costs a consistent
~1.5-1.7KB peak RSS per row (real `fr`, 12.93M rows: 19.4GB peak). Holding
only an identity set plus one input chunk, and flushing each partition's
buffer once it reaches `chunk_size` rows, cut that by 60-85% in the same
benchmark (`fr`: 3.3GB).

Collisions are reported, not presumed absent. Canonical does not guarantee
non-null/unique `system_uri` by construction: the real 2026-06-21 GLEIF
snapshot carries 321 genuine duplicate `system_uri` groups, so a source
system we don't control can and does deliver them. Unlike the perturbation
stage's `PerturbationCollisionError` -- a hard fail, because that collision
would mean a bug in our own code -- a collision here can be legitimate
source data, so it resolves deterministically (keep first) and surfaces in
a written report rather than blocking the pipeline.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .output_chunking import (
    NULL_PARTITION_SENTINEL,
    FlatChunkWriter,
    PartitionedChunkWriter,
)
from .row_progress import RowProgress

DEDUPE_REPORT_FILE_NAME = "_dedupe_report.json"

DROPPED_REASON_COLUMN = "dropped_reason"

DROPPED_DUPLICATE_IDENTITY = "duplicate_system_uri"
DROPPED_MISSING_IDENTITY = "missing_system_uri"

CAPTURE_ROLE_COLUMN = "capture_role"

CAPTURE_ROLE_DROPPED = "dropped"
CAPTURE_ROLE_KEPT = "kept_counterpart"
"""Roles a row plays in the collision capture.

A `dropped` row is one the dedupe removed. A `kept_counterpart` is the row
that survived the same collision, copied in so the capture answers "how did
these two source rows differ" on its own.

Without it the only thing to compare a dropped row against is the canonical
view, and by the time anyone reads the file that view has moved: the
Canonical stage's later steps rewrite `name` (the primary-name override) and
`previous_names` (the successor chain) on surviving rows only. On real GLEIF
data those two columns account for 319 and 51 of the apparent differences
across 321 collisions, none of which is a difference between the source rows
at all. The counterpart is captured before any of that runs, so both sides
of the comparison are the same vintage.
"""

_PARTITION_KEY_COLUMN = "_partition_key"

MAX_REPORTED_DUPLICATE_GROUPS = 10_000
"""Cap on how many colliding identity values the JSON report lists individually.

The group count, row count and every other total are always exact; only the
per-value listing is truncated. A run that finds millions of collisions has a
source-data emergency, not a reporting need, and an uncapped listing would
turn that into a multi-gigabyte JSON file written while the pipeline is
already in trouble. Every dropped row is captured in the parquet companion
regardless, so nothing is lost to the cap.
"""


def _tag_dropped(frame: pl.DataFrame, reason: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(CAPTURE_ROLE_DROPPED).alias(CAPTURE_ROLE_COLUMN),
        pl.lit(reason).alias(DROPPED_REASON_COLUMN),
    )


def _tag_kept_counterpart(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(CAPTURE_ROLE_KEPT).alias(CAPTURE_ROLE_COLUMN),
        pl.lit(None, dtype=pl.Utf8).alias(DROPPED_REASON_COLUMN),
    )


@dataclass(frozen=True)
class DedupeFindings:
    """What the dedupe pass saw. Written to disk every run, empty or not."""

    input_rows: int
    written_rows: int
    dropped_missing_identity_rows: int
    duplicate_group_count: int
    duplicate_rows_dropped: int
    duplicate_groups: tuple[tuple[str, int], ...]
    """(identity value, rows dropped for it), truncated to
    MAX_REPORTED_DUPLICATE_GROUPS entries; `duplicate_group_count` is exact."""
    dropped_rows_paths: tuple[Path, ...] = ()
    """The `{code}-duplicates-{NNN}.parquet` files holding every dropped row."""

    @property
    def has_findings(self) -> bool:
        return self.duplicate_group_count > 0 or self.dropped_missing_identity_rows > 0

    @property
    def duplicate_groups_truncated(self) -> bool:
        return len(self.duplicate_groups) < self.duplicate_group_count


def resolve_partition_key_expr(partition_by: str) -> pl.Expr:
    """The partition-directory naming key: `partition_by` lowercased, with
    nulls mapped to the Hive null sentinel.

    Matches what Cleanse's merge derived for the cleansed view, so Canonical
    and Cleanse name the same directories for the same rows. The source
    column itself is left untouched in the written rows -- real
    `jurisdiction_code` values are uppercase (`GB`, `AE-DU`), and only the
    directory name is normalized.
    """
    return (
        pl.col(partition_by)
        .cast(pl.Utf8, strict=False)
        .str.to_lowercase()
        .fill_null(NULL_PARTITION_SENTINEL)
        .alias(_PARTITION_KEY_COLUMN)
    )


def write_deduped_output(
    *,
    chunk_sources: list[Path],
    output_dir: Path,
    chunk_size: int | None,
    frame_transformer: Callable[[Path], pl.DataFrame],
    identity_column: str = "system_uri",
    partition_by: str | None = None,
    flat_prefix: str | None = None,
    dropped_rows_prefix: str,
    dropped_rows_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
    progress_label: str = "dedupe",
) -> tuple[list[Path], DedupeFindings]:
    """Stream `chunk_sources` through an identity dedupe into either a
    `{partition_by}={value}/part-{NNNNN}.parquet` view (when `partition_by`
    is given) or flat `{flat_prefix}-{NNN}.parquet` files.

    The first row seen for each identity value is kept and every later row
    for that value is dropped -- the same resolution Cleanse's merge applied,
    moved a stage earlier so Canonical's own output is the deduped one. A row
    whose `identity_column` is null or empty is dropped too: it has no
    identity, so it cannot be a record in a view keyed on one.

    Nothing is dropped silently. Every dropped row is written verbatim to
    `{dropped_rows_prefix}-{NNN}.parquet` under `dropped_rows_dir` (default:
    `output_dir`), so a collision is inspectable as real rows rather than as
    a count, and each collision's surviving row is copied in beside it. Two
    columns say which is which: `capture_role` (`dropped` or
    `kept_counterpart`) and `dropped_reason` (null for a counterpart). Both
    sides are captured here, before any later step enriches the view, so the
    file answers "how did these two source rows differ" without a join to
    something that has since changed.

    The Canonical stage points `dropped_rows_dir` at the snapshot root
    rather than at the entity view's own directory: the capture is
    entity-shaped, and leaving it inside `primary/` would put rows the
    dedupe deliberately dropped back inside the tree every consumer of the
    entity view scans.
    """
    if partition_by is None and not flat_prefix:
        raise ValueError(
            "write_deduped_output needs one of partition_by or flat_prefix"
        )

    writer: PartitionedChunkWriter | FlatChunkWriter
    if partition_by is not None:
        writer = PartitionedChunkWriter(
            output_dir=output_dir,
            partition_by=_PARTITION_KEY_COLUMN,
            chunk_size=chunk_size,
            label=partition_by,
            include_key=False,
        )
    else:
        assert flat_prefix is not None  # nosec B101 - narrowed by the guard above
        writer = FlatChunkWriter(
            output_dir=output_dir, prefix=flat_prefix, chunk_size=chunk_size
        )
    dropped_writer = FlatChunkWriter(
        output_dir=output_dir if dropped_rows_dir is None else dropped_rows_dir,
        prefix=dropped_rows_prefix,
        chunk_size=chunk_size,
    )

    seen: set[str] = set()
    duplicate_counts: dict[str, int] = {}
    input_rows = 0
    dropped_missing = 0
    reporter = RowProgress(label=progress_label, progress=progress)

    for chunk_source in chunk_sources:
        frame = frame_transformer(chunk_source)
        if frame.height == 0:
            continue
        input_rows += frame.height
        reporter.advance(frame.height)

        if identity_column not in frame.columns:
            raise ValueError(
                f"{chunk_source.name} has no '{identity_column}' column; the canonical "
                "dedupe pass cannot establish row identity without it."
            )
        if partition_by is not None and partition_by not in frame.columns:
            raise ValueError(
                f"{chunk_source.name} has no '{partition_by}' column; the canonical "
                "dedupe pass cannot partition by a column that is not present."
            )

        frame = frame.with_columns(
            pl.col(identity_column).cast(pl.Utf8, strict=False).alias(identity_column)
        )
        has_identity = pl.col(identity_column).is_not_null() & (
            pl.col(identity_column) != ""
        )
        identity_less = frame.filter(~has_identity)
        if identity_less.height > 0:
            dropped_missing += identity_less.height
            dropped_writer.add(_tag_dropped(identity_less, DROPPED_MISSING_IDENTITY))
        frame = frame.filter(has_identity)
        if frame.height == 0:
            continue

        # One pass over the column in Python rather than a polars join
        # against the accumulated set: `seen` grows to the whole dataset's
        # identity values, so rebuilding it as a Series per chunk to use
        # `is_in` would be O(n) work per chunk. This also folds the
        # within-chunk and cross-chunk duplicate cases into one rule.
        keep_flags: list[bool] = []
        for identity in frame.get_column(identity_column).to_list():
            if identity in seen:
                keep_flags.append(False)
                duplicate_counts[identity] = duplicate_counts.get(identity, 0) + 1
            else:
                seen.add(identity)
                keep_flags.append(True)

        if not all(keep_flags):
            keep_mask = pl.Series(keep_flags)
            dropped_writer.add(
                _tag_dropped(frame.filter(~keep_mask), DROPPED_DUPLICATE_IDENTITY)
            )
            frame = frame.filter(keep_mask)
            if frame.height == 0:
                continue

        if partition_by is not None:
            frame = frame.with_columns(resolve_partition_key_expr(partition_by))

        writer.add(frame)

    written_paths = writer.close()
    reporter.finish()

    # Copy each collision's surviving row into the capture, so the file
    # answers "how did these two source rows differ" on its own. Done here,
    # inside the write, for two reasons: `duplicate_counts` still holds every
    # colliding identity (the report's listing is capped, this is not), and
    # nothing has yet enriched the written view -- the Canonical stage's
    # later steps rewrite `name` and `previous_names` on surviving rows only,
    # which is precisely what makes a comparison against the finished view
    # misleading.
    if duplicate_counts and written_paths:
        counterparts = (
            pl.scan_parquet(written_paths)
            .filter(pl.col(identity_column).is_in(list(duplicate_counts)))
            .collect()
        )
        if counterparts.height:
            dropped_writer.add(_tag_kept_counterpart(counterparts))

    dropped_paths = dropped_writer.close()

    ranked_duplicates = sorted(
        duplicate_counts.items(), key=lambda item: (-item[1], item[0])
    )
    findings = DedupeFindings(
        input_rows=input_rows,
        # Exactly one row survives per identity value, so the identity set's
        # own size is the written row count.
        written_rows=len(seen),
        dropped_missing_identity_rows=dropped_missing,
        duplicate_group_count=len(duplicate_counts),
        duplicate_rows_dropped=sum(duplicate_counts.values()),
        duplicate_groups=tuple(ranked_duplicates[:MAX_REPORTED_DUPLICATE_GROUPS]),
        dropped_rows_paths=tuple(dropped_paths),
    )
    return written_paths, findings


def write_dedupe_report(
    *,
    output_dir: Path,
    system: str,
    snapshot_date: str,
    findings: DedupeFindings,
) -> Path:
    """Write `_dedupe_report.json` into `output_dir`, whether or not the run
    found anything. An always-present report is what makes "no collisions" an
    observed result rather than an assumption -- one real GLEIF data
    contradicts.
    """
    report_path = output_dir / DEDUPE_REPORT_FILE_NAME
    report_path.write_text(
        json.dumps(
            {
                "system": system,
                "snapshot_date": snapshot_date,
                "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "input_rows": findings.input_rows,
                "written_rows": findings.written_rows,
                "dropped_missing_system_uri_rows": (
                    findings.dropped_missing_identity_rows
                ),
                "duplicate_system_uri_groups": findings.duplicate_group_count,
                "duplicate_rows_dropped": findings.duplicate_rows_dropped,
                "duplicate_groups_truncated": findings.duplicate_groups_truncated,
                "dropped_rows_files": [
                    path.name for path in findings.dropped_rows_paths
                ],
                "duplicate_groups": [
                    {"system_uri": identity, "rows_dropped": rows_dropped}
                    for identity, rows_dropped in findings.duplicate_groups
                ],
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
