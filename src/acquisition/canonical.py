"""The Canonical stage: map each system's shards to the OpenCorporates-style schema, plus a name-variant sidecar.

The entity pass and the name-row chain run concurrently, and the name rows are finalised once both finish. An identity field that cannot be derived from the source, such as a jurisdiction or company number, is left null rather than filled with a speculative default.

The name-variant sidecar (`<system>-names-*.parquet`) holds every other name a source records: GLEIF's previous names and predecessor names resolved through successor entities, Wikidata's labels, aliases, official and short names with each value's `language_code`, previous names for GB and OffeneRegister, and FR's SIRENE acronym, usual, usage and pseudonym names. IE's bulk source carries one current name per company, so it has none. Each row's `system_uri` is composed by `workspace.derived_uri.name_variant_uri` from the entity, the name's type and its value, so it is stable across reruns; rows sharing that triple across shards are collapsed before `name_variant_uri`'s uniqueness check, which can then fire only on a real hash collision.

`canonicalize_system_primary_name_override` promotes GLEIF's curated ASCII transliteration into canonical `name` for a non-Latin-script entity that has one. Few do, so canonical `name` is not always Latin-script.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

from workspace.data_file_naming import (
    COMPANION_DUPLICATES,
    COMPANION_NAMES,
    COMPANION_SIDECAR,
    COMPANION_SUCCESSORS,
    companion_data_file_glob,
    is_primary_data_file,
    primary_data_file_glob,
)
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    canonical_snapshot_dir,
    system_layer_dir,
)
from workspace.derived_uri import name_variant_uri
from workspace.layer_layout import (
    PARTITION_COLUMN,
    PARTITION_ROWS_PER_FILE,
    SAMPLE_FILE_NAME,
    STAGING_DIR_NAME,
    fresh_layer_staging_dir,
    names_family_dir,
    primary_family_dir,
    resolve_name_files,
    resolve_primary_files,
    swap_layer_into_place,
)
from workspace.roots import WorkspaceRoots

from .canonical_dedupe import (
    DEDUPE_REPORT_FILE_NAME,
    write_dedupe_report,
    write_deduped_output,
)
from .canonical_frame_transform import canonicalize_frame
from .canonical_sampling import compute_z_sample_size, write_canonical_sample_file
from .canonical_utils import TRANSLITERATION_NAME_TYPES, dedupe_names, has_latin_char
from .company_type_excludes import get_company_type_exclusions
from .company_type_registry import get_system_company_type_mapping
from .constants_status import require_runnable_plan
from .fr_name_rows import derive_fr_name_rows
from .gb_name_rows import derive_gb_name_rows
from .name_variant_uri import (
    check_name_variant_uniqueness,
)
from .offeneregister_name_rows import derive_offeneregister_name_rows
from .output_chunking import write_chunked_output_files_from_frame
from .plan_registry import get_system_plan
from .row_progress import RowProgress
from .sharding import _default_system_uri_materializer
from .wikidata_name_rows import derive_wikidata_name_rows

SAMPLE_CONFIDENCE_Z = 2.576
SAMPLE_MARGIN_OF_ERROR = 0.01
SAMPLE_PROPORTION = 0.5

PROJECT_IDENTITY_COLUMNS = [
    "system_uri",
]

OPENCORPORATES_COLUMNS = [
    "jurisdiction_code",
    "company_number",
    "name",
    "alternative_names",
    "previous_names",
    "company_type",
    "current_status",
    "incorporation_date",
    "dissolution_date",
    "inactive",
    "branch",
    "branch_status",
    "registered_address_in_full",
    "registry_url",
    "opencorporates_url",
    "lei",
    "vat",
    "tax_id",
    "match_uri",
]

METADATA_COLUMNS = [
    "metadata_generated_utc",
]

CANONICAL_OUTPUT_COLUMNS = (
    *PROJECT_IDENTITY_COLUMNS,
    *OPENCORPORATES_COLUMNS,
    *METADATA_COLUMNS,
)


def _resolve_system_field_candidates(plan) -> dict[str, list[str]]:
    raw_candidates = getattr(plan, "system_field_candidates", None)
    if not raw_candidates:
        raise RuntimeError(
            f"No system_field_candidates configured for system '{plan.code}'."
        )

    system_field_candidates: dict[str, list[str]] = {
        str(field_name): [candidate for candidate in candidates]
        for field_name, candidates in raw_candidates
    }

    return system_field_candidates


def _require_runnable_canonical_plan(plan, *, allow_research: bool) -> None:
    return require_runnable_plan(plan, allow_research=allow_research, stage="canonical")


def _resolve_latest_snapshot_date(
    root: Path,
    *,
    run_date: str | None,
    source_resource,
    has_match: Callable[[Path], bool],
) -> str:
    """Resolve the snapshot date to read from `root`.

    Delegates to `source_resource.resolve_snapshot_date()` when `run_date` is
    given. Otherwise scans `root` for the newest (by name, descending) child
    directory for which `has_match` returns True, so a partially-written or
    unrelated directory doesn't get picked as the latest snapshot.
    """
    if run_date is not None:
        return source_resource.resolve_snapshot_date(run_date)
    if not root.exists():
        return ""
    for candidate_dir in sorted(
        (path for path in root.glob("*") if path.is_dir()), reverse=True
    ):
        if has_match(candidate_dir):
            return candidate_dir.name
    return ""


def _canonicalize_frame(
    frame: pl.DataFrame,
    *,
    country: str,
    source_company_type_col: str | None,
    exclude_company_type_values: set[str],
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None = None,
    system_field_candidates: dict[str, list[str]] | None = None,
    canonical_derived_fields: tuple[tuple[str, tuple[str, ...], str], ...]
    | None = None,
) -> pl.DataFrame:
    return canonicalize_frame(
        frame,
        country=country,
        source_company_type_col=source_company_type_col,
        exclude_company_type_values=exclude_company_type_values,
        canonical_source_column_aliases=canonical_source_column_aliases,
        system_field_candidates=system_field_candidates or {},
        canonical_derived_fields=canonical_derived_fields,
        company_type_mapping_resolver=get_system_company_type_mapping,
        project_identity_columns=PROJECT_IDENTITY_COLUMNS,
        opencorporates_columns=OPENCORPORATES_COLUMNS,
        metadata_columns=METADATA_COLUMNS,
    )


def _compute_z_sample_size(
    *,
    population_size: int,
    z_score: float = SAMPLE_CONFIDENCE_Z,
    margin_of_error: float = SAMPLE_MARGIN_OF_ERROR,
    proportion: float = SAMPLE_PROPORTION,
) -> int:
    return compute_z_sample_size(
        population_size=population_size,
        z_score=z_score,
        margin_of_error=margin_of_error,
        proportion=proportion,
    )


def _write_canonical_sample_file(
    *, output_dir: Path, written_paths: list[Path], seed: int = 20260621
) -> None:
    write_canonical_sample_file(
        output_dir=output_dir,
        written_paths=written_paths,
        sample_file_name=SAMPLE_FILE_NAME,
        seed=seed,
        z_score=SAMPLE_CONFIDENCE_Z,
        margin_of_error=SAMPLE_MARGIN_OF_ERROR,
        proportion=SAMPLE_PROPORTION,
    )


def canonicalize_system_shards(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    chunk_size: int | None = None,
    partition_rows_per_file: int | None = None,
    allow_research: bool = False,
    progress: Callable[[str], None] | None = print,
) -> list[Path]:
    """Write a system's deduped canonical view from its Shard-stage files.

    `chunk_size` sizes the flat fallback layout (and is what
    `SourceResource.resolve_output_chunk_size` overrides);
    `partition_rows_per_file` sizes the partitioned one, defaulting to
    `PARTITION_ROWS_PER_FILE`. They are separate because the two
    layouts want different file sizes and only the flat one has ever been
    driven from the CLI's `--chunk-size`.
    """
    plan = get_system_plan(system)
    effective_chunk_size = 100_000 if chunk_size is None else chunk_size

    if effective_chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    SYSTEM_FIELD_CANDIDATES = _resolve_system_field_candidates(plan)

    source_resource = plan.resources[0]
    resolve_chunk_size = getattr(source_resource, "resolve_output_chunk_size", None)
    if callable(resolve_chunk_size):
        effective_chunk_size = resolve_chunk_size(effective_chunk_size)
    exclude_company_type_values = get_company_type_exclusions(plan.code)

    source_root = system_layer_dir(roots, plan.code, layer="source")
    snapshot_date = _resolve_latest_snapshot_date(
        source_root,
        run_date=run_date,
        source_resource=source_resource,
        has_match=lambda candidate_dir: any(
            is_primary_data_file(file_name=path.name, system_code=plan.code)
            for path in candidate_dir.glob("*.parquet")
        ),
    )

    output_dir = canonical_snapshot_dir(roots, system=plan.code, run_date=snapshot_date)
    primary_dir = primary_family_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # `primary/` holds nothing but this pass's own output, so it can be
    # cleared wholesale -- the family split into `primary/` and `names/` is
    # what makes that safe. In one shared directory this cleanup would have to
    # filter by companion suffix to avoid unlinking a names file the
    # concurrent name-row pass had just written, and the two threads could
    # race to unlink the same stale file.
    #
    # Everything else cleared here sits at the snapshot's top level and is
    # still this pass's own: the dropped-rows capture, the dedupe report, the
    # scoped sample, and the staging directory. Entity files the older flat
    # layout wrote at the top level are swept too, so a snapshot regenerated
    # into the family layout does not leave its old one beside it to be
    # picked up by the fallback in `resolve_primary_files`.
    primary_staging_dir = fresh_layer_staging_dir(primary_dir)
    for stale_file in output_dir.glob(primary_data_file_glob(plan.code)):
        if is_primary_data_file(
            file_name=stale_file.name, system_code=plan.code
        ) or stale_file.name.startswith(f"{plan.code}-{COMPANION_DUPLICATES}-"):
            stale_file.unlink()
    for stale_partition_dir in output_dir.glob("*=*"):
        if stale_partition_dir.is_dir():
            shutil.rmtree(stale_partition_dir)
    sample_file = output_dir / SAMPLE_FILE_NAME
    if sample_file.exists():
        sample_file.unlink()
    stale_report = output_dir / DEDUPE_REPORT_FILE_NAME
    if stale_report.exists():
        stale_report.unlink()
    chunks_dir = output_dir / STAGING_DIR_NAME
    shutil.rmtree(chunks_dir, ignore_errors=True)

    shard_dir = source_root / snapshot_date
    shard_files = sorted(
        path
        for path in shard_dir.glob("*.parquet")
        if is_primary_data_file(file_name=path.name, system_code=plan.code)
    )
    if not shard_files:
        raise FileNotFoundError(
            f"No sharded files found in {shard_dir} matching {plan.code}-*.parquet. Run sharding first."
        )

    def _transform_shard(shard_file: Path) -> pl.DataFrame:
        frame = pl.read_parquet(shard_file)
        return _canonicalize_frame(
            frame,
            country=plan.code,
            source_company_type_col=plan.company_type_column,
            exclude_company_type_values=exclude_company_type_values,
            canonical_source_column_aliases=plan.canonical_source_column_aliases,
            system_field_candidates=SYSTEM_FIELD_CANDIDATES,
            canonical_derived_fields=plan.canonical_derived_fields,
        )

    chunks_dir.mkdir(parents=True, exist_ok=True)
    staged_paths: list[Path] = []
    transform_progress = RowProgress(
        label=f"{plan.code} canonical transform", progress=progress
    )
    for index, shard_file in enumerate(shard_files, start=1):
        frame = _transform_shard(shard_file)
        if frame.height == 0:
            continue
        staged_path = chunks_dir / f"{plan.code}-{index:03d}.parquet"
        frame.write_parquet(staged_path, compression="snappy")
        staged_paths.append(staged_path)
        transform_progress.advance(frame.height)
    transform_progress.finish()

    resolve_partition_by = getattr(source_resource, "resolve_output_partition_by", None)
    partition_by = (
        resolve_partition_by(default_partition_by=PARTITION_COLUMN)
        if callable(resolve_partition_by)
        else PARTITION_COLUMN
    )

    written_paths, findings = write_deduped_output(
        chunk_sources=staged_paths,
        output_dir=primary_staging_dir,
        dropped_rows_dir=output_dir,
        chunk_size=(
            (
                PARTITION_ROWS_PER_FILE
                if partition_rows_per_file is None
                else partition_rows_per_file
            )
            if partition_by
            else effective_chunk_size
        ),
        frame_transformer=pl.read_parquet,
        partition_by=partition_by,
        flat_prefix=plan.code,
        dropped_rows_prefix=f"{plan.code}-{COMPANION_DUPLICATES}",
        progress=progress,
        progress_label=f"{plan.code} canonical write",
    )
    write_dedupe_report(
        output_dir=output_dir,
        system=plan.code,
        snapshot_date=snapshot_date,
        findings=findings,
    )

    # The entity view is swapped in only now that it is complete, so an
    # interrupted run leaves the previous one readable rather than a
    # half-written mixture of both (a lesson learned more than once).
    swap_layer_into_place(staging=primary_staging_dir, live=primary_dir)
    written_paths = [
        primary_dir / path.relative_to(primary_staging_dir) for path in written_paths
    ]

    # Staging goes either way: the dropped rows themselves are now captured
    # in the `{code}-duplicates-*.parquet` companion, and the row each
    # collision resolved to is in the canonical view, so retaining `chunks/`
    # would only duplicate on disk what is already inspectable.
    shutil.rmtree(chunks_dir, ignore_errors=True)

    if findings.has_findings:
        print(
            f"[warn] [{plan.code}] Canonical dedupe dropped "
            f"{findings.duplicate_rows_dropped:,} row(s) across "
            f"{findings.duplicate_group_count:,} duplicate system_uri group(s) and "
            f"{findings.dropped_missing_identity_rows:,} row(s) with no system_uri; "
            f"see {output_dir / DEDUPE_REPORT_FILE_NAME} and "
            f"{plan.code}-{COMPANION_DUPLICATES}-*.parquet for the dropped rows."
        )

    _write_canonical_sample_file(output_dir=output_dir, written_paths=written_paths)

    return written_paths


def _name_identifier_as_id(
    frame: pl.DataFrame, *, identifier_candidates: Sequence[str]
) -> pl.DataFrame:
    """`frame` with the entity identifier its Shard stage named natively
    renamed to `id`, the column every other system's name rows already carry.

    The Shard stage keeps each system's own column names, so GLEIF's name
    rows arrive carrying `LEI` where the rest carry `id`. Canonical is where
    one shape is promised, so the rename happens here rather than in the
    system's own shard writer, which is right to spell it as GLEIF does.
    A frame already carrying `id` is left alone.
    """
    if "id" in frame.columns:
        return frame
    for candidate in identifier_candidates:
        if candidate in frame.columns:
            return frame.rename({candidate: "id"})
    return frame


def canonicalize_system_name_rows(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    allow_research: bool = False,
) -> list[Path]:
    """Map Shard-stage native name-form rows to the shared name_type vocabulary.

    A no-op for any system without a `name_variant_type_map` configured (`fr`,
    `gb`, `gleif`, `offeneregister` and `wikidata` today) or without any
    Shard-stage names files on disk, so it is safe to call unconditionally
    without special-casing any one system by name.
    """
    plan = get_system_plan(system)

    raw_type_map = getattr(plan, "name_variant_type_map", None)
    if not raw_type_map:
        return []

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    type_map = {
        str(source_type): str(name_type) for source_type, name_type in raw_type_map
    }

    source_resource = plan.resources[0]
    source_root = system_layer_dir(roots, plan.code, layer="source")
    snapshot_date = _resolve_latest_snapshot_date(
        source_root,
        run_date=run_date,
        source_resource=source_resource,
        # A Shard-layer probe, so the plain companion glob rather than
        # `resolve_name_files`: `source/` keeps both families flat
        # side by side and the family split applies only to `canonical/`.
        has_match=lambda candidate_dir: any(
            candidate_dir.glob(
                companion_data_file_glob(system_code=plan.code, family=COMPANION_NAMES)
            )
        ),
    )

    shard_dir = source_root / snapshot_date
    name_shard_files = sorted(
        shard_dir.glob(
            companion_data_file_glob(system_code=plan.code, family=COMPANION_NAMES)
        )
    )
    if not name_shard_files:
        return []

    snapshot_dir = canonical_snapshot_dir(
        roots, system=plan.code, run_date=snapshot_date
    )
    output_dir = names_family_dir(snapshot_dir)
    # `names/` holds nothing but this pass's output, so a wholesale clear is
    # both safe and enough -- including of the older flat layout's copies at
    # the snapshot's own top level, which `resolve_name_files` would
    # otherwise keep falling back to.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in snapshot_dir.glob(
        companion_data_file_glob(system_code=plan.code, family=COMPANION_NAMES)
    ):
        stale_file.unlink()

    written_paths: list[Path] = []
    for shard_file in name_shard_files:
        frame = _name_identifier_as_id(
            pl.read_parquet(shard_file),
            identifier_candidates=plan.system_uri_identifier_candidates or (),
        )
        try:
            frame = frame.with_columns(
                pl.col("source_type")
                .replace_strict(type_map, return_dtype=pl.Utf8)
                .alias("name_type")
            )
        except pl.exceptions.InvalidOperationError as error:
            raise RuntimeError(
                f"{plan.code}: {shard_file.name} contains a source_type value not present in "
                f"name_variant_type_map ({error}). Add it to the catalog's "
                "name_variant_type_map before re-running, rather than silently dropping or "
                "mis-tagging a name form."
            ) from error
        output_path = output_dir / shard_file.name
        frame.write_parquet(output_path, compression="snappy")
        written_paths.append(output_path)

    return written_paths


_NAME_ROW_DERIVERS: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "wikidata": derive_wikidata_name_rows,
    "gb": derive_gb_name_rows,
    "offeneregister": derive_offeneregister_name_rows,
    "fr": derive_fr_name_rows,
}
"""Systems whose Shard-stage name-variant rows are derived *from
already-written shard output* rather than emitted by the Shard stage itself.

GLEIF writes `gleif-names-*.parquet` during Shard, straight off the parsed
XML tree, because its per-name `@type` and `xml:lang` attributes only exist
there. Wikidata's naming fields all survive into the main shard parquet as
ordinary columns, so re-reading that parquet is strictly cheaper
and leaves the performance-sensitive Wikidata shard path untouched. GB's
`PreviousName_N.CompanyName` columns survive into its *sidecar* parquet
instead (see `_NAME_ROW_DERIVER_READS_SIDECAR` below) -- they aren't in
`_extend_gb_preferred_columns`'s preferred-main-column list, so the ordinary
main/sidecar split routes them there. OffeneRegister's `previous_names_list`
column (see `sharding_offeneregister.py`) lives only in its sidecar for the
same reason -- it isn't in `system_field_candidates`' preferred-main-column
list either. FR's `sigleUniteLegale`/`denominationUsuelleNUniteLegale`/
`nomUsageUniteLegale`/`pseudonymeUniteLegale` columns live only in its
sidecar too, for the same reason.
"""

_NAME_ROW_DERIVER_READS_SIDECAR: frozenset[str] = frozenset(
    {"gb", "offeneregister", "fr"}
)
"""Systems in `_NAME_ROW_DERIVERS` whose deriver reads the Shard-stage
*sidecar* file (`{code}-sidecar-NNN.parquet`) instead of the main per-entity
shard file. GB's previous-name columns, OffeneRegister's
`previous_names_list` column, and FR's `sigleUniteLegale`/
`denominationUsuelleNUniteLegale`/`nomUsageUniteLegale`/`pseudonymeUniteLegale`
columns all live only in the sidecar (see above); every other registered
deriver reads the main shard, matching GLEIF's native `{code}-names-*.parquet`
writer, which is also main-shard-derived data.
"""


def derive_system_name_rows(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    allow_research: bool = False,
) -> list[Path]:
    """Derive Shard-shaped name-variant rows from a system's shard files.

    Writes `data/<code>/source/<snapshot>/<code>-names-NNN.parquet`, one per
    input shard file, index-aligned -- the same location, naming and column
    shape GLEIF's Shard stage produces -- so `canonicalize_system_name_rows`
    picks them up with no changes. Reads the main per-entity shard file for
    every registered system except those listed in
    `_NAME_ROW_DERIVER_READS_SIDECAR` (GB today), which read the sidecar
    file instead since that's where their naming columns actually live.

    A no-op for any system with no entry in `_NAME_ROW_DERIVERS`, so it is
    safe to call unconditionally without special-casing Wikidata by name.

    Always fully recomputes (deleting any existing `<code>-names-*.parquet`
    first): a bare `--processes shard` re-run does not refresh this file (it
    reads shard output, it isn't produced by Shard), so this keeps the names
    file from ever going stale relative to the current shard content instead
    of silently drifting between shard and canonical runs.
    """
    plan = get_system_plan(system)

    deriver = _NAME_ROW_DERIVERS.get(plan.code)
    if deriver is None:
        return []

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    source_resource = plan.resources[0]
    source_root = system_layer_dir(roots, plan.code, layer="source")
    snapshot_date = _resolve_latest_snapshot_date(
        source_root,
        run_date=run_date,
        source_resource=source_resource,
        has_match=lambda candidate_dir: any(
            is_primary_data_file(file_name=path.name, system_code=plan.code)
            for path in candidate_dir.glob(primary_data_file_glob(plan.code))
        ),
    )

    reads_sidecar = plan.code in _NAME_ROW_DERIVER_READS_SIDECAR
    shard_dir = source_root / snapshot_date
    if reads_sidecar:
        shard_files = sorted(
            shard_dir.glob(
                companion_data_file_glob(
                    system_code=plan.code, family=COMPANION_SIDECAR
                )
            )
        )
        source_name_prefix = f"{plan.code}-sidecar-"
    else:
        shard_files = sorted(
            path
            for path in shard_dir.glob(primary_data_file_glob(plan.code))
            if is_primary_data_file(file_name=path.name, system_code=plan.code)
        )
        source_name_prefix = f"{plan.code}-"
    if not shard_files:
        return []

    for stale_file in shard_dir.glob(
        companion_data_file_glob(system_code=plan.code, family=COMPANION_NAMES)
    ):
        stale_file.unlink()

    written_paths: list[Path] = []
    for shard_file in shard_files:
        rows = deriver(pl.read_parquet(shard_file))
        output_path = shard_dir / shard_file.name.replace(
            source_name_prefix, f"{plan.code}-names-", 1
        )
        rows.write_parquet(output_path, compression="snappy")
        written_paths.append(output_path)

    return written_paths


def canonicalize_system_primary_name_override(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    allow_research: bool = False,
) -> list[Path]:
    """Promote a script-appropriate name-variant row into canonical `name`
    when the current primary name has no Latin-script character.

    A no-op for any system without `primary_name_override_type_priority`
    configured (every system except GLEIF today), so it is safe to call
    unconditionally without special-casing GLEIF by name. Overrides `name`
    in place on the canonical per-entity files already written by
    `canonicalize_system_shards`. The demoted native name is left exactly
    as-is in the canonical names file under its ordinary source-native
    `name_type` (e.g. "primary" for GLEIF's LegalName-sourced rows) --
    that tag describes where the row came from, not whatever canonical
    `name` currently equals, so nothing there needs to change.
    """
    plan = get_system_plan(system)

    raw_override_priority = getattr(plan, "primary_name_override_type_priority", None)
    if not raw_override_priority:
        return []

    # Only ever honor genuine transliterations here, regardless of what the
    # catalog lists -- a `previous`/`trading`/`alternative_language` row is a
    # different name, not a script rendering of the current one, and
    # promoting it on a Latin-script match alone would misrepresent the
    # entity. A misconfigured catalog value is simply not used, never a crash.
    override_priority = [
        name_type
        for name_type in raw_override_priority
        if name_type in TRANSLITERATION_NAME_TYPES
    ]
    if not override_priority:
        return []

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    source_resource = plan.resources[0]
    canonical_root = system_layer_dir(roots, plan.code, layer=CANONICAL_LAYER_NAME)
    snapshot_date = _resolve_latest_snapshot_date(
        canonical_root,
        run_date=run_date,
        source_resource=source_resource,
        has_match=lambda candidate_dir: bool(
            resolve_name_files(candidate_dir, system_code=plan.code)
        ),
    )

    snapshot_dir = canonical_root / snapshot_date
    entity_files = resolve_primary_files(snapshot_dir, system_code=plan.code)
    name_files = resolve_name_files(snapshot_dir, system_code=plan.code)
    if not entity_files or not name_files:
        return []

    rank_by_type = {name_type: rank for rank, name_type in enumerate(override_priority)}
    override_frame = (
        pl.scan_parquet(name_files)
        .filter(pl.col("name_type").is_in(list(override_priority)))
        .with_columns(
            pl.col("name_type")
            .replace_strict(rank_by_type, return_dtype=pl.Int64)
            .alias("_override_rank")
        )
        .sort(["system_uri", "_override_rank"])
        .group_by("system_uri", maintain_order=True)
        .agg(pl.col("name").first().alias("_override_name"))
        .collect()
    )

    written_paths: list[Path] = []
    for entity_file in entity_files:
        frame = pl.read_parquet(entity_file)
        if "name" not in frame.columns or "system_uri" not in frame.columns:
            continue

        joined = frame.join(override_frame, on="system_uri", how="left")
        candidates = joined.filter(
            pl.col("name").is_not_null() & pl.col("_override_name").is_not_null()
        )
        if candidates.height == 0:
            continue

        non_latin_uris = set(
            candidates.filter(
                ~pl.col("name").map_elements(has_latin_char, return_dtype=pl.Boolean)
            )["system_uri"].to_list()
        )
        if not non_latin_uris:
            continue

        updated = joined.with_columns(
            pl.when(
                pl.col("system_uri").is_in(non_latin_uris)
                & pl.col("_override_name").is_not_null()
            )
            .then(pl.col("_override_name"))
            .otherwise(pl.col("name"))
            .alias("name")
        ).drop("_override_name")
        updated.write_parquet(entity_file, compression="snappy")
        written_paths.append(entity_file)

    return written_paths


_SUCCESSOR_CHAIN_SOURCE_TYPE = "SUCCESSOR_CHAIN_PREDECESSOR"
_SUCCESSOR_CHAIN_HOP_CAP = 50


def _resolve_successor_chain_terminal(
    start: str, edge_map: dict[str, str], *, hop_cap: int = _SUCCESSOR_CHAIN_HOP_CAP
) -> str | None:
    """Walk an unambiguous (one-outgoing-edge-per-node) successor chain from
    `start` to its terminal (a node with no further outgoing edge).

    Returns `None` if the walk hits a cycle or exceeds `hop_cap` before
    reaching a terminal -- malformed data is skipped, never looped forever
    or attributed to a node that isn't actually the chain's end.
    """
    visited = {start}
    node = start
    for _ in range(hop_cap):
        nxt = edge_map.get(node)
        if nxt is None:
            return node
        if nxt in visited:
            return None
        visited.add(nxt)
        node = nxt
    return None


def canonicalize_system_successor_chain(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    allow_research: bool = False,
) -> list[Path]:
    """Fold a predecessor's canonical name onto its successor(s) via GLEIF's
    SuccessorLEI edges.

    Two folds, both applied: every predecessor's name is folded directly
    onto every LEI its own SuccessorLEI names, and -- for a predecessor that
    sits on an unambiguous (single-successor-per-node) chain -- also onto
    that chain's ultimate terminal entity, so multi-hop history (e.g. a
    three-way merger) lands on whichever entity is still active rather than
    being scattered across dead intermediate LEI records. A node that
    splits into more than one successor has no single "surviving entity" to
    walk onward to, so the chain walk stops there; its direct folds are
    still correct and unaffected.

    A no-op for any system without Shard-stage successor sidecar files on
    disk (only GLEIF produces these today), so it is safe to call
    unconditionally without special-casing GLEIF by name.

    Idempotent: recomputes its full contribution from source data every run
    and replaces (rather than accumulates onto) whatever it wrote last time,
    so re-running without an intervening `canonicalize_system_shards` does
    not duplicate `previous_names` entries or names-file rows.
    """
    plan = get_system_plan(system)

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    source_resource = plan.resources[0]
    canonical_root = system_layer_dir(roots, plan.code, layer=CANONICAL_LAYER_NAME)
    snapshot_date = _resolve_latest_snapshot_date(
        canonical_root,
        run_date=run_date,
        source_resource=source_resource,
        has_match=lambda candidate_dir: bool(
            resolve_name_files(candidate_dir, system_code=plan.code)
        ),
    )

    shard_dir = system_layer_dir(roots, plan.code, layer="source") / snapshot_date
    successor_shard_files = sorted(
        shard_dir.glob(
            companion_data_file_glob(system_code=plan.code, family=COMPANION_SUCCESSORS)
        )
    )
    if not successor_shard_files:
        return []

    snapshot_dir = canonical_root / snapshot_date
    entity_files = resolve_primary_files(snapshot_dir, system_code=plan.code)
    name_files = resolve_name_files(snapshot_dir, system_code=plan.code)
    if not entity_files or not name_files:
        return []

    # predecessor_name/LEI come from the Shard-stage extraction itself (the
    # predecessor's own Entity/LegalName, captured while the full record was
    # still in hand) rather than a canonical-stage lookup: canonical entity
    # output already drops any row whose own SuccessorLEI is populated (see
    # canonical_frame_transform.py), so a predecessor's canonical entity row
    # essentially never exists to join against -- only its successor's does.
    edges = (
        pl.scan_parquet(successor_shard_files)
        .select(["system_uri", "LEI", "predecessor_name", "successor_lei"])
        .rename({"LEI": "predecessor_lei"})
        .filter(pl.col("successor_lei").is_not_null())
        .collect()
    )
    if edges.height == 0:
        return []

    successor_uris = _default_system_uri_materializer(
        edges.select("successor_lei").unique().rename({"successor_lei": "LEI"}),
        plan.code,
        plan.system_uri_identifier_candidates,
    ).rename({"system_uri": "successor_system_uri", "LEI": "successor_lei"})

    edges = edges.join(successor_uris, on="successor_lei", how="left").filter(
        pl.col("successor_system_uri").is_not_null()
    )
    if edges.height == 0:
        return []

    # Direct-hop fold: every predecessor -> every successor LEI it names.
    direct_pairs = edges.select(
        pl.col("successor_system_uri").alias("target_system_uri"),
        pl.col("system_uri").alias("predecessor_system_uri"),
        pl.col("predecessor_name"),
        pl.col("predecessor_lei"),
    )

    # Multi-hop terminal fold: only walk chains through nodes with exactly
    # one outgoing successor edge -- see _resolve_successor_chain_terminal.
    successor_counts = edges.group_by("system_uri").agg(
        pl.col("successor_system_uri").n_unique().alias("_successor_count")
    )
    unambiguous_edges = (
        edges.join(successor_counts, on="system_uri")
        .filter(pl.col("_successor_count") == 1)
        .unique(subset=["system_uri"])
    )
    edge_map: dict[str, str] = dict(
        zip(
            unambiguous_edges["system_uri"].to_list(),
            unambiguous_edges["successor_system_uri"].to_list(),
        )
    )
    predecessor_name_by_uri: dict[str, str] = dict(
        zip(edges["system_uri"].to_list(), edges["predecessor_name"].to_list())
    )
    predecessor_lei_by_uri: dict[str, str] = dict(
        zip(edges["system_uri"].to_list(), edges["predecessor_lei"].to_list())
    )

    terminal_pair_rows: list[dict[str, str | None]] = []
    unresolved_chain_count = 0
    for start in edge_map:
        terminal = _resolve_successor_chain_terminal(start, edge_map)
        if terminal is None:
            unresolved_chain_count += 1
            continue
        terminal_pair_rows.append(
            {
                "target_system_uri": terminal,
                "predecessor_system_uri": start,
                "predecessor_name": predecessor_name_by_uri.get(start),
                "predecessor_lei": predecessor_lei_by_uri.get(start),
            }
        )
    if unresolved_chain_count:
        print(
            f"[warn] {plan.code}: skipped {unresolved_chain_count} successor "
            "chain(s) that formed a cycle or exceeded the hop cap instead of "
            "reaching a terminal entity."
        )

    pair_schema = {
        "target_system_uri": pl.Utf8,
        "predecessor_system_uri": pl.Utf8,
        "predecessor_name": pl.Utf8,
        "predecessor_lei": pl.Utf8,
    }
    terminal_pairs = (
        pl.DataFrame(terminal_pair_rows, schema=pair_schema)
        if terminal_pair_rows
        else pl.DataFrame(schema=pair_schema)
    )

    all_pairs = pl.concat([direct_pairs, terminal_pairs]).unique()
    if all_pairs.height == 0:
        return []

    relevant_uris = list(set(all_pairs["target_system_uri"].to_list()))
    name_lookup = (
        pl.scan_parquet(entity_files)
        .select(["system_uri", "lei", "name"])
        .filter(pl.col("system_uri").is_in(relevant_uris))
        .collect()
    )

    resolved = (
        all_pairs.join(
            name_lookup.rename(
                {
                    "system_uri": "target_system_uri",
                    "lei": "target_lei",
                    "name": "target_name",
                }
            ),
            on="target_system_uri",
            how="inner",
        )
        .filter(
            pl.col("predecessor_name").is_not_null()
            & pl.col("predecessor_lei").is_not_null()
            & pl.col("target_lei").is_not_null()
            & pl.col("target_name").is_not_null()
            & (pl.col("predecessor_name") != pl.col("target_name"))
        )
        .unique(subset=["target_system_uri", "predecessor_system_uri"])
    )
    if resolved.height == 0:
        return []

    # Exclude any (target, name) pair already present in the names-file
    # family under any name_type -- most commonly a native PREVIOUS_LEGAL_NAME
    # row that already recorded this exact predecessor name for this target,
    # so a chain-derived row would otherwise duplicate it.
    existing_names_for_targets = (
        pl.scan_parquet(name_files)
        .select(["system_uri", "name"])
        .filter(pl.col("system_uri").is_in(relevant_uris))
        .unique()
        .collect()
        .rename({"system_uri": "target_system_uri", "name": "predecessor_name"})
    )
    resolved = resolved.join(
        existing_names_for_targets,
        on=["target_system_uri", "predecessor_name"],
        how="anti",
    )
    if resolved.height == 0:
        return []

    written_paths: list[Path] = []

    new_names_by_target = resolved.group_by("target_system_uri").agg(
        pl.col("predecessor_name").alias("new_previous_names")
    )

    for entity_file in entity_files:
        frame = pl.read_parquet(entity_file)
        if "system_uri" not in frame.columns or "previous_names" not in frame.columns:
            continue

        joined = frame.join(
            new_names_by_target,
            left_on="system_uri",
            right_on="target_system_uri",
            how="left",
        )
        if joined.filter(pl.col("new_previous_names").is_not_null()).height == 0:
            continue

        updated = joined.with_columns(
            pl.struct(["previous_names", "new_previous_names"])
            .map_elements(
                lambda row: dedupe_names(
                    (row["previous_names"] or []) + (row["new_previous_names"] or [])
                ),
                return_dtype=pl.List(pl.Utf8),
            )
            .alias("previous_names")
        ).drop("new_previous_names")
        updated.write_parquet(entity_file, compression="snappy")
        written_paths.append(entity_file)

    names_target_file = name_files[0]
    names_frame = pl.read_parquet(names_target_file)
    if "source_type" in names_frame.columns:
        names_frame = names_frame.filter(
            pl.col("source_type") != _SUCCESSOR_CHAIN_SOURCE_TYPE
        )
    new_name_rows = resolved.select(
        pl.col("target_system_uri").alias("system_uri"),
        pl.col("target_lei").alias("id"),
        pl.col("predecessor_name").alias("name"),
        pl.lit(_SUCCESSOR_CHAIN_SOURCE_TYPE).alias("source_type"),
        pl.lit(None, dtype=pl.Utf8).alias("language_code"),
        pl.lit("previous").alias("name_type"),
        (
            pl.lit("successor chain (predecessor LEI=")
            + pl.col("predecessor_lei")
            + pl.lit(")")
        ).alias("derivation_note"),
    ).select(names_frame.columns)
    # "vertical_relaxed": a names file where an entire column (e.g.
    # language_code) happens to be all-null infers as Null dtype on disk,
    # which plain "vertical" concat rejects against new_name_rows' Utf8;
    # relaxed upcasts to the common supertype instead of erroring.
    combined_names = pl.concat([names_frame, new_name_rows], how="vertical_relaxed")
    combined_names.write_parquet(names_target_file, compression="snappy")
    written_paths.append(names_target_file)

    return written_paths


def finalize_canonical_name_rows(
    system: str,
    *,
    run_date: str | None = None,
    roots: WorkspaceRoots,
    allow_research: bool = False,
) -> list[Path]:
    """Consolidate a system's name-variant sidecar, give every row the
    `jurisdiction_code` of the entity it describes, and give every row its
    own unique identity.

    Four jobs, all of which need every other Canonical step to have finished:

    1. **Consolidate.** `canonicalize_system_name_rows` writes one file per
       Shard-stage names file, which for GB means 57 files averaging 0.36MB
       -- the same small-file problem the entity view's consolidation fixed,
       left standing in the other family. This rewrites them as
       `PARTITION_ROWS_PER_FILE`-bounded files, which every real
       system today fits inside one of.

    2. **Attach `jurisdiction_code`.** Name rows carry `system_uri` but no
       jurisdiction, so selecting one country's aliases meant reading the
       whole sidecar and joining to the entity view by hand. The column is
       joined in from the entity view here, giving row-group pushdown for
       `filter(jurisdiction_code == "gb")` without partitioning the family
       across the ~200 jurisdictions GLEIF spans. It also leaves an eventual
       `names/jurisdiction_code=` split purely additive, since the column it
       would partition on already exists.

    3. **Carry the source-level exclusion.** The join is inner, so a variant
       whose entity Canonical excluded is dropped rather than kept with a
       null jurisdiction. GLEIF excludes FUND-type entities at the source
       because nothing downstream can match them; retaining their variants
       would inflate every count taken from the sidecar (9.1% of it on the
       real 2026-06-21 snapshot) purely to discard them at a later stage.

    4. **Give each row its own `system_uri`.** Every alias/variant row of one
       entity previously shared that entity's own `system_uri` -- a
       many-to-one shape incompatible with anything downstream that treats
       `system_uri` as a unique row identity. The jurisdiction join above
       still uses the entity's own `system_uri` (it's the join key into the
       entity view), so identity composition runs after it: the entity's
       `system_uri` is renamed to `source_uri` (a straight rename, a
       back-reference to the row's primary entity, not a recompute), and a
       fresh `system_uri` is composed from it through `workspace`,
       `name://<system>/<id>/<hash>`, the hash over the name type and value. This is the layer's last write, so no later stage needs
       to supply either column itself, or rewrite output this stage already
       wrote. `match_uri` is deliberately not part of this: it does not
       exist until the Match stage runs, so it isn't written here --
       consumers that need it resolve it themselves by joining this row's
       `source_uri` against `data/<code>/matched/`'s own `system_uri`.
       Rows sharing a composed identity are collapsed to a single survivor
       first: two rows composing the same `system_uri` share the same
       `(source_uri, name_type, value)` triple and are the same fact, so
       keeping either is equally correct -- the same source capturing one
       entity's name twice must not hard-fail the uniqueness check below.
       Only after that collapse does `check_name_variant_uniqueness` run, so
       it can now only ever fire on a genuine hash collision between two
       *different* triples.

    Runs last because all four jobs depend on earlier steps: the jurisdiction
    comes from the entity view `canonicalize_system_shards` writes, and
    `canonicalize_system_successor_chain` *appends* rows to this sidecar, so
    consolidating before it would leave its rows unconsolidated and without
    a jurisdiction. A no-op for any system with no sidecar on disk.
    """
    plan = get_system_plan(system)

    source_resource = plan.resources[0]
    canonical_root = system_layer_dir(roots, plan.code, layer=CANONICAL_LAYER_NAME)
    snapshot_date = _resolve_latest_snapshot_date(
        canonical_root,
        run_date=run_date,
        source_resource=source_resource,
        has_match=lambda candidate_dir: bool(
            resolve_name_files(candidate_dir, system_code=plan.code)
        ),
    )

    snapshot_dir = canonical_root / snapshot_date
    name_files = resolve_name_files(snapshot_dir, system_code=plan.code)
    if not name_files:
        return []

    _require_runnable_canonical_plan(plan, allow_research=allow_research)

    entity_files = resolve_primary_files(snapshot_dir, system_code=plan.code)
    if not entity_files:
        raise FileNotFoundError(
            f"{plan.code}: found a name-variant sidecar under {snapshot_dir} but no "
            "canonical entity view to resolve each row's jurisdiction_code from. Run "
            "the canonical entity pass for this snapshot before finalizing its names."
        )

    names_frame = pl.read_parquet(name_files)
    jurisdiction_by_uri = (
        pl.scan_parquet(entity_files)
        .select([PARTITION_COLUMN, "system_uri"])
        .unique(subset=["system_uri"])
        .collect()
    )

    # Inner join: a source-level exclusion has to carry through to every
    # family, not just the entity view. GLEIF drops FUND-type entities at
    # the source because nothing downstream can ever match them, so keeping
    # 365,383 of their name variants (9.1% of the sidecar on the real
    # 2026-06-21 snapshot) would inflate every count built from it only for
    # a later stage to discard them again. The rows are dropped here, and
    # counted, rather than surviving with a null jurisdiction.
    before_rows = names_frame.height
    enriched = names_frame.drop(PARTITION_COLUMN, strict=False).join(
        jurisdiction_by_uri, on="system_uri", how="inner"
    )
    excluded_rows = before_rows - enriched.height

    # Give each row its own unique identity: `system_uri` (the
    # entity's own identity, and this join's key above) is renamed to
    # `source_uri`, a back-reference to the primary entity the row
    # describes, and a fresh per-row `system_uri` is composed over
    # `(source_uri, name_type, name)`. Must run after the jurisdiction join,
    # which still needs `system_uri` to be the entity's own identity.
    enriched = enriched.rename({"system_uri": "source_uri"}).with_columns(
        pl.struct(["source_uri", "name_type", "name"])
        .map_elements(
            lambda row: name_variant_uri(
                source_uri=row["source_uri"],
                name_type=row["name_type"],
                value=row["name"],
            ),
            return_dtype=pl.Utf8,
        )
        .alias("system_uri")
    )

    # Collapse rows sharing a composed identity across the whole sidecar
    # (the name-row dedupe, carried forward to this stage): lossless, since two
    # rows composing the same system_uri share the same
    # (source_uri, name_type, value) triple and are the same fact. Must
    # happen before the uniqueness check, which should only ever fire on a
    # genuine hash collision between two *different* triples.
    enriched = enriched.unique(subset=["system_uri"], keep="first")
    check_name_variant_uniqueness(enriched, system_code=plan.code)

    output_dir = names_family_dir(snapshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in name_files:
        stale_file.unlink()

    written_paths = write_chunked_output_files_from_frame(
        frame=enriched,
        output_dir=output_dir,
        prefix=f"{plan.code}-{COMPANION_NAMES}",
        chunk_size=PARTITION_ROWS_PER_FILE,
    )

    if excluded_rows:
        print(
            f"[info] [{plan.code}] dropped {excluded_rows:,} name-variant row(s) "
            f"({excluded_rows / before_rows:.1%}) whose entity is excluded from the "
            "canonical view, carrying the source-level exclusion through to the "
            "name family."
        )

    return written_paths
