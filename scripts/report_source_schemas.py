"""Write the data dictionary of each system's source layer, read from the
files themselves.

The source layer is what a system publishes, before any mapping into the
canonical schema (`docs/canonical_schema.md`): every system's columns are its
own, and a sidecar or names file beside the entity file carries its own shape
again. One table per file family per system, each column with its type, how
much of the sample is null, how many distinct values that sample holds, an
example, and the canonical field the system's catalog entry maps it to where
it maps one.

The row counts are every file's, read from the parquet footers; the other
figures are read from one sample of `--sample-rows` rows, since a distinct
count over `fr`'s 12.9M rows costs far more than the dictionary is worth.
The sample is stated in the document beside each table.

Usage:
    .venv/Scripts/python.exe scripts/report_source_schemas.py --systems ie fr
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
import pyarrow.parquet as pq
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_external_destination_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from acquisition.canonical import CANONICAL_OUTPUT_COLUMNS
from workspace.data_layout import system_layer_dir
from workspace.repository import docs_dir
from workspace.roots import WorkspaceRoots

# The systems a blocking run reads today. `gleif` is left out: it is the
# source side of every pairing rather than a target register, and `dbpedia` is
# research-only.
DEFAULT_SYSTEMS: tuple[str, ...] = ("ie", "fr", "gb", "offeneregister", "wikidata")

OUTPUT_NAME = "source_schemas.md"
DEFAULT_SAMPLE_ROWS = 20_000

# Where a system's field mapping is declared, the same file
# `acquisition.canonical` resolves its canonical columns through.
CATALOG_DIR = Path("src/acquisition/catalog/systems")


@dataclass(frozen=True, slots=True)
class ColumnFacts:
    """One column of one file family, as the sample shows it."""

    name: str
    dtype: str
    null_share: float | None
    distinct: int | None
    example: str
    canonical_field: str


@dataclass(frozen=True, slots=True)
class FamilyFacts:
    """One family of files inside a snapshot: the entity files, or a sidecar
    or names companion beside them."""

    name: str
    files: int
    rows: int
    sampled: int
    columns: tuple[ColumnFacts, ...]


@dataclass(frozen=True, slots=True)
class SystemFacts:
    """One system's source snapshot."""

    system: str
    snapshot: str
    families: tuple[FamilyFacts, ...]
    # The catalog's `name_variant_type_map`: what a names family's own
    # `source_type` values become in the shared name-type vocabulary.
    name_variant_types: tuple[tuple[str, str], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the data dictionary of each system's source layer, read "
            "from the parquet files of its latest snapshot."
        ),
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="The systems to report. Default: %(default)s.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Rows read per file family for the null, distinct and example "
        "figures. Default: %(default)s.",
    )
    add_external_destination_arg(
        parser,
        "--out",
        help_text="Where the document is written, relative to the checkout. "
        f"Default: docs/{OUTPUT_NAME}.",
    )
    add_dry_run_arg(parser)
    return parser


def latest_snapshot_dir(source_root: Path) -> Path | None:
    """The system's latest dated source snapshot, the one a canonical pass
    would read, or `None` when it holds no snapshot with parquet files."""
    dated = [
        directory
        for directory in sorted(source_root.glob("*"))
        if directory.is_dir() and any(directory.glob("*.parquet"))
    ]
    return dated[-1] if dated else None


def family_of(path: Path, *, system: str) -> str:
    """The file family a shard belongs to: `<system>-001.parquet` is the
    entity family, `<system>-sidecar-001.parquet` its sidecar."""
    stem = path.stem
    if "-" in stem:
        stem = stem.rsplit("-", 1)[0]
    return stem if stem != system else f"{system} (entities)"


def canonical_fields(system: str, *, checkout: Path) -> dict[str, str]:
    """Each source column the system's catalog entry maps, by the canonical
    field or fields it feeds. A column the catalog does not name is absent.

    Five keys name a source column, and a column reached through more than
    one carries them all: `canonical_source_column_aliases`, which renames a
    column before anything else reads it (`ie`'s `company_num` is
    `company_number` by the time the mapping runs); `system_field_candidates`
    and `company_type_column`, the direct mappings;
    `system_uri_identifier_candidates`, which composes `system_uri`; and
    `canonical_derived_fields`, which builds one canonical field out of
    several columns.
    """
    catalog = checkout / CATALOG_DIR / f"{system}.json"
    if not catalog.is_file():
        return {}
    entry = json.loads(catalog.read_text(encoding="utf-8"))
    aliases = {
        str(column): str(target)
        for column, target in (
            entry.get("canonical_source_column_aliases") or {}
        ).items()
    }
    mapped: dict[str, list[str]] = {}

    def record(column: object, field: str) -> None:
        name = str(column)
        fields = mapped.setdefault(name, [])
        if field not in fields:
            fields.append(field)

    for column, target in aliases.items():
        record(column, target)
    for field, candidates in (entry.get("system_field_candidates") or {}).items():
        for column in candidates or ():
            record(column, str(field))
    company_type_column = entry.get("company_type_column")
    if company_type_column:
        # Not a mapping: the column the canonical pass drops excluded
        # company types on (`canonical_frame_transform.canonicalize_frame`).
        record(company_type_column, "(row filter)")
    for field, derivation in (entry.get("canonical_derived_fields") or {}).items():
        for column in (derivation or {}).get("parts") or ():
            record(column, f"{field} (part)")
    # A column already named for a canonical field feeds it with no mapping
    # at all, the fallback `canonical_frame_transform._field_candidates_or_
    # same_name()` adds, which is how `ie`'s `company_type` is read. A field
    # the catalog maps or derives elsewhere is not reached this way, so only
    # the fields nothing else claims are taken.
    claimed = {field for fields in mapped.values() for field in fields}
    for field in CANONICAL_OUTPUT_COLUMNS:
        if field not in claimed and f"{field} (part)" not in claimed:
            record(field, field)
    for column in entry.get("system_uri_identifier_candidates") or ():
        # The identifier candidates are read after the aliases, so a renamed
        # column is named there by its canonical name, not its own.
        source = next(
            (alias for alias, target in aliases.items() if target == str(column)),
            str(column),
        )
        record(source, "system_uri")
    return {column: ", ".join(fields) for column, fields in mapped.items()}


def _example(series: pl.Series) -> str:
    """One non-null value of the column, short enough for a table cell."""
    values = series.drop_nulls()
    if values.is_empty():
        return ""
    text = str(values[0]).replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= 60 else f"{text[:57]}..."


def read_family(
    files: list[Path], *, name: str, sample_rows: int, mapped: dict[str, str]
) -> FamilyFacts:
    """One family's facts: every file's rows from the parquet footers, and
    the column figures from a sample of the first files."""
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    sample = pl.read_parquet(files[0], n_rows=sample_rows)
    for path in files[1:]:
        if sample.height >= sample_rows:
            break
        sample = pl.concat(
            [sample, pl.read_parquet(path, n_rows=sample_rows - sample.height)],
            how="vertical_relaxed",
        )
    columns = tuple(
        ColumnFacts(
            name=column,
            dtype=str(sample.schema[column]),
            null_share=(
                sample.get_column(column).null_count() / sample.height
                if sample.height
                else None
            ),
            distinct=sample.get_column(column).n_unique() if sample.height else None,
            example=_example(sample.get_column(column)),
            canonical_field=mapped.get(column, ""),
        )
        for column in sample.columns
    )
    return FamilyFacts(
        name=name,
        files=len(files),
        rows=rows,
        sampled=sample.height,
        columns=columns,
    )


def read_system(
    roots: WorkspaceRoots, *, system: str, sample_rows: int
) -> SystemFacts | None:
    """One system's latest source snapshot, family by family, or `None` when
    it has no source layer on disk."""
    snapshot = latest_snapshot_dir(system_layer_dir(roots, system, layer="source"))
    if snapshot is None:
        return None
    mapped = canonical_fields(system, checkout=roots.checkout)
    families: dict[str, list[Path]] = {}
    for path in sorted(snapshot.glob("*.parquet")):
        families.setdefault(family_of(path, system=system), []).append(path)
    # The canonical pass reads the entity files alone, so only that family's
    # columns carry a canonical field: a column a sidecar holds is one the
    # canonical schema does not take, which is why it is in the sidecar.
    entity_family = f"{system} (entities)"
    catalog = roots.checkout / CATALOG_DIR / f"{system}.json"
    entry = json.loads(catalog.read_text(encoding="utf-8")) if catalog.is_file() else {}
    return SystemFacts(
        system=system,
        snapshot=snapshot.name,
        families=tuple(
            read_family(
                files,
                name=name,
                sample_rows=sample_rows,
                mapped=mapped if name == entity_family else {},
            )
            for name, files in sorted(families.items())
        ),
        name_variant_types=tuple(
            (str(source_type), str(name_type))
            for source_type, name_type in (
                entry.get("name_variant_type_map") or {}
            ).items()
        ),
    )


def render(systems: list[SystemFacts], *, sample_rows: int) -> str:
    """The document: one section per system, one table per file family."""
    lines = [
        "# Source Schemas",
        "",
        "What each system publishes, as its own source layer holds it in",
        "`data/<system>/source/<snapshot>/`, before the mapping into the shared",
        "canonical schema (`docs/canonical_schema.md`). Generated by",
        "`scripts/report_source_schemas.py`; regenerate it rather than editing it.",
        "",
        (
            f"`Nulls`, `Distinct` and `Example` are read from a sample of up to "
            f"{sample_rows:,} rows per family, taken from the head of its first "
            "files rather than at random, so a column the register writes in order "
            "reads as more distinct, and rarer values may be missing altogether. "
            "`Rows` is every file's. `Canonical "
            "field` is what the system's catalog entry "
            "(`src/acquisition/catalog/systems/<system>.json`) maps the column to; "
            "a blank means the canonical schema does not read that column."
        ),
        "",
    ]
    for facts in systems:
        lines += [f"## `{facts.system}`", "", f"Snapshot `{facts.snapshot}`.", ""]
        for family in facts.families:
            lines += [
                f"### `{family.name}`",
                "",
                (
                    f"{family.files} file(s), {family.rows:,} rows, "
                    f"{family.sampled:,} sampled."
                ),
                "",
            ]
            # Only the entity family can carry a canonical field, so the
            # column is left out of a table where nothing could fill it.
            mapped = any(column.canonical_field for column in family.columns)
            lines += (
                [
                    "| Column | Type | Nulls | Distinct | Example | Canonical field |",
                    "|---|---|---|---|---|---|",
                ]
                if mapped
                else [
                    "| Column | Type | Nulls | Distinct | Example |",
                    "|---|---|---|---|---|",
                ]
            )
            for column in family.columns:
                nulls = "" if column.null_share is None else f"{column.null_share:.1%}"
                distinct = "" if column.distinct is None else f"{column.distinct:,}"
                row = (
                    f"| `{column.name}` | {column.dtype} | {nulls} | {distinct} | "
                    f"{column.example} |"
                )
                lines.append(f"{row} {column.canonical_field} |" if mapped else row)
            lines.append("")
            if family.name.endswith("-sidecar"):
                lines += [
                    (
                        "The canonical pass reads the entity files alone, so no column "
                        "here feeds a canonical field: what the schema takes is in the "
                        "entity family, and the sidecar keeps the rest of the "
                        "register's own record, joined on `system_uri`."
                    ),
                    "",
                ]
            if family.name.endswith("-names") and facts.name_variant_types:
                translated = ", ".join(
                    f"`{source_type}` to `{name_type}`"
                    for source_type, name_type in facts.name_variant_types
                )
                lines += [
                    (
                        "Its rows become `alternative_names` and `previous_names` on the "
                        "canonical row, its `source_type` read through the catalog's "
                        f"`name_variant_type_map`: {translated}."
                    ),
                    "",
                ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    if args.out is None:
        out_path = docs_dir(roots.checkout) / OUTPUT_NAME
    else:
        out_path = args.out if args.out.is_absolute() else roots.checkout / args.out

    if args.dry_run:
        report_dry_run(
            "report_source_schemas",
            systems=args.systems,
            sample_rows=args.sample_rows,
        )
        report_output_plan("[dry-run]  ", [PlannedOutput(out_path, OUTPUT_CLEAR)])
        return 0

    facts: list[SystemFacts] = []
    for system in args.systems:
        read = read_system(roots, system=system, sample_rows=args.sample_rows)
        if read is None:
            print(f"[source-schemas] {system}: no source layer on disk, skipped")
            continue
        print(
            f"[source-schemas] {system}: snapshot {read.snapshot}, "
            f"{len(read.families)} file family(ies)"
        )
        facts.append(read)
    if not facts:
        print("[source-schemas] error: no system has a source layer on disk")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(facts, sample_rows=args.sample_rows), encoding="utf-8")
    print(f"[source-schemas] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
