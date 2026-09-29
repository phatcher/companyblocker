"""Materialise a public two-table entity-matching benchmark into the blocking
loader's shape, so it runs through `scripts/run_blocking.py` unchanged.

Reads the three-file shape the DeepMatcher/DeepBlocker/Magellan family of
benchmark repositories ships (`tableA.csv`, `tableB.csv`, a `matches.csv` of
`ltable_id`/`rtable_id` pairs, or labelled `train`/`valid`/`test` splits in
its place): a Zenodo archive of that shape is the input, never the archive
itself, since extracting `data_ea.tar.gz` is a one-off step outside this
module's job.

**Two systems, one row shape.** Table A becomes the indexed system
(`<benchmark>-a`), scored as a blocking target: it needs no ground truth,
the same as any other target the blocking loader resolves under
`require_ground_truth=False`. Table B becomes the queried system
(`<benchmark>-b`), scored as the source: every row carries `source_uri`,
null for a record with no true match and the indexed side's `system_uri`
for one that has exactly one, the perturbed-dataset pattern
(`validation.perturbation_materializer`, `blocking.truth.ColumnTruth`) this
module reuses rather than inventing a second one. `--match-col source_uri`
on `scripts/run_blocking.py` reads it.

**Direction** follows the analysis this benchmark validates against
(`docs/external_alignment.md`): the first collection is indexed, the second
is posed as queries, matching how `ltable_id`/`rtable_id` are conventionally
assigned in the reference repositories (`l` = table A, `r` = table B).

**Multiple true matches.** A table B record with more than one true
`ltable_id` cannot carry more than one `source_uri` on a single row, and
`source_uri` is read as a plain column (`ColumnTruth`), not a list -- so
such a record is emitted once per true pair instead of once per record,
each copy's `system_uri` disambiguated with the matched `ltable_id`
(`<system>://<id>~<ltable_id>`) so the two duplicates remain distinct rows.
A record with zero or one true match keeps the plain `<system>://<id>`
form. `PublicBenchmarkMaterializationResult.multi_match_source_count` is
how many table B records needed this, for a caller to record beside a
recall figure a scorer that only reads a source's first candidate would
undercount.

**Name.** `name` is the concatenation, in the CSV's own column order and
joined by a single space, of every column other than the id column whose
inferred dtype is a string: a numeric attribute such as `price` is not
text and plays no part in it. The sentence encoder this dataset is meant
to be scored under lowercases its own input, so no cleanse stage needs to
run over this name before embedding it.

**Jurisdiction.** Neither benchmark carries one, so every row gets the
same fixed placeholder, `DEFAULT_JURISDICTION_CODE` -- an ISO 3166-1
user-assigned code, chosen because it cannot collide with a real one, not
because it means anything about the rows it partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots

DEFAULT_ID_COLUMN = "id"

DEFAULT_JURISDICTION_CODE = "xx"

_MATCHES_FILE_NAME = "matches.csv"

_SPLIT_FILE_NAMES = ("train.csv", "valid.csv", "test.csv")

_LEFT_ID_COLUMN = "ltable_id"

_RIGHT_ID_COLUMN = "rtable_id"

_LABEL_COLUMN = "label"

_MAX_REPORTED_UNKNOWN_IDS = 20


class BenchmarkFormatError(ValueError):
    """A benchmark root does not hold the expected table/match file shape."""


@dataclass(frozen=True)
class PublicBenchmarkMaterializationConfig:
    roots: WorkspaceRoots

    benchmark_root: Path

    benchmark_name: str

    id_col: str = DEFAULT_ID_COLUMN

    jurisdiction_code: str = DEFAULT_JURISDICTION_CODE


@dataclass(frozen=True)
class PublicBenchmarkMaterializationResult:
    indexed_system: str

    queried_system: str

    indexed_dir: Path

    queried_dir: Path

    indexed_rows: int

    queried_rows: int

    truth_pairs: int

    multi_match_source_count: int


def _text_columns(frame: pl.DataFrame, *, id_col: str) -> list[str]:
    return [
        name
        for name, dtype in frame.schema.items()
        if name != id_col and dtype == pl.Utf8
    ]


def read_benchmark_table(
    csv_path: Path, *, id_col: str = DEFAULT_ID_COLUMN
) -> pl.DataFrame:
    """One row per record of a `tableA.csv`/`tableB.csv`-shaped file: its own
    `id` (cast to string) and `name`, the space-joined concatenation of every
    other string-typed column in file order. Raises `BenchmarkFormatError`
    when `id_col` is absent or there is no textual attribute to build a name
    from, rather than materializing an empty or nonsensical name column."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark table not found: {csv_path}")

    frame = pl.read_csv(csv_path, infer_schema_length=None)

    if id_col not in frame.columns:
        raise BenchmarkFormatError(
            f"{csv_path} has no {id_col!r} column; columns are {frame.columns!r}"
        )

    text_cols = _text_columns(frame, id_col=id_col)
    if not text_cols:
        raise BenchmarkFormatError(
            f"{csv_path} has no textual attribute column besides {id_col!r} to "
            f"build a name from; columns are {frame.columns!r}"
        )

    name_expr = pl.concat_str(
        [pl.col(column).fill_null("").cast(pl.Utf8) for column in text_cols],
        separator=" ",
    )
    return frame.select(
        pl.col(id_col).cast(pl.Utf8).alias("id"),
        name_expr.alias("name"),
    )


def load_benchmark_matches(benchmark_root: Path) -> pl.DataFrame:
    """`(ltable_id, rtable_id)` truth pairs for a benchmark roots.

    Reads `matches.csv` directly when present. Otherwise reads whichever of
    `train.csv`/`valid.csv`/`test.csv` exist and keeps the rows labelled
    `1`, the way DeepBlocker's own reference code derives its match set from
    the labelled splits rather than a dedicated matches file. Deduplicated
    either way, since a pair repeated across splits (or within one file) is
    one true pair, not two.
    """

    matches_path = benchmark_root / _MATCHES_FILE_NAME
    if matches_path.exists():
        frame = pl.read_csv(matches_path, infer_schema_length=None)
        pairs = frame.select(
            pl.col(_LEFT_ID_COLUMN).cast(pl.Utf8),
            pl.col(_RIGHT_ID_COLUMN).cast(pl.Utf8),
        )
    else:
        split_frames = [
            pl.read_csv(split_path, infer_schema_length=None)
            for split_name in _SPLIT_FILE_NAMES
            if (split_path := benchmark_root / split_name).exists()
        ]
        if not split_frames:
            raise BenchmarkFormatError(
                f"{benchmark_root} has neither {_MATCHES_FILE_NAME!r} nor any of "
                f"{_SPLIT_FILE_NAMES!r} to derive truth pairs from"
            )
        combined = pl.concat(split_frames, how="vertical_relaxed")
        pairs = combined.filter(pl.col(_LABEL_COLUMN) == 1).select(
            pl.col(_LEFT_ID_COLUMN).cast(pl.Utf8),
            pl.col(_RIGHT_ID_COLUMN).cast(pl.Utf8),
        )

    return pairs.unique().sort([_LEFT_ID_COLUMN, _RIGHT_ID_COLUMN])


def _check_matches_reference_known_ids(
    matches: pl.DataFrame, *, table_a_ids: set[str], table_b_ids: set[str]
) -> None:
    unknown_left = sorted(set(matches.get_column(_LEFT_ID_COLUMN)) - table_a_ids)
    unknown_right = sorted(set(matches.get_column(_RIGHT_ID_COLUMN)) - table_b_ids)
    if unknown_left or unknown_right:
        raise BenchmarkFormatError(
            "matches reference ids absent from the tables they name: "
            f"{len(unknown_left)} ltable_id(s) not in tableA "
            f"{unknown_left[:_MAX_REPORTED_UNKNOWN_IDS]!r}, "
            f"{len(unknown_right)} rtable_id(s) not in tableB "
            f"{unknown_right[:_MAX_REPORTED_UNKNOWN_IDS]!r}"
        )


def _build_indexed_frame(
    table: pl.DataFrame, *, system: str, jurisdiction_code: str
) -> pl.DataFrame:
    return table.select(
        (pl.lit(f"{system}://") + pl.col("id")).alias("system_uri"),
        pl.col("name"),
        pl.lit(jurisdiction_code).alias("jurisdiction_code"),
    )


def _build_queried_frame(
    table: pl.DataFrame,
    *,
    matches: pl.DataFrame,
    indexed_system: str,
    queried_system: str,
    jurisdiction_code: str,
) -> tuple[pl.DataFrame, int]:
    """Every table B record as one row, its `source_uri` null unless matches
    names it a true pair -- and, for a record with more than one true match,
    one row per pair instead of one row per record (see module docstring)."""

    matches_by_right: dict[str, list[str]] = {}
    for right_id, left_id in matches.select(
        _RIGHT_ID_COLUMN, _LEFT_ID_COLUMN
    ).iter_rows():
        matches_by_right.setdefault(right_id, []).append(left_id)

    system_uris: list[str] = []
    names: list[str] = []
    source_uris: list[str | None] = []
    multi_match_source_count = 0

    for row_id, name in table.select("id", "name").iter_rows():
        left_ids = sorted(matches_by_right.get(row_id, []))
        if not left_ids:
            system_uris.append(f"{queried_system}://{row_id}")
            names.append(name)
            source_uris.append(None)
        elif len(left_ids) == 1:
            system_uris.append(f"{queried_system}://{row_id}")
            names.append(name)
            source_uris.append(f"{indexed_system}://{left_ids[0]}")
        else:
            multi_match_source_count += 1
            for left_id in left_ids:
                system_uris.append(f"{queried_system}://{row_id}~{left_id}")
                names.append(name)
                source_uris.append(f"{indexed_system}://{left_id}")

    frame = pl.DataFrame(
        {
            "system_uri": system_uris,
            "name": names,
            "jurisdiction_code": [jurisdiction_code] * len(system_uris),
            "source_uri": source_uris,
        },
        schema={
            "system_uri": pl.Utf8,
            "name": pl.Utf8,
            "jurisdiction_code": pl.Utf8,
            "source_uri": pl.Utf8,
        },
    )
    return frame, multi_match_source_count


def _write_cleansed_system(
    *, roots: WorkspaceRoots, system: str, frame: pl.DataFrame, jurisdiction_code: str
) -> Path:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    partition_dir = layer_partition_dir(cleansed_dir, value=jurisdiction_code)
    partition_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(partition_dir / "part-00000.parquet")
    return cleansed_dir


def materialize_public_benchmark(
    config: PublicBenchmarkMaterializationConfig,
) -> PublicBenchmarkMaterializationResult:
    """Read `config.benchmark_root`'s tables and matches and write the two
    systems `<benchmark_name>-a` (indexed) and `<benchmark_name>-b`
    (queried) under `config.roots`' data root, in the shape
    `blocking.loader.load_dataset_descriptor` reads unchanged. Overwrites
    either system's single output file if it already exists -- there is no
    batching or staging to reconcile, only two small tables."""

    table_a = read_benchmark_table(
        config.benchmark_root / "tableA.csv", id_col=config.id_col
    )
    table_b = read_benchmark_table(
        config.benchmark_root / "tableB.csv", id_col=config.id_col
    )
    matches = load_benchmark_matches(config.benchmark_root)
    _check_matches_reference_known_ids(
        matches,
        table_a_ids=set(table_a.get_column("id")),
        table_b_ids=set(table_b.get_column("id")),
    )

    indexed_system = f"{config.benchmark_name}-a"
    queried_system = f"{config.benchmark_name}-b"

    indexed_frame = _build_indexed_frame(
        table_a, system=indexed_system, jurisdiction_code=config.jurisdiction_code
    )
    queried_frame, multi_match_source_count = _build_queried_frame(
        table_b,
        matches=matches,
        indexed_system=indexed_system,
        queried_system=queried_system,
        jurisdiction_code=config.jurisdiction_code,
    )

    indexed_dir = _write_cleansed_system(
        roots=config.roots,
        system=indexed_system,
        frame=indexed_frame,
        jurisdiction_code=config.jurisdiction_code,
    )
    queried_dir = _write_cleansed_system(
        roots=config.roots,
        system=queried_system,
        frame=queried_frame,
        jurisdiction_code=config.jurisdiction_code,
    )

    return PublicBenchmarkMaterializationResult(
        indexed_system=indexed_system,
        queried_system=queried_system,
        indexed_dir=indexed_dir,
        queried_dir=queried_dir,
        indexed_rows=indexed_frame.height,
        queried_rows=queried_frame.height,
        truth_pairs=matches.height,
        multi_match_source_count=multi_match_source_count,
    )


__all__ = [
    "DEFAULT_ID_COLUMN",
    "DEFAULT_JURISDICTION_CODE",
    "BenchmarkFormatError",
    "PublicBenchmarkMaterializationConfig",
    "PublicBenchmarkMaterializationResult",
    "load_benchmark_matches",
    "materialize_public_benchmark",
    "read_benchmark_table",
]
