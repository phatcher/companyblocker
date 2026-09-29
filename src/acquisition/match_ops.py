"""The Match stage: label one source's canonical rows against one or more targets'.

Reads each system's latest canonical snapshot and writes `data/<source>/matched/jurisdiction_code=<country>/`, canonical's row with `match_uri` filled in. A rerun replaces only the partitions it writes, so runs for different jurisdictions coexist. The join keys are canonical fields normalised here, so Match reads no cleansed layer and a system can be matched before any cleanse configuration exists: `company_number_key` tolerates leading zeros, strips Germany's trailing court-register suffix and reduces France's SIRET to its SIREN, on top of stripping non-alphanumerics and uppercasing.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    MATCHED_LAYER_NAME,
    latest_canonical_snapshot_dir,
    system_layer_dir,
)
from workspace.layer_layout import (
    layer_partition_dir,
    partition_values,
    resolve_partition_dir,
)
from workspace.roots import WorkspaceRoots


@dataclass(frozen=True)
class MatchRunSummary:
    """One match run: the named source labelled against the named target, for
    the one jurisdiction they share."""

    source_system: str
    source_country: str
    target_system: str
    output_file_count: int
    rows_written: int
    rows_matched: int
    source_rows_total: int
    source_valid_keys: int
    source_invalid_keys: int
    target_rows_total: int
    target_valid_keys: int
    target_invalid_keys: int
    source_duplicate_keys: int
    target_duplicate_keys: int
    elapsed_seconds: float


def _canonical_view_dir(roots: WorkspaceRoots, system: str) -> Path | None:
    """The canonical snapshot Match reads a system from, or None when the
    system has none. Match joins on canonical-schema fields alone, so it
    reads the rows canonical wrote and no layer derived from them."""
    return latest_canonical_snapshot_dir(
        system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME)
    )


def _partition_countries(*, system_merge_dir: Path) -> list[str]:
    return partition_values(system_merge_dir)


def _resolve_explicit_target_system(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    target_system: str,
) -> tuple[str, Path, list[str]]:
    normalized_target = target_system.strip().lower()
    if not normalized_target:
        raise ValueError("Explicit match target system cannot be empty.")
    if normalized_target == source_system:
        raise ValueError("Explicit match target system must differ from source system.")

    target_merge_dir = _canonical_view_dir(roots, normalized_target)
    if target_merge_dir is None:
        raise FileNotFoundError(
            f"Explicit match target system '{normalized_target}' has no canonical snapshot."
        )

    target_countries = _partition_countries(system_merge_dir=target_merge_dir)
    return normalized_target, target_merge_dir, target_countries


def _resolve_match_country_for_explicit_target(
    *,
    source_system: str,
    source_countries: list[str],
    target_system: str,
    target_countries: list[str],
) -> str:
    source_set = set(source_countries)
    target_set = set(target_countries)

    if len(source_countries) == 1:
        source_country = source_countries[0]
        if source_country not in target_set:
            raise ValueError(
                f"Explicit match target system '{target_system}' does not contain jurisdiction '{source_country}'."
            )
        return source_country

    if len(target_countries) == 1:
        target_country = target_countries[0]
        if target_country not in source_set:
            raise ValueError(
                f"Source system '{source_system}' does not contain jurisdiction '{target_country}' required by target '{target_system}'."
            )
        return target_country

    overlap = sorted(source_set & target_set)
    if len(overlap) == 1:
        return overlap[0]

    raise ValueError(
        "Cannot resolve match jurisdiction for explicit target because both systems expose multiple overlapping jurisdictions. "
        "Add a single-jurisdiction target or extend match args with an explicit jurisdiction selector."
    )


def _as_glob(path: Path) -> str:
    return path.as_posix()


def _sql_string_list(values: list[str]) -> str:
    escaped = ["'" + value.replace("'", "''") + "'" for value in values]
    return "[" + ", ".join(escaped) + "]"


_COMPANY_NUMBER_BASE_KEY_SQL = (
    "upper(regexp_replace(trim(coalesce(cast(company_number AS VARCHAR), '')), "
    "'[^0-9A-Za-z]', '', 'g'))"
)
_JURISDICTION_KEY_SQL = "lower(trim(coalesce(cast(jurisdiction_code AS VARCHAR), '')))"


def _leading_zero_tolerant_key_sql(base_key_sql: str) -> str:
    """Strip leading zeros from a purely-numeric key, tolerating
    inconsistent zero-padding across systems (e.g. Ireland's '000123' vs
    '123'). Keys that contain letters pass through unchanged.
    """
    stripped = f"regexp_replace({base_key_sql}, '^0+', '')"
    return (
        f"CASE WHEN {base_key_sql} ~ '^[0-9]+$' "
        f"THEN CASE WHEN {stripped} = '' THEN '0' ELSE {stripped} END "
        f"ELSE {base_key_sql} END"
    )


def _enriched_key_columns_sql() -> str:
    """`jurisdiction_key`/`_numeric_key` columns shared by the source- and
    target-side enrichment queries; `_company_number_key_sql` derives the
    final `company_number_key` from these two, so both sides always compute
    it identically.
    """
    numeric_key = _leading_zero_tolerant_key_sql(_COMPANY_NUMBER_BASE_KEY_SQL)
    return (
        f"{_JURISDICTION_KEY_SQL} AS jurisdiction_key,\n"
        f"    ({numeric_key}) AS _numeric_key"
    )


def _company_number_key_sql() -> str:
    """`company_number_key`, derived from `jurisdiction_key`/`_numeric_key`
    (see `_enriched_key_columns_sql`).

    Normalization beyond the base strip-non-alphanumeric-and-uppercase pass
    is scoped to real failure shapes surveyed against real
    `gleif -> {gb, ie, fr, offeneregister}` joins:

      - Leading-zero tolerance applies to every jurisdiction (folded into
        `_numeric_key` above) -- confirmed real for Ireland.
      - Germany (`de`): strip a trailing letter suffix -- OffeneRegister and
        GLEIF disagree on court-specific suffixes like 'HL'/'B'/'KI' appended
        after the register number -- while preserving the register-type
        prefix (HRA/HRB/GnR/...), which legitimately distinguishes different
        registrations that happen to share the same number.
      - France (`fr`): a 14-digit numeric key is a SIRET (9-digit SIREN plus
        a 5-digit establishment code); reduce it to its leading 9 digits so
        it matches the other side's SIREN-only key.

    GB was surveyed too and showed no fixable pattern -- its mismatched keys
    are genuinely different numbers, not a normalization gap -- so no GB
    branch was added rather than adding one speculatively.
    """
    return (
        "CASE\n"
        "        WHEN jurisdiction_key = 'de' "
        "THEN regexp_replace(_numeric_key, '[A-Za-z]+$', '')\n"
        "        WHEN jurisdiction_key = 'fr' AND _numeric_key ~ '^[0-9]{14}$' "
        "THEN left(_numeric_key, 9)\n"
        "        ELSE _numeric_key\n"
        "    END"
    )


def _create_enriched_keys_table(
    con: duckdb.DuckDBPyConnection,
    *,
    source_table: str,
    target_table: str,
    include_row_id: bool,
) -> None:
    row_id_expr = "row_number() OVER () AS _row_id," if include_row_id else ""
    create_sql = f"CREATE TEMP TABLE {target_table} AS\n"
    create_sql += "SELECT\n"
    create_sql += f"    {row_id_expr}\n"
    create_sql += f"    {source_table}.*,\n"
    create_sql += f"    {_enriched_key_columns_sql()},\n"
    create_sql += f"    {_company_number_key_sql()} AS company_number_key\n"
    create_sql += f"FROM {source_table}"
    con.execute(create_sql)


def _add_valid_key_column(con: duckdb.DuckDBPyConnection, *, table_name: str) -> None:
    con.execute(
        f"""
        ALTER TABLE {table_name}
        ADD COLUMN is_valid_key BOOLEAN
        """
    )
    # nosec B608 - table_name is one of this module's own fixed table literals, never
    # external input.
    update_sql = f"UPDATE {table_name}\nSET is_valid_key = (jurisdiction_key <> '' AND company_number_key <> '')"  # nosec B608
    con.execute(update_sql)


def _count_rows(
    con: duckdb.DuckDBPyConnection, *, table_name: str, where_sql: str | None = None
) -> int:
    where_clause = f" WHERE {where_sql}" if where_sql else ""
    # nosec B608 - table_name/where_sql are this module's own fixed literals, never
    # external input.
    row = con.execute(f"SELECT count(*) FROM {table_name}{where_clause}").fetchone()  # nosec B608
    return int(row[0] if row is not None else 0)


def _count_duplicate_keys(con: duckdb.DuckDBPyConnection, *, table_name: str) -> int:
    # nosec B608 - table_name is this module's own fixed table literal, never external
    # input.
    duplicate_sql = (
        f"SELECT count(*)\n"  # nosec B608
        "FROM (\n"
        "    SELECT jurisdiction_key, company_number_key\n"
        f"    FROM {table_name}\n"
        "    WHERE is_valid_key\n"
        "    GROUP BY jurisdiction_key, company_number_key\n"
        "    HAVING count(*) > 1\n"
        ")"
    )
    row = con.execute(duplicate_sql).fetchone()
    return int(row[0] if row is not None else 0)


def _write_duplicate_key_report(
    con: duckdb.DuckDBPyConnection,
    *,
    output_root: Path,
    has_source_enriched: bool,
    has_target_enriched: bool,
) -> None:
    """Write a per-key (not per-row) summary of ambiguous match keys.

    A hard failure on duplicate keys breaks the pipeline; a silently-picked
    arbitrary match hides a real problem. Instead the ambiguous keys are
    excluded from the join (see _build_target_lookup) and their identity
    (side, key, row count) is written here so the run completes and the
    ambiguity is still investigable afterwards -- always written, even when
    empty, so its row count is a reliable signal without special-casing
    "file might not exist".
    """
    # nosec B608 (below) - the interpolated identifiers ('source_enriched',
    # 'target_enriched') are this module's own fixed table literals, never external
    # input.
    selects: list[str] = []
    if has_source_enriched:
        selects.append(
            "SELECT 'source' AS side, jurisdiction_key, company_number_key, count(*) AS row_count\n"  # nosec B608
            "FROM source_enriched\n"
            "WHERE is_valid_key\n"
            "GROUP BY jurisdiction_key, company_number_key\n"
            "HAVING count(*) > 1"
        )
    if has_target_enriched:
        selects.append(
            "SELECT 'target' AS side, jurisdiction_key, company_number_key, count(*) AS row_count\n"  # nosec B608
            "FROM target_enriched\n"
            "WHERE is_valid_key\n"
            "GROUP BY jurisdiction_key, company_number_key\n"
            "HAVING count(*) > 1"
        )

    output_path = output_root / "_duplicate_keys.parquet"
    if not selects:
        con.execute(
            """
            COPY (
                SELECT
                    cast(NULL AS VARCHAR) AS side,
                    cast(NULL AS VARCHAR) AS jurisdiction_key,
                    cast(NULL AS VARCHAR) AS company_number_key,
                    cast(NULL AS BIGINT) AS row_count
                WHERE FALSE
            ) TO ? (FORMAT PARQUET)
            """,
            [str(output_path)],
        )
        return

    union_sql = "\nUNION ALL\n".join(selects)
    # nosec B608 - union_sql is built entirely from this module's own fixed SELECT
    # literals above; the output path is bound through the `?` placeholder.
    con.execute(
        f"COPY ({union_sql}) TO ? (FORMAT PARQUET)",  # nosec B608
        [str(output_path)],
    )


def _source_key_stats(con: duckdb.DuckDBPyConnection) -> tuple[int, int, int, int]:
    source_rows_total = _count_rows(con, table_name="source_enriched")
    source_valid_keys = _count_rows(
        con, table_name="source_enriched", where_sql="is_valid_key"
    )
    source_invalid_keys = source_rows_total - source_valid_keys
    source_duplicate_keys = _count_duplicate_keys(con, table_name="source_enriched")
    return (
        source_rows_total,
        source_valid_keys,
        source_invalid_keys,
        source_duplicate_keys,
    )


def _build_target_lookup(
    con: duckdb.DuckDBPyConnection,
    *,
    target_partition_globs: list[str],
) -> tuple[int, int, int, int]:
    target_rows_total = 0
    target_valid_keys = 0
    target_invalid_keys = 0
    target_duplicate_keys = 0

    if not target_partition_globs:
        con.execute(
            """
            CREATE TEMP TABLE target_lookup (
                jurisdiction_key VARCHAR,
                company_number_key VARCHAR,
                target_match_uri VARCHAR
            )
            """
        )
        return (
            target_rows_total,
            target_valid_keys,
            target_invalid_keys,
            target_duplicate_keys,
        )

    target_glob_literals = _sql_string_list(target_partition_globs)
    # nosec B608 (below, and in target_enriched_sql further down) - the interpolated
    # SQL is built from this module's own fixed literals; target_glob_literals is a
    # quoted string list produced by _sql_string_list, not raw user input.
    target_raw_sql = (
        f"CREATE TEMP TABLE target_raw AS\n"  # nosec B608
        "SELECT *\n"
        f"FROM read_parquet({target_glob_literals}, hive_partitioning=1)"
    )
    con.execute(target_raw_sql)

    target_columns = [
        str(row[0]) for row in con.execute("DESCRIBE target_raw").fetchall()
    ]
    if "company_number" not in target_columns:
        raise ValueError(
            "Target canonical view is missing required column 'company_number'."
        )
    if "system_uri" not in target_columns:
        raise ValueError(
            "Target canonical view is missing required column 'system_uri'."
        )

    has_target_match_uri = "match_uri" in target_columns
    target_match_uri_expr = (
        "coalesce(cast(system_uri AS VARCHAR), cast(match_uri AS VARCHAR))"
        if has_target_match_uri
        else "cast(system_uri AS VARCHAR)"
    )
    target_enriched_sql = (
        "CREATE TEMP TABLE target_enriched AS\n"  # nosec B608
        "SELECT\n"
        f"    {_enriched_key_columns_sql()},\n"
        f"    {_company_number_key_sql()} AS company_number_key,\n"
        f"    {target_match_uri_expr} AS target_match_uri\n"
        "FROM target_raw"
    )
    con.execute(target_enriched_sql)
    _add_valid_key_column(con, table_name="target_enriched")

    target_rows_total = _count_rows(con, table_name="target_enriched")
    target_valid_keys = _count_rows(
        con, table_name="target_enriched", where_sql="is_valid_key"
    )
    target_invalid_keys = target_rows_total - target_valid_keys
    target_duplicate_keys = _count_duplicate_keys(con, table_name="target_enriched")

    # A key with more than one target row is ambiguous -- which row it should
    # resolve to can't be determined, so it's excluded from the lookup
    # entirely (source rows referencing it come back as no-match) rather than
    # guessed at via an arbitrary pick. See _write_duplicate_key_report for
    # where these get surfaced for investigation instead of failing the run.
    con.execute(
        """
        CREATE TEMP TABLE target_lookup AS
        SELECT jurisdiction_key, company_number_key, max(target_match_uri) AS target_match_uri
        FROM target_enriched
        WHERE is_valid_key
        GROUP BY jurisdiction_key, company_number_key
        HAVING count(*) = 1
        """
    )
    return (
        target_rows_total,
        target_valid_keys,
        target_invalid_keys,
        target_duplicate_keys,
    )


def _build_select_sql(*, source_columns: list[str]) -> str:
    select_columns: list[str] = []
    for column_name in source_columns:
        quoted = '"' + column_name.replace('"', '""') + '"'
        if column_name == "match_uri":
            select_columns.append("j.target_match_uri AS match_uri")
        else:
            select_columns.append(f"j.{quoted}")
    if "match_uri" not in source_columns:
        select_columns.append("j.target_match_uri AS match_uri")
    return ",\n                ".join(select_columns)


def _write_joined_match_chunks(
    con: duckdb.DuckDBPyConnection,
    *,
    source_files: list[Path],
    output_partition_dir: Path,
    select_sql: str,
) -> tuple[int, int, int]:
    matched_rows = 0
    written_rows = 0
    output_file_count = 0

    for file_index, source_file in enumerate(source_files, start=1):
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE source_chunk_raw AS
            SELECT *
            FROM read_parquet(?)
            """,
            [source_file.as_posix()],
        )
        _create_enriched_keys_table(
            con,
            source_table="source_chunk_raw",
            target_table="source_chunk_enriched",
            include_row_id=True,
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE joined_rows AS
            SELECT
                s._row_id,
                s.*,
                t.target_match_uri
            FROM source_chunk_enriched AS s
            LEFT JOIN target_lookup AS t
              ON t.jurisdiction_key = s.jurisdiction_key
             AND t.company_number_key = s.company_number_key
            """
        )

        matched_rows += _count_rows(
            con, table_name="joined_rows", where_sql="target_match_uri IS NOT NULL"
        )
        chunk_rows = _count_rows(con, table_name="joined_rows")
        written_rows += chunk_rows
        output_file_count += 1

        output_file = output_partition_dir / f"part-{file_index:05d}.parquet"
        # nosec B608 - select_sql is built from this module's own fixed column
        # literals, never external input.
        copy_sql = (
            f"COPY (\n"  # nosec B608
            "    SELECT\n"
            f"        {select_sql}\n"
            "    FROM joined_rows AS j\n"
            "    ORDER BY j._row_id\n"
            ") TO ? (FORMAT PARQUET)"
        )
        con.execute(
            copy_sql,
            [output_file.as_posix()],
        )

    return matched_rows, written_rows, output_file_count


def _merge_matched_target_systems(
    output_root: Path, *, target_systems: list[str]
) -> list[str]:
    """Union `target_systems` with whatever `_match_metadata.json` already
    records, so writing one source country's match run doesn't erase the
    `target_systems` another country's earlier run already recorded.
    `_match_metadata.json` is one shared file per source system, not
    per-partition, so `src/blocking.loader._load_matched_target_systems()`
    (its only consumer) needs the union across every country ever matched,
    not just the country this invocation just wrote.
    """
    metadata_path = output_root / "_match_metadata.json"
    existing_targets: list[str] = []
    if metadata_path.exists():
        try:
            existing_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
        existing_targets = [
            str(target) for target in existing_payload.get("target_systems") or []
        ]
    return list(dict.fromkeys([*existing_targets, *target_systems]))


def materialize_match_uri_artifact(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    target_system: str,
) -> MatchRunSummary:
    """Label `source_system`'s rows against `target_system`, writing the source's
    `matched/` partition for the one jurisdiction the two share.

    A match is a statement of intent, this source against that target, so
    both are named by the caller and nothing else is written: no system is
    discovered from what happens to be on disk, and the direction given is
    the direction run."""
    started_at = time.perf_counter()
    source_merge_dir = _canonical_view_dir(roots, source_system)
    if source_merge_dir is None:
        raise FileNotFoundError(
            f"No canonical snapshot found for source system '{source_system}'."
        )

    source_countries = _partition_countries(system_merge_dir=source_merge_dir)
    resolved_target, target_merge_dir, target_countries = (
        _resolve_explicit_target_system(
            roots=roots, source_system=source_system, target_system=target_system
        )
    )
    source_country = _resolve_match_country_for_explicit_target(
        source_system=source_system,
        source_countries=source_countries,
        target_system=resolved_target,
        target_countries=target_countries,
    )

    output_root = system_layer_dir(roots, source_system, layer=MATCHED_LAYER_NAME)
    output_root.mkdir(parents=True, exist_ok=True)
    # Mirrors the layer it derives from: matched/ carries the same family
    # and partition shape as canonical, because a stage that constructs its
    # own output shape is a stage that can disagree with its input.
    output_partition_dir = layer_partition_dir(output_root, value=source_country)
    if output_partition_dir.exists():
        shutil.rmtree(output_partition_dir)
    output_partition_dir.mkdir(parents=True, exist_ok=True)

    source_partition_dir = resolve_partition_dir(source_merge_dir, value=source_country)
    if source_partition_dir is None:
        raise FileNotFoundError(
            f"No '{source_country}' partition found for system '{source_system}' "
            f"under {source_merge_dir}."
        )
    source_partition_glob = _as_glob(source_partition_dir / "*.parquet")
    target_partition_dir = resolve_partition_dir(target_merge_dir, value=source_country)
    if target_partition_dir is None:
        raise FileNotFoundError(
            f"No '{source_country}' partition found under canonical for target "
            f"system '{resolved_target}'."
        )
    target_partition_globs = [_as_glob(target_partition_dir / "*.parquet")]
    source_files = sorted(source_partition_dir.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(
            f"No source parquet files found for system '{source_system}' and jurisdiction '{source_country}'."
        )

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=4")

        con.execute(
            """
            CREATE TEMP TABLE source_raw AS
            SELECT *
            FROM read_parquet(?, hive_partitioning=1)
            """,
            [source_partition_glob],
        )

        source_columns = [
            str(row[0]) for row in con.execute("DESCRIBE source_raw").fetchall()
        ]
        if "company_number" not in source_columns:
            raise ValueError(
                f"Source system '{source_system}' canonical view is missing required column 'company_number'."
            )

        _create_enriched_keys_table(
            con,
            source_table="source_raw",
            target_table="source_enriched",
            include_row_id=True,
        )
        _add_valid_key_column(con, table_name="source_enriched")

        (
            source_rows_total,
            source_valid_keys,
            source_invalid_keys,
            source_duplicate_keys,
        ) = _source_key_stats(con)

        (
            target_rows_total,
            target_valid_keys,
            target_invalid_keys,
            target_duplicate_keys,
        ) = _build_target_lookup(
            con,
            target_partition_globs=target_partition_globs,
        )

        _write_duplicate_key_report(
            con,
            output_root=output_root,
            has_source_enriched=True,
            has_target_enriched=bool(target_partition_globs),
        )

        select_sql = _build_select_sql(source_columns=source_columns)

        matched_rows, written_rows, output_file_count = _write_joined_match_chunks(
            con,
            source_files=source_files,
            output_partition_dir=output_partition_dir,
            select_sql=select_sql,
        )

        merged_target_systems = _merge_matched_target_systems(
            output_root, target_systems=[resolved_target]
        )
        metadata = {
            "source_system": source_system,
            "source_country": source_country,
            "target_systems": merged_target_systems,
            "output_file_count": output_file_count,
            "rows_written": written_rows,
            "rows_matched": matched_rows,
            "source_rows_total": source_rows_total,
            "source_valid_keys": source_valid_keys,
            "source_invalid_keys": source_invalid_keys,
            "target_rows_total": target_rows_total,
            "target_valid_keys": target_valid_keys,
            "target_invalid_keys": target_invalid_keys,
            "source_duplicate_keys": source_duplicate_keys,
            "target_duplicate_keys": target_duplicate_keys,
            "join_type": "left",
            "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        }
        (output_root / "_match_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        return MatchRunSummary(
            source_system=source_system,
            source_country=source_country,
            target_system=resolved_target,
            output_file_count=output_file_count,
            rows_written=written_rows,
            rows_matched=matched_rows,
            source_rows_total=source_rows_total,
            source_valid_keys=source_valid_keys,
            source_invalid_keys=source_invalid_keys,
            target_rows_total=target_rows_total,
            target_valid_keys=target_valid_keys,
            target_invalid_keys=target_invalid_keys,
            source_duplicate_keys=source_duplicate_keys,
            target_duplicate_keys=target_duplicate_keys,
            elapsed_seconds=time.perf_counter() - started_at,
        )
    finally:
        con.close()
