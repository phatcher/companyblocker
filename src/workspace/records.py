"""What was produced, from what, and how: one record per location, and a query over them.

Everything a producer writes goes through `produce`. The producer hands it the
output reference, the references it consumed by role, its parameters and the
invocation that ran it, and writes its own files into the staging directory it
is given. On success `produce` writes the location's record last and swaps the
staged directory into place; on failure it removes the staging directory and
nothing is recorded. A location is therefore complete exactly when it holds a
record, whatever the area writes beside it. The record, `_reference.json`, holds
the location's URI and key, each input's URI and consumed content digest, the
parameters, the roots, the commit, the command line without its root flags,
and the start and finish times.

**The record lives inside the location it describes.** `data/` and
`artifacts/` are written by many sessions and worktrees at once, so one shared
database file would be a lock every writer contends for, and it could disagree
with what is on disk. A record moves and disappears with its files instead, and
a record counts only where its own URI locates it, so one left inside an
abandoned staging directory is never read.

**The query is a view, rebuilt on demand.** `open_catalog` reads every record
under the registered side roots into three DuckDB tables, `records`,
`segments` and `inputs`, so a report selects by any segment and follows an
input's URI to everything that consumed it, rather than walking a tree.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from .identity import ConsumedReference, current_commit, reference_key
from .reference import (
    InvalidReferenceError,
    Reference,
    locate,
    parse_reference,
    reference_at,
    registered_layouts,
    side_root,
)
from .roots import WorkspaceRoots

RECORD_FILENAME = "_reference.json"
"""The one file name inside a location that belongs to `workspace`, not the area."""


class ProductionError(ValueError):
    """A production refused before any work: an incomplete output or a missing input."""


@dataclass(frozen=True)
class ConsumedInput:
    """One input as recorded: its URI and the content digest the producer consumed."""

    uri: str
    content_digest: str | None


@dataclass(frozen=True)
class Record:
    """What one location holds, how it was made, and from what."""

    uri: str
    key: str
    inputs: Mapping[str, ConsumedInput]
    parameters: Mapping[str, object]
    invocation: tuple[str, ...]
    roots: Mapping[str, Mapping[str, str]]
    commit: str | None
    started_at: str
    finished_at: str
    content_digest: str | None

    @property
    def reference(self) -> Reference:
        return parse_reference(self.uri)

    def to_json(self) -> str:
        ref = self.reference
        return json.dumps(
            {
                "uri": self.uri,
                "kind": ref.kind.value,
                "side": ref.side.value,
                "segments": ref.fields,
                "key": self.key,
                "inputs": {
                    role: {
                        "uri": consumed.uri,
                        "content_digest": consumed.content_digest,
                    }
                    for role, consumed in sorted(self.inputs.items())
                },
                "parameters": dict(self.parameters),
                "invocation": list(self.invocation),
                "roots": {name: dict(root) for name, root in self.roots.items()},
                "commit": self.commit,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "content_digest": self.content_digest,
            },
            sort_keys=True,
            indent=2,
            default=str,
        )

    @classmethod
    def from_json(cls, text: str) -> Record:
        raw = json.loads(text)
        return cls(
            uri=raw["uri"],
            key=raw["key"],
            inputs={
                role: ConsumedInput(
                    uri=value["uri"], content_digest=value["content_digest"]
                )
                for role, value in raw["inputs"].items()
            },
            parameters=raw["parameters"],
            invocation=tuple(raw["invocation"]),
            roots=raw["roots"],
            commit=raw["commit"],
            started_at=raw["started_at"],
            finished_at=raw["finished_at"],
            content_digest=raw["content_digest"],
        )


@dataclass
class Production:
    """The staging directory a producer writes into, and what it declares on the way.

    A mutable output must declare its own content digest before the production
    ends, since a later run's key reads it from here rather than re-hashing.
    """

    output: Reference
    directory: Path
    content_digest: str | None = field(default=None)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def produce(
    roots: WorkspaceRoots,
    output: Reference,
    *,
    inputs: Mapping[str, ConsumedReference],
    parameters: Mapping[str, object],
    invocation: Sequence[str],
    defaults: Mapping[str, object] | None = None,
    key: str | None = None,
) -> Iterator[Production]:
    """Stage a write to `output`, and on success record it and put it in place.

    Refuses before any work when `output` is a selection or an input's location
    does not exist. An immutable output that another producer finished first is
    left as it is, since its reference names the same content; a mutable one,
    or a location left with no record, is replaced.

    The record's key is `key` when the producer already derives its output's
    identity itself, such as a blocking run keyed by the rows it consumed, so
    the record and the location it names agree; otherwise it is
    `reference_key` over `inputs` and `parameters`.
    """
    if not output.complete:
        raise ProductionError(
            f"{output.uri} is a selection; a production writes one reference."
        )
    missing = [
        consumed.reference.uri
        for consumed in inputs.values()
        if not locate(roots, consumed.reference).is_dir()
    ]
    if missing:
        raise ProductionError(f"{output.uri} consumes {missing}, which do not exist.")
    if key is None:
        key = reference_key(inputs, parameters, defaults=defaults)
    started_at = _now()
    location = locate(roots, output)
    location.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{location.name}.", dir=location.parent))
    production = Production(output=output, directory=staging)
    try:
        yield production
        if not output.immutable and production.content_digest is None:
            raise ProductionError(
                f"{output.uri} is mutable, so its production must declare its content digest."
            )
        if output.immutable and production.content_digest is not None:
            raise ProductionError(
                f"{output.uri} is immutable and names its content; it declares no digest."
            )
        record = Record(
            uri=output.uri,
            key=key,
            inputs={
                role: ConsumedInput(
                    uri=consumed.reference.uri, content_digest=consumed.content_digest
                )
                for role, consumed in inputs.items()
            },
            parameters=dict(parameters),
            invocation=tuple(invocation),
            roots=roots.to_manifest(),
            commit=current_commit(roots.checkout),
            started_at=started_at,
            finished_at=_now(),
            content_digest=production.content_digest,
        )
        (staging / RECORD_FILENAME).write_text(record.to_json(), encoding="utf-8")
        _put_in_place(staging, location, replace=not output.immutable)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _put_in_place(staging: Path, location: Path, *, replace: bool) -> None:
    if location.exists() and (replace or not (location / RECORD_FILENAME).is_file()):
        aside = location.parent / f".{location.name}.replaced.{uuid.uuid4().hex}"
        location.rename(aside)
        try:
            staging.rename(location)
        finally:
            shutil.rmtree(aside, ignore_errors=True)
        return
    try:
        staging.rename(location)
    except OSError:
        # Another producer finished the same immutable reference first:
        # Windows reports that as FileExistsError, POSIX as ENOTEMPTY.
        if not (location / RECORD_FILENAME).is_file():
            raise


def read_record(roots: WorkspaceRoots, ref: Reference) -> Record | None:
    """The record of a complete reference, or None when its location is incomplete."""
    path = locate(roots, ref) / RECORD_FILENAME
    if not path.is_file():
        return None
    record = Record.from_json(path.read_text(encoding="utf-8"))
    return record if record.uri == ref.uri else None


def iter_records(roots: WorkspaceRoots) -> Iterator[Record]:
    """Every record under every registered side root that its own URI locates."""
    sides = dict.fromkeys((layout.kind, layout.side) for layout in registered_layouts())
    for kind, side in sides:
        base = side_root(roots, kind, side)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob(RECORD_FILENAME)):
            try:
                found = reference_at(roots, path.parent)
            except InvalidReferenceError:
                continue
            record = Record.from_json(path.read_text(encoding="utf-8"))
            if record.uri == found.uri:
                yield record


def open_catalog(roots: WorkspaceRoots) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection holding every record as three tables.

    - `records(uri, kind, side, key, commit, started_at, finished_at, content_digest, parameters, invocation)`
    - `segments(uri, name, value)`, one row per segment of each record's reference
    - `inputs(uri, role, input_uri, content_digest)`, one row per consumed reference
    """
    record_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    for record in iter_records(roots):
        ref = record.reference
        record_rows.append(
            {
                "uri": record.uri,
                "kind": ref.kind.value,
                "side": ref.side.value,
                "key": record.key,
                "commit": record.commit,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "content_digest": record.content_digest,
                "parameters": json.dumps(
                    dict(record.parameters), sort_keys=True, default=str
                ),
                "invocation": json.dumps(list(record.invocation)),
            }
        )
        segment_rows.extend(
            {"uri": record.uri, "name": name, "value": value}
            for name, value in ref.values
        )
        input_rows.extend(
            {
                "uri": record.uri,
                "role": role,
                "input_uri": consumed.uri,
                "content_digest": consumed.content_digest,
            }
            for role, consumed in record.inputs.items()
        )
    connection = duckdb.connect()
    for name, rows, columns in (
        ("records", record_rows, _RECORD_COLUMNS),
        ("segments", segment_rows, _SEGMENT_COLUMNS),
        ("inputs", input_rows, _INPUT_COLUMNS),
    ):
        column_list = ", ".join(f"{column} VARCHAR" for column in columns)
        connection.execute(f"CREATE TABLE {name} ({column_list})")  # nosec B608 - fixed names
        if rows:
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {name} VALUES ({placeholders})",  # nosec B608 - fixed names
                [[row[column] for column in columns] for row in rows],
            )
    return connection


_RECORD_COLUMNS = (
    "uri",
    "kind",
    "side",
    "key",
    "commit",
    "started_at",
    "finished_at",
    "content_digest",
    "parameters",
    "invocation",
)
_SEGMENT_COLUMNS = ("uri", "name", "value")
_INPUT_COLUMNS = ("uri", "role", "input_uri", "content_digest")


def select_recorded(
    connection: duckdb.DuckDBPyConnection, **segments: str
) -> list[str]:
    """The URIs of every record whose reference has all the given segment values,
    whatever kind or side, in URI order."""
    if not segments:
        return [
            row[0]
            for row in connection.execute(
                "SELECT uri FROM records ORDER BY uri"
            ).fetchall()
        ]
    clauses = " OR ".join("(name = ? AND value = ?)" for _ in segments)
    arguments: list[str] = [part for item in segments.items() for part in item]
    rows = connection.execute(
        f"SELECT uri FROM segments WHERE {clauses} "  # nosec B608 - placeholders only
        "GROUP BY uri HAVING count(DISTINCT name) = ? ORDER BY uri",
        [*arguments, len(segments)],
    ).fetchall()
    return [row[0] for row in rows]


def consumers_of(connection: duckdb.DuckDBPyConnection, ref: Reference) -> list[str]:
    """The URIs of every record that consumed `ref`, in URI order."""
    rows = connection.execute(
        "SELECT DISTINCT uri FROM inputs WHERE input_uri = ? ORDER BY uri", [ref.uri]
    ).fetchall()
    return [row[0] for row in rows]


__all__ = [
    "RECORD_FILENAME",
    "ConsumedInput",
    "Production",
    "ProductionError",
    "Record",
    "consumers_of",
    "iter_records",
    "open_catalog",
    "produce",
    "read_record",
    "select_recorded",
]
