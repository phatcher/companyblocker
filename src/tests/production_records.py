"""The production record a test writes when it lays a finished output out by hand.

A location is complete exactly when it holds its record
(`workspace.records`), so a test that builds a run directory file by file,
rather than producing it, marks it finished through here. The record names
nothing consumed: a test that cares what a record says produces its output
through `workspace.records.produce` instead.
"""

from __future__ import annotations

from pathlib import Path

from workspace.records import RECORD_FILENAME, Record
from workspace.reference import reference_at
from workspace.roots import WorkspaceRoots


def write_record(roots: WorkspaceRoots, directory: Path) -> None:
    """Mark `directory`, a complete reference's location, as finished."""
    ref = reference_at(roots, directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = Record(
        uri=ref.uri,
        key=ref.fields.get("key", ""),
        inputs={},
        parameters={},
        invocation=(),
        roots=roots.to_manifest(),
        commit=None,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:00+00:00",
        content_digest=None,
    )
    (directory / RECORD_FILENAME).write_text(record.to_json(), encoding="utf-8")
