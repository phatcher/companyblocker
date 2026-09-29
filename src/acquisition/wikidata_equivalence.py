from __future__ import annotations

import json
import subprocess  # nosec B404 - runs the deployed wikisieve binary with a fixed argv
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RecordMismatch:
    record_id: str
    differing_fields: tuple[str, ...]
    expected: Mapping[str, object]
    actual: Mapping[str, object]


@dataclass(frozen=True)
class EquivalenceResult:
    expected_count: int
    actual_count: int
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    mismatches: tuple[RecordMismatch, ...]

    @property
    def is_equivalent(self) -> bool:
        return not self.missing_ids and not self.extra_ids and not self.mismatches


def load_jsonl_records_by_id(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"{path}: line {line_number} must decode to an object")

            record_id = payload.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"{path}: line {line_number} missing non-empty string id"
                )
            if record_id in records:
                raise ValueError(
                    f"{path}: duplicate record id '{record_id}' on line {line_number}"
                )

            records[record_id] = payload

    return records


def compare_projected_record_sets(
    # `Mapping`, not `dict`: both are only read here, and `dict`'s value type
    # is invariant, so a caller's `dict[str, dict[str, str]]` would not fit.
    expected: Mapping[str, Mapping[str, object]],
    actual: Mapping[str, Mapping[str, object]],
) -> EquivalenceResult:
    expected_ids = set(expected)
    actual_ids = set(actual)

    missing_ids = tuple(sorted(expected_ids - actual_ids))
    extra_ids = tuple(sorted(actual_ids - expected_ids))

    mismatches: list[RecordMismatch] = []
    for record_id in sorted(expected_ids & actual_ids):
        expected_record = expected[record_id]
        actual_record = actual[record_id]
        if expected_record == actual_record:
            continue

        differing_fields = tuple(
            sorted(
                field_name
                for field_name in set(expected_record) | set(actual_record)
                if expected_record.get(field_name) != actual_record.get(field_name)
            )
        )
        mismatches.append(
            RecordMismatch(
                record_id=record_id,
                differing_fields=differing_fields,
                expected=expected_record,
                actual=actual_record,
            )
        )

    return EquivalenceResult(
        expected_count=len(expected),
        actual_count=len(actual),
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        mismatches=tuple(mismatches),
    )


def write_resolved_spec(
    *, spec_path: Path, p279_path: Path, destination_path: Path
) -> None:
    """Copy a projection spec with every `qid_closure_file` marker pointed at
    `p279_path`, absolute, since the tracked spec carries a bare relative name
    that only resolves from the pipeline's own working directory."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for marker in spec.get("markers", []):
        match = marker.get("match", {})
        if match.get("type") == "qid_closure_file":
            match["path"] = str(p279_path.resolve())
    destination_path.write_text(json.dumps(spec), encoding="utf-8")


def run_wikisieve(
    *,
    wikisieve_binary: Path,
    source_path: Path,
    spec_path: Path,
    destination_path: Path,
    summary_path: Path,
    max_rows: int | None,
    manifest_path: Path | None = None,
) -> tuple[int, str]:
    """Run `wikisieve` over `source_path` with `spec_path`, writing JSONL to
    `destination_path`, and return the records emitted with the run's stdout.

    The binary's own `--summary-json` carries no timing, so wall-clock is
    measured around the call and, with `manifest_path`, written beside the
    summary as `<engine>-manifest.json`.
    """
    command = [
        str(wikisieve_binary),
        "--input",
        str(source_path),
        "--spec",
        str(spec_path),
        "--output",
        str(destination_path),
        "--summary-json",
        str(summary_path),
        "--output-mode",
        "jsonl",
    ]
    if max_rows is not None:
        command.extend(["--max-rows", str(max_rows)])

    started_utc = datetime.now(UTC)
    started_perf = time.perf_counter()
    # A fixed argv, the binary plus its own flags; not shell=True, and nothing
    # untrusted reaches the command line.
    completed = subprocess.run(  # nosec B603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed_seconds = time.perf_counter() - started_perf
    finished_utc = datetime.now(UTC)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[0]} failed with exit code {completed.returncode}:\n{completed.stdout}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    emitted = summary.get("records_emitted")
    if not isinstance(emitted, int):
        raise TypeError(f"summary missing integer records_emitted: {summary_path}")

    if manifest_path is not None:

        def _rate(count: object) -> float | None:
            if not isinstance(count, int) or elapsed_seconds <= 0:
                return None
            return count / elapsed_seconds

        manifest = {
            "engine": "wikisieve",
            "command": command,
            "started_utc": started_utc.isoformat(),
            "finished_utc": finished_utc.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "dump_input_path": str(source_path.resolve()),
            "dump_input_bytes": source_path.stat().st_size,
            "summary": summary,
            "lines_per_sec": _rate(summary.get("lines_scanned")),
            "candidates_per_sec": _rate(summary.get("candidates_scanned")),
            "records_per_sec": _rate(summary.get("records_emitted")),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return emitted, completed.stdout
