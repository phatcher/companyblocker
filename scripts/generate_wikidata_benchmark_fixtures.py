"""Cut a deterministic sample of a Wikidata dump into paired `.bz2` and `.gz` fixtures with a checksum metadata file.

The pair under `src/tests/acquisition/fixtures/` lets `scripts/benchmark_wikidata_line_filter.py` compare readers on identical lines without touching a live dump.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap
from cli_common import resolve_existing_path_and_rows, run_reporting_argument_errors
from company_tokenize import to_portable_path_str

from acquisition.wikidata_pipeline_helpers import iter_clean_wikidata_json_lines

REPO_ROOT = _bootstrap.REPO_ROOT


def _sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_fixtures(
    *, source_path: Path, fixture_base: Path, rows: int, project_root: Path = REPO_ROOT
) -> dict[str, object]:
    fixture_base.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    uncompressed_bytes = 0
    canonical_hasher = hashlib.sha256()

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        for line in iter_clean_wikidata_json_lines(source_path):
            json.loads(line)
            payload = (line + "\n").encode("utf-8")
            tmp_file.write(line)
            tmp_file.write("\n")
            canonical_hasher.update(payload)
            uncompressed_bytes += len(payload)
            line_count += 1
            if line_count >= rows:
                break

    if line_count == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("No rows extracted from source")

    bz2_path = fixture_base.with_suffix(".jsonl.bz2")
    gz_path = fixture_base.with_suffix(".jsonl.gz")

    with tmp_path.open("rb") as src, bz2.open(bz2_path, "wb", compresslevel=9) as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    with (
        tmp_path.open("rb") as src,
        gz_path.open("wb") as gz_handle,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=gz_handle, compresslevel=9, mtime=0
        ) as dst,
    ):
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)

    tmp_path.unlink(missing_ok=True)

    metadata = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_path": to_portable_path_str(source_path, project_root=project_root),
        "rows": line_count,
        "uncompressed_bytes": uncompressed_bytes,
        "canonical_jsonl_sha256": canonical_hasher.hexdigest(),
        "fixtures": {
            bz2_path.name: {
                "path": to_portable_path_str(bz2_path, project_root=project_root),
                "bytes": bz2_path.stat().st_size,
                "sha256": _sha256_path(bz2_path),
            },
            gz_path.name: {
                "path": to_portable_path_str(gz_path, project_root=project_root),
                "bytes": gz_path.stat().st_size,
                "sha256": _sha256_path(gz_path),
            },
        },
    }
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate permanent Wikidata benchmark fixtures (.bz2 and .gz) with identical content"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to live/source Wikidata dump (.bz2/.gz/.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="src/tests/acquisition/fixtures",
        help="Directory to place generated fixture files",
    )
    parser.add_argument(
        "--name",
        default="wikidata-benchmark-sample",
        help="Fixture base name (without extensions)",
    )
    parser.add_argument(
        "--rows", type=int, default=50000, help="Rows to extract into fixture"
    )
    args = parser.parse_args()

    source_path, rows = resolve_existing_path_and_rows(
        source=args.source, rows=args.rows
    )

    output_dir = Path(args.output_dir)
    fixture_base = output_dir / args.name
    metadata_path = output_dir / f"{args.name}.metadata.json"

    metadata = generate_fixtures(
        source_path=source_path, fixture_base=fixture_base, rows=rows
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated fixture rows: {metadata['rows']}")
    print(f"Wrote: {fixture_base.with_suffix('.jsonl.bz2')}")
    print(f"Wrote: {fixture_base.with_suffix('.jsonl.gz')}")
    print(f"Wrote metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
