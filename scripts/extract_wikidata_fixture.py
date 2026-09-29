from __future__ import annotations

import argparse
import json
from pathlib import Path

from acquisition.wikidata_pipeline_helpers import iter_clean_wikidata_json_lines
from scripts.cli_common import resolve_existing_path_and_rows


def _iter_wikidata_json_lines(source: Path):
    yield from iter_clean_wikidata_json_lines(source)


def extract_fixture(source: Path, output: Path, max_rows: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8", newline="\n") as out:
        for line in _iter_wikidata_json_lines(source):
            # Validate JSON shape while preserving original payload text.
            json.loads(line)
            out.write(line)
            out.write("\n")
            rows_written += 1
            if rows_written >= max_rows:
                break
    return rows_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a small Wikidata JSONL fixture from a local dump without re-downloading.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to local Wikidata .json.bz2 or .jsonl file",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output JSONL fixture file"
    )
    parser.add_argument(
        "--rows", type=int, default=20, help="Maximum rows to write (default: 20)"
    )
    args = parser.parse_args()

    source, rows = resolve_existing_path_and_rows(source=args.source, rows=args.rows)
    output = Path(args.output)

    written = extract_fixture(source, output, rows)
    print(f"Wrote {written} row(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
