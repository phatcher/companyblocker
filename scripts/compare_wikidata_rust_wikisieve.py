"""Diff a `wikisieve` run against a stored reference output, record by record.

Runs `wikisieve` (`tools/bin/wikisieve.exe` by default, or `--wikisieve-binary`) with the
production spec over an input, then diffs every field of every record against `--reference`,
a JSONL file this repository keeps rather than an engine it runs. The default reference is
`src/tests/acquisition/fixtures/wikisieve-expected-output.jsonl`, the reviewed output of the
production spec over the tracked 10,000-row benchmark sample, which
`src/tests/acquisition/test_wikisieve_conformance.py` holds the deployed binary to on every
suite run. This script is the same check by hand, at any scale: a full-dump check passes the
real acquired snapshot as `--input` and the stored full extract as `--reference`, with
`--report-json`/`--keep-output-dir` so results survive the process.

The name is historical: it once ran the hardcoded Rust extractor as the other side. That
extractor is being removed, and the reference is a file so nothing here needs it built.

A deliberate change to the production spec or the projection regenerates the default reference
with `--write-reference`, and commits it with the diff reviewed field by field: the test
passing against a reference nobody read proves nothing.

The full-scale check runs the acquired dump against the stored extract of the same dump,
`--input data/wikidata/acquire/2026-07-16/wikidata-all.json.gz --reference
data/wikidata/reference/2026-07-16/wikidata-companies.jsonl --report-json <path>`. That extract
is the one the hardcoded Rust extractor made, kept under `reference/`, where no prepare run
writes.
"""
# isort: skip_file

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import _bootstrap
from cli_common import run_reporting_argument_errors

from acquisition.wikidata_equivalence import (
    compare_projected_record_sets,
    load_jsonl_records_by_id,
    run_wikisieve,
    write_resolved_spec,
)
from workspace.reference_inputs import wikidata_legal_form_closure
from workspace.roots import default_workspace_roots

REPO_ROOT = _bootstrap.REPO_ROOT
REPO_ROOTS = default_workspace_roots(REPO_ROOT)
DEFAULT_INPUT = (
    REPO_ROOT
    / "src"
    / "tests"
    / "acquisition"
    / "fixtures"
    / "wikidata-benchmark-sample.jsonl.gz"
)
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "src"
    / "tests"
    / "acquisition"
    / "fixtures"
    / "wikisieve-expected-output.jsonl"
)
DEFAULT_P279 = wikidata_legal_form_closure(REPO_ROOTS)
DEFAULT_WIKISIEVE_BINARY = REPO_ROOT / "tools" / "bin" / "wikisieve.exe"
DEFAULT_WIKISIEVE_SPEC = (
    REPO_ROOT
    / "src"
    / "acquisition"
    / "catalog"
    / "projections"
    / "wikidata-company.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diff a wikisieve run against a stored reference output"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="The JSONL output the run is diffed against (default: the reviewed "
        "expected output over the tracked benchmark sample).",
    )
    parser.add_argument(
        "--write-reference",
        action="store_true",
        help="Write the run's output to --reference instead of diffing against it, "
        "for a deliberate spec or projection change; review the diff before committing.",
    )
    parser.add_argument("--p279", type=Path, default=DEFAULT_P279)
    parser.add_argument(
        "--wikisieve-binary", type=Path, default=DEFAULT_WIKISIEVE_BINARY
    )
    parser.add_argument("--wikisieve-spec", type=Path, default=DEFAULT_WIKISIEVE_SPEC)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--keep-output-dir", type=Path)
    parser.add_argument("--show-diff-limit", type=int, default=10)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help=(
            "Persist the full comparison report (counts, missing/extra ids, and a sample "
            "of mismatching records) to this path instead of leaving it only in stdout."
        ),
    )
    parser.add_argument(
        "--report-mismatch-detail-limit",
        type=int,
        default=200,
        help="Max number of mismatches to include with full expected/actual payloads in --report-json.",
    )
    return parser


def _print_result_summary(
    result,
    *,
    reference_path: Path,
    wikisieve_rows: int,
    wikisieve_stdout: str,
    show_diff_limit: int,
) -> None:
    print(f"Reference:                    {reference_path}")
    print(f"Reference records:            {result.expected_count:,}")
    print(f"wikisieve rows emitted:       {wikisieve_rows:,}")
    print(f"wikisieve records:            {result.actual_count:,}")
    print(f"Missing ids:                  {len(result.missing_ids):,}")
    print(f"Extra ids:                    {len(result.extra_ids):,}")
    print(f"Content mismatches:           {len(result.mismatches):,}")

    wikisieve_summary_lines = [
        line for line in wikisieve_stdout.splitlines() if line.strip()
    ]
    if wikisieve_summary_lines:
        print("wikisieve output tail:")
        for line in wikisieve_summary_lines[-5:]:
            print(f"  {line}")

    if result.missing_ids:
        print("First missing ids:")
        for record_id in result.missing_ids[:show_diff_limit]:
            print(f"  {record_id}")
    if result.extra_ids:
        print("First extra ids:")
        for record_id in result.extra_ids[:show_diff_limit]:
            print(f"  {record_id}")
    if result.mismatches:
        print("First content mismatches:")
        for mismatch in result.mismatches[:show_diff_limit]:
            fields = ", ".join(mismatch.differing_fields)
            print(f"  {mismatch.record_id}: {fields}")


def _write_comparison_report(
    report_path: Path,
    *,
    result,
    reference_path: Path,
    wikisieve_rows: int,
    wikisieve_manifest_path: Path,
    detail_limit: int,
) -> None:
    """Persist the full comparison report to disk.

    `_print_result_summary` only ever went to stdout; a full-dump run can plausibly surface
    far more mismatches than the sample-scale check this script runs by default, so this
    keeps the full missing/extra id lists and every mismatch's `record_id`/`differing_fields`
    (cheap, string-only), plus a capped sample of full expected/actual payloads for the
    mismatches (`--report-mismatch-detail-limit`) so an investigation doesn't require
    re-running the whole comparison.
    """
    report = {
        "is_equivalent": result.is_equivalent,
        "reference_path": str(reference_path),
        "wikisieve_rows_emitted": wikisieve_rows,
        "expected_count": result.expected_count,
        "actual_count": result.actual_count,
        "missing_ids_count": len(result.missing_ids),
        "extra_ids_count": len(result.extra_ids),
        "mismatch_count": len(result.mismatches),
        "missing_ids": list(result.missing_ids),
        "extra_ids": list(result.extra_ids),
        "mismatches_summary": [
            {"record_id": m.record_id, "differing_fields": list(m.differing_fields)}
            for m in result.mismatches
        ],
        "mismatches_detail_sample": [
            asdict(m) for m in result.mismatches[:detail_limit]
        ],
        "wikisieve_manifest_path": str(wikisieve_manifest_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")
    if not args.p279.exists():
        parser.error(f"p279 file not found: {args.p279}")
    if not args.wikisieve_binary.exists():
        parser.error(f"wikisieve binary not found: {args.wikisieve_binary}")
    if not args.wikisieve_spec.exists():
        parser.error(f"wikisieve spec not found: {args.wikisieve_spec}")
    if not args.write_reference and not args.reference.exists():
        parser.error(f"reference file not found: {args.reference}")

    keep_output_dir = args.keep_output_dir
    temp_dir = None
    if keep_output_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="wikisieve-conformance-")
        work_dir = Path(temp_dir.name)
    else:
        work_dir = keep_output_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved_spec_path = work_dir / "wikisieve-spec.json"
        write_resolved_spec(
            spec_path=args.wikisieve_spec,
            p279_path=args.p279,
            destination_path=resolved_spec_path,
        )

        wikisieve_output_path = work_dir / "wikisieve-projection.jsonl"
        wikisieve_summary_path = work_dir / "wikisieve-summary.json"
        wikisieve_manifest_path = work_dir / "wikisieve-manifest.json"

        print("Running wikisieve...")
        wikisieve_rows, wikisieve_stdout = run_wikisieve(
            wikisieve_binary=args.wikisieve_binary,
            source_path=args.input,
            spec_path=resolved_spec_path,
            destination_path=wikisieve_output_path,
            summary_path=wikisieve_summary_path,
            max_rows=args.max_rows,
            manifest_path=wikisieve_manifest_path,
        )

        if args.write_reference:
            args.reference.parent.mkdir(parents=True, exist_ok=True)
            args.reference.write_bytes(wikisieve_output_path.read_bytes())
            print(
                f"Reference written: {args.reference} ({wikisieve_rows:,} records); "
                "review its diff before committing."
            )
            return 0

        reference_records = load_jsonl_records_by_id(args.reference)
        wikisieve_records = load_jsonl_records_by_id(wikisieve_output_path)
        result = compare_projected_record_sets(reference_records, wikisieve_records)
        _print_result_summary(
            result,
            reference_path=args.reference,
            wikisieve_rows=wikisieve_rows,
            wikisieve_stdout=wikisieve_stdout,
            show_diff_limit=args.show_diff_limit,
        )

        if keep_output_dir is not None:
            print(f"Outputs kept in {keep_output_dir}")
            print(f"Run manifest: {wikisieve_manifest_path}")

        if args.report_json is not None:
            _write_comparison_report(
                args.report_json,
                result=result,
                reference_path=args.reference,
                wikisieve_rows=wikisieve_rows,
                wikisieve_manifest_path=wikisieve_manifest_path,
                detail_limit=args.report_mismatch_detail_limit,
            )
            print(f"Comparison report written to {args.report_json}")

        if not result.is_equivalent:
            return 1
        print("Conformance check passed: every record matches the reference.")
        return 0
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
