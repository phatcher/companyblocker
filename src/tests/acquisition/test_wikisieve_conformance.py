"""This repository's conformance test for the deployed `wikisieve` binary: the
real production spec over the tracked benchmark sample must reproduce the
checked-in expected output, record for record, and a prepare run stopped part
way and resumed must write the same extract as one that ran straight through.

The crate's own tests are wikisieve's; this one holds the tool to this
repository's spec and sample, and stays here when the crate moves out. A
deliberate spec change regenerates the expected output with
`scripts/compare_wikidata_rust_wikisieve.py --write-reference` and commits
the reviewed diff (see that script's docstring).
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pytest

from acquisition.downloader_wikidata import extract_wikidata_company_projection_two_pass
from acquisition.wikidata_equivalence import (
    compare_projected_record_sets,
    load_jsonl_records_by_id,
    run_wikisieve,
    write_resolved_spec,
)
from workspace.reference_inputs import wikidata_legal_form_closure
from workspace.roots import default_workspace_roots

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURES / "wikidata-benchmark-sample.jsonl.gz"
EXPECTED_OUTPUT = FIXTURES / "wikisieve-expected-output.jsonl"
SPEC = (
    REPO_ROOT
    / "src"
    / "acquisition"
    / "catalog"
    / "projections"
    / "wikidata-company.json"
)
DEPLOYED_BINARY = REPO_ROOT / "tools" / "bin" / "wikisieve.exe"
BUILT_BINARY = (
    REPO_ROOT / "src" / "rust" / "wikisieve" / "target" / "release" / "wikisieve.exe"
)


def _wikisieve_binary() -> Path:
    """The deployed binary, else the crate's release build, else skip: the
    binary is not tracked, and a checkout without one has nothing to conform."""
    for candidate in (DEPLOYED_BINARY, BUILT_BINARY):
        if candidate.is_file():
            return candidate
    pytest.skip(f"no wikisieve binary at {DEPLOYED_BINARY} or {BUILT_BINARY}")


@pytest.mark.integration
def test_deployed_wikisieve_reproduces_the_expected_output_over_the_sample(
    tmp_path: Path,
) -> None:
    binary = _wikisieve_binary()
    resolved_spec = tmp_path / "wikisieve-spec.json"
    write_resolved_spec(
        spec_path=SPEC,
        p279_path=wikidata_legal_form_closure(default_workspace_roots(REPO_ROOT)),
        destination_path=resolved_spec,
    )
    output = tmp_path / "wikisieve-projection.jsonl"
    emitted, _ = run_wikisieve(
        wikisieve_binary=binary,
        source_path=SAMPLE,
        spec_path=resolved_spec,
        destination_path=output,
        summary_path=tmp_path / "wikisieve-summary.json",
        max_rows=None,
    )

    expected = load_jsonl_records_by_id(EXPECTED_OUTPUT)
    actual = load_jsonl_records_by_id(output)
    result = compare_projected_record_sets(expected, actual)

    assert emitted == len(expected)
    assert result.is_equivalent, (
        f"missing={result.missing_ids[:5]} extra={result.extra_ids[:5]} "
        f"mismatches={[(m.record_id, m.differing_fields) for m in result.mismatches[:5]]}"
    )


def _prepare_run(run_dir: Path, binary: Path) -> tuple[Path, Path, dict[str, object]]:
    """The sample and its p279.json side by side in `run_dir`, as a prepare run
    finds a dump, plus the destination and settings for the deployed binary."""
    run_dir.mkdir()
    source = run_dir / SAMPLE.name
    shutil.copyfile(SAMPLE, source)
    shutil.copyfile(
        wikidata_legal_form_closure(default_workspace_roots(REPO_ROOT)),
        run_dir / "p279.json",
    )
    defaults: dict[str, object] = {
        "engine": "wikisieve",
        "binary_path": str(binary),
        "spec_path": str(SPEC),
    }
    return source, run_dir / "wikidata-companies.jsonl", defaults


def _read_gz_text(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return handle.read().decode("utf-8")


@pytest.mark.integration
def test_a_stopped_and_resumed_prepare_run_matches_an_uninterrupted_one(
    tmp_path: Path,
) -> None:
    binary = _wikisieve_binary()

    flat_source, flat_destination, defaults = _prepare_run(tmp_path / "flat", binary)
    flat_rows = extract_wikidata_company_projection_two_pass(
        flat_source, flat_destination, stream_resume=False, projection_defaults=defaults
    )

    source, destination, defaults = _prepare_run(tmp_path / "resumed", binary)
    # Stopped part way: a bounded run keeps its chunks and writes no extract.
    extract_wikidata_company_projection_two_pass(
        source,
        destination,
        max_lines=4_000,
        stream_resume=True,
        projection_defaults=defaults,
    )
    assert not destination.exists()
    resumed_rows = extract_wikidata_company_projection_two_pass(
        source, destination, stream_resume=True, projection_defaults=defaults
    )

    assert resumed_rows == flat_rows == len(load_jsonl_records_by_id(EXPECTED_OUTPUT))
    assert destination.read_bytes() == flat_destination.read_bytes()
    raw_cache = "wikidata-companies-raw.jsonl.gz"
    assert _read_gz_text(destination.with_name(raw_cache)) == _read_gz_text(
        flat_destination.with_name(raw_cache)
    )
    assert not destination.with_name("wikidata-companies.chunks").exists()
