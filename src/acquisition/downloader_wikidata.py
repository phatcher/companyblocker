"""Project Wikidata's gzipped entity dump to the company extract the Shard stage reads.

Two engines, chosen by the resource's catalog metadata: `wikisieve`, the default, a spec-driven Rust extractor deployed separately to `tools/bin` and not compiled on demand, and a native Python path. The spec is `catalog/projections/wikidata-company.json`.

The extract is `data/wikidata/prepare/<snapshot>/wikidata-companies.jsonl`, reused on later runs while non-empty. A failed extract never removes or truncates the live extract or its raw-candidate cache: a flat `wikisieve` run writes `.tmp` files renamed over the live ones only after the process exits 0 and the staged row count matches its summary, and a resumable run writes chunks that `wikisieve merge-chunks` assembles by the same check against the summary and the resume state. A resume keeps only the chunks its last checkpoint covers, so a run killed between sealing a chunk and saving its checkpoint neither loses nor repeats rows.

Every `wikisieve` run writes `wikidata-run-manifest.json` beside the extract: the binary's path and sha256, and the version, commit and dirty flag it reported, null for a binary built without that provenance.
"""

from __future__ import annotations

# isort: skip_file
# fmt: off
import functools
import hashlib
import json
import re
import subprocess  # nosec B404 - runs the deployed wikisieve binary; see nosec B603 below
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from acquisition.io_eta_helpers import (_READ_CHUNK_SIZE_BYTES,
                                        _estimate_stream_eta,
                                        _format_eta_band_and_stability,
                                        _indexed_bzip2, _json_loads,
                                        _open_eta_enabled_source)
from acquisition.wikidata_pipeline_helpers import \
    _is_wikidata_company_candidate_raw_line as \
    _is_wikidata_company_candidate_raw_line_impl
from acquisition.wikidata_pipeline_helpers import \
    _iter_wikidata_content_lines_from_handle
from acquisition.wikidata_pipeline_helpers import \
    extract_wikidata_company_projection_two_pass as \
    _extract_wikidata_company_projection_two_pass_impl
from acquisition.wikidata_projection_helpers import \
    _extract_claim_text_values_with_english as \
    _extract_claim_text_values_with_english_impl
from acquisition.wikidata_projection_helpers import \
    _project_wikidata_company_record_line
from acquisition.wikidata_runtime import (WikidataProgressEmitter,
                                          build_wikidata_reader_strategy)

# fmt: on

_WIKIDATA_PROJECTION_ENGINES = {"python", "wikisieve"}
_WIKIDATA_DEFAULT_CHUNK_DIR_NAME = "wikidata-companies.chunks"
_WIKIDATA_DEFAULT_CHUNK_PREFIX = "wikidata-companies-part-"
_WIKIDATA_DEFAULT_RESUME_STATE_NAME = "wikidata-rust-resume-state.json"
_WIKIDATA_DEFAULT_WIKISIEVE_BINARY_PATH = "tools/bin/wikisieve.exe"
_WIKIDATA_DEFAULT_WIKISIEVE_SPEC_PATH = (
    "src/acquisition/catalog/projections/wikidata-company.json"
)
_WIKIDATA_DEFAULT_RAW_CANDIDATE_CACHE_NAME = "wikidata-companies-raw.jsonl.gz"
_WIKIDATA_DEFAULT_RUN_MANIFEST_NAME = "wikidata-run-manifest.json"
_RUN_MANIFEST_HASH_CHUNK_SIZE = 1024 * 1024


def _normalize_wikidata_projection_engine(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in _WIKIDATA_PROJECTION_ENGINES:
        valid_modes = ", ".join(sorted(_WIKIDATA_PROJECTION_ENGINES))
        raise ValueError(f"wikidata projection engine must be one of: {valid_modes}")
    return normalized


def _resolve_wikidata_projection_defaults(
    projection_defaults: Mapping[str, object] | None,
) -> tuple[str, Path, str, Path]:
    defaults = projection_defaults or {}
    engine = defaults.get("engine", "python")

    if not isinstance(engine, str):
        raise TypeError("wikidata projection engine must be a string when provided")
    normalized_engine = _normalize_wikidata_projection_engine(engine)

    # An explicit binary_path in projection_defaults always wins.
    binary_path = defaults.get("binary_path", _WIKIDATA_DEFAULT_WIKISIEVE_BINARY_PATH)
    if not isinstance(binary_path, str) or not binary_path.strip():
        raise ValueError(
            "wikidata projection binary_path must be a non-empty string when provided"
        )
    normalized_binary_path = Path(binary_path)
    if not normalized_binary_path.is_absolute():
        normalized_binary_path = Path.cwd() / normalized_binary_path

    output_mode = defaults.get("output_mode", "jsonl")
    if not isinstance(output_mode, str) or output_mode not in {"jsonl", "ids"}:
        raise ValueError(
            "wikidata projection output_mode must be 'jsonl' or 'ids' when provided"
        )

    # spec_path only matters for wikisieve (python has no spec file), but is
    # always resolved so callers get a consistent 4-tuple regardless of engine.
    spec_path = defaults.get("spec_path", _WIKIDATA_DEFAULT_WIKISIEVE_SPEC_PATH)
    if not isinstance(spec_path, str) or not spec_path.strip():
        raise ValueError(
            "wikidata projection spec_path must be a non-empty string when provided"
        )
    normalized_spec_path = Path(spec_path)
    if not normalized_spec_path.is_absolute():
        normalized_spec_path = Path.cwd() / normalized_spec_path

    return normalized_engine, normalized_binary_path, output_mode, normalized_spec_path


def _resolve_wikidata_raw_candidate_cache_path(
    projection_defaults: Mapping[str, object] | None,
    *,
    destination_path: Path,
) -> Path | None:
    """Resolves where the wikisieve run's raw-candidate cache lands, or None to
    capture nothing.

    On by default, and the storage tradeoff behind that default was measured
    rather than assumed: over 19,372 real matched candidates the raw dump line
    behind each emitted row costs about 2.5 KB gzipped against a 700-byte
    projected record, so a full run's cache is roughly 2 GB beside a 612 MB
    projected artifact -- and about 1.4% of the 155 GB source dump that has to
    be kept anyway. Capture costs 4-5% of the extract run's wall time. What it
    buys is that adding a projected field later replays 884K cached lines in
    seconds instead of rescanning 121M dump lines for about an hour.

    `projection_defaults["raw_candidate_cache"]` overrides: False switches
    capture off, and a string names an explicit path instead of the default
    sibling of `destination_path`.
    """
    defaults = projection_defaults or {}
    setting = defaults.get("raw_candidate_cache", True)

    if isinstance(setting, bool):
        if not setting:
            return None
        return destination_path.with_name(_WIKIDATA_DEFAULT_RAW_CANDIDATE_CACHE_NAME)

    if isinstance(setting, str) and setting.strip():
        cache_path = Path(setting)
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        return cache_path

    raise ValueError(
        "wikidata projection raw_candidate_cache must be a bool or a non-empty "
        "path string when provided"
    )


def _resolve_wikidata_projection_resume_defaults(
    *,
    stream_resume: bool,
    destination_path: Path,
) -> tuple[bool, Path, Path, str]:
    """Where a resumable run keeps its chunks and state, beside `destination_path`.

    Each finished chunk is written under a temp name and renamed into place, so a
    finished chunk is durable and a stopped run continues from it; wikisieve drops
    any chunk sealed after its last checkpoint on its own. Resume follows the
    caller's request.
    """
    resume_enabled = stream_resume
    state_path = destination_path.parent / _WIKIDATA_DEFAULT_RESUME_STATE_NAME
    chunk_dir_path = destination_path.parent / _WIKIDATA_DEFAULT_CHUNK_DIR_NAME
    chunk_prefix = _WIKIDATA_DEFAULT_CHUNK_PREFIX

    return (
        resume_enabled,
        state_path,
        chunk_dir_path,
        chunk_prefix,
    )


_WIKISIEVE_PROGRESS_COUNTERS_RE = re.compile(
    r"lines_scanned=(\d+) candidates=(\d+) emitted=(\d+)"
)


def _enrich_wikisieve_progress_line(line: str, *, elapsed_seconds: float) -> str:
    """Append line/emit rates to a wikisieve progress or complete line.

    wikisieve reports raw cumulative counters only (lines_scanned, candidates,
    emitted) -- unlike the native Python engine's progress messages, it has no
    rate or share figure. This computes the same kind of rate by timing arrivals
    of these lines on the Python side, so both engines report comparable stats.
    """
    match = _WIKISIEVE_PROGRESS_COUNTERS_RE.search(line)
    if match is None or elapsed_seconds <= 0:
        return line
    lines_scanned, _candidates, emitted = (int(value) for value in match.groups())
    line_rate = lines_scanned / elapsed_seconds
    emitted_rate = emitted / elapsed_seconds
    company_share = (emitted / lines_scanned * 100) if lines_scanned else 0.0
    return (
        f"{line} elapsed={elapsed_seconds:.0f}s line_rate={line_rate:,.0f}/s "
        f"emitted_rate={emitted_rate:,.1f}/s company_share={company_share:.2f}%"
    )


def _run_wikisieve_process(
    *,
    command: list[str],
    progress: Callable[[str], None] | None,
) -> None:
    """Runs a wikisieve subprocess to completion, forwarding its stderr progress.

    wikisieve writes its real output straight to disk (chunk files or a flat
    file), never through stdout, so stdout is discarded.
    """
    # command is a fixed argv (the deployed wikisieve binary + this module's own flags);
    # not shell=True, no untrusted input reaches the command line.
    process = subprocess.Popen(  # nosec B603
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.stderr is None:
        raise RuntimeError("wikisieve stderr pipe is not available")

    # Progress lines go to stderr while the run lasts, about 79 minutes on the
    # full dump. stderr is drained on its own thread so progress forwards live,
    # and so an unread pipe never fills its OS buffer and stalls the subprocess.
    stderr_lines: list[str] = []
    subprocess_started = time.perf_counter()

    def _drain_stderr() -> None:
        assert process.stderr is not None  # nosec B101 - narrows Optional for mypy
        for raw_line in process.stderr:
            stripped = raw_line.strip()
            if not stripped:
                continue
            stderr_lines.append(stripped)
            if progress is not None:
                elapsed = time.perf_counter() - subprocess_started
                progress(
                    _enrich_wikisieve_progress_line(stripped, elapsed_seconds=elapsed)
                )

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()
    stderr_thread.join()
    returncode = process.wait()

    if returncode != 0:
        step = "merge-chunks" if command[1:2] == ["merge-chunks"] else "projection"
        raise RuntimeError(
            f"wikidata wikisieve {step} failed with exit code "
            f"{returncode}: {' '.join(stderr_lines).strip()}"
        )


def _count_non_empty_jsonl_rows_in_dir(path: Path) -> int:
    total = 0
    for chunk_path in sorted(path.rglob("*.jsonl")):
        if not chunk_path.is_file():
            continue
        with chunk_path.open("rb") as handle:
            total += sum(1 for line in handle if line.strip())
    return total


def _count_non_empty_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _read_records_emitted(summary_path: Path) -> int:
    if not summary_path.exists():
        raise RuntimeError(
            f"wikidata wikisieve projection did not produce summary file: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records_emitted = summary.get("records_emitted")
    if not isinstance(records_emitted, int):
        raise TypeError("wikisieve summary did not include integer records_emitted")
    return records_emitted


def _validate_wikisieve_chunk_summary(
    *, summary_path: Path, chunk_dir_path: Path
) -> int:
    """Validates the completed run and returns its true emitted row count.

    Counts rows across the chunk directory's finalized .jsonl files rather than
    trusting the summary alone. A mismatch never deletes anything: each chunk's
    write-then-rename is atomic, so a finalized chunk is durable data a later
    resumed run can still build on.
    """
    records_emitted = _read_records_emitted(summary_path)
    rows_emitted = _count_non_empty_jsonl_rows_in_dir(chunk_dir_path)
    if rows_emitted != records_emitted:
        raise RuntimeError(
            "wikidata wikisieve emitted row count mismatch: "
            f"chunk_dir={rows_emitted} summary={records_emitted}"
        )
    return rows_emitted


def _staging_path(path: Path) -> Path:
    """`<path>.tmp`, a same-directory sibling, so committing it is an atomic rename."""
    return path.with_name(path.name + ".tmp")


def _build_wikisieve_merge_chunks_command(
    *,
    binary_path: Path,
    chunk_dir_path: Path,
    chunk_prefix: str,
    destination_path: Path,
    summary_path: Path,
    resume_state_path: Path,
    raw_candidate_cache_path: Path | None,
) -> list[str]:
    """Builds the `wikisieve merge-chunks` invocation that assembles a finished
    resumable run into `destination_path`.

    The expected row count comes from both the run's summary and its resume
    state: `merge-chunks` refuses when the two disagree, and refuses to replace
    the destination unless the merged count equals them.
    """
    command = [
        str(binary_path),
        "merge-chunks",
        "--chunk-dir",
        str(chunk_dir_path),
        "--chunk-prefix",
        chunk_prefix,
        "--output",
        str(destination_path),
        "--summary-json",
        str(summary_path),
        "--state-path",
        str(resume_state_path),
    ]
    if raw_candidate_cache_path is not None:
        command.extend(["--raw-candidate-output", str(raw_candidate_cache_path)])
    return command


def _write_resolved_wikisieve_spec(
    *, spec_path: Path, p279_path: Path, destination_path: Path
) -> None:
    """Writes a copy of `spec_path` with its `qid_closure_file` marker path
    resolved to an absolute path pointing at this run's real p279.json.

    wikisieve resolves `qid_closure_file` paths relative to the spec file's
    own directory (`CompiledSpec::load`), and the checked-in
    `src/acquisition/catalog/projections/wikidata-company.json` ships without
    a real p279.json beside it -- the closure file is kept beside the raw dump instead
    (`source_path.with_name("p279.json")`).
    """
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    for marker in spec.get("markers", []):
        match = marker.get("match", {})
        if match.get("type") == "qid_closure_file":
            match["path"] = str(p279_path.resolve())
    destination_path.write_text(json.dumps(spec), encoding="utf-8")


def _build_wikisieve_command(
    *,
    binary_path: Path,
    source_path: Path,
    spec_path: Path,
    summary_path: Path,
    output_mode: str,
    resume_enabled: bool,
    resume_state_path: Path,
    chunk_dir_path: Path,
    chunk_prefix: str,
    destination_path: Path,
    max_lines: int | None,
    max_companies: int | None,
    raw_candidate_cache_path: Path | None = None,
) -> list[str]:
    """Builds the wikisieve CLI invocation (`src/rust/wikisieve/src/cli.rs`).

    Chunked output is active only when `--resume` is passed; a non-resuming run
    gets wikisieve's flat-file sink at `--output <destination_path>`, which the
    caller points at a staging path rather than the live file.

    `--raw-candidate-output` follows that same split, which is why it is passed
    unconditionally when capture is on: it is where the cache lands in flat
    mode, and in chunked mode only the flag's presence matters, since capture
    then goes to a companion beside each chunk and `merge-chunks` assembles them.
    """
    output_target = chunk_dir_path if resume_enabled else destination_path
    command = [
        str(binary_path),
        "--input",
        str(source_path),
        "--spec",
        str(spec_path),
        "--output",
        str(output_target),
        "--summary-json",
        str(summary_path),
        "--output-mode",
        output_mode,
    ]
    if resume_enabled:
        command.extend(
            [
                "--resume",
                "--state-path",
                str(resume_state_path),
                "--chunk-dir",
                str(chunk_dir_path),
                "--chunk-prefix",
                chunk_prefix,
            ]
        )
    if raw_candidate_cache_path is not None:
        command.extend(["--raw-candidate-output", str(raw_candidate_cache_path)])
    if max_lines is not None:
        command.extend(["--max-rows", str(max_lines)])
    if max_companies is not None:
        command.extend(["--max-records", str(max_companies)])
    return command


def _hash_file_sha256(path: Path) -> str:
    """A sha256 digest of a file's bytes, read in fixed-size chunks so a multi-MB binary
    never has to be held in memory at once."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_RUN_MANIFEST_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_wikidata_run_manifest(
    *,
    manifest_path: Path,
    binary_path: Path,
    engine: str,
    summary: Mapping[str, object],
) -> Path:
    """Ties a Wikidata prepare run to the binary that produced it: the binary's own path and
    sha256 digest, plus whatever build provenance it self-reported in its `--summary-json`
    (`crate_version`/`git_commit`/`git_dirty`, from `wikisieve`'s `write_summary`). The
    `reported_*` fields come back `None` for a binary whose build never embedded that
    provenance -- absence here is itself informative, not an error, since a manifest is
    written for every run regardless of what its binary knows about itself.
    """
    payload = {
        "engine": engine,
        "binary_path": str(binary_path),
        "binary_sha256": _hash_file_sha256(binary_path),
        "reported_version": summary.get("crate_version"),
        "reported_commit": summary.get("git_commit"),
        "reported_dirty": summary.get("git_dirty"),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _commit_flat_wikisieve_output(
    *,
    staged_destination_path: Path,
    destination_path: Path,
    staged_raw_candidate_cache_path: Path | None,
    raw_candidate_cache_path: Path | None,
    records_emitted: int,
) -> None:
    """Renames a flat run's staged outputs onto the live paths once the staged
    extract holds exactly the rows the run reported, and refuses otherwise."""
    staged_rows = _count_non_empty_jsonl_rows(staged_destination_path)
    if staged_rows != records_emitted:
        raise RuntimeError(
            "wikidata wikisieve flat output row count mismatch: "
            f"staged={staged_rows} summary={records_emitted}"
        )
    staged_destination_path.replace(destination_path)
    if (
        staged_raw_candidate_cache_path is not None
        and raw_candidate_cache_path is not None
    ):
        staged_raw_candidate_cache_path.replace(raw_candidate_cache_path)


def _extract_wikidata_company_projection_two_pass_wikisieve(
    source_path: Path,
    destination_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    max_companies: int | None = None,
    max_lines: int | None = None,
    stream_resume: bool = False,
    projection_defaults: Mapping[str, object] | None = None,
) -> int:
    """Runs wikisieve over `source_path` into `destination_path`.

    A failed run never removes or truncates the live extract or raw-candidate
    cache: a flat run writes `.tmp` siblings that are renamed over the live files
    only once the run has exited cleanly and its row count checks out, and a
    resumable run writes chunks that `wikisieve merge-chunks` assembles the same
    way. A failed flat run's staging files are removed; a failed resumable run's
    chunks stay for the next `--resume`.
    """
    engine, binary_path, output_mode, spec_path = _resolve_wikidata_projection_defaults(
        projection_defaults
    )
    if engine != "wikisieve":
        raise ValueError(f"expected wikisieve projection engine, got: {engine}")
    if output_mode != "jsonl":
        raise ValueError(
            "wikidata wikisieve projection output_mode must be 'jsonl' for pipeline compatibility"
        )
    if not binary_path.exists():
        raise FileNotFoundError(
            f"wikisieve binary not found: {binary_path}. Run scripts/build_wikisieve.ps1 "
            "(and deploy it to tools/bin) first."
        )
    if not spec_path.exists():
        raise FileNotFoundError(f"wikisieve spec not found: {spec_path}")

    company_types_path = source_path.with_name("p279.json")
    if not company_types_path.exists():
        raise FileNotFoundError(
            f"Wikidata company-type seed not found: {company_types_path}"
        )

    (
        resume_enabled,
        resume_state_path,
        chunk_dir_path,
        chunk_prefix,
    ) = _resolve_wikidata_projection_resume_defaults(
        stream_resume=stream_resume,
        destination_path=destination_path,
    )
    raw_candidate_cache_path = _resolve_wikidata_raw_candidate_cache_path(
        projection_defaults, destination_path=destination_path
    )
    staged_destination_path = _staging_path(destination_path)
    staged_raw_candidate_cache_path = (
        _staging_path(raw_candidate_cache_path)
        if raw_candidate_cache_path is not None
        else None
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wikisieve-run-") as temp_dir:
        summary_path = Path(temp_dir) / "summary.json"
        resolved_spec_path = Path(temp_dir) / "wikisieve-resolved-spec.json"
        _write_resolved_wikisieve_spec(
            spec_path=spec_path,
            p279_path=company_types_path,
            destination_path=resolved_spec_path,
        )
        command = _build_wikisieve_command(
            binary_path=binary_path,
            source_path=source_path,
            spec_path=resolved_spec_path,
            summary_path=summary_path,
            output_mode=output_mode,
            resume_enabled=resume_enabled,
            resume_state_path=resume_state_path,
            chunk_dir_path=chunk_dir_path,
            chunk_prefix=chunk_prefix,
            destination_path=staged_destination_path,
            max_lines=max_lines,
            max_companies=max_companies,
            raw_candidate_cache_path=(
                raw_candidate_cache_path
                if resume_enabled
                else staged_raw_candidate_cache_path
            ),
        )

        if progress is not None:
            progress(
                f"[wikidata] wikisieve started: source={source_path.name}, binary={binary_path.name}"
            )
            progress(f"[wikidata] wikisieve resume settings: enabled={resume_enabled}")
            if resume_enabled:
                progress(
                    f"[wikidata] wikisieve streaming chunks to {chunk_dir_path.name}"
                )
            if raw_candidate_cache_path is not None:
                progress(
                    "[wikidata] wikisieve capturing raw candidates to "
                    f"{raw_candidate_cache_path.name}"
                )

        if not resume_enabled:
            try:
                _run_wikisieve_process(command=command, progress=progress)
                records_emitted = _read_records_emitted(summary_path)
                _commit_flat_wikisieve_output(
                    staged_destination_path=staged_destination_path,
                    destination_path=destination_path,
                    staged_raw_candidate_cache_path=staged_raw_candidate_cache_path,
                    raw_candidate_cache_path=raw_candidate_cache_path,
                    records_emitted=records_emitted,
                )
            finally:
                staged_destination_path.unlink(missing_ok=True)
                if staged_raw_candidate_cache_path is not None:
                    staged_raw_candidate_cache_path.unlink(missing_ok=True)
        else:
            _run_wikisieve_process(command=command, progress=progress)

        summary_for_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
        _write_wikidata_run_manifest(
            manifest_path=destination_path.with_name(
                _WIKIDATA_DEFAULT_RUN_MANIFEST_NAME
            ),
            binary_path=binary_path,
            engine=engine,
            summary=summary_for_manifest,
        )

        if not resume_enabled:
            return records_emitted

        rows_emitted = _validate_wikisieve_chunk_summary(
            summary_path=summary_path, chunk_dir_path=chunk_dir_path
        )

        # A max_lines/max_companies run is intentionally partial, so its chunks
        # stay in place for a later higher-limit invocation to resume and extend
        # rather than starting over.
        if max_lines is None and max_companies is None:
            if progress is not None:
                progress(
                    f"[wikidata] wikisieve assembling chunks into {destination_path.name}"
                )
            _run_wikisieve_process(
                command=_build_wikisieve_merge_chunks_command(
                    binary_path=binary_path,
                    chunk_dir_path=chunk_dir_path,
                    chunk_prefix=chunk_prefix,
                    destination_path=destination_path,
                    summary_path=summary_path,
                    resume_state_path=resume_state_path,
                    raw_candidate_cache_path=raw_candidate_cache_path,
                ),
                progress=progress,
            )

        return rows_emitted


def _is_wikidata_company_candidate_raw_line(raw_line: bytes) -> bool:
    return _is_wikidata_company_candidate_raw_line_impl(raw_line)


def _extract_claim_text_values_with_english(
    entity: dict[str, object],
    property_id: str,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    return _extract_claim_text_values_with_english_impl(entity, property_id)


def _iter_wikidata_entities_from_handle(handle):
    for line_number, line in _iter_wikidata_content_lines_from_handle(handle):
        payload = _json_loads(line)
        if payload is not None:
            yield line_number, payload


def _iter_wikidata_entities(path: Path):
    if path.suffix.lower() == ".bz2":
        with _indexed_bzip2.open(str(path), parallelization=2) as handle:
            yield from _iter_wikidata_entities_from_handle(handle)
        return

    with _open_eta_enabled_source(path) as handle:
        yield from _iter_wikidata_entities_from_handle(handle)


def _emit_prepare_projection_progress_if_due(
    *,
    source_path: Path,
    source_size: int,
    source_handle,
    lines_processed: int,
    rows_written: int,
    started: float,
    last_progress_emit: float,
    progress_every_entities: int,
    progress_every_seconds: float,
    progress_samples: list[tuple[float, int]],
    emitter: WikidataProgressEmitter,
) -> float:
    elapsed = max(time.perf_counter() - started, 1e-9)
    since_last_progress = time.perf_counter() - last_progress_emit
    should_emit_progress = False
    if progress_every_entities > 0 and lines_processed % progress_every_entities == 0:
        should_emit_progress = True
    if progress_every_seconds > 0 and since_last_progress >= progress_every_seconds:
        should_emit_progress = True

    if not should_emit_progress:
        return last_progress_emit

    eta_text, compressed_progress_text = _build_prepare_projection_progress_text(
        source_path=source_path,
        source_size=source_size,
        source_handle=source_handle,
        lines_processed=lines_processed,
        elapsed=elapsed,
        progress_samples=progress_samples,
    )

    emitter.emit_phase_counter_progress(
        phase="prepare projection",
        lines_processed=lines_processed,
        secondary_count_label="written",
        secondary_count=rows_written,
        compressed_progress_text=compressed_progress_text,
        eta_text=eta_text,
        line_rate=lines_processed / elapsed,
        secondary_rate_label="write_rate",
        secondary_rate=rows_written / elapsed,
    )
    return time.perf_counter()


def _build_prepare_projection_progress_text(
    *,
    source_path: Path,
    source_size: int,
    source_handle,
    lines_processed: int,
    elapsed: float,
    progress_samples: list[tuple[float, int]],
) -> tuple[str, str]:
    eta_details = _estimate_stream_eta(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=lines_processed,
        elapsed_seconds=elapsed,
    )
    if eta_details is None:
        return "eta=unknown", "compressed_bytes=unknown"

    estimated_total_lines, compressed_bytes_read, eta_seconds = eta_details
    now = time.perf_counter()
    progress_samples.append((now, lines_processed))
    cutoff = now - 600.0
    while len(progress_samples) > 1 and progress_samples[0][0] < cutoff:
        progress_samples.pop(0)

    recent_line_rate: float | None = None
    if len(progress_samples) >= 2:
        first_time, first_lines = progress_samples[0]
        delta_seconds = now - first_time
        delta_lines = lines_processed - first_lines
        if delta_seconds > 0 and delta_lines > 0:
            recent_line_rate = delta_lines / delta_seconds

    eta_band_text, eta_stability = _format_eta_band_and_stability(
        global_eta_seconds=eta_seconds,
        estimated_total_lines=estimated_total_lines,
        lines_processed=lines_processed,
        recent_line_rate=recent_line_rate,
    )
    eta_text = f"{eta_band_text}, eta_stability={eta_stability}"
    compressed_progress_text = (
        f"compressed_bytes={compressed_bytes_read:,}/{source_size:,}, "
        f"estimated_total_lines={estimated_total_lines:,}"
    )
    return eta_text, compressed_progress_text


def _emit_prepare_projection_early_stop_for_lines(
    *,
    emitter: WikidataProgressEmitter,
    should_stop: bool,
    lines_processed: int,
    max_lines: int | None,
) -> bool:
    # `should_stop` is only ever set when `max_lines` is not None (it is
    # `max_lines is not None and lines_processed >= max_lines`), but the two
    # arrive as independent arguments, so re-check rather than assume.
    if not should_stop or max_lines is None:
        return False
    emitter.emit_early_stop_lines(
        phase="prepare projection",
        lines_processed=lines_processed,
        max_lines=max_lines,
    )
    return True


def _build_prepare_projection_final_eta_text(
    *,
    source_path: Path,
    source_handle,
    lines_processed: int,
    elapsed: float,
) -> str:
    final_eta_details = _estimate_stream_eta(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=lines_processed,
        elapsed_seconds=elapsed,
    )
    if final_eta_details is None:
        return "eta=unknown"

    estimated_total_lines, _, eta_seconds = final_eta_details
    eta_band_text, eta_stability = _format_eta_band_and_stability(
        global_eta_seconds=eta_seconds,
        estimated_total_lines=estimated_total_lines,
        lines_processed=lines_processed,
        recent_line_rate=None,
    )
    return f"{eta_band_text}, eta_stability={eta_stability}"


def _process_prepare_projection_line(
    *,
    line: bytes,
    line_number: int,
    rows_written: int,
    max_companies: int | None,
    max_lines: int | None,
    handle,
    emitter: WikidataProgressEmitter,
    source_path: Path,
    source_size: int,
    source_handle,
    started: float,
    last_progress_emit: float,
    progress_every_entities: int,
    progress_every_seconds: float,
    progress_samples: list[tuple[float, int]],
) -> tuple[bool, int, float]:
    lines_processed = line_number
    stop_after_this_line = max_lines is not None and lines_processed >= max_lines

    if not _is_wikidata_company_candidate_raw_line(line):
        if _emit_prepare_projection_early_stop_for_lines(
            emitter=emitter,
            should_stop=stop_after_this_line,
            lines_processed=lines_processed,
            max_lines=max_lines,
        ):
            return True, rows_written, last_progress_emit
        last_progress_emit = _emit_prepare_projection_progress_if_due(
            source_path=source_path,
            source_size=source_size,
            source_handle=source_handle,
            lines_processed=lines_processed,
            rows_written=rows_written,
            started=started,
            last_progress_emit=last_progress_emit,
            progress_every_entities=progress_every_entities,
            progress_every_seconds=progress_every_seconds,
            progress_samples=progress_samples,
            emitter=emitter,
        )
        return False, rows_written, last_progress_emit

    projected_line = _project_wikidata_company_record_line(line)
    if projected_line is None:
        if _emit_prepare_projection_early_stop_for_lines(
            emitter=emitter,
            should_stop=stop_after_this_line,
            lines_processed=lines_processed,
            max_lines=max_lines,
        ):
            return True, rows_written, last_progress_emit
        last_progress_emit = _emit_prepare_projection_progress_if_due(
            source_path=source_path,
            source_size=source_size,
            source_handle=source_handle,
            lines_processed=lines_processed,
            rows_written=rows_written,
            started=started,
            last_progress_emit=last_progress_emit,
            progress_every_entities=progress_every_entities,
            progress_every_seconds=progress_every_seconds,
            progress_samples=progress_samples,
            emitter=emitter,
        )
        return False, rows_written, last_progress_emit

    handle.write(projected_line)
    rows_written += 1

    if max_companies is not None and rows_written >= max_companies:
        emitter.emit_early_stop_rows(
            phase="prepare projection",
            rows_written=rows_written,
            max_companies=max_companies,
        )
        return True, rows_written, last_progress_emit

    if _emit_prepare_projection_early_stop_for_lines(
        emitter=emitter,
        should_stop=stop_after_this_line,
        lines_processed=lines_processed,
        max_lines=max_lines,
    ):
        return True, rows_written, last_progress_emit

    last_progress_emit = _emit_prepare_projection_progress_if_due(
        source_path=source_path,
        source_size=source_size,
        source_handle=source_handle,
        lines_processed=lines_processed,
        rows_written=rows_written,
        started=started,
        last_progress_emit=last_progress_emit,
        progress_every_entities=progress_every_entities,
        progress_every_seconds=progress_every_seconds,
        progress_samples=progress_samples,
        emitter=emitter,
    )
    return False, rows_written, last_progress_emit


def extract_wikidata_company_projection(
    source_path: Path,
    destination_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every_entities: int = 50_000,
    progress_every_seconds: float = 30.0,
    max_companies: int | None = None,
    max_lines: int | None = None,
    resolved_read_options: dict[str, object] | None = None,
) -> int:
    """Standalone single-pass filter+project over a raw Wikidata source.

    Only matches the 6 hardcoded root instance-of QIDs -- it does not load
    p279.json, so it misses companies that are only identifiable via a P1454
    legal-form claim. Independent of `extract_wikidata_company_projection_two_pass`.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    lines_processed = 0
    started = time.perf_counter()
    last_progress_emit = started
    progress_samples: list[tuple[float, int]] = []

    emitter = WikidataProgressEmitter(progress)

    emitter.emit_started(phase="prepare projection", source_name=source_path.name)
    emitter.emit_scanning(phase="prepare projection")

    source_size = source_path.stat().st_size
    reader_strategy = build_wikidata_reader_strategy(
        source_path=source_path,
        resolved_read_options=resolved_read_options,
        iter_content_lines_fn=_iter_wikidata_content_lines_from_handle,
        default_chunk_size_bytes=_READ_CHUNK_SIZE_BYTES,
    )
    with destination_path.open("wb") as handle:
        source_handle = reader_strategy.open_source()
        with source_handle:
            for line_number, line in reader_strategy.iter_content_lines(source_handle):
                lines_processed = line_number
                should_break, rows_written, last_progress_emit = (
                    _process_prepare_projection_line(
                        line=line,
                        line_number=line_number,
                        rows_written=rows_written,
                        max_companies=max_companies,
                        max_lines=max_lines,
                        handle=handle,
                        emitter=emitter,
                        source_path=source_path,
                        source_size=source_size,
                        source_handle=source_handle,
                        started=started,
                        last_progress_emit=last_progress_emit,
                        progress_every_entities=progress_every_entities,
                        progress_every_seconds=progress_every_seconds,
                        progress_samples=progress_samples,
                    )
                )
                if should_break:
                    break

    elapsed = max(time.perf_counter() - started, 1e-9)
    final_eta_text = _build_prepare_projection_final_eta_text(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=lines_processed,
        elapsed=elapsed,
    )
    emitter.emit_prepare_complete(
        lines_processed=lines_processed,
        rows_written=rows_written,
        eta_text=final_eta_text,
        elapsed=elapsed,
        line_rate=lines_processed / elapsed,
        write_rate=rows_written / elapsed,
    )

    return rows_written


@functools.lru_cache(maxsize=8)
def _load_wikidata_company_legal_form_qids(company_types_path: Path) -> frozenset[str]:
    """Parse p279.json's subclass URIs into "Q<n>" tokens.

    Returns the p279 subclass-of-company-legal-form closure (~49K QIDs as of
    2026-07-16): entities whose P1454 legal-form claim resolves into this set
    (e.g. "GmbH", "S.A.") are companies even without a direct P31 root match.
    The Rust CLI's `load_company_type_qids` loads the same file.
    """
    with company_types_path.open("rb") as handle:
        entries = json.load(handle)
    qids: set[str] = set()
    for entry in entries:
        subclass = entry.get("subclass") if isinstance(entry, dict) else None
        if not isinstance(subclass, str):
            continue
        qid = subclass.rsplit("/", 1)[-1]
        if qid.startswith("Q") and qid[1:].isdigit():
            qids.add(qid)
    return frozenset(qids)


def extract_wikidata_company_projection_two_pass(
    source_path: Path,
    destination_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every_entities: int = 50_000,
    progress_every_seconds: float = 30.0,
    max_companies: int | None = None,
    max_lines: int | None = None,
    phase1_wiring_mode: str = "stream",
    stream_resume: bool = False,
    resolved_read_options: dict[str, object] | None = None,
    projection_defaults: Mapping[str, object] | None = None,
) -> int:
    """Wikidata company extraction, dispatched to the wikisieve or native Python
    engine per `projection_defaults`.

    Neither engine reads `phase1_wiring_mode`: wikisieve's resume is a plain
    boolean (`stream_resume`), and the native Python engine always runs
    single-scan. The caller's prepare stage uses the same setting to choose
    between chunked and flat output.
    """
    projection_engine, _, _, _ = _resolve_wikidata_projection_defaults(
        projection_defaults
    )
    if projection_engine == "wikisieve":
        return _extract_wikidata_company_projection_two_pass_wikisieve(
            source_path,
            destination_path,
            progress=progress,
            max_companies=max_companies,
            max_lines=max_lines,
            stream_resume=stream_resume,
            projection_defaults=projection_defaults,
        )

    legal_form_qids: frozenset[str] | None = None
    company_types_path = source_path.with_name("p279.json")
    if company_types_path.exists():
        legal_form_qids = _load_wikidata_company_legal_form_qids(company_types_path)
    elif progress is not None:
        progress(
            f"[wikidata] no p279.json beside {source_path.name}: python engine will "
            "only match the 6 root instance-of QIDs (legal-form matching skipped)"
        )

    return _extract_wikidata_company_projection_two_pass_impl(
        source_path=source_path,
        destination_path=destination_path,
        progress=progress,
        progress_every_entities=progress_every_entities,
        progress_every_seconds=progress_every_seconds,
        max_companies=max_companies,
        max_lines=max_lines,
        stream_resume=stream_resume,
        resolved_read_options=resolved_read_options,
        legal_form_qids=legal_form_qids,
    )
