"""
Benchmark line-by-line processing with actual filtering (P31 check).
Tests different chunking/buffering strategies for BZIP2 decompression.
Compares: bz2.open vs indexed_bzip2 with various chunk sizes
Note: rapidgzip is for gzip files only, not compatible with bz2

Also supports benchmarking the Wikidata company projection pipeline across
engines (native Python vs the wikisieve binary), both driven through the same
`extract_wikidata_company_projection_two_pass` entry point so the only
difference between runs is `projection_defaults.engine`.
"""

import argparse
import bz2
import gzip
import os
import statistics
import sys
import time
from pathlib import Path

import _bootstrap
import indexed_bzip2

from acquisition.downloader_wikidata import extract_wikidata_company_projection_two_pass

REPO_ROOT = _bootstrap.REPO_ROOT
FIXTURE_PATHS: dict[str, Path] = {
    "wikidata-benchmark-bz2": REPO_ROOT
    / "src"
    / "tests"
    / "acquisition"
    / "fixtures"
    / "wikidata-benchmark-sample.jsonl.bz2",
    "wikidata-benchmark-gz": REPO_ROOT
    / "src"
    / "tests"
    / "acquisition"
    / "fixtures"
    / "wikidata-benchmark-sample.jsonl.gz",
}


class TimeoutError(Exception):
    """Raised when benchmark exceeds time limit."""


def get_optimal_bzip2_threads() -> int:
    """Calculates optimal thread count for indexed_bzip2 on Xeon systems."""
    try:
        # Best for Linux/HPC/Docker: gets cores allocated to this specific job
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for Windows/Mac or general systems
        cores = os.cpu_count() or 1

    # 1. Ensure we use at least 2 threads if multi-core is available
    # 2. Cap at 24 to prevent the single-threaded block scanner bottleneck
    optimal_threads = max(2, min(cores, 24))
    return optimal_threads


def benchmark_startup_time(name: str, source_path: Path) -> float | None:
    """Measure time from open to reading first line (index build cost)."""
    thread_count = get_optimal_bzip2_threads()

    if name == "line_iter_bz2":

        def open_fn():
            return bz2.open(source_path, "rb")
    elif name == "line_iter_gz":

        def open_fn():
            return gzip.open(source_path, "rb")
    elif name.startswith(("line_iter_indexed", "chunked_indexed")):

        def open_fn():
            return indexed_bzip2.open(str(source_path), parallelization=thread_count)
    elif name.startswith("chunked_bz2"):

        def open_fn():
            return bz2.open(source_path, "rb")
    elif name.startswith("chunked_gz"):

        def open_fn():
            return gzip.open(source_path, "rb")
    else:
        return None

    try:
        started = time.perf_counter()
        with open_fn() as f:
            f.readline()
        elapsed = time.perf_counter() - started
        return elapsed
    except Exception as e:  # noqa: BLE001 -- benchmark probe: report and skip, never abort the sweep
        print(f"  ERROR reading first line: {e}")
        return None


def _resolve_benchmark_open_fn(name: str, source_path: Path, thread_count: int):
    if name == "line_iter_bz2":
        return lambda: bz2.open(source_path, "rb")
    if name == "line_iter_gz":
        return lambda: gzip.open(source_path, "rb")
    if "indexed" in name:
        return lambda: indexed_bzip2.open(
            str(source_path), parallelization=thread_count
        )
    if "bz2" in name:
        return lambda: bz2.open(source_path, "rb")
    if "gz" in name:
        return lambda: gzip.open(source_path, "rb")
    return None


def _resolve_source_path(*, input_path: str | None, fixture: str | None) -> Path:
    if fixture is not None:
        return FIXTURE_PATHS[fixture]
    if input_path is None:
        raise ValueError("Provide either --input or --fixture")
    return Path(input_path)


def _default_line_filter_approaches(source_path: Path) -> list[str]:
    suffix = source_path.suffix.lower()
    if suffix == ".gz":
        return ["line_iter_gz", "chunked_gz_16mb", "chunked_gz_32mb", "chunked_gz_64mb"]
    return [
        "line_iter_bz2",
        "chunked_bz2_16mb",
        "chunked_indexed_16mb",
        "chunked_indexed_32mb",
        "chunked_indexed_64mb",
    ]


def _resolve_benchmark_chunk_size(name: str, chunk_size: int | None) -> int | None:
    if chunk_size is not None or "_" not in name:
        return chunk_size
    try:
        size_part = name.split("_")[-1]  # e.g., "16mb" or "64mb"
        return int(size_part.replace("mb", "")) * 1024 * 1024
    except (ValueError, IndexError):
        return None


def _run_chunked_filter_pass(
    *,
    open_fn,
    chunk_size: int | None,
    max_lines: int,
    p31_bytes: bytes,
    report_progress,
) -> tuple[int, int]:
    lines_read = 0
    rows_matched = 0
    remainder = b""
    with open_fn() as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                if remainder:
                    lines_read += 1
                break

            data = remainder + chunk
            parts = data.split(b"\n")

            for line in parts[:-1]:
                lines_read += 1
                if p31_bytes in line:
                    rows_matched += 1

                report_progress(lines_read)
                if lines_read >= max_lines:
                    break

            remainder = parts[-1]
            if lines_read >= max_lines:
                break

    return lines_read, rows_matched


def _run_line_iter_filter_pass(
    *,
    open_fn,
    max_lines: int,
    p31_bytes: bytes,
    report_progress,
) -> tuple[int, int]:
    lines_read = 0
    rows_matched = 0
    with open_fn() as f:
        for line in f:
            lines_read += 1
            if p31_bytes in line:
                rows_matched += 1

            report_progress(lines_read)
            if lines_read >= max_lines:
                break

    return lines_read, rows_matched


def benchmark_line_filter(
    name: str,
    source_path: Path,
    max_lines: int = 100_000,
    chunk_size: int | None = None,
    timeout_sec: int = 60,
) -> None:
    """Benchmark with line-by-line processing and P31 filtering.

    Args:
        name: Test name (line_iter_bz2, chunked_bz2_16mb, chunked_indexed_16mb, etc.)
        source_path: Path to bz2 file
        max_lines: Max lines to process
        chunk_size: If provided, use chunked reading with this size (bytes)
        timeout_sec: Timeout in seconds (0 = no timeout)
    """
    print(f"\n[{name}] (timeout={timeout_sec}s)")

    thread_count = get_optimal_bzip2_threads()

    open_fn = _resolve_benchmark_open_fn(name, source_path, thread_count)
    if open_fn is None:
        print(f"  ERROR: unknown approach {name}")
        return

    chunk_size = _resolve_benchmark_chunk_size(name, chunk_size)

    # Measure startup time (open + first read)
    startup_time = benchmark_startup_time(name, source_path)
    if startup_time is not None:
        print(f"  startup_time={startup_time:.3f}s (open + read first line)")

    rows_matched = 0
    started = time.perf_counter()
    last_progress = started
    p31_bytes = b'"P31"'
    progress_interval_lines = 10000
    progress_interval_sec = 5.0

    def _report_progress_and_check_timeout(current_lines_read: int, now: float) -> None:
        nonlocal last_progress

        if (
            current_lines_read % progress_interval_lines != 0
            and (now - last_progress) < progress_interval_sec
        ):
            return

        elapsed = now - started
        rate = current_lines_read / elapsed if elapsed > 0 else 0
        print(f"    {current_lines_read:>7,} lines in {elapsed:.1f}s ({rate:>7,.0f}/s)")
        last_progress = now

        # Check timeout (wall-clock, works on all platforms)
        if timeout_sec > 0 and elapsed > timeout_sec:
            raise TimeoutError(f"Exceeded {timeout_sec}s timeout")

    try:
        if "chunked" in name:
            lines_read, rows_matched = _run_chunked_filter_pass(
                open_fn=open_fn,
                chunk_size=chunk_size,
                max_lines=max_lines,
                p31_bytes=p31_bytes,
                report_progress=lambda line_count: _report_progress_and_check_timeout(
                    line_count, time.perf_counter()
                ),
            )
        else:
            lines_read, rows_matched = _run_line_iter_filter_pass(
                open_fn=open_fn,
                max_lines=max_lines,
                p31_bytes=p31_bytes,
                report_progress=lambda line_count: _report_progress_and_check_timeout(
                    line_count, time.perf_counter()
                ),
            )
    except TimeoutError:
        elapsed = time.perf_counter() - started
        rate = lines_read / elapsed if elapsed > 0 else 0
        print(
            f"  TIMEOUT after {elapsed:.1f}s: processed {lines_read:,} lines ({rate:,.0f}/s)"
        )
        return
    except Exception as e:  # noqa: BLE001 -- benchmark probe: report and skip, never abort the sweep
        elapsed = time.perf_counter() - started
        print(f"  ERROR after {elapsed:.1f}s ({lines_read:,} lines): {e}")
        return

    elapsed = time.perf_counter() - started
    read_rate = lines_read / elapsed if elapsed > 0 else 0

    print(f"  COMPLETE: lines_read={lines_read:,}, matched={rows_matched:,}")
    print(f"  elapsed={elapsed:.2f}s, rate={read_rate:,.0f}/s")


def _resolve_engine_projection_defaults(
    engine: str, *, wikisieve_binary: Path
) -> dict[str, object]:
    if engine == "wikisieve":
        return {
            "engine": "wikisieve",
            "binary_path": str(wikisieve_binary),
            "output_mode": "jsonl",
        }
    return {"engine": "python"}


def _run_two_pass_single(
    *,
    source_path: Path,
    output_path: Path,
    engine: str,
    wikisieve_binary: Path,
    max_lines: int,
) -> tuple[float, int, int]:
    started = time.perf_counter()
    rows_written = extract_wikidata_company_projection_two_pass(
        source_path=source_path,
        destination_path=output_path,
        progress=None,
        max_companies=None,
        max_lines=max_lines if max_lines > 0 else None,
        projection_defaults=_resolve_engine_projection_defaults(
            engine, wikisieve_binary=wikisieve_binary
        ),
    )
    elapsed = time.perf_counter() - started
    output_size = _resolve_output_size(output_path)
    return elapsed, rows_written, output_size


def _resolve_output_size(output_path: Path) -> int:
    if output_path.exists():
        return output_path.stat().st_size
    # The native Python engine writes resumable chunk files beside the
    # nominal destination instead of a flat file.
    chunk_dir = output_path.parent / "wikidata-companies.chunks"
    if chunk_dir.exists():
        return sum(path.stat().st_size for path in chunk_dir.rglob("*.jsonl"))
    return 0


def benchmark_two_pass_engines(
    *,
    source_path: Path,
    output_dir: Path,
    engines: list[str],
    wikisieve_binary: Path,
    max_lines: int,
    runs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "wikisieve" in engines and not wikisieve_binary.exists():
        print(f"ERROR: wikisieve binary not found: {wikisieve_binary}")
        print("Run scripts/deploy_wikisieve.ps1 first, or pass --wikisieve-binary.")
        return

    print("\n[Two-pass engine benchmark]")
    print(f"  source={source_path}")
    print(f"  max_lines={max_lines if max_lines > 0 else 'all'}")
    print(f"  runs={runs}")
    print(f"  engines={','.join(engines)}")

    summary: dict[str, list[float]] = {engine: [] for engine in engines}
    rows_by_engine: dict[str, list[int]] = {engine: [] for engine in engines}

    for engine in engines:
        print(f"\n  engine={engine}")
        for run_index in range(1, runs + 1):
            output_path = output_dir / f"wikidata-{engine}-run{run_index}.jsonl"
            elapsed, rows_written, output_size = _run_two_pass_single(
                source_path=source_path,
                output_path=output_path,
                engine=engine,
                wikisieve_binary=wikisieve_binary,
                max_lines=max_lines,
            )
            summary[engine].append(elapsed)
            rows_by_engine[engine].append(rows_written)
            rate = rows_written / elapsed if elapsed > 0 else 0.0
            print(
                f"    run={run_index}: elapsed={elapsed:.2f}s, "
                f"rows_written={rows_written:,}, row_rate={rate:,.0f}/s, output_bytes={output_size:,}"
            )

    print("\n[Two-pass summary]")
    mean_by_engine: dict[str, float] = {}
    for engine in engines:
        values = summary[engine]
        mean_elapsed = statistics.mean(values)
        mean_by_engine[engine] = mean_elapsed
        distinct_rows = set(rows_by_engine[engine])
        rows_text = (
            f"{distinct_rows.pop():,}"
            if len(distinct_rows) == 1
            else f"varied {sorted(distinct_rows)}"
        )
        print(
            f"  engine={engine}: mean={mean_elapsed:.2f}s, min={min(values):.2f}s, "
            f"max={max(values):.2f}s, runs={len(values)}, rows_written={rows_text}"
        )

    all_rows = {
        engine: set(rows_by_engine[engine])
        for engine in engines
        if rows_by_engine[engine]
    }
    row_counts_match = len({frozenset(v) for v in all_rows.values()}) <= 1
    if not row_counts_match:
        print(
            "  WARNING: engines produced different row counts -- speedup figure "
            "below is not comparing like-for-like output"
        )

    if "python" in mean_by_engine and "wikisieve" in mean_by_engine:
        python_mean = mean_by_engine["python"]
        wikisieve_mean = mean_by_engine["wikisieve"]
        if wikisieve_mean > 0:
            print(f"  wikisieve speedup over python: {python_mean / wikisieve_mean:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Wikidata line processing")
    parser.add_argument(
        "--input", help="Path to wikidata compressed file (.bz2 or .gz)"
    )
    parser.add_argument(
        "--fixture",
        choices=sorted(FIXTURE_PATHS.keys()),
        help="Named local benchmark fixture under src/tests/acquisition/fixtures",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=500_000,
        help=(
            "Max lines to process. Two-pass mode defaults to 500K since smaller "
            "samples are dominated by fixed overhead (subprocess startup, "
            "p279/legal-form-set loading) rather than steady-state throughput."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout per benchmark in seconds (0=no timeout)",
    )
    parser.add_argument(
        "--mode",
        choices=["line-filter", "two-pass"],
        default="two-pass",
        help="Benchmark mode. 'two-pass' compares projection engines.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=None,
        help=(
            "Approaches to benchmark. BZ2 examples: line_iter_bz2, chunked_bz2_16mb, "
            "chunked_indexed_16mb, chunked_indexed_32mb, chunked_indexed_64mb. "
            "GZ examples: line_iter_gz, chunked_gz_16mb, chunked_gz_32mb, chunked_gz_64mb."
        ),
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=["python", "wikisieve"],
        default=["python", "wikisieve"],
        help="Two-pass projection engines to benchmark and compare.",
    )
    parser.add_argument(
        "--wikisieve-binary",
        default="tools/bin/wikisieve.exe",
        help="Path to the deployed wikisieve binary (used when --engines includes wikisieve).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs per engine in two-pass benchmark mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/perf/wikidata_two_pass_benchmark",
        help="Output directory for two-pass benchmark JSONL outputs.",
    )

    args = parser.parse_args()
    if args.input and args.fixture:
        print("ERROR: provide either --input or --fixture, not both")
        return 1
    source_path = _resolve_source_path(input_path=args.input, fixture=args.fixture)

    if not source_path.exists():
        print(f"ERROR: {source_path} not found")
        return 1

    print(f"Benchmarking {source_path.name} ({source_path.stat().st_size / 1e9:.1f}GB)")
    print(f"Selected mode: {args.mode}")

    if args.mode == "line-filter":
        approaches = args.approaches or _default_line_filter_approaches(source_path)
        print(f"Max lines per test: {args.max_lines:,}")
        print(f"Approaches: {', '.join(approaches)}")
        for approach in approaches:
            # Extract chunk size from approach name if present
            chunk_size = None
            if "_" in approach:
                try:
                    size_part = approach.split("_")[-1]  # e.g., "16mb", "32mb", "64mb"
                    chunk_size = int(size_part.replace("mb", "")) * 1024 * 1024
                except (ValueError, IndexError):
                    pass

            benchmark_line_filter(
                approach, source_path, args.max_lines, chunk_size, args.timeout
            )
    else:
        if args.runs < 1:
            print("ERROR: --runs must be >= 1")
            return 1
        benchmark_two_pass_engines(
            source_path=source_path,
            output_dir=Path(args.output_dir),
            engines=args.engines,
            wikisieve_binary=Path(args.wikisieve_binary),
            max_lines=args.max_lines,
            runs=args.runs,
        )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
