#!/usr/bin/env python3
"""Profile the Wikidata bz2 read path, in one of two modes.

Both modes read the same dump the same way (`indexed_bzip2` at
`get_optimal_bzip2_threads()` parallelization, counting `"P31"` byte hits up to
`--max-lines`), so their results describe the same work and can be read together.

    --mode timings   Phase-by-phase wall-clock breakdown (open, read, summary)
                     with periodic throughput milestones. Answers "where does
                     the time go, and does throughput hold up over a long run".

    --mode cprofile  The same read under cProfile, reporting the top functions
                     by cumulative and by self time. Answers "which call is
                     responsible".

The two read loops are deliberately kept separate rather than sharing one
parameterised body: `timings` prints from inside the loop, which is what makes
its throughput numbers meaningful and exactly what would distort cProfile's.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import indexed_bzip2

from scripts.wikidata_profile_common import (
    format_size_gb,
    get_optimal_bzip2_threads,
    validate_input_path,
)

P31_BYTES = b'"P31"'
MILESTONE_INTERVAL = 50_000


def profile_timings(source_path: Path, max_lines: int = 100_000) -> None:
    """Phase-by-phase timing breakdown with throughput milestones."""

    thread_count = get_optimal_bzip2_threads()

    print(f"Thread count: {thread_count}")
    print(f"Max lines: {max_lines:,}")
    print()

    # Phase 1: Open file
    print("[Phase 1] Opening file...")
    start_open = time.perf_counter()
    f = indexed_bzip2.open(str(source_path), parallelization=thread_count)
    elapsed_open = time.perf_counter() - start_open
    print(f"  Time: {elapsed_open:.3f}s")

    # Phase 2: Iterate and process (with milestones)
    print("\n[Phase 2] Reading and filtering...")
    lines_read = 0
    rows_matched = 0

    start_total = time.perf_counter()
    milestone_time = start_total

    try:
        for line in f:
            lines_read += 1
            if P31_BYTES in line:
                rows_matched += 1

            # Progress milestone
            if lines_read % MILESTONE_INTERVAL == 0:
                elapsed_milestone = time.perf_counter() - milestone_time
                rate = (
                    MILESTONE_INTERVAL / elapsed_milestone
                    if elapsed_milestone > 0
                    else 0
                )
                cumulative_elapsed = time.perf_counter() - start_total
                cumulative_rate = (
                    lines_read / cumulative_elapsed if cumulative_elapsed > 0 else 0
                )
                print(
                    f"  {lines_read:>7,} lines: {elapsed_milestone:.2f}s "
                    f"({rate:>7,.0f}/s current, {cumulative_rate:>7,.0f}/s avg)"
                )
                milestone_time = time.perf_counter()

            if lines_read >= max_lines:
                break
    finally:
        f.close()

    total_elapsed = time.perf_counter() - start_total

    print("\n[Phase 3] Summary")
    print(f"  Lines read: {lines_read:,}")
    print(f"  Rows matched (P31): {rows_matched:,}")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(
        f"  Open overhead: {elapsed_open:.3f}s ({100 * elapsed_open / total_elapsed:.1f}%)"
    )
    print(f"  Processing: {total_elapsed:.2f}s")
    print(f"  Throughput: {lines_read / total_elapsed:,.0f} lines/sec")

    # Theoretical breakdown
    print("\n[Analysis]")
    print("  Since almost all time is in C/Cython (indexed_bzip2), bottleneck is:")
    print("  - Parallel decompression threads reading from disk")
    print("  - Data transfer from decompression buffer to Python")
    print("  - Python line iteration/splitting (minimal)")
    print("  - P31 byte search (minimal)")


def profile_cprofile(source_path: Path, max_lines: int = 100_000) -> None:
    """The same read under cProfile, reported by cumulative and self time."""

    thread_count = get_optimal_bzip2_threads()

    def read_and_filter():
        lines_read = 0
        rows_matched = 0

        with indexed_bzip2.open(str(source_path), parallelization=thread_count) as f:
            for line in f:
                lines_read += 1
                if P31_BYTES in line:
                    rows_matched += 1
                if lines_read >= max_lines:
                    break

        return lines_read, rows_matched

    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()

    lines_read, rows_matched = read_and_filter()

    profiler.disable()

    # Print stats
    print(
        f"\nProcessed: {lines_read:,} lines, {rows_matched:,} matched (thread_count={thread_count})\n"
    )

    print("=== Top 20 by total time (cumulative) ===")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    print("\n=== Top 20 by time in function (self) ===")
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("time")
    ps.print_stats(20)
    print(s.getvalue())


MODES = {
    "timings": profile_timings,
    "cprofile": profile_cprofile,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Path to wikidata bz2 file")
    parser.add_argument(
        "--max-lines", type=int, default=100_000, help="Max lines to process"
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODES),
        default="timings",
        help="Which profile to run (default: timings).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source_path = Path(args.input)
    if not validate_input_path(source_path):
        return 1

    print(f"Profiling {source_path} ({format_size_gb(source_path)}) [{args.mode}]")
    print(f"Max lines: {args.max_lines:,}\n")
    MODES[args.mode](source_path, args.max_lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
