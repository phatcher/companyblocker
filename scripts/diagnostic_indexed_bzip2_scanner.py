#!/usr/bin/env python3
"""
Diagnostic scanner for indexed_bzip2 crash investigation.

Focused experimentation: find crash point, then test if thread count fixes it.

Usage:
  # Find where crash happens (full file, 8 threads, 120s timeout)
  python diagnostic_indexed_bzip2_scanner.py \
    --input data/wikidata/acquire/2026-07-02/wikidata-all-2026-07-02.json.bz2 \
    --chunks -1 --threads 8 --timeout 120

  # Test region 1000-2000 with 4 threads
  python diagnostic_indexed_bzip2_scanner.py \
    --input data/wikidata/acquire/2026-07-02/wikidata-all-2026-07-02.json.bz2 \
    --chunks 1000:2000 --threads 4 --timeout 120

  # Test first 100 chunks with varying threads
  for threads in 2 4 8 12 16; do
    python diagnostic_indexed_bzip2_scanner.py \
      --input data/wikidata/acquire/2026-07-02/wikidata-all-2026-07-02.json.bz2 \
      --chunks 100 --threads $threads --timeout 60
  done
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import indexed_bzip2

CHUNK_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB
_BYTE_NEWLINE = b"\n"


def iter_lines_from_handle(handle, max_lines: int = -1):
    """
    Generator that yields lines, matching production pattern exactly.
    Uses chunked reading with remainder carry-over.
    """
    line_number = 0
    carry = b""

    while True:
        chunk = handle.read(CHUNK_SIZE_BYTES)
        if not chunk:
            break

        if carry:
            chunk = carry + chunk

        if chunk.endswith(_BYTE_NEWLINE):
            parts = chunk.split(_BYTE_NEWLINE)
            complete_lines = parts[:-1]
            carry = b""
        else:
            parts = chunk.split(_BYTE_NEWLINE)
            complete_lines = parts[:-1]
            carry = parts[-1]

        for raw_line in complete_lines:
            line_number += 1
            if max_lines > 0 and line_number > max_lines:
                return
            yield line_number, raw_line

    if carry:
        line_number += 1
        if max_lines < 0 or line_number <= max_lines:
            yield line_number, carry


def scan_chunks(
    file_path: Path,
    chunk_count: int,
    threads: int,
    timeout_seconds: int,
    max_lines: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Scan file using indexed_bzip2 with specified chunk count and thread count.

    Args:
        file_path: Path to bz2 file
        chunk_count: Number of chunks to read (-1 for full file)
        threads: Thread count for parallelization
        timeout_seconds: Wall-clock timeout in seconds
        verbose: Print progress messages

    Returns:
        Dictionary with results: chunks_completed, lines_processed, elapsed_time,
        crash_chunk (if any), exception (if any)
    """
    result = {
        "file": str(file_path),
        "chunk_count_requested": chunk_count,
        "threads": threads,
        "timeout_seconds": timeout_seconds,
        "chunks_completed": 0,
        "lines_processed": 0,
        "elapsed_time": 0.0,
        "crash_chunk": None,
        "exception": None,
        "exception_traceback": None,
    }

    if not file_path.exists():
        result["exception"] = f"File not found: {file_path}"
        return result

    start_time = time.perf_counter()

    try:
        with indexed_bzip2.open(str(file_path), parallelization=threads) as handle:
            # Use the exact production generator pattern
            for line_number, raw_line in iter_lines_from_handle(
                handle, max_lines=max_lines if max_lines > 0 else -1
            ):
                result["lines_processed"] = line_number

                # Estimate chunk index (roughly every 1.8M lines per 16MB chunk)
                result["chunks_completed"] = max(1, line_number // 1800000)

                elapsed = time.perf_counter() - start_time
                result["elapsed_time"] = elapsed

                # Check timeout
                if timeout_seconds > 0 and elapsed > timeout_seconds:
                    result["exception"] = f"TIMEOUT after {elapsed:.1f}s"
                    break

                # Check if we've consumed enough chunks
                chunks_completed = result["chunks_completed"]
                if (
                    chunk_count > 0
                    and isinstance(chunks_completed, int)
                    and chunks_completed >= chunk_count
                ):
                    break

                if verbose and line_number % 100000 == 0:
                    rate = line_number / elapsed if elapsed > 0 else 0
                    print(
                        f"[Lines {line_number:10d}] "
                        f"chunks≈{result['chunks_completed']:6d} "
                        f"elapsed={elapsed:8.1f}s "
                        f"rate={rate:8.0f} lines/sec"
                    )

    except Exception as e:  # noqa: BLE001 -- diagnostic scanner: capture the crash into the result payload rather than propagate
        result["exception"] = str(e)
        result["exception_traceback"] = traceback.format_exc()
        result["crash_chunk"] = result["chunks_completed"]

    result["elapsed_time"] = time.perf_counter() - start_time
    return result


def parse_chunk_spec(spec: str) -> tuple[int, int | None]:
    """
    Parse chunk specification: int, -1 (full file), or range "N:M".
    Returns (start_chunk, end_chunk) for range, or (count, None) for simple count.
    """
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid range: {spec}")
        try:
            start = int(parts[0])
            end = int(parts[1])
            return start, end
        except ValueError:
            raise ValueError(f"Invalid range: {spec}")
    else:
        try:
            count = int(spec)
            return count, None
        except ValueError:
            raise ValueError(f"Invalid chunk count: {spec}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic scanner for indexed_bzip2 crash investigation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to bz2 file",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="-1",
        help="Chunk count: integer for N chunks, -1 for full file, N:M for range (default: -1)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Maximum lines to process (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Thread count for parallelization (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds (0 = no timeout, default: 120)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress every 10 chunks",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Append results to CSV file",
    )

    args = parser.parse_args()

    # Parse chunk specification
    try:
        chunk_spec = parse_chunk_spec(args.chunks)
        if chunk_spec[1] is not None:
            # Range - do binary search bisection or just one run
            chunk_count = chunk_spec[1] - chunk_spec[0]
            print(
                f"Testing chunk range {chunk_spec[0]}:{chunk_spec[1]} "
                f"({chunk_count} chunks)"
            )
        else:
            chunk_count = chunk_spec[0]
            if chunk_count == -1:
                print("Testing full file")
            else:
                print(f"Testing {chunk_count} chunks")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Input: {args.input}\n"
        f"Threads: {args.threads}\n"
        f"Timeout: {args.timeout}s\n"
        f"Verbose: {args.verbose}\n"
    )

    # Run scan
    result = scan_chunks(
        args.input,
        chunk_count
        if chunk_spec[1] is None
        else -1,  # -1 means full file for this test
        args.threads,
        args.timeout,
        max_lines=args.max_lines,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Chunks completed: {result['chunks_completed']}")
    print(f"Lines processed: {result['lines_processed']}")
    print(f"Elapsed time: {result['elapsed_time']:.1f}s")
    if result["lines_processed"] > 0 and result["elapsed_time"] > 0:
        rate = result["lines_processed"] / result["elapsed_time"]
        print(f"Rate: {rate:.0f} lines/sec")
    if result["exception"]:
        print(f"Exception: {result['exception']}")
        if result["exception_traceback"] and args.verbose:
            print(f"Traceback:\n{result['exception_traceback']}")
    if result["crash_chunk"]:
        print(f"Crash at chunk: {result['crash_chunk']}")

    # Append to CSV if requested
    if args.csv:
        csv_path = args.csv
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "threads",
                    "chunks_requested",
                    "chunks_completed",
                    "lines_processed",
                    "elapsed_time",
                    "rate_lines_per_sec",
                    "crash_chunk",
                    "exception",
                ],
            )
            if write_header:
                writer.writeheader()

            rate = (
                result["lines_processed"] / result["elapsed_time"]
                if result["elapsed_time"] > 0
                else 0
            )
            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "threads": args.threads,
                    "chunks_requested": args.chunks,
                    "chunks_completed": result["chunks_completed"],
                    "lines_processed": result["lines_processed"],
                    "elapsed_time": f"{result['elapsed_time']:.1f}",
                    "rate_lines_per_sec": f"{rate:.0f}",
                    "crash_chunk": result["crash_chunk"] or "",
                    "exception": result["exception"] or "",
                }
            )
        print(f"\nResults appended to {csv_path}")

    # Exit with error if crashed
    if result["exception"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
