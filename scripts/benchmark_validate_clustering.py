from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - dev-tooling script; see nosec B603 at its call site
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_root_arg, run_reporting_argument_errors

from workspace.artifact_layout import perf_artifact_root
from workspace.roots import default_workspace_roots

_RUNTIME_RE = re.compile(
    r"\[validation\] completed in (?P<seconds>[0-9]+(?:\.[0-9]+)?)s"
)
_PROGRESS_RE = re.compile(
    r"\[info\] progress .*source_rows_processed=(?P<source_rows_processed>\d+) "
    r"source_rows_found=(?P<source_rows_found>\d+) "
    r"edges_found=(?P<edges_found>\d+)"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark validate_clustering similarity backends."
    )
    parser.add_argument("--sources", dest="source_systems", nargs="+", required=True)
    parser.add_argument("--targets", dest="target_systems", nargs="+", required=True)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--source-chunk-size", type=int, default=1_000)
    parser.add_argument(
        "--backends", nargs="+", default=["sklearn", "sparse_dot_topn", "svd_rerank"]
    )
    parser.add_argument("--svd-candidates", type=int, default=200)
    parser.add_argument("--svd-dimensions", type=int, default=128)
    add_root_arg(parser, help_text="Project root path.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _run_backend(args: argparse.Namespace, backend: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(args.root).resolve() / "scripts" / "validate_clustering.py"),
        "--root",
        str(Path(args.root).resolve()),
        "--sources",
        *args.source_systems,
        "--targets",
        *args.target_systems,
        "--max-rows",
        str(args.max_rows),
        "--source-chunk-size",
        str(args.source_chunk_size),
        "--similarity-backend",
        backend,
    ]
    if backend == "svd_rerank":
        command.extend(
            [
                "--additional-args",
                f"backend.svd_candidates={args.svd_candidates}",
                f"backend.svd_dimensions={args.svd_dimensions}",
            ]
        )

    started = time.perf_counter()
    # A hardcoded argv (sys.executable + an in-repo script path); not shell=True, no
    # untrusted input reaches the command line.
    completed = subprocess.run(  # nosec B603
        command, capture_output=True, text=True, check=False
    )
    wall_seconds = time.perf_counter() - started

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    runtime_seconds = None
    runtime_match = _RUNTIME_RE.search(stdout)
    if runtime_match:
        runtime_seconds = float(runtime_match.group("seconds"))

    progress = {
        "source_rows_processed": 0,
        "source_rows_found": 0,
        "edges_found": 0,
    }
    for match in _PROGRESS_RE.finditer(stdout):
        progress = {
            "source_rows_processed": int(match.group("source_rows_processed")),
            "source_rows_found": int(match.group("source_rows_found")),
            "edges_found": int(match.group("edges_found")),
        }

    return {
        "backend": backend,
        "exit_code": int(completed.returncode),
        "wall_seconds": wall_seconds,
        "runtime_seconds": runtime_seconds,
        **progress,
        "stdout": stdout,
        "stderr": stderr,
    }


def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    results = []
    for backend in args.backends:
        print(f"[benchmark] running backend={backend}")
        outcome = _run_backend(args, backend)
        results.append(outcome)
        print(
            f"[benchmark] backend={backend} exit_code={outcome['exit_code']} "
            f"runtime_seconds={outcome['runtime_seconds']} wall_seconds={outcome['wall_seconds']:.2f} "
            f"source_rows_processed={outcome['source_rows_processed']} edges_found={outcome['edges_found']}"
        )

    output_path = (
        Path(args.output)
        if args.output
        else perf_artifact_root(roots) / "validate_clustering_backend_benchmark.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_systems": args.source_systems,
        "target_systems": args.target_systems,
        "max_rows": int(args.max_rows),
        "source_chunk_size": int(args.source_chunk_size),
        "svd_candidates": int(args.svd_candidates),
        "svd_dimensions": int(args.svd_dimensions),
        "results": results,
    }
    output_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[benchmark] wrote {output_path}")

    failed = [
        entry
        for entry in results
        if isinstance(entry["exit_code"], int) and entry["exit_code"] != 0
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
