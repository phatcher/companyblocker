from __future__ import annotations

import os
from pathlib import Path


def get_optimal_bzip2_threads() -> int:
    """Calculate a bounded thread count for indexed_bzip2."""
    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cores = os.cpu_count() or 1

    return max(2, min(cores, 24))


def validate_input_path(source_path: Path) -> bool:
    if source_path.exists():
        return True
    print(f"ERROR: {source_path} not found")
    return False


def format_size_gb(source_path: Path) -> str:
    return f"{source_path.stat().st_size / 1e9:.1f}GB"
