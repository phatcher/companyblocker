"""Find the child directories of a root that hold parquet output."""

from __future__ import annotations

from pathlib import Path


def discover_child_dirs_with_parquet(
    *,
    parent_dir: Path,
    subdir_name: str | None = None,
    required_file: str | None = None,
) -> list[Path]:
    if not parent_dir.exists() or not parent_dir.is_dir():
        return []

    discovered: list[Path] = []
    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir():
            continue

        candidate = child / subdir_name if subdir_name else child
        if not candidate.exists() or not candidate.is_dir():
            continue

        if required_file is not None:
            if (candidate / required_file).exists():
                discovered.append(child)
            continue

        if any(candidate.glob("*.parquet")):
            discovered.append(child)

    return discovered
