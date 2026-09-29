"""Where a published report sits: `docs/reports/` in the checkout, tracked in git.

A run's own output stays under `artifacts/` and is never tracked. What is
reported on, a promoted tokenizer's report and pictures or a comparison across
scopes, is copied here under fixed names, so git keeps the picture and its
history. This module is the one place that composes a path under
`docs/reports/`, and it writes nothing but what a caller hands it.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from .roots import WorkspaceRoots

TOKENIZER_REPORTS = "tokenizers"


def tokenizer_reports_dir(roots: WorkspaceRoots, *, scope: str | None = None) -> Path:
    """`docs/reports/tokenizers/`, or one scope's directory beneath it."""
    base = roots.checkout / "docs" / "reports" / TOKENIZER_REPORTS
    return base if scope is None else base / scope


def publish(
    directory: Path,
    *,
    texts: Mapping[str, str] | None = None,
    files: Mapping[str, Path] | None = None,
    replace: bool = False,
) -> None:
    """Write `texts` and copy `files` into `directory` under the names given.

    With `replace`, the files already directly in `directory` are removed
    first, so a picture the new report no longer has does not outlive it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if replace:
        for existing in directory.iterdir():
            if existing.is_file():
                existing.unlink()
    for name, text in (texts or {}).items():
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    for name, source in (files or {}).items():
        shutil.copyfile(source, directory / name)


__all__ = ["TOKENIZER_REPORTS", "publish", "tokenizer_reports_dir"]
