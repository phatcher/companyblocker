"""Asserts this package's own isolation, since the boundary checker cannot see it here.

`tach` 0.35 on Windows treats an import crossing source roots as an external distribution and
does not check it, so a forbidden import between this package and `company_vectorize` or the
repo's `src` areas would pass `tach check` silently. This test walks this package's own source
files with `ast` and fails on any such import directly, rather than relying on a tool that
cannot see across these roots on this platform.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_TOP_LEVEL_MODULES = {
    "company_vectorize",
    "acquisition",
    "analysis",
    "blocking",
    "training",
    "validation",
    "workspace",
}

PACKAGE_SRC = Path(__file__).resolve().parent.parent / "src" / "company_resolvers"


def _imported_top_level_modules(source_file: Path) -> set[str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            modules.add(node.module.split(".")[0])
    return modules


def test_package_source_files_exist() -> None:
    """Guards the test itself: an empty glob would make the forbidden-import test vacuous."""
    source_files = list(PACKAGE_SRC.rglob("*.py"))
    assert source_files, f"expected at least one source file under {PACKAGE_SRC}"


def test_no_forbidden_imports_from_company_vectorize_or_src_areas() -> None:
    for source_file in PACKAGE_SRC.rglob("*.py"):
        imported = _imported_top_level_modules(source_file)
        forbidden_hits = imported & FORBIDDEN_TOP_LEVEL_MODULES
        assert not forbidden_hits, (
            f"{source_file} imports forbidden module(s) {forbidden_hits}; "
            "company_resolvers must not import company_vectorize or any repo src area"
        )
