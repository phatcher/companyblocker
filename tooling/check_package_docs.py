#!/usr/bin/env python3
"""Verify each packages/*/README.md stays self-contained and documents its public API.

Packages under packages/ are meant to be independently consumable (e.g. as a git
dependency from another project), so their README must not silently assume the
rest of this monorepo is present, and must not silently drift out of sync with
what the package actually exports.

Checks, per packages/<name>/README.md:
  1. Every relative markdown link resolves to a real file within the package's own
     directory tree. A link that escapes the package, or points at a file that
     doesn't exist at all, fails. Cross-repo context belongs in plain prose (a
     backticked path, not a clickable link) -- a package README names a monorepo
     file as plain text.
  2. Every name in packages/<name>/src/<name>/__init__.py's __all__ appears
     somewhere in the README text, so newly-exported public API doesn't silently
     go undocumented.

Checks, per packages/<name>/src/**/*.py (see packages/README.md's Input contract):
  3. No file imports `workspace` (or a `workspace.*` submodule) -- that package
     encodes this monorepo's on-disk layout, and importing it makes a package's
     correctness depend on a layout that does not exist once the package is used
     standalone.
  4. No file hand-builds a `jurisdiction_code=`-style partition path, in a plain
     string or an f-string, outside of a docstring. That is the same layout
     knowledge as #3, spelled out literally instead of imported.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from _tooling_common import repository_root

REPO_ROOT = repository_root(Path(__file__))
PACKAGES_DIR = REPO_ROOT / "packages"

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _is_external_link(target: str) -> bool:
    return not target or target.startswith(("http://", "https://", "mailto:", "#"))


def check_readme_links(package_dir: Path, readme: Path) -> list[str]:
    errors: list[str] = []
    text = readme.read_text(encoding="utf-8")
    for match in MD_LINK_RE.finditer(text):
        target = match.group(1).strip()
        if _is_external_link(target):
            continue

        path_part = target.split("#", 1)[0].strip()
        if not path_part:
            continue

        resolved = (readme.parent / path_part).resolve()
        try:
            resolved.relative_to(package_dir.resolve())
        except ValueError:
            errors.append(
                f"{readme.relative_to(REPO_ROOT)}: link '{target}' escapes the "
                "package directory -- use plain prose for monorepo-only context, "
                "not a clickable link"
            )
            continue

        if not resolved.exists():
            errors.append(
                f"{readme.relative_to(REPO_ROOT)}: link '{target}' points at a "
                "nonexistent file"
            )
    return errors


def _load_all_names(init_path: Path) -> list[str] | None:
    if not init_path.exists():
        return None
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "__all__" not in target_names or not isinstance(
            node.value, (ast.List, ast.Tuple)
        ):
            continue
        return [
            elt.value
            for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        ]
    return None


def check_exports_documented(
    package_dir: Path, package_name: str, readme: Path
) -> list[str]:
    init_path = package_dir / "src" / package_name / "__init__.py"
    all_names = _load_all_names(init_path)
    if not all_names:
        return []

    readme_text = readme.read_text(encoding="utf-8")
    errors = []
    for name in all_names:
        if not re.search(rf"\b{re.escape(name)}\b", readme_text):
            errors.append(
                f"{readme.relative_to(REPO_ROOT)}: exported name '{name}' (in "
                "__all__) is not mentioned anywhere in the README"
            )
    return errors


_PARTITION_MARKER = "jurisdiction_code="


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Identify the `Constant` nodes that serve as a docstring.

    A docstring is an ordinary string literal in the AST, so a text scan
    that does not exclude these would flag prose like packages/README.md's
    own description of the partition-path shape. Collected by identity
    (`id()`) since `ast.Constant` isn't hashable in a way we want to rely on
    for equality.
    """
    docstring_ids: set[int] = set()
    doc_bearing = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, doc_bearing) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstring_ids.add(id(first.value))
    return docstring_ids


def _check_no_workspace_layout_in_file(path: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path.relative_to(repo_root)}: could not parse ({exc})"]

    docstring_ids = _docstring_constant_ids(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "workspace" or alias.name.startswith("workspace."):
                    errors.append(
                        f"{path.relative_to(repo_root)}:{node.lineno}: imports "
                        f"'{alias.name}' -- a package must never import this "
                        "monorepo's src/workspace, see packages/README.md's "
                        "Input contract"
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "workspace" or module.startswith("workspace."):
                errors.append(
                    f"{path.relative_to(repo_root)}:{node.lineno}: imports from "
                    f"'{module}' -- a package must never import this monorepo's "
                    "src/workspace, see packages/README.md's Input contract"
                )
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _PARTITION_MARKER in node.value
            and id(node) not in docstring_ids
        ):
            errors.append(
                f"{path.relative_to(repo_root)}:{node.lineno}: hand-builds a "
                f"'{_PARTITION_MARKER}'-style partition path -- that is "
                "src/workspace-owned layout knowledge, see packages/README.md's "
                "Input contract"
            )
    return errors


def check_no_workspace_layout(package_dir: Path, repo_root: Path) -> list[str]:
    """Fail if any file under a package's own src/ leans on a workspace layout.

    Enforces the two forms packages/README.md's Input contract names: importing
    `workspace` outright, and hand-building the same layout knowledge as a
    `jurisdiction_code=`-style partition-path literal.
    """
    src_dir = package_dir / "src"
    if not src_dir.exists():
        return []

    errors: list[str] = []
    for py_file in sorted(src_dir.rglob("*.py")):
        errors.extend(_check_no_workspace_layout_in_file(py_file, repo_root))
    return errors


def main() -> int:
    all_errors: list[str] = []
    checked = 0

    for package_dir in sorted(p for p in PACKAGES_DIR.iterdir() if p.is_dir()):
        all_errors.extend(check_no_workspace_layout(package_dir, REPO_ROOT))

        readme = package_dir / "README.md"
        if not readme.exists():
            continue
        checked += 1
        all_errors.extend(check_readme_links(package_dir, readme))
        all_errors.extend(
            check_exports_documented(package_dir, package_dir.name, readme)
        )

    if all_errors:
        print("Package documentation consistency check failed:\n")
        for error in all_errors:
            print(f"  - {error}")
        print(f"\n{len(all_errors)} issue(s) found.")
        return 1

    print(f"[OK] All {checked} package README(s) validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
