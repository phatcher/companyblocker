"""List the modules that no test imports directly.

A module has a direct test when a test file in the module's own test root
(``src/tests/<area>/`` for an area module, ``packages/<package>/tests/`` for a
package module) imports it by its own path, or imports from its package a name
that the package's ``__init__`` re-exports from it. Coverage reached only through
a caller, however high, does not count, and neither does an import from another
area's tests: a stage-level test proves the system as wired together works
today, not that the module's own decision rule is asserted anywhere, and a
module imported by another area's tests only as a helper is that same shape.
``docs/TESTSTYLE.md`` states the rule; ``src/tests/tooling/scripts/test_direct_tests.py``
pins the backlog this reports so it can only shrink.

Modules that define nothing (no top-level ``def`` or ``class``) are not counted,
nor are ``__init__.py`` and ``__main__.py``. Scripts under ``scripts/`` are not
counted either: they are thin wrappers over the areas, and their tests live
under the area owning the entry point each script runs, or under
``src/tests/tooling/scripts`` for the ones that drive no single area.

Usage::

    uv run --no-sync python tooling/check_direct_tests.py            # list, by root
    uv run --no-sync python tooling/check_direct_tests.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from _tooling_common import repository_root

PROJECT_ROOT = repository_root(Path(__file__))

SRC_AREAS = (
    "acquisition",
    "analysis",
    "blocking",
    "training",
    "validation",
    "workspace",
)
SKIP_FILES = {"__init__.py", "__main__.py"}


@dataclass(frozen=True)
class ModuleRoot:
    """A directory whose files import as `prefix.<relative dotted path>`."""

    directory: Path
    prefix: str


def module_roots() -> list[ModuleRoot]:
    roots = [ModuleRoot(PROJECT_ROOT / "src" / area, area) for area in SRC_AREAS]
    for pkg_dir in sorted((PROJECT_ROOT / "packages").glob("*/src/*")):
        if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
            roots.append(ModuleRoot(pkg_dir, pkg_dir.name))
    return [r for r in roots if r.directory.is_dir()]


def _defines_something(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )


def modules_under(root: ModuleRoot) -> dict[str, Path]:
    """Import path -> file, for every module under `root` that defines something."""
    found: dict[str, Path] = {}
    for path in sorted(root.directory.rglob("*.py")):
        if "__pycache__" in path.parts or path.name in SKIP_FILES:
            continue
        if not _defines_something(path):
            continue
        rel = path.relative_to(root.directory).with_suffix("")
        found[".".join((root.prefix, *rel.parts))] = path
    return found


def reexports(root: ModuleRoot) -> dict[str, str]:
    """Name -> defining module, from `__init__.py`'s eager and lazy re-exports.

    Eager: `from .mod import name`. Lazy: a module-level dict whose values are
    `(".mod", "name")` string pairs, the shape a `__getattr__` resolves on demand.
    Both count as the package re-exporting `name` from `mod`, which is what the
    direct-test rule in `docs/TESTSTYLE.md` says.
    """
    init = root.directory / "__init__.py"
    if not init.exists():
        return {}
    tree = ast.parse(init.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            target = f"{root.prefix}.{node.module}"
            for alias in node.names:
                if alias.name != "*":
                    mapping[alias.asname or alias.name] = target
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Dict):
                continue
            for key, item in zip(value.keys, value.values):
                if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                    continue
                if not (isinstance(item, ast.Tuple) and len(item.elts) == 2):
                    continue
                mod, _attr = item.elts
                if (
                    isinstance(mod, ast.Constant)
                    and isinstance(mod.value, str)
                    and mod.value.startswith(".")
                ):
                    mapping[key.value] = f"{root.prefix}.{mod.value.lstrip('.')}"
    return mapping


def own_test_root(root: ModuleRoot) -> Path:
    """The one directory whose tests count as a root's own, per TESTSTYLE's placement rule.

    An area's tests live in `src/tests/<area>/`; a package's in
    `packages/<package>/tests/`, two levels above its `src/<name>` directory.
    """
    if root.directory.parent == PROJECT_ROOT / "src":
        return PROJECT_ROOT / "src" / "tests" / root.prefix
    return root.directory.parents[1] / "tests"


def own_test_files(root: ModuleRoot) -> list[Path]:
    directory = own_test_root(root)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.rglob("test_*.py") if "__pycache__" not in p.parts
    )


def modules_imported_by(
    path: Path, known: set[str], reexport_map: dict[str, dict[str, str]]
) -> set[str]:
    """Every known module `path` reaches by import, resolving package re-exports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in known:
                    reached.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            base = node.module
            if base in known:
                reached.add(base)
            for alias in node.names:
                dotted = f"{base}.{alias.name}"
                if dotted in known:
                    reached.add(dotted)
                elif alias.name in reexport_map.get(base, {}):
                    reached.add(reexport_map[base][alias.name])
    return reached


def measure() -> dict[str, list[str]]:
    """Modules with no direct test, keyed by root prefix."""
    roots = module_roots()
    by_root = {root.prefix: modules_under(root) for root in roots}
    known = {m for mods in by_root.values() for m in mods}
    reexport_map = {root.prefix: reexports(root) for root in roots}
    missing: dict[str, list[str]] = {}
    for root in roots:
        tested: set[str] = set()
        for path in own_test_files(root):
            tested |= modules_imported_by(path, known, reexport_map)
        missing[root.prefix] = sorted(
            m for m in by_root[root.prefix] if m not in tested
        )
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true", help="emit {root: [modules]} as JSON"
    )
    args = parser.parse_args(argv)
    missing = measure()
    if args.json:
        print(json.dumps(missing, indent=2))
        return 0
    total = 0
    for prefix, modules in missing.items():
        if not modules:
            continue
        print(f"{prefix} ({len(modules)})")
        for module in modules:
            print(f"    {module}")
        total += len(modules)
    print(f"\n{total} module(s) with no direct test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
