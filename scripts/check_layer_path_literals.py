"""Fail when a file being committed takes or builds a path the workspace should own.

Two rule sets share this checker, each held to its own baseline of known offenders
under `src/tests/baselines/`, so a commit fails only on an offence beyond what its
files already carried.

**Layer partition literals** (`layer_path_literals.txt`). `jurisdiction_code=<value>`
is `workspace.layer_layout`'s own partition-directory naming (`PARTITION_COLUMN`), not
public vocabulary a caller should reproduce: a module that still constructs it -- in an
f-string interpolation (`f"jurisdiction_code={country}"`) or a literal glob
(`"jurisdiction_code=*"`) -- is a module that will disagree with the layer the next time
its shape changes, which is exactly what happened across several real call sites once
the `data/` layers gained a `primary/` family directory. `resolve_partition_dir`,
`resolve_primary_files`, `layer_partition_dir` and `partition_values`
(`src/workspace/layer_layout.py`) already express every shape a caller needs.

**The workspace path contract** (`workspace_paths.txt`). A script takes the three
workspace roots and otherwise only types and parameters; `workspace` decides where
everything lives. Each rule below is one way a file steps outside that:

- `path-flag`: an argument whose flag names a location (`--*-dir`, `--*-path`,
  `--*-out`, `--*-file`, `--*-root`, `--root`) or whose type is `Path`, anywhere but
  `cli_common`'s roots helper and its external-destination helper.
- `path-from-flag`: `Path(args.<name>)`, a location parsed from a flag.
- `root-join`: `/ "data"`, `/ "artifacts"`, `/ "tmp"` or `/ "config"`.
- `root-literal`: a string starting `data/`, `artifacts/`, `tmp/` or `config/`.
- `cwd`: `Path.cwd()` or `os.getcwd()`.
- `tempfile`: a `tempfile` call given no `dir=`, so scratch lands outside the temp root.
- `default-roots`: `default_workspace_roots(...)`, which ignores the roots' environment
  overrides.
- `repo-join`: a join below the repository root (`REPO_ROOT / ...`,
  `repository_root(...) / ...`, `Path(__file__).resolve().parents[n] / ...`) outside
  `workspace.repository`.

Scoped to code under `src/` and `scripts/`, and to non-docstring strings: a docstring
or a `#` comment describing a shape is not a bypass. `src/workspace/` owns the shapes
and `src/tests/` builds fixture trees, so both are exempt; `tooling/` is repository
gate code whose subject is the checkout itself, and `packages/` cannot depend on
`workspace` at all.

`src/tests/workspace/test_layer_path_literals.py` holds the repo-wide half: counts
that cannot grow.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from baseline import read_baseline

PATTERN = "jurisdiction_code="

LAYER_BASELINE = "layer_path_literals"
WORKSPACE_BASELINE = "workspace_paths"

SCAN_ROOTS = ("src", "scripts")

# workspace owns the shape it names; scoped, one-off scratch/synthetic-fixture code that
# builds its own throwaway directory tree is not reading or writing the real data/ layer.
EXEMPT_FILE_PREFIXES: tuple[str, ...] = ("src/workspace/", "src/tests/")

# A checker's own subject matter -- its docstring, argparse help text, and print output
# all name the pattern it looks for, same reason check_docstring_item_ids.py exempts
# itself. check_package_docs.py is here for the same reason: its own workspace-layout
# check must name this pattern to detect a package hand-building it. `_bootstrap.py`
# puts the source roots on `sys.path` before any first-party import can resolve, so it
# cannot ask `workspace.repository` for them.
EXEMPT_FILES: frozenset[str] = frozenset(
    {
        "scripts/check_layer_path_literals.py",
        "tooling/check_package_docs.py",
        "scripts/_bootstrap.py",
    }
)

LOCATION_FLAG_HELPERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("scripts/cli_common.py", "add_workspace_roots_args"),
        ("scripts/cli_common.py", "add_external_destination_arg"),
    }
)
"""The functions allowed to add location flags: the three workspace roots, and a
destination the workspace does not own, declared as such at each call."""

_LOCATION_FLAG = re.compile(
    r"^--(?:root|(?![\w-]*-per-file$)[\w-]+-(?:dir|path|out|file|root))$"
)
"""A flag naming a location; `--rows-per-file` and its kind count rows, not paths."""
_ROOT_NAMES = frozenset({"data", "artifacts", "tmp", "config"})
_ROOT_LITERAL = re.compile(r"^(?:data|artifacts|tmp|config)/")
_REPO_ROOT_NAMES = frozenset(
    {"REPO_ROOT", "PROJECT_ROOT", "REPOSITORY_ROOT", "repo_root", "project_root"}
)
_ARGS_NAMES = frozenset({"args", "arguments", "namespace", "parsed"})


@dataclass(frozen=True)
class Offence:
    lineno: int
    rule: str
    text: str


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """`id()` of every `ast.Constant` that is a module/class/function docstring."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def _fstring_segment_ids(tree: ast.AST) -> set[int]:
    # ast.walk visits a JoinedStr's own Constant segments as separate nodes
    # too, so they are tracked and skipped -- otherwise an f-string match is
    # counted twice, once per node.
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant):
                    ids.add(id(value))
    return ids


def _parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every non-docstring string in `tree`: a plain literal, or an f-string's
    literal segment reported at the f-string's line."""
    docstring_ids = _docstring_node_ids(tree)
    segment_ids = _fstring_segment_ids(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.append((node.lineno, value.value))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_ids
            and id(node) not in segment_ids
        ):
            found.append((node.lineno, node.value))
    return found


def layer_path_literals(path: Path) -> list[tuple[int, str]]:
    """`(line number, literal text)` for every non-docstring string literal in `path`
    (plain or an f-string segment) whose text contains `jurisdiction_code=`."""
    tree = _parse(path)
    if tree is None:
        return []
    return [(lineno, text) for lineno, text in _strings(tree) if PATTERN in text]


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_repo_root(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _REPO_ROOT_NAMES
    if isinstance(node, ast.Call):
        return _call_name(node.func) == "repository_root"
    if isinstance(node, ast.Subscript):
        value = node.value
        return isinstance(value, ast.Attribute) and value.attr == "parents"
    return False


def _functions_by_node(tree: ast.AST) -> dict[int, str]:
    """`id()` of every node to the name of the innermost function holding it."""
    owner: dict[int, str] = {}

    def visit(node: ast.AST, name: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        if name is not None:
            owner[id(node)] = name
        for child in ast.iter_child_nodes(node):
            visit(child, name)

    visit(tree, None)
    return owner


def workspace_path_offences(
    path: Path, *, relative: str | None = None
) -> list[Offence]:
    """Every place `path` steps outside the workspace path contract, by rule."""
    tree = _parse(path)
    if tree is None:
        return []
    rel = relative if relative is not None else Path(path).as_posix()
    owners = _functions_by_node(tree)
    found: list[Offence] = []

    for lineno, text in _strings(tree):
        if _ROOT_LITERAL.match(text):
            found.append(Offence(lineno, "root-literal", text))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "add_argument":
                in_roots_helper = (rel, owners.get(id(node))) in LOCATION_FLAG_HELPERS
                flags = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                typed_path = any(
                    keyword.arg == "type" and _call_name(keyword.value) == "Path"
                    for keyword in node.keywords
                    if isinstance(keyword.value, (ast.Name, ast.Attribute))
                )
                if not in_roots_helper and (
                    typed_path or any(_LOCATION_FLAG.match(flag) for flag in flags)
                ):
                    found.append(Offence(node.lineno, "path-flag", ", ".join(flags)))
            elif name == "Path" and node.args:
                first = node.args[0]
                if (
                    isinstance(first, ast.Attribute)
                    and isinstance(first.value, ast.Name)
                    and first.value.id in _ARGS_NAMES
                ):
                    found.append(
                        Offence(
                            node.lineno, "path-from-flag", f"Path(args.{first.attr})"
                        )
                    )
            elif name in {"cwd", "getcwd"} and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id in {"Path", "os"}:
                    found.append(Offence(node.lineno, "cwd", f"{owner.id}.{name}()"))
            elif name == "default_workspace_roots":
                found.append(Offence(node.lineno, "default-roots", name))
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tempfile"
                and not any(keyword.arg == "dir" for keyword in node.keywords)
            ):
                found.append(
                    Offence(node.lineno, "tempfile", f"tempfile.{node.func.attr}()")
                )
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if (
                isinstance(right, ast.Constant)
                and isinstance(right.value, str)
                and right.value in _ROOT_NAMES
            ):
                found.append(Offence(node.lineno, "root-join", f'/ "{right.value}"'))
            if _is_repo_root(node.left):
                found.append(Offence(node.lineno, "repo-join", ast.unparse(node)))

    return sorted(found, key=lambda offence: (offence.lineno, offence.rule))


def in_scope(relative: str) -> bool:
    """Whether a repository-relative path is code this checker governs."""
    return (
        relative.endswith(".py")
        and relative.split("/", 1)[0] in SCAN_ROOTS
        and relative not in EXEMPT_FILES
        and not any(relative.startswith(prefix) for prefix in EXEMPT_FILE_PREFIXES)
    )


def workspace_path_counts(path: Path, relative: str) -> dict[str, int]:
    """Baseline keys `<file> <rule>` to how many offences `path` carries."""
    counts = Counter(
        offence.rule for offence in workspace_path_offences(path, relative=relative)
    )
    return {f"{relative} {rule}": count for rule, count in counts.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    layer_allowed = read_baseline(LAYER_BASELINE)
    workspace_allowed = read_baseline(WORKSPACE_BASELINE)
    failed = False
    for name in args.filenames:
        path = Path(name)
        relative = path.as_posix()
        if not in_scope(relative) or not path.exists():
            continue

        layer_hits = layer_path_literals(path)
        if len(layer_hits) > layer_allowed.get(relative, 0):
            failed = True
            print(f"{relative}: hand-built jurisdiction_code= path outside workspace")
            for lineno, literal in layer_hits:
                print(f"  line {lineno}: {literal!r}")
            print(
                "  Route through workspace.layer_layout's resolve_partition_dir/"
                "resolve_primary_files/layer_partition_dir/partition_values instead."
            )

        offences = workspace_path_offences(path, relative=relative)
        for key, count in workspace_path_counts(path, relative).items():
            allowed = workspace_allowed.get(key, 0)
            if count <= allowed:
                continue
            failed = True
            rule = key.rsplit(" ", 1)[1]
            print(
                f"{relative}: {count} {rule} offence(s), {allowed} allowed by the baseline"
            )
            for offence in offences:
                if offence.rule == rule:
                    print(f"  line {offence.lineno}: {offence.text}")
    if failed:
        print(
            "\nA script takes the workspace roots and otherwise only types and parameters; "
            "name what it reads and writes with workspace.reference and a cli_common "
            "reference helper. See src/workspace/README.md."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
