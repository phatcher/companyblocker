"""Classify a test by what it does, not by where it sits or a marker on it.

`docs/TESTSTYLE.md`'s unit tier is the default and is meant to exclude a test
that materializes a layer, starts a subprocess, or reads a real corpus --
the three things that make a test slow or broad regardless of which area it
lives in. Those were decided by a marker someone remembered to add, which is
why almost nothing carried one: the fast selection was measured deselecting
only a few percent of the suite by count while still running most of its
wall time, because the cost sits in tests nobody tagged rather than in the
ones that were.

This module reads a test function's own source, plus the source of any
same-module helper it calls directly, and looks for three behavioural
signatures:

- **subprocess**: a call to `subprocess.run`/`Popen`/`call`/`check_call`/
  `check_output`.
- **layer materialization**: a call to a `materialize_<something>` function
  or to `swap_layer_into_place`, the repo's naming convention for the
  functions that publish a layer for real (`acquisition.match_ops.
  materialize_match_uri_artifact`, `validation.perturbation_materializer.
  materialize_perturbations`, `workspace.layer_layout.
  swap_layer_into_place`, and others matching the same shape).
- **corpus**: a repo-root computation (`Path(__file__).resolve()
  .parents[N]`) together with a `"data"` path segment, anywhere in the same
  block. Neither alone is a signal -- a repo root can be computed for other
  reasons, and almost every synthetic fixture uses `"data"` as a literal
  segment to mirror the production layout under `tmp_path` -- but the
  combination means the real `data/` directory is being resolved from the
  test file's own location rather than built fresh under `tmp_path`.

A match on any of the three means the test touches something the unit tier
excludes. The check is by behaviour, not by path or file location, so it
survives a test file moving between directories, which several are while
this rule is being decided.

The behaviour is usually not in the test's own body: the common shape here
is a fixture that does the heavy work (a `repo` fixture that shells out to
`git init`, say) and a test that only calls the module under test. So
`tooling/check_test_tier_markers.py`, which applies this rule, checks not
just a test's own source but every fixture it depends on, transitively, and
reports the test as broad if any of them match -- a test is exactly as broad
as the heaviest thing it depends on.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

SUBPROCESS_PATTERN = re.compile(
    r"\bsubprocess\.(?:Popen|run|call|check_call|check_output)\s*\("
)
MATERIALIZE_PATTERN = re.compile(r"\b(?:materialize_\w+|swap_layer_into_place)\s*\(")
# A repo-root computation (`Path(__file__).resolve().parents[N]`) and a
# `"data"` path segment, present anywhere in the same block: neither alone
# is a signal (a fixture may compute a repo root for other reasons, and
# almost every synthetic fixture uses "data" as a literal path segment to
# mirror the production layout under `tmp_path`), but the combination means
# the real `data/` directory is being resolved from the test file's own
# location rather than built fresh under `tmp_path`.
REAL_ROOT_PATTERN = re.compile(r"resolve\(\)\s*\.\s*parents\[\d+\]")
DATA_SEGMENT_PATTERN = re.compile(r"[\"']data[\"']")

REASON_PATTERNS = {
    "subprocess": SUBPROCESS_PATTERN,
    "layer": MATERIALIZE_PATTERN,
}


@dataclass(frozen=True)
class TierSignal:
    """What behaviour a test's source was found to contain, and why."""

    reasons: frozenset[str]

    def __bool__(self) -> bool:
        return bool(self.reasons)


def classify_source(source: str) -> TierSignal:
    """Return the behavioural signals a single block of source matches."""
    reasons = {
        name for name, pattern in REASON_PATTERNS.items() if pattern.search(source)
    }
    if REAL_ROOT_PATTERN.search(source) and DATA_SEGMENT_PATTERN.search(source):
        reasons.add("corpus")
    return TierSignal(frozenset(reasons))


def _module_level_function_sources(module_source: str) -> dict[str, str]:
    """Map each module-level function's name to its own source text.

    Only direct, same-module helpers are resolved -- one level of
    indirection, which covers the shape every real example in this repo
    uses (a test calling a private helper defined beside it in the same
    file). A helper that itself calls another helper is not followed
    further; the rule is deliberately shallow rather than a full call-graph
    walk.
    """
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return {}
    lines = module_source.splitlines(keepends=True)
    sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            sources[node.name] = "".join(lines[node.lineno - 1 : end])
    return sources


def _called_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def classify_callable(function_source: str, module_source: str) -> TierSignal:
    """Classify one function given its own source and its module's.

    Used for a test function itself, and equally for each fixture a test
    depends on (the checker calls this once per fixture too, since
    a fixture that starts a subprocess or materializes a layer -- setting
    up a throwaway git repo, say -- makes the test that depends on it just
    as broad as if the test called it directly).

    Checks the callable's own source, then the source of any module-level
    helper it calls directly (see `_module_level_function_sources`), and
    unions whatever each matches.
    """
    signal = classify_source(function_source)
    if signal:
        return signal
    helpers = _module_level_function_sources(module_source)
    if not helpers:
        return signal
    reasons: set[str] = set()
    for name in _called_names(function_source):
        helper_source = helpers.get(name)
        if helper_source is None:
            continue
        reasons |= classify_source(helper_source).reasons
    return TierSignal(frozenset(reasons))
