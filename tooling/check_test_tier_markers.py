"""Flag a test that matches a broad-tier behavioural signature but carries no marker.

`docs/TESTSTYLE.md`'s unit tier excludes a test that starts a subprocess, materializes
a layer, or reads the real `data/` corpus; `tooling/test_tier_classifier.py` is the
static rule for what counts. Enforcing that rule first went through a
collection-time `conftest.py` hook that silently reclassified every matching test on
every run, on every branch, with its reason recorded only in `item.user_properties`,
which nothing read: a standing runtime mechanism serving a population of 39 tests,
with no reviewable artefact anywhere. That was withdrawn. The 39 are now real
`@pytest.mark.integration` decorators in their own test files -- a reviewable diff --
and this is the ratchet half that keeps them honest: a test added later that matches
one of the three signatures but carries no marker fails the gate instead of quietly
running in the wrong tier forever.

A test's tier depends not just on its own source but on every fixture it depends on
transitively (a test can look narrow while a fixture it uses shells out to `git init`,
say), and fixture resolution -- following a name through `conftest.py` inheritance and
parametrisation -- is pytest's own machinery, not something worth re-deriving
statically. So the measurement here runs a real `pytest --collect-only` pass with this
module registered as a plugin, rather than walking the AST the way
`check_direct_tests.py` and `check_layer_path_literals.py` do; borrowing pytest's own
resolution for one collection pass is deliberate, not a shortcut around the rule.

This module doubles as that plugin. Run directly, it prints every unmarked match found
during a real collection of the suite:

    pytest --collect-only -q -p check_test_tier_markers

Whole-tree collection is the expensive half of that measurement, not the classification
that follows it, and a full suite run has already paid for one by the time the ratchet
executes. So a run that wants the saving registers this module with `-p check_test_tier_markers`:
the hook below keeps that run's own collection, and `measure_in_session()` classifies it
in place. The hook marks nothing, deselects nothing and classifies nothing, which is what
separates it from the reclassifying hook described above; it holds a list and lets the
reader do the work. Registration stays on the invocation rather than in `pytest.ini`,
since it is an optimization and every editor and tool would otherwise inherit it.

`measure()` remains the fallback, running the pass in a subprocess (collection has side
effects on pytest's internal state that make a second, nested `pytest.main()` call
inside an already-running suite unreliable). It is what a run collecting only a subset
uses, since a subset containing no unmarked match says nothing about the tests it never
collected. Either way the answer reaches
`src/tests/tooling/scripts/test_check_test_tier_markers.py`, which checks it against
`src/tests/baselines/test_tier_markers.txt`.

Usage::

    uv run python tooling/check_test_tier_markers.py            # list
    uv run python tooling/check_test_tier_markers.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess  # nosec B404 - dev-tooling script; see nosec B603 at its call site
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from _tooling_common import repository_root

PROJECT_ROOT = repository_root(Path(__file__))
# Importable as `test_tier_classifier` under pytest (pythonpath=".") but not
# when this file is run directly, since a directly-run script's own directory, not the
# repo root, is what lands on sys.path -- same shim `scripts/acquire_companies.py` uses.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test_tier_classifier import TierSignal, classify_callable

REPORT_PREFIX = "TIER-MARKER-MISSING"
REPORT_LINE = re.compile(rf"^{REPORT_PREFIX} \[(?P<reasons>[^\]]*)\] (?P<nodeid>.+)$")

# `tooling/test_tier_classifier.py`'s own direct tests pass a sample of matching source
# as a plain string (`classify_source(source)`) to assert what the classifier finds in
# it. That sample text lands, verbatim, inside the test function's own real source too
# -- a nested string literal the checker's raw-text pattern match cannot tell apart
# from a live call -- so this file's tests are its own subject matter, the same reason
# `check_layer_path_literals.py` exempts itself.
EXEMPT_TEST_FILES = frozenset({"src/tests/tooling/test_test_tier_classifier.py"})

# Keyed by (function's module, qualname): most fixtures are shared across many tests
# (`tmp_path`, `repo`, ...), so classifying one once and reusing the result is what
# keeps a 6500-test collection pass affordable.
_signal_cache: dict[tuple[str, str], TierSignal] = {}

# The surrounding run's own collection, kept by the hook below when that run collected
# the whole tree, so the ratchet can read it instead of starting a second one. `None`
# means no such collection was seen and the subprocess pass is the only measurement.
_collected_items: list[pytest.Item] | None = None


def _classify_cached(function: Callable[..., object]) -> TierSignal:
    key = (getattr(function, "__module__", ""), getattr(function, "__qualname__", ""))
    if key in _signal_cache:
        return _signal_cache[key]
    module = inspect.getmodule(function)
    if module is None:
        signal = TierSignal(frozenset())
    else:
        try:
            function_source = inspect.getsource(function)
            module_source = inspect.getsource(module)
        except (OSError, TypeError):
            signal = TierSignal(frozenset())
        else:
            signal = classify_callable(function_source, module_source)
    _signal_cache[key] = signal
    return signal


def _fixture_functions(item: pytest.Item) -> list[Callable[..., object]]:
    info = getattr(item, "_fixtureinfo", None)
    fixturenames = getattr(item, "fixturenames", None)
    if info is None or fixturenames is None:
        return []
    functions = []
    for name in fixturenames:
        fixturedefs = info.name2fixturedefs.get(name)
        if not fixturedefs:
            continue
        func = getattr(fixturedefs[-1], "func", None)
        if func is not None:
            functions.append(func)
    return functions


def unmarked_matches(items: list[pytest.Item]) -> dict[str, str]:
    """nodeid -> comma-joined reasons, for every collected item that matches a
    broad-tier signature but carries neither `integration` nor `performance`."""
    matches: dict[str, str] = {}
    for item in items:
        function = getattr(item, "function", None)
        if function is None:
            continue
        if item.nodeid.split("::", 1)[0].replace("\\", "/") in EXEMPT_TEST_FILES:
            continue
        if item.get_closest_marker("integration") or item.get_closest_marker(
            "performance"
        ):
            continue
        reasons: set[str] = set(_classify_cached(function).reasons)
        for fixture_function in _fixture_functions(item):
            reasons |= _classify_cached(fixture_function).reasons
        if reasons:
            matches[item.nodeid] = ",".join(sorted(reasons))
    return matches


def _collects_the_whole_tree(config: pytest.Config) -> bool:
    """Whether this run's collection covers the default tree `pytest.ini` names.

    A run given explicit paths or a `-k` expression collects a subset, and a subset
    that happens to contain no unmarked match is not evidence that none exists.
    """
    return not getattr(config.option, "file_or_dir", None) and not getattr(
        config.option, "keyword", ""
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Keep the whole-tree collection for `measure_in_session`, and print each match
    under `--collect-only`.

    Marks nothing and deselects nothing -- see the module docstring for why a standing
    marking hook was rejected. `tryfirst` is what makes the kept list usable: a `-m`
    filter deselects inside this same hook, so running after it would hand the ratchet
    a collection with the marked tests already removed, which is exactly the population
    it exists to check.

    Classification is deliberately not done here. It costs a few seconds, and a run
    that never reaches the ratchet test (the unit selection deselects it) would pay
    that for nothing, so the work is left to whoever reads the list.
    """
    global _collected_items
    if _collects_the_whole_tree(config):
        _collected_items = list(items)
    if config.option.collectonly:
        for nodeid, reasons in sorted(unmarked_matches(items).items()):
            print(f"{REPORT_PREFIX} [{reasons}] {nodeid}")


def measure_with_reasons() -> dict[str, str]:
    """nodeid -> comma-joined reasons, for every unmarked test matching a broad-tier
    signature. Runs a real `pytest --collect-only` pass, in a subprocess, with this
    module as the collection plugin."""
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "check_test_tier_markers",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    found: dict[str, str] = {}
    for line in result.stdout.splitlines():
        match = REPORT_LINE.match(line)
        if match:
            found[match.group("nodeid")] = match.group("reasons")
    return found


def measure() -> dict[str, int]:
    """nodeid -> 1, for every unmarked test matching a broad-tier signature.

    The shape `src/tests/baselines/check_baseline` expects; see
    `measure_with_reasons` for the reason each one matched.
    """
    return {nodeid: 1 for nodeid in measure_with_reasons()}


def measure_in_session() -> dict[str, int]:
    """`measure()`'s answer, taken from the surrounding run's own collection when that
    run collected the whole tree, and from a subprocess pass when it did not.

    A full run already pays for one whole-tree collection, which is the expensive part
    of this measurement rather than the classification that follows it; starting a
    second one inside that run doubles it. A narrower run has no such collection to
    read, so it still spawns its own rather than checking a subset and reporting a
    clean result for the tests it never looked at.
    """
    if _collected_items is None:
        return measure()
    return {nodeid: 1 for nodeid in unmarked_matches(_collected_items)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true", help="emit {nodeid: reasons} as JSON"
    )
    args = parser.parse_args(argv)
    found = measure_with_reasons()
    if args.json:
        print(json.dumps(found, indent=2))
        return 0
    for nodeid, reasons in sorted(found.items()):
        print(f"  [{reasons}] {nodeid}")
    print(f"\n{len(found)} unmarked test(s) match a broad-tier signature")
    return 0


if __name__ == "__main__":
    sys.exit(main())
