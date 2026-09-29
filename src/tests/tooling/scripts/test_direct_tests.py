"""No module may lack a direct test beyond the ones the baseline already lists.

`tooling/check_direct_tests.py` is the measurement: a module has a direct test when
a test file in its own test root imports it by its own path, or imports from its
package a name the package's `__init__` re-exports from it; an import from another
area's tests is a caller's coverage. `src/tests/baselines/direct_tests.txt` lists
the modules that still lack one, so a new module cannot ship with nothing asserting
its own contract, however much incidental coverage it gets through its callers.
Delete a module's line in the same commit that gives it a direct test; never add one.
`docs/TESTSTYLE.md` states the rule.
"""

from __future__ import annotations

from pathlib import Path

from baseline import check_baseline
from check_direct_tests import measure, module_roots, own_test_root


def test_test_root_is_the_modules_own_area_or_package(repo_root: Path) -> None:
    """A workspace module imported only from acquisition or pipeline tests has a
    caller's coverage, not a direct test, so each root reads one directory."""
    by_prefix = {root.prefix: root for root in module_roots()}
    assert (
        own_test_root(by_prefix["workspace"])
        == repo_root / "src" / "tests" / "workspace"
    )
    assert (
        own_test_root(by_prefix["company_cleanse"])
        == repo_root / "packages" / "company_cleanse" / "tests"
    )


def test_measurement_sees_every_root() -> None:
    """A root that silently drops out would empty the measurement without any test being written."""
    prefixes = {root.prefix for root in module_roots()}
    assert {
        "acquisition",
        "workspace",
        "company_cleanse",
        "company_vectorize",
    } <= prefixes


def test_no_module_lacks_a_direct_test_beyond_the_baseline() -> None:
    current = {module: 1 for modules in measure().values() for module in modules}
    check_baseline("direct_tests", current)
