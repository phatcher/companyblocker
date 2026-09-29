"""Put this repository's source roots on `sys.path`, once, for scripts in this directory.

Import this first, before any first-party import, and bare rather than dotted::

    import _bootstrap  # noqa: F401

    from blocking.contracts import BlockingRunConfig

**Bare, because a dotted import would need what this module supplies.**
`from scripts._bootstrap import ...` requires the repository root to already be
on `sys.path`, which is the thing being arranged. A directly-run script has its
own directory on `sys.path[0]` for free, so the bare form resolves with no setup
at all -- and `pytest.ini` puts `scripts` on `pythonpath` so the same bare import
resolves when a test imports a script as `scripts.<name>` instead of running it.

**First, because the roots must exist before the imports that need them.** This
module therefore imports nothing first-party itself; anything it imported would
hit the same missing roots. That rules out putting this in `cli_common`, which
imports `acquisition`.

The root is derived from this file's own location, so a script resolves against
the checkout it actually sits in rather than whichever checkout an editable
install happens to name. That matters in a worktree, where the two differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def source_roots() -> list[Path]:
    """Every directory holding importable first-party code, nearest first.

    `packages/*/src` is globbed rather than listed, so a new package needs no
    edit here. Only directories that exist are returned, which keeps a partial
    checkout from putting a phantom entry on the path.
    """

    roots = [REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tooling"]
    roots.extend(sorted((REPO_ROOT / "packages").glob("*/src")))
    return [root for root in roots if root.is_dir()]


def install() -> None:
    """Prepend every source root to `sys.path`, preserving order, without duplicates.

    Idempotent: a root already present is left where it is rather than moved, so
    importing this twice, or importing it from a script a test has also put on
    the path, changes nothing.
    """

    for root in reversed(source_roots()):
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)


install()
