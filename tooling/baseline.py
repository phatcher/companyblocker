"""Compare a measurement against its baseline file of known offenders.

A rule such as "every module has a direct test" cannot be switched on while
offenders exist, so each ratchet keeps them in `src/tests/baselines/<name>.txt`,
one per line, optionally followed by a count. The test asserts that what it finds
today is within the baseline: a new offender, or a higher count for a known one,
fails with the exact line it would take to add, which is the line not to add.

A line the current run no longer needs is pruned from the file automatically, and
warned about so it is visible in the diff and included in the same commit. That
only happens in the main checkout, detected by `.git` being a directory: a worktree's
run sees its own branch, and a line asserts a fact about `main` that a branch cannot
see, so there a stale line is reported for the lander to clear at landing from a
fresh run on the rebased branch. Pruning is never arithmetic on a count -- the file is
rewritten from what the current run actually measured, which is why the measurement
is a mapping and not a single pinned total.

Lines starting with `#` and blank lines are ignored.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from _tooling_common import repository_root

PROJECT_ROOT = repository_root(Path(__file__))
BASELINE_DIR = PROJECT_ROOT / "src" / "tests" / "baselines"


def read_baseline(name: str) -> dict[str, int]:
    """Key -> allowed count, from `<name>.txt`; a bare key allows one."""
    path = BASELINE_DIR / f"{name}.txt"
    allowed: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, count = line.rpartition(" ")
        if key and count.isdigit():
            allowed[key] = int(count)
        else:
            allowed[line] = 1
    return allowed


def _is_main_checkout() -> bool:
    """True in the main checkout, False in a worktree, whose `.git` is a file."""
    return (PROJECT_ROOT / ".git").is_dir()


def _prune_baseline(name: str, current: dict[str, int]) -> None:
    """Rewrite `<name>.txt` from what this run measured, keeping the leading comments."""
    path = BASELINE_DIR / f"{name}.txt"
    header: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            header.append(raw)
            continue
        break
    body = [f"{k} {v}" if v != 1 else k for k, v in sorted(current.items())]
    path.write_text("\n".join([*header, *body]) + "\n", encoding="utf-8")


def check_baseline(name: str, current: dict[str, int]) -> None:
    """Fail on anything beyond the baseline; prune lines it no longer needs."""
    rel = (BASELINE_DIR / f"{name}.txt").relative_to(PROJECT_ROOT).as_posix()
    allowed = read_baseline(name)
    grown = {k: v for k, v in current.items() if v > allowed.get(k, 0)}
    stale = {k: v for k, v in allowed.items() if current.get(k, 0) < v}
    if grown:
        lines = "\n".join(
            f"  {k} {v}" if v != 1 else f"  {k}" for k, v in sorted(grown.items())
        )
        raise AssertionError(
            f"{name}: {len(grown)} offender(s) beyond {rel}. Fix them rather than adding these lines:\n{lines}"
        )
    if stale:
        lines = "\n".join(f"  {k}" for k in sorted(stale))
        if _is_main_checkout():
            _prune_baseline(name, current)
            warnings.warn(
                f"{name}: pruned {len(stale)} line(s) no longer needed from {rel}; include the "
                f"file in this commit so the smaller backlog stays protected:\n{lines}",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{name}: {len(stale)} baseline line(s) in {rel} no longer needed, left in place "
                f"because a branch cannot assert what {rel} records about main; report them so "
                f"they are cleared at landing from a fresh run on the merged state:\n{lines}",
                stacklevel=2,
            )
