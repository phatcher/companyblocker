"""`baseline.py`'s own contract: what it allows, what it fails on, and what it prunes.

Five ratchet tests import `check_baseline` to police their own measurement, so it was
reachable at high coverage with nothing asserting its rules directly -- a caller's
coverage, which `docs/TESTSTYLE.md`'s direct-test rule does not count. Its fail path and
both stale-line paths were the parts no caller exercised, which is the wrong half of a
gate to leave unasserted: this is the code that decides whether a ratchet fires.

Every test points the module at a `tmp_path` baseline directory rather than the real
`src/tests/baselines/`, since the prune path rewrites the file it is given.
"""

from __future__ import annotations

from pathlib import Path

import baseline
import pytest


@pytest.fixture
def baseline_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Both roots move together: the failure message names the file relative to
    `PROJECT_ROOT`, so pointing only `BASELINE_DIR` at `tmp_path` raises before any
    of the behaviour under test runs."""
    directory = tmp_path / "src" / "tests" / "baselines"
    directory.mkdir(parents=True)
    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(baseline, "BASELINE_DIR", directory)
    return directory


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_read_baseline_reads_a_bare_key_as_allowing_one(baseline_dir: Path) -> None:
    """A line with no trailing count is the common shape, and a key whose own text
    ends in a non-digit word must not be mistaken for a key-and-count pair."""
    _write(baseline_dir, "b", "# a comment\n\nsrc/a.py\nsrc/b.py 3\nsrc/c d\n")

    assert baseline.read_baseline("b") == {
        "src/a.py": 1,
        "src/b.py": 3,
        "src/c d": 1,
    }


def test_check_baseline_fails_on_an_offender_the_file_does_not_list(
    baseline_dir: Path,
) -> None:
    """The message carries the exact line that would silence it, which is the line
    the reader is being told not to add."""
    _write(baseline_dir, "b", "src/a.py\n")

    with pytest.raises(AssertionError) as excinfo:
        baseline.check_baseline("b", {"src/a.py": 1, "src/new.py": 1})

    message = str(excinfo.value)
    assert "1 offender(s)" in message
    assert "src/new.py" in message
    assert "src/a.py" not in message


def test_check_baseline_fails_when_a_listed_key_grows(baseline_dir: Path) -> None:
    """A known offender getting worse is a regression, so the count is a ceiling
    rather than a label, and the message shows the count it would take."""
    _write(baseline_dir, "b", "src/a.py 2\n")

    with pytest.raises(AssertionError) as excinfo:
        baseline.check_baseline("b", {"src/a.py": 3})

    assert "src/a.py 3" in str(excinfo.value)


def test_check_baseline_accepts_a_measurement_within_the_baseline(
    baseline_dir: Path,
) -> None:
    path = _write(baseline_dir, "b", "src/a.py 2\n")

    baseline.check_baseline("b", {"src/a.py": 2})

    assert path.read_text(encoding="utf-8") == "src/a.py 2\n"


def test_check_baseline_prunes_a_cleared_line_in_the_main_checkout(
    baseline_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pruning rewrites the file from what this run measured rather than doing
    arithmetic on the old count, and keeps the leading comments so the file still
    explains itself."""
    path = _write(baseline_dir, "b", "# why this file exists\n\nsrc/a.py\nsrc/b.py 2\n")
    monkeypatch.setattr(baseline, "_is_main_checkout", lambda: True)

    with pytest.warns(UserWarning, match="pruned 1 line"):
        baseline.check_baseline("b", {"src/b.py": 2})

    assert path.read_text(encoding="utf-8") == (
        "# why this file exists\n\nsrc/b.py 2\n"
    )


def test_check_baseline_leaves_a_cleared_line_alone_in_a_worktree(
    baseline_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch cannot assert what the file records about `main`, so a worktree
    reports the stale line instead of removing it."""
    original = "src/a.py\nsrc/b.py\n"
    path = _write(baseline_dir, "b", original)
    monkeypatch.setattr(baseline, "_is_main_checkout", lambda: False)

    with pytest.warns(UserWarning, match="no longer needed"):
        baseline.check_baseline("b", {"src/b.py": 1})

    assert path.read_text(encoding="utf-8") == original


def test_is_main_checkout_distinguishes_a_directory_from_a_worktrees_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree's `.git` is a file pointing at the real one; the main checkout's
    is a directory. That is the whole rule, and it decides whether pruning runs."""
    monkeypatch.setattr(baseline, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()
    assert baseline._is_main_checkout() is True

    (tmp_path / ".git").rmdir()
    (tmp_path / ".git").write_text("gitdir: ../.git/worktrees/x\n", encoding="utf-8")
    assert baseline._is_main_checkout() is False
