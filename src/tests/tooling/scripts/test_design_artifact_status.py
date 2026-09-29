from __future__ import annotations

import datetime
import os
import subprocess
from pathlib import Path

import design_artifact_status as das
import pytest

# Fixed base instant for deterministic, sub-second-race-free commit/mtime ordering.
_BASE_TIME = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _iso_at_offset(offset_seconds: int) -> str:
    return (_BASE_TIME + datetime.timedelta(seconds=offset_seconds)).isoformat()


def _git(root: Path, *args: str, commit_offset_seconds: int | None = None) -> None:
    env = None
    if commit_offset_seconds is not None:
        commit_date = _iso_at_offset(commit_offset_seconds)
        env = dict(os.environ)
        env["GIT_AUTHOR_DATE"] = commit_date
        env["GIT_COMMITTER_DATE"] = commit_date
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        env=env,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    (root / "src").mkdir()
    (root / "src" / "placeholder.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "initial commit", commit_offset_seconds=0)


def _touch_src_commit(root: Path, name: str, *, offset_seconds: int) -> None:
    (root / "src" / name).write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(
        root,
        "commit",
        "-q",
        "-m",
        f"touch {name}",
        commit_offset_seconds=offset_seconds,
    )


@pytest.mark.integration
def test_check_graphify_status_counts_commits_since_last_tracked_update(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)

    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    (graphify_dir / "GRAPH_REPORT.md").write_text("report\n", encoding="utf-8")
    _git(tmp_path, "add", "graphify-out/GRAPH_REPORT.md")
    _git(tmp_path, "commit", "-q", "-m", "add graph report", commit_offset_seconds=100)

    for i in range(3):
        _touch_src_commit(tmp_path, f"file_{i}.py", offset_seconds=200 + i)

    status = das.check_graphify_status(tmp_path)

    assert status.last_run_at is not None
    assert status.commits_since == 3
    assert status.is_stale is False  # below STALE_THRESHOLD_COMMITS


@pytest.mark.integration
def test_check_graphify_status_flags_stale_past_threshold(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    graphify_dir = tmp_path / "graphify-out"
    graphify_dir.mkdir()
    (graphify_dir / "GRAPH_REPORT.md").write_text("report\n", encoding="utf-8")
    _git(tmp_path, "add", "graphify-out/GRAPH_REPORT.md")
    _git(tmp_path, "commit", "-q", "-m", "add graph report", commit_offset_seconds=100)

    for i in range(das.STALE_THRESHOLD_COMMITS):
        _touch_src_commit(tmp_path, f"file_{i}.py", offset_seconds=200 + i)

    status = das.check_graphify_status(tmp_path)

    assert status.commits_since == das.STALE_THRESHOLD_COMMITS
    assert status.is_stale is True


def test_check_graphify_status_never_tracked(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    status = das.check_graphify_status(tmp_path)

    assert status.last_run_at is None
    assert status.commits_since is None
    assert status.is_stale is True


def test_check_pyscn_status_counts_commits_since_report_mtime(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    reports_dir = tmp_path / ".pyscn" / "reports"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / "analyze_20260101_000000.html"
    report_path.write_text("x", encoding="utf-8")
    report_mtime = (_BASE_TIME + datetime.timedelta(seconds=100)).timestamp()
    os.utime(report_path, (report_mtime, report_mtime))

    for i in range(2):
        _touch_src_commit(tmp_path, f"file_{i}.py", offset_seconds=200 + i)

    status = das.check_pyscn_status(tmp_path)

    assert status.last_run_at is not None
    assert status.commits_since == 2
    assert status.is_stale is False


def test_check_pyscn_status_no_reports_dir(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    status = das.check_pyscn_status(tmp_path)

    assert status.last_run_at is None
    assert status.is_stale is True


def test_build_design_artifact_statuses_returns_both(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    statuses = das.build_design_artifact_statuses(tmp_path)

    names = {status.name for status in statuses}
    assert names == {"graphify", "pyscn"}


def test_main_never_fails_even_when_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(das, "REPO_ROOT", tmp_path)
    _init_repo(tmp_path)

    exit_code = das.main()

    assert exit_code == 0


def test_format_status_line_never_generated() -> None:
    status = das.DesignArtifactStatus(
        name="graphify", last_run_at=None, commits_since=None, is_stale=True
    )

    line = das.format_status_line(status)

    assert "never generated" in line


def test_format_status_line_stale_notes_regeneration() -> None:
    status = das.DesignArtifactStatus(
        name="pyscn",
        last_run_at=datetime.datetime.now().astimezone(),
        commits_since=das.STALE_THRESHOLD_COMMITS,
        is_stale=True,
    )

    line = das.format_status_line(status)

    assert "consider regenerating" in line
    assert str(das.STALE_THRESHOLD_COMMITS) in line
