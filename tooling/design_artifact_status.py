from __future__ import annotations

import subprocess  # nosec B404 - dev-tooling script; see nosec B603/B607 at its call site
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _tooling_common import repository_root

REPO_ROOT = repository_root(Path(__file__))

# Paths whose commits count toward "design artifacts are stale". Both graphify
# (whole-repo knowledge graph) and pyscn (Python architecture/quality analysis)
# are scoped to the same code the diagrams/reports actually describe.
SCOPED_PATHS = ("src", "scripts", "tooling", "packages")

# Below this many commits touching SCOPED_PATHS since the last run, don't bother
# flagging it - regenerating pyscn/graphify has a real cost, so this is a
# reminder threshold, not a hard rule. Tune to taste.
STALE_THRESHOLD_COMMITS = 10


@dataclass(frozen=True)
class DesignArtifactStatus:
    name: str
    last_run_at: datetime | None
    commits_since: int | None
    is_stale: bool


def _run_git(root: Path, *args: str) -> str:
    # `git` is resolved via PATH (developer tooling, not attacker-controlled), and
    # every call site in this module passes fixed subcommand literals, not shell=True.
    result = subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip()


def _commits_since_sha(root: Path, sha: str) -> int | None:
    raw = _run_git(root, "rev-list", "--count", f"{sha}..HEAD", "--", *SCOPED_PATHS)
    return int(raw) if raw.isdigit() else None


def _commits_since_iso(root: Path, since_iso: str) -> int | None:
    raw = _run_git(
        root, "rev-list", "--count", f"--since={since_iso}", "HEAD", "--", *SCOPED_PATHS
    )
    return int(raw) if raw.isdigit() else None


def check_graphify_status(root: Path) -> DesignArtifactStatus:
    # graphify-out/GRAPH_REPORT.md is the one graphify output still tracked in
    # git (see .gitignore), so its own commit history tells us when it was
    # last regenerated without needing a separate marker file.
    last_commit = _run_git(
        root, "log", "-1", "--format=%H", "--", "graphify-out/GRAPH_REPORT.md"
    )
    if not last_commit:
        return DesignArtifactStatus(
            name="graphify", last_run_at=None, commits_since=None, is_stale=True
        )

    commit_date_raw = _run_git(root, "log", "-1", "--format=%cI", last_commit)
    last_run_at = datetime.fromisoformat(commit_date_raw) if commit_date_raw else None
    commits_since = _commits_since_sha(root, last_commit)
    is_stale = commits_since is not None and commits_since >= STALE_THRESHOLD_COMMITS
    return DesignArtifactStatus("graphify", last_run_at, commits_since, is_stale)


def check_pyscn_status(root: Path) -> DesignArtifactStatus:
    # pyscn reports are entirely gitignored, so there's no commit to anchor on;
    # use the newest report file's own mtime and count commits since then.
    reports_dir = root / ".pyscn" / "reports"
    report_files = (
        [path for path in reports_dir.iterdir() if path.is_file()]
        if reports_dir.exists()
        else []
    )
    if not report_files:
        return DesignArtifactStatus(
            name="pyscn", last_run_at=None, commits_since=None, is_stale=True
        )

    newest = max(report_files, key=lambda path: path.stat().st_mtime)
    last_run_at = datetime.fromtimestamp(newest.stat().st_mtime).astimezone()
    commits_since = _commits_since_iso(root, last_run_at.isoformat())
    is_stale = commits_since is not None and commits_since >= STALE_THRESHOLD_COMMITS
    return DesignArtifactStatus("pyscn", last_run_at, commits_since, is_stale)


def build_design_artifact_statuses(root: Path) -> list[DesignArtifactStatus]:
    return [check_graphify_status(root), check_pyscn_status(root)]


def format_status_line(status: DesignArtifactStatus) -> str:
    if status.last_run_at is None:
        return f"{status.name}: never generated (or output not found) - run it before relying on it."
    when = status.last_run_at.strftime("%Y-%m-%d %H:%M")
    count = status.commits_since if status.commits_since is not None else "?"
    marker = " - consider regenerating" if status.is_stale else ""
    return f"{status.name}: last run {when}, {count} commit(s) since touching src/scripts/packages{marker}"


def main() -> int:
    statuses = build_design_artifact_statuses(REPO_ROOT)
    stale = [status for status in statuses if status.is_stale]
    if stale:
        print("[design-artifact-status] design artifacts may be out of date:")
        for status in statuses:
            print(f"[design-artifact-status]   {format_status_line(status)}")
    # Always exit 0: this is an FYI, not a gate - never block a commit/push on it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
