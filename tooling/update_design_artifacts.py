from __future__ import annotations

import argparse
import subprocess  # nosec B404 - dev-tooling script; see nosec B603 at its call site
from pathlib import Path

from _tooling_common import repository_root

REPO_ROOT = repository_root(Path(__file__))


def _run(cmd: list[str], *, cwd: Path, label: str) -> bool:
    print(f"[update-design-artifacts] {label}: {' '.join(cmd)}")
    # Every caller passes a fixed argv (pyscn/graphify literals below); not shell=True.
    result = subprocess.run(cmd, cwd=cwd, check=False)  # nosec B603
    if result.returncode != 0:
        print(f"[update-design-artifacts] {label} failed (exit {result.returncode})")
        return False
    return True


def update_pyscn(root: Path) -> bool:
    return _run(
        ["uv", "run", "pyscn", "analyze", "--html", "."],
        cwd=root,
        label="pyscn analyze",
    )


def update_graphify(root: Path) -> bool:
    # This is a cheap incremental refresh, not the full `/graphify --update` skill
    # pipeline: it skips the read-only health check and (via --no-label) the LLM
    # community-naming pass, so communities keep their placeholder names. Run the
    # full skill pipeline in a Claude Code session periodically for the real thing.
    if not _run(
        ["uv", "run", "graphify", "update", "."], cwd=root, label="graphify update"
    ):
        return False

    if not _run(
        ["uv", "run", "graphify", "cluster-only", ".", "--no-label"],
        cwd=root,
        label="graphify cluster-only",
    ):
        return False

    print(
        "[update-design-artifacts] graphify: cheap incremental refresh only "
        "(re-extraction + reclustering, no LLM calls) - community names are placeholders "
        "and the read-only health check was skipped. Run `/graphify --update` in a "
        "Claude Code session periodically for the full pipeline."
    )
    print(
        "[update-design-artifacts] graphify-out/GRAPH_REPORT.md changed - commit it so "
        "tooling/design_artifact_status.py picks up the refresh."
    )
    return True


# Extend this as more design tooling gets a regeneration step.
TOOLS = {
    "pyscn": update_pyscn,
    "graphify": update_graphify,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate design artifacts (pyscn architecture/quality report, graphify "
            "knowledge graph) - one command for the tools tooling/design_artifact_status.py "
            "checks freshness for."
        )
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=None,
        help="Checkout to analyze (default: the checkout this script sits in).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(TOOLS),
        default=None,
        help="Only regenerate specific tools. Defaults to all.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = args.scan.resolve() if args.scan is not None else REPO_ROOT

    selected = args.only or sorted(TOOLS)
    failures = [name for name in selected if not TOOLS[name](root)]

    if failures:
        print(f"[update-design-artifacts] failed: {', '.join(failures)}")
        return 1
    print("[update-design-artifacts] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
