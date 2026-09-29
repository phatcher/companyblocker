from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cli_common import (
    add_workspace_roots_args,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from workspace.artifact_archive import prune_artifact, prune_stale_temporary_dirs
from workspace.artifact_layout import artifact_store_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove one keyed artifact or stale temporary store directories."
    )
    add_workspace_roots_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--key", help="Key path relative to artifacts/store.")
    group.add_argument(
        "--stale-temporary",
        action="store_true",
        help="Remove temporary store directories older than the module ceiling.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store_root = artifact_store_root(resolve_workspace_roots_from_args(args))
    if args.stale_temporary:
        for path in prune_stale_temporary_dirs(store_root):
            print(f"Removed stale temporary directory: {path}")
        return 0

    candidate = (store_root / args.key).resolve()
    if store_root.resolve() not in candidate.parents:
        raise SystemExit("--key must stay beneath artifacts/store")
    if prune_artifact(candidate):
        print(f"Removed artifact: {candidate}")
    else:
        print(f"Artifact not found: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
