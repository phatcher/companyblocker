from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from cli_common import (
    add_workspace_roots_args,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from analysis.sif_word_probabilities import (
    DEFAULT_PREPARATION,
    GLOBAL_LIST_NAME,
    resolve_word_probabilities,
)
from analysis.token_zipf import NAME_TIER_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Word document frequency, converted to a SIF token "
            "probability, over one system's names or the "
            f"{GLOBAL_LIST_NAME!r} list across every runnable system, at "
            "word level under a stated preparation. Writes a "
            "content-addressed artefact under artifacts/store/, recording "
            "the list, its preparation, its number of names and its mean "
            "tokens per name; an unchanged list and preparation reuse the "
            "existing artefact rather than rewriting it."
        )
    )
    add_workspace_roots_args(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--system", help="Count over one system's own names.")
    group.add_argument(
        "--global",
        dest="global_list",
        action="store_true",
        help=f"Count over the {GLOBAL_LIST_NAME!r} list: every runnable system's names.",
    )
    parser.add_argument(
        "--preparation",
        default=DEFAULT_PREPARATION,
        choices=sorted(NAME_TIER_COLUMNS),
        help=f"Name-column tier the words are counted from (default: {DEFAULT_PREPARATION!r}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    roots = resolve_workspace_roots_from_args(args)
    entry = resolve_word_probabilities(
        roots,
        system=None if args.global_list else args.system,
        preparation=args.preparation,
    )
    print(f"Word probabilities: {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
