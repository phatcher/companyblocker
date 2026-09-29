from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_root_arg, run_reporting_argument_errors
from company_tokenize import (
    compare_archive_entries,
    read_archive_index,
    resolve_archive_dir,
)

from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two archived tokenizer candidates' metrics and report whether "
            "one actually improves on the other. Use --list to see what's archived "
            "for a scope/system first."
        )
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument("--scope", choices=["country", "global"], default="country")
    parser.add_argument(
        "--system", default=None, help="System code, required for --scope country."
    )
    parser.add_argument(
        "--profile", default="default", help="Global tokenizer profile, if applicable."
    )
    parser.add_argument("--label-a", default=None, help="Baseline label.")
    parser.add_argument("--label-b", default=None, help="Candidate label to compare.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List archived labels for this scope/system and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.scope == "country" and not args.system:
        parser.error("--system is required when --scope country")
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)
    tokenizer_root = tokenizer_artifact_root(roots)

    if args.list:
        archive_dir = resolve_archive_dir(
            tokenizer_root=tokenizer_root,
            scope=args.scope,
            system=args.system,
            profile=args.profile,
        )
        entries = read_archive_index(archive_dir)
        if not entries:
            print(f"No archived entries under {archive_dir}")
            return 0
        for entry in entries:
            print(
                f"{entry['label']:30s} kind={entry.get('kind'):20s} "
                f"vocab={entry.get('vocab_size_requested')} "
                f"min_freq={entry.get('min_frequency')}"
            )
        return 0

    if not args.label_a or not args.label_b:
        parser.error("--label-a and --label-b are required unless --list is given")

    result = compare_archive_entries(
        tokenizer_root=tokenizer_root,
        scope=args.scope,
        system=args.system,
        profile=args.profile,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    print(
        f"{result['label_a']} ({result['kind_a']}) vs {result['label_b']} ({result['kind_b']})"
    )
    print(
        f"vocab: {result['vocab_size_requested_a']} -> {result['vocab_size_requested_b']}, "
        f"min_frequency: {result['min_frequency_a']} -> {result['min_frequency_b']}"
    )
    print()
    for metric, values in result["metric_deltas"].items():
        delta = values["delta"]
        delta_text = f"{delta:+.4f}" if isinstance(delta, (int, float)) else "n/a"
        print(
            f"  {metric:24s} {values['a']!r:>12} -> {values['b']!r:>12}  ({delta_text})"
        )
    print()
    print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
