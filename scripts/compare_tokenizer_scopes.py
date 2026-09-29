"""Compare every tokenizer scope's current-operational tokenizer in one shape.

Consolidates the metrics each country/system scope's and `global`'s promoted
candidate was stored with and (where it exists) its optimize-run benchmark
evidence into one standardised comparison artifact -- see
`training.scope_comparison` for the consolidation logic this only wraps. This
is the one command that produces every scope's comparison in one shape, in
place of reading each scope's promoted candidate separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_root_arg, run_reporting_argument_errors

from training.scope_comparison import (
    DEFAULT_COUNTRY_SYSTEMS,
    build_scope_comparison,
    render_scope_comparison_markdown,
    scope_comparison_as_dict,
)
from workspace.artifact_layout import tokenizer_artifact_root
from workspace.published_reports import publish, tokenizer_reports_dir
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every tokenizer scope's current-operational tokenizer "
            "(the country/system scopes plus 'global') in one standardised "
            "shape: a quality-summary table and a benchmark table, written "
            "as both markdown and JSON, plus printed to stdout."
        )
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--tokenizer",
        choices=["wordpiece", "sentencepiece"],
        default="wordpiece",
        help="Tokenizer backend to compare across scopes.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_COUNTRY_SYSTEMS),
        help=(
            "Country/system codes to compare, 'global' is always included "
            "in addition. Comma-separated values are accepted. Defaults to "
            f"{', '.join(DEFAULT_COUNTRY_SYSTEMS)!s}."
        ),
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Do not publish a copy to docs/reports/tokenizers/.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    systems: list[str] = []
    for raw in args.systems:
        systems.extend(code.strip().lower() for code in raw.split(",") if code.strip())

    build = build_scope_comparison(
        roots=roots,
        trainer=args.tokenizer,
        systems=systems,
    )
    markdown = render_scope_comparison_markdown(build)
    payload = scope_comparison_as_dict(build)

    output_dir = tokenizer_artifact_root(roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"scope_comparison.{args.tokenizer}.md"
    json_path = output_dir / f"scope_comparison.{args.tokenizer}.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(markdown)
    print(f"[compare_tokenizer_scopes] wrote {markdown_path}")
    print(f"[compare_tokenizer_scopes] wrote {json_path}")
    if args.publish:
        published_dir = tokenizer_reports_dir(roots)
        publish(published_dir, texts={markdown_path.name: markdown})
        print(
            f"[compare_tokenizer_scopes] published {published_dir / markdown_path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
