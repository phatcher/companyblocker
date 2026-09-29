"""Overlay each scope's per-country fertility/UNK at its own elbow vocab.

One point per country per scope: a country scope's own single point (itself,
at that scope's own winning vocab size) alongside `global`'s per-country
breakdown (one point per constituent country, all sharing `global`'s one
winning vocab size). Deliberately two different vocab sizes on the same
x-axis per scope -- normalising them onto one shared axis would erase the
very difference this overlay exists to show, so scopes are deliberately never
normalised onto a shared vocab size to make them comparable.

Reads `training.scope_comparison`'s already-consolidated overlay data (itself
read from the optimize summary each scope's promoted candidate was stored with
-- see that module for the consolidation logic); computes nothing new.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
from cli_common import add_root_arg, run_reporting_argument_errors

from training.scope_comparison import DEFAULT_COUNTRY_SYSTEMS, build_scope_comparison
from workspace.artifact_layout import tokenizer_artifact_root
from workspace.published_reports import publish, tokenizer_reports_dir
from workspace.roots import WorkspaceRoots, default_workspace_roots


def plot_country_global_overlay(
    *,
    roots: WorkspaceRoots,
    trainer: str,
    systems: list[str] | None = None,
) -> Path | None:
    """Draw the two-panel (fertility, UNK rate) overlay and save it.

    Returns `None`, writing nothing, when no scope in `systems` (plus
    `global`) has ever been optimize-swept for `trainer` -- there is nothing
    to overlay yet, not an error.
    """
    build = build_scope_comparison(roots=roots, trainer=trainer, systems=systems)
    points = [p for p in build.overlay_points if p.vocab_size_resolved is not None]
    if not points:
        print(
            f"[plot_country_global_overlay] no optimize-swept scope found for "
            f"trainer={trainer} -- nothing to overlay"
        )
        return None

    fig, (ax_fertility, ax_unk) = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")
    countries = sorted(
        {
            str(row["system"])
            for point in points
            for row in point.per_system
            if row.get("system")
        }
    )
    color_by_country = {
        country: cmap(idx % 10) for idx, country in enumerate(countries)
    }

    for point in points:
        # A country scope's own point is itself, marked "o"; global's
        # per-country breakdown is marked "^" so the two are visually
        # distinct even where they land at the same vocab size by
        # coincidence.
        marker = "^" if point.scope == "global" else "o"
        for row in point.per_system:
            country = str(row.get("system"))
            color = color_by_country.get(country, "black")
            label = f"{country} (via {point.scope}, vocab={point.vocab_size_resolved})"
            fertility = row.get("fertility")
            unk_rate = row.get("unk_rate")
            if fertility is not None:
                ax_fertility.scatter(
                    [point.vocab_size_resolved],
                    [fertility],
                    color=color,
                    marker=marker,
                    s=90,
                    edgecolors="black" if point.scope == "global" else "none",
                    linewidths=0.8,
                    label=label,
                )
            if unk_rate is not None:
                ax_unk.scatter(
                    [point.vocab_size_resolved],
                    [unk_rate],
                    color=color,
                    marker=marker,
                    s=90,
                    edgecolors="black" if point.scope == "global" else "none",
                    linewidths=0.8,
                    label=label,
                )

    ax_fertility.set_xlabel("Resolved vocab size (each scope's own elbow winner)")
    ax_fertility.set_ylabel("Fertility")
    ax_fertility.set_title("Fertility by country, at each scope's own vocab")
    ax_unk.set_xlabel("Resolved vocab size (each scope's own elbow winner)")
    ax_unk.set_ylabel("UNK rate")
    ax_unk.set_title("UNK rate by country, at each scope's own vocab")
    ax_unk.set_ylim(bottom=0)

    handles, labels = ax_fertility.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=7,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(f"Country vs. global tokenizer overlay -- trainer={trainer}")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    output_path = (
        tokenizer_artifact_root(roots) / f"scope_comparison_overlay.{trainer}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" rather than relying on the rect margin alone --
    # the legend sits below the axes via bbox_to_anchor, which fig.tight_layout's
    # rect margin doesn't reliably reserve enough room for once the country
    # count (and so the legend's row count) grows past a handful.
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_country_global_overlay] wrote {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay fertility and UNK rate against vocab size, per country, "
            "at each scope's own elbow-optimized winning vocab -- one point "
            "per country scope plus global's per-country breakdown at "
            "global's own vocab, deliberately not normalised onto one axis."
        )
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--tokenizer",
        choices=["wordpiece", "sentencepiece"],
        default="wordpiece",
        help="Tokenizer backend to overlay.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_COUNTRY_SYSTEMS),
        help=(
            "Country/system codes to overlay, 'global' is always included in "
            f"addition. Comma-separated values are accepted. Defaults to "
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

    output_path = plot_country_global_overlay(
        roots=roots, trainer=args.tokenizer, systems=systems
    )
    if output_path is not None and args.publish:
        published_dir = tokenizer_reports_dir(roots)
        publish(published_dir, files={output_path.name: output_path})
        print(
            f"[plot_country_global_overlay] published {published_dir / output_path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
