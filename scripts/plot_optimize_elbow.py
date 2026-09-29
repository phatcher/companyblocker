from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cli_common import (
    add_root_arg,
    add_systems_arg,
    add_tokenizer_profile_arg,
    add_tokenizer_scope_arg,
    run_reporting_argument_errors,
)
from company_tokenize import (
    resolve_active_optimize_sweep_dir as _resolve_active_optimize_sweep_dir,
)
from company_tokenize import resolve_tokenizer_paths
from company_tokenize.training import (
    OPTIMIZE_RUN_LOG_FILENAME,
    OPTIMIZE_SUMMARY_FILENAME,
)
from company_tokenize.training import resolve_optimize_dir as _resolve_optimize_dir

from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import WorkspaceRoots, default_workspace_roots


def resolve_optimize_dir(
    *,
    roots: WorkspaceRoots,
    scope: str,
    system: str | None,
    profile: str,
    trainer: str,
    tokenizer_encoding: str | None = None,
) -> Path:
    # `optimize/` in a tokenizer's own folder is the parent its sweep leaves
    # nest under (see resolve_active_optimize_sweep_dir).
    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope=scope,
        system=system,
        profile=profile,
    )
    return _resolve_optimize_dir(
        scope_directory=tokenizer_paths.directory,
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )


def resolve_active_optimize_sweep_dir(
    *,
    roots: WorkspaceRoots,
    scope: str,
    system: str | None,
    profile: str,
    trainer: str,
    tokenizer_encoding: str | None = None,
) -> Path | None:
    """Thin wrapper kept local so existing imports of this name keep working
    -- the real implementation lives in `company_tokenize.training` (shared
    with `src/training/tokenizer_corpus_report.py`, which can't depend on
    `scripts` per the module dependency graph).
    """
    return _resolve_active_optimize_sweep_dir(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )


# The untuned defaults every optimize winner is graded against, the naive
# side of the naive-vs-optimized deltas. Marked from the sweep's own run log when the
# grid happens to contain this point, never from an archived entry's metrics:
# archive metrics are computed over the full corpus, so plotting one on this
# validation-split axis would compare two different measurements.
NAIVE_BASELINE_VOCAB = 40000
NAIVE_BASELINE_MIN_FREQUENCY = 1


def _elbow_note(summary: dict) -> str:
    min_points = summary.get("elbow_min_points")
    min_improvement = summary.get("elbow_min_improvement")
    extra_steps = summary.get("effective_elbow_extra_steps")
    if min_points is None or min_improvement is None or extra_steps is None:
        return "elbow signal: not recorded in this summary"
    return (
        f"elbow signal (diagnostic only -- every vocab point is still evaluated):\n"
        f"flags a trajectory after >= {min_points} vocab points once fertility-distance\n"
        f"improvement < {min_improvement} for {extra_steps} consecutive larger-vocab step(s)"
    )


def plot_optimize_elbow(
    *,
    roots: WorkspaceRoots,
    scope: str,
    system: str | None,
    profile: str,
    trainer: str,
    tokenizer_encoding: str | None = None,
) -> Path | None:
    label = system if scope == "country" else f"global:{profile}"

    sweep_dir = resolve_active_optimize_sweep_dir(
        roots=roots,
        scope=scope,
        system=system,
        profile=profile,
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )
    if sweep_dir is None:
        print(
            f"[plot_optimize_elbow] [{label}] no optimize sweep recorded for "
            f"trainer={trainer} tokenizer_encoding={tokenizer_encoding or 'bpe'}"
        )
        return None
    run_log_path = sweep_dir / OPTIMIZE_RUN_LOG_FILENAME
    summary_path = sweep_dir / OPTIMIZE_SUMMARY_FILENAME

    if not run_log_path.exists():
        print(f"[plot_optimize_elbow] [{label}] no run log found at {run_log_path}")
        return None

    runs = pl.read_parquet(run_log_path)
    if runs.height == 0:
        print(f"[plot_optimize_elbow] [{label}] run log at {run_log_path} is empty")
        return None

    summary: dict | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cmap = plt.get_cmap("tab10")
    min_frequencies = sorted(runs["min_frequency"].unique().to_list())

    for idx, min_freq in enumerate(min_frequencies):
        color = cmap(idx % 10)
        freq_rows = runs.filter(pl.col("min_frequency") == min_freq)

        for seed in sorted(freq_rows["seed"].unique().to_list()):
            seed_rows = freq_rows.filter(pl.col("seed") == seed).sort(
                "vocab_size_resolved"
            )
            xs = seed_rows["vocab_size_resolved"].to_list()
            ys = seed_rows["fertility_distance"].to_list()
            rejected = seed_rows["rejected"].to_list()
            ax.plot(xs, ys, color=color, alpha=0.3, linewidth=1, zorder=1)
            accepted = [(x, y) for x, y, r in zip(xs, ys, rejected) if not r]
            declined = [(x, y) for x, y, r in zip(xs, ys, rejected) if r]
            if accepted:
                acc_x, acc_y = zip(*accepted)
                ax.scatter(acc_x, acc_y, color=color, marker="o", s=20, zorder=2)
            if declined:
                dec_x, dec_y = zip(*declined)
                ax.scatter(dec_x, dec_y, color=color, marker="x", s=28, zorder=2)

        median_by_vocab = (
            freq_rows.group_by("vocab_size_resolved")
            .agg(pl.col("fertility_distance").median().alias("median_dist"))
            .sort("vocab_size_resolved")
        )
        ax.plot(
            median_by_vocab["vocab_size_resolved"].to_list(),
            median_by_vocab["median_dist"].to_list(),
            color=color,
            linewidth=2.5,
            label=f"min_frequency={min_freq} (median across seeds)",
            zorder=3,
        )

    # token_count_p95 on a twin axis, alongside fertility_distance rather
    # than in a separate chart: it's a step function over vocab size
    # (optimize_search_methodology.md Part 7 -- typically two or three
    # integer values across the whole grid), and whether its one real step
    # lands inside or outside the fertility elbow is exactly what decides
    # whether the gate matters for this system. Guarded on column presence
    # for older run logs that predate token_count_p95 being recorded.
    p95_axis = None
    if "token_count_p95" in runs.columns:
        p95_axis = ax.twinx()
        for idx, min_freq in enumerate(min_frequencies):
            color = cmap(idx % 10)
            freq_rows = runs.filter(pl.col("min_frequency") == min_freq)
            median_p95_by_vocab = (
                freq_rows.group_by("vocab_size_resolved")
                .agg(pl.col("token_count_p95").median().alias("median_p95"))
                .sort("vocab_size_resolved")
            )
            p95_axis.step(
                median_p95_by_vocab["vocab_size_resolved"].to_list(),
                median_p95_by_vocab["median_p95"].to_list(),
                where="post",
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                label=f"min_frequency={min_freq} token_count_p95 (median, dashed)",
                zorder=2.5,
            )
        p95_axis.set_ylabel("token_count_p95 (validation, dashed lines)")

    naive_rows = runs.filter(
        (pl.col("vocab_size_resolved") == NAIVE_BASELINE_VOCAB)
        & (pl.col("min_frequency") == NAIVE_BASELINE_MIN_FREQUENCY)
    )
    # Median across seeds, matching how the per-min_frequency trend lines above
    # are aggregated, so the marker sits on the same footing.
    naive_distance = (
        naive_rows.select(pl.col("fertility_distance").median()).item()
        if naive_rows.height
        else None
    )
    if naive_distance is not None:
        ax.scatter(
            [NAIVE_BASELINE_VOCAB],
            [naive_distance],
            facecolors="none",
            edgecolors="crimson",
            marker="D",
            s=150,
            linewidths=1.8,
            zorder=4,
            label=(
                f"naive baseline (vocab={NAIVE_BASELINE_VOCAB}, "
                f"min_freq={NAIVE_BASELINE_MIN_FREQUENCY})"
            ),
        )

    # Winner detail goes in the title (above the axes), not an in-plot
    # annotation box -- an offset text box anchored to the winner point
    # reliably lands on top of real data, since the lowest-distance lines
    # from every min_frequency series tend to converge right around the
    # winner's vocab, wherever that happens to be for a given system.
    winner_detail: str | None = None
    if summary is not None:
        winner = summary.get("winner_candidate", {})
        winner_vocab = winner.get("vocab_size_resolved")
        winner_dist = winner.get("median_fertility_distance")
        if winner_vocab is not None and winner_dist is not None:
            ax.scatter(
                [winner_vocab],
                [winner_dist],
                color="black",
                marker="*",
                s=280,
                zorder=5,
                label="winner",
            )
            winner_detail = (
                f"winner: vocab={winner_vocab}, min_freq={winner.get('min_frequency')}, "
                f"fertility_dist={winner_dist:.4f}, "
                f"score={winner.get('median_selection_score', float('nan')):.4f}"
            )

        fertility_tolerance = summary.get("fertility_tolerance")
        if fertility_tolerance is not None:
            ax.axhspan(
                0,
                fertility_tolerance,
                color="green",
                alpha=0.06,
                label=f"within fertility_tolerance ({fertility_tolerance})",
                zorder=0,
            )

    ax.set_xlabel("Resolved vocab size")
    ax.set_ylabel("Fertility distance (validation, lower is better)")
    title = f"Tokenizer optimize sweep -- {label} ({trainer})"
    if winner_detail is not None:
        title += f"\n{winner_detail}"
    ax.set_title(title, fontsize=11)
    # One combined legend on ax (not ax2) -- test_plot_optimize_elbow.py's
    # _legend_labels reads captured["ax"].get_legend(), and a reader
    # shouldn't have to check two legend boxes for one chart anyway.
    handles, labels = ax.get_legend_handles_labels()
    if p95_axis is not None:
        p95_handles, p95_labels = p95_axis.get_legend_handles_labels()
        handles += p95_handles
        labels += p95_labels
    ax.legend(handles, labels, fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)
    # Reserve a margin below the axes for the elbow note instead of drawing
    # it inside the plot (ax.text at a fixed corner) -- the best-fitting
    # points (lowest fertility_distance) tend to cluster near the bottom of
    # the chart wherever the winner's vocab lands, so a fixed in-axes corner
    # box ends up sitting on top of real data more often than not.
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    if summary is not None:
        fig.text(
            0.02,
            0.02,
            _elbow_note(summary),
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "gray", "alpha": 0.9},
        )

    output_path = sweep_dir / f"elbow_curve.{trainer}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot_optimize_elbow] [{label}] wrote {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the vocab-size optimization curve (fertility distance vs. vocab "
            "size) from a `train_tokenizer.py --mode optimize` run, marking "
            "rejected candidates, the fertility-tolerance band, and the winning "
            "candidate, so it's visible why the sweep stopped where it did."
        )
    )
    add_systems_arg(
        parser,
        action="plot the optimize sweep for",
        detail="Use '--tokenizer-scope global' with --profile instead for a global tokenizer sweep.",
    )
    add_root_arg(parser, help_text="Project root containing artifacts/.")
    add_tokenizer_scope_arg(parser)
    add_tokenizer_profile_arg(parser)
    parser.add_argument(
        "--tokenizer",
        choices=["wordpiece", "sentencepiece"],
        default="wordpiece",
        help="Tokenizer backend whose optimize run should be plotted.",
    )
    parser.add_argument(
        "--encoding",
        dest="tokenizer_encoding",
        choices=["bpe", "unigram"],
        default="bpe",
        help=(
            "SentencePiece encoding whose sweep should be plotted, when "
            "--tokenizer sentencepiece. A bpe and a unigram sweep for the same "
            "system are stored separately -- this picks which one. Ignored "
            "for --tokenizer wordpiece."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    if args.tokenizer_scope == "global":
        plot_optimize_elbow(
            roots=roots,
            scope="global",
            system=None,
            profile=args.tokenizer_profile,
            trainer=args.tokenizer,
            tokenizer_encoding=args.tokenizer_encoding,
        )
        return 0

    for system in args.systems:
        for code in system.split(","):
            code = code.strip().lower()
            if not code:
                continue
            plot_optimize_elbow(
                roots=roots,
                scope="country",
                system=code,
                profile=args.tokenizer_profile,
                trainer=args.tokenizer,
                tokenizer_encoding=args.tokenizer_encoding,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
