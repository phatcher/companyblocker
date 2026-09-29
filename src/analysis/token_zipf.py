"""Zipf and out-of-vocabulary divergence of each system's names from general language.

Fits rank-frequency curves per system and name-column tier (raw, basic, cleansed) against a `wordfreq` reference, and reports type-level and occurrence-weighted OOV% and name-level hapax and OOV incidence. A second sweep holds the tier fixed and varies the tokenize-stage noise-word trim setting. Run through `scripts/analyze_token_zipf.py`.

Each system and tier gets two pictures: `<system>_<tier>_zipf.png`, the rank-frequency curve, and `<system>_<tier>_head_words.png`, the thirty most frequent words in rank order, named on the axis, which reads out the part of the curve a log-log plot compresses into a corner. They are recorded as `plot_path` and `head_words_plot_path`, which `training.tokenizer_corpus_report` embeds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import polars as pl
import powerlaw
from company_tokenize import to_portable_path_str
from company_tokenize.paths import scope_directory_files
from company_tokenize.tfidf import resolve_noise_words
from wordfreq import top_n_list, zipf_frequency

from analysis.report_layout import metrics_dir as _metrics_dir
from analysis.report_layout import reports_dir as _reports_dir
from analysis.report_layout import viz_dir as _viz_dir
from analysis.system_discovery import filter_runnable_systems
from analysis.token_rarity import attach_language_frequency, resolve_system_language
from workspace.artifact_layout import (
    analysis_report_run_dir,
    analysis_report_runs_root,
    tokenizer_scope_dir,
)
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    data_root,
    system_layer_dir,
)
from workspace.layer_layout import resolve_primary_files
from workspace.roots import WorkspaceRoots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Name-column tiers swept for the raw-vs-basic-vs-cleansed divergence
# evidence. "raw" is included deliberately with its known confound (spaced
# single-character acronym runs like "I B M" score as ordinary English at the
# per-fragment level -- see docs/findings/token-zipf.md) rather than
# pre-collapsed, so that confound is visible in the report instead of hidden.
NAME_TIER_COLUMNS: dict[str, str] = {
    "raw": "name",
    "basic": "name_cleansed_basic",
    "cleansed": "name_cleansed",
}

# Unicode letter/number, not [A-Z0-9] -- an ASCII-only class fragments any
# accented word into pieces at each diacritic (found running this for real on
# offeneregister: "beschränkter" split into "BESCHR"/"NKTER"; same issue would
# hit fr's "société" etc.). Only matters for the "raw" tier in practice, since
# basic/cleansed already transliterate diacritics to ASCII upstream (the
# default normalization chain's `diacritics` operation) before this module
# ever sees the text.
_WORD_TOKEN_PATTERN = r"[\p{L}\p{N}]+"


@dataclass(frozen=True)
class TokenZipfPaths:
    run_root: Path
    stats_dir: Path
    viz_dir: Path
    reports_dir: Path
    report_path: Path
    summary_path: Path


def resolve_token_zipf_paths(
    roots: WorkspaceRoots, run_date: str, *, force: bool = False
) -> TokenZipfPaths:
    """Create and return one run's output directories.

    A run is keyed on its date alone and its summary is written whole, so a
    second run on the same date replaces the first's summary with only its own
    systems: two same-day `--systems` invocations left one system's stats
    parquet files on disk with no summary row naming them, invisible to every
    reader. Refuses a date that already holds a summary unless `force`, which
    is the caller saying it means to replace that run.
    """
    run_root = analysis_report_run_dir(roots, "token_zipf", run_date)
    stats_dir = _metrics_dir(run_root)
    viz_dir = _viz_dir(run_root)
    reports_dir = _reports_dir(run_root)
    summary_path = stats_dir / "token_zipf_summary.parquet"
    if summary_path.exists() and not force:
        raise FileExistsError(
            f"A token_zipf run for {run_date!r} already exists at {run_root}; its "
            "summary would be replaced by this run's systems alone. Pass a "
            "different run date, or force to replace it."
        )
    for directory in (stats_dir, viz_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return TokenZipfPaths(
        run_root=run_root,
        stats_dir=stats_dir,
        viz_dir=viz_dir,
        reports_dir=reports_dir,
        report_path=reports_dir / "token_zipf_summary.md",
        summary_path=summary_path,
    )


def find_latest_zipf_summary_for_system(
    roots: WorkspaceRoots, system: str
) -> tuple[str, pl.DataFrame] | None:
    """Most recent regular (non trim-sweep) `run_token_zipf_analysis` run that
    covers `system`, for read-only consumption by other consolidation tooling
    (the per-system consolidation report). Scans `artifacts/analysis/token_zipf/runs/`
    newest-run-date-first (ISO date directory names sort correctly as
    strings) and returns the first one whose persisted
    `token_zipf_summary.parquet` has rows for `system`. Returns `None` if no
    run has ever covered it -- callers should treat that as "not yet run",
    not an error, since none of this module's outputs are durable/cached.

    Deliberately skips `-trim-sweep` run directories: those carry a
    different summary schema (`trim_setting` instead of `tier`) and answer a
    different question: the trim-profile axis, not the name-tier sweep the
    per-system report consolidates.
    """
    runs_root = analysis_report_runs_root(roots, "token_zipf")
    if not runs_root.exists():
        return None
    candidate_dirs = sorted(
        (
            child
            for child in runs_root.iterdir()
            if child.is_dir() and not child.name.endswith("-trim-sweep")
        ),
        key=lambda child: child.name,
        reverse=True,
    )
    for run_dir in candidate_dirs:
        summary_path = _metrics_dir(run_dir) / "token_zipf_summary.parquet"
        if not summary_path.exists():
            continue
        summary_df = pl.read_parquet(summary_path)
        if summary_df.height == 0 or "system" not in summary_df.columns:
            continue
        filtered = summary_df.filter(pl.col("system") == system)
        if filtered.height > 0:
            return run_dir.name, filtered
    return None


def load_persisted_word_token_stats(
    roots: WorkspaceRoots, system: str
) -> dict[str, tuple[pl.DataFrame, int]] | None:
    """Per-tier token/document-frequency stats from the most recent
    `run_token_zipf_analysis` run that covered `system`, so read-only
    consumers (the exploratory notebook) don't repeat that run's full-corpus
    scan just to look at the numbers it already produced. Returns `None` if
    no run has ever covered this system, or its per-tier stats parquet files
    predate this being persisted -- callers should fall back to
    `compute_word_token_stats` directly in that case.
    """
    found = find_latest_zipf_summary_for_system(roots, system)
    if found is None:
        return None
    run_name, summary_df = found
    stats_dir = _metrics_dir(analysis_report_run_dir(roots, "token_zipf", run_name))

    stats_by_tier: dict[str, tuple[pl.DataFrame, int]] = {}
    for row in summary_df.iter_rows(named=True):
        tier = row["tier"]
        stats_path = stats_dir / f"{system}_{tier}_token_stats.parquet"
        if not stats_path.exists():
            continue
        stats_by_tier[tier] = (pl.read_parquet(stats_path), int(row["total_docs"]))
    return stats_by_tier or None


def load_persisted_power_law_fit(
    roots: WorkspaceRoots, system: str, tier: str
) -> dict[str, float] | None:
    """The persisted power-law/log-normal comparison for one system/tier,
    from the most recent `run_token_zipf_analysis` run that covered it, so
    read-only consumers (the exploratory notebook) can reuse a result
    already computed with `compare_power_law=True` instead of repeating the
    fit. Returns `None` if no run has ever covered this system/tier, or
    that run didn't have the comparison enabled -- callers should fall back
    to `compare_power_law_and_lognormal` directly in that case.

    Uses `_POWER_LAW_SUMMARY_FIELDS` for both the column names to read and
    the returned dict's keys, so this can't drift out of step with what
    `run_token_zipf_analysis` actually persists.
    """
    found = find_latest_zipf_summary_for_system(roots, system)
    if found is None:
        return None
    _, summary_df = found
    matching = summary_df.filter(pl.col("tier") == tier)
    if matching.height == 0:
        return None
    record = matching.row(0, named=True)
    if record.get(f"power_law_{_POWER_LAW_SUMMARY_FIELDS[0]}") is None:
        return None
    return {
        field: float(record[f"power_law_{field}"])
        for field in _POWER_LAW_SUMMARY_FIELDS
    }


def discover_cleansed_files(roots: WorkspaceRoots, system: str) -> list[Path]:
    """A system's real cleansed shards.

    Delegates to `src/workspace`, which owns the layout. This used to carry
    its own copy of the rule -- partitioned view wins, staging never counts
    -- kept "in step" with `company_tokenize`'s copy by hand, which is
    exactly how a blind glob once picked up `chunks/` staging as though it
    were cleansed output and the fix landed in only one of the two.
    """
    return resolve_primary_files(
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME),
        system_code=system,
    )


def _files_have_column(files: list[Path], name_col: str) -> bool:
    """Check the first shard's schema rather than assuming every system's
    cleansed data uses the standard name-tier columns. Not every source has
    migrated to the standardized `name_cleansed`/`name_cleansed_basic`
    contract yet (for example dbpedia's cleansed data still carries
    `companyname_cleansed` and no basic tier at all), and a
    system missing one tier shouldn't abort the whole sweep."""
    if not files:
        return False
    return name_col in pl.read_parquet_schema(files[0])


def _non_empty_text_lazyframe(files: list[Path], name_col: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet([str(path) for path in files])
        .select(name_col)
        .filter(
            pl.col(name_col).is_not_null() & (pl.col(name_col).str.strip_chars() != "")
        )
    )


def trim_text_expr(source_expr: pl.Expr, noise_words: frozenset[str]) -> pl.Expr:
    """Vectorized whole-word noise-word trim.

    Semantically equivalent to `company_tokenize.tfidf.trim_text_before_tokenization`
    (case-insensitive exact whole-word match) but implemented natively in
    polars instead of that helper's pure-Python per-row loop, which doesn't
    scale to this sweep's multi-million-row corpora (`gb` ~5.7M rows, `fr`
    ~12.9M rows). Reuses `resolve_noise_words()` for the actual token-set
    resolution/precedence -- only the row-level trimming is reimplemented.
    """
    if not noise_words:
        return source_expr
    lowered_noise_words = [token.lower() for token in noise_words]
    return (
        source_expr.str.split(" ")
        .list.eval(
            pl.element().filter(
                ~pl.element().str.to_lowercase().is_in(lowered_noise_words)
            )
        )
        .list.join(" ")
    )


_EMPTY_TOKEN_STATS = pl.DataFrame(
    {"token": [], "document_frequency": []},
    schema={"token": pl.Utf8, "document_frequency": pl.UInt32},
)
_EMPTY_INCIDENCE: dict[str, float] = {
    "total_names": 0,
    "names_with_hapax_pct": 0.0,
    "names_with_unseen_in_wordfreq_pct": 0.0,
    "occurrence_weighted_unseen_pct": 0.0,
}


@dataclass(frozen=True)
class TokenizedCorpus:
    """One corpus column read once, held as a token list per name.

    Both figures this module reports for a system and tier, the per-token
    document frequencies and the name-level incidence of rare tokens, are
    read from these lists rather than from the files, so the corpus is
    scanned once per tier rather than once per figure. `rows` carries
    `_row_id` and `_tokens`, every lowercased word of the trimmed text in
    order, and one row per non-empty name.
    """

    rows: pl.DataFrame

    @property
    def total_docs(self) -> int:
        return self.rows.height

    def word_token_stats(self) -> pl.DataFrame:
        """Per-token document-frequency stats.

        Uses document frequency (names containing the token at least once),
        not raw occurrence count, as the ranking measure -- company names
        rarely repeat a word, so DF and TF are effectively the same signal
        here, and DF matches the convention this repo's token-rarity stats
        already use.
        """
        if self.total_docs == 0:
            return _EMPTY_TOKEN_STATS
        return (
            self.rows.select(pl.col("_tokens").list.unique().explode().alias("token"))
            .drop_nulls("token")
            .group_by("token")
            .len()
            .rename({"len": "document_frequency"})
        )

    def name_level_incidence(self, rarity_lookup: pl.DataFrame) -> dict[str, float]:
        """Name-level (not token-level) rarity/OOV incidence.

        Reports both the fraction of *names* touched by a hapax/OOV token
        (the operationally relevant number for blocking) and the
        occurrence-weighted OOV percentage (mass of token occurrences that
        are OOV, not just the count of OOV token *types*), alongside the
        type-level percentage callers can compute from `rarity_lookup`
        directly. Type-level rarity alone overstates operational impact,
        which is why both are reported.
        """
        total_names = self.total_docs
        if total_names == 0 or rarity_lookup.height == 0:
            return dict(_EMPTY_INCIDENCE)

        exploded = (
            self.rows.explode("_tokens")
            .rename({"_tokens": "token"})
            .drop_nulls("token")
        )
        total_token_occurrences = exploded.height
        if total_token_occurrences == 0:
            return {**_EMPTY_INCIDENCE, "total_names": total_names}

        has_unseen_col = "unseen_in_wordfreq" in rarity_lookup.columns
        flag_columns = [pl.col("document_frequency").eq(1).alias("is_hapax")]
        if has_unseen_col:
            flag_columns.append(pl.col("unseen_in_wordfreq"))
        flags = rarity_lookup.select("token", *flag_columns)

        joined = exploded.join(flags, on="token", how="left").with_columns(
            pl.col("is_hapax").fill_null(False)
        )
        if has_unseen_col:
            joined = joined.with_columns(pl.col("unseen_in_wordfreq").fill_null(False))

        agg_columns = [pl.col("is_hapax").any().alias("has_hapax")]
        if has_unseen_col:
            agg_columns.append(pl.col("unseen_in_wordfreq").any().alias("has_unseen"))
        per_name = joined.group_by("_row_id").agg(*agg_columns)

        names_with_hapax = int(per_name.get_column("has_hapax").sum())
        names_with_unseen = (
            int(per_name.get_column("has_unseen").sum()) if has_unseen_col else 0
        )
        unseen_occurrences = (
            int(joined.get_column("unseen_in_wordfreq").sum()) if has_unseen_col else 0
        )

        return {
            "total_names": total_names,
            "names_with_hapax_pct": names_with_hapax / total_names,
            "names_with_unseen_in_wordfreq_pct": (
                names_with_unseen / total_names if has_unseen_col else 0.0
            ),
            "occurrence_weighted_unseen_pct": (
                unseen_occurrences / total_token_occurrences if has_unseen_col else 0.0
            ),
        }


def tokenize_corpus(
    files: list[Path],
    name_col: str,
    *,
    noise_words: frozenset[str] | None = None,
) -> TokenizedCorpus:
    """Read one column of a corpus once into per-name token lists.

    Operates on `name_col` text directly rather than any tokenizer output, so
    results are unaffected by tokenize-stage noise-word trimming (see
    src/tests/company_tokenize/scripts/test_noise_reduction_layers.py for why that distinction
    matters) -- unless `noise_words` is supplied, in which case that trim is
    applied deliberately, to measure its effect: the trim-profile axis.
    """
    if not files:
        return TokenizedCorpus(
            pl.DataFrame(
                schema={"_row_id": pl.UInt32, "_tokens": pl.List(pl.Utf8)},
            )
        )
    text_expr = trim_text_expr(pl.col(name_col), noise_words or frozenset())
    rows = (
        _non_empty_text_lazyframe(files, name_col)
        .with_row_index("_row_id")
        .select(
            "_row_id",
            text_expr.str.to_lowercase()
            .str.extract_all(_WORD_TOKEN_PATTERN)
            .alias("_tokens"),
        )
        .collect()
    )
    return TokenizedCorpus(rows)


def compute_word_token_stats(
    files: list[Path],
    name_col: str,
    *,
    noise_words: frozenset[str] | None = None,
) -> tuple[pl.DataFrame, int]:
    """`TokenizedCorpus.word_token_stats` and its name count from a fresh read.

    For a caller wanting only the stats; a caller wanting the incidence too
    holds the `tokenize_corpus` result and asks it for both.
    """
    corpus = tokenize_corpus(files, name_col, noise_words=noise_words)
    return corpus.word_token_stats(), corpus.total_docs


def fit_zipf_slope(ranks: list[float], frequencies: list[float]) -> float:
    """Least-squares slope of log(frequency) vs log(rank).

    Classic Zipf's law predicts a slope near -1 for natural language.
    """
    pairs = [(r, f) for r, f in zip(ranks, frequencies) if r > 0 and f > 0]
    if len(pairs) < 2:
        return 0.0
    log_ranks = [math.log(r) for r, _ in pairs]
    log_freqs = [math.log(f) for _, f in pairs]
    n = len(log_ranks)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
    denominator = sum((x - mean_x) ** 2 for x in log_ranks)
    return numerator / denominator if denominator else 0.0


def build_wordfreq_reference_curve(language: str, top_n: int = 5000) -> pl.DataFrame:
    """Rank-ordered general-language reference curve for overlay plotting."""
    words = top_n_list(language, top_n)
    if not words:
        return pl.DataFrame({"rank": [], "zipf_frequency": []})
    ranks = list(range(1, len(words) + 1))
    zipf_values = [zipf_frequency(word, language) for word in words]
    return pl.DataFrame({"rank": ranks, "zipf_frequency": zipf_values})


def _relative_to_top(values: list[float]) -> list[float]:
    if not values or values[0] <= 0:
        return values
    top = values[0]
    return [value / top for value in values]


_LEGEND_FONTSIZE = 8
# How many of the most frequent words the head-words picture names. Thirty
# reaches past the descriptor shelf on every register looked at so far
# (`ltd`, then ten or so words at 2% to 4% of names, then the slide into the
# tail), which is the region a rank-frequency curve cannot show by itself.
HEAD_WORDS_TOP_N = 30


def draw_zipf_curve(
    ax: plt.Axes,
    corpus_stats: pl.DataFrame,
    wordfreq_curve: pl.DataFrame | None,
    *,
    title: str,
) -> dict[str, float]:
    """Draw a log-log rank-vs-relative-frequency Zipf curve onto `ax`.

    Both curves are normalized to their own rank-1 value so that shape
    (decay slope), not absolute scale, is what's compared -- corpus document
    frequency and wordfreq's per-billion-word scale are not the same unit.
    Returns the fitted slope(s) so callers can persist them alongside the plot.

    Public and axis-level rather than figure-level: this is the seam both
    consumers need. `render_zipf_plot` wraps it to write a standalone file and
    `render_zipf_overview_grid` calls it per panel, while a notebook supplies
    its own axis to get the identical curve inline. Anything that draws a Zipf
    curve should call this rather than reimplement the normalization.

    The slope is one least-squares line over every rank, so on a million-type
    vocabulary the hapax plateau dominates it and the legal-form spike at the
    lowest ranks barely moves it. This once split the curve at a share
    threshold and fitted the head and tail apart; the split marked how many
    legal-form words the tier carried, not where the curve's slope changed,
    and a head of one or two ranks has no slope. Where the head ends is shown
    by `draw_head_words`, which names the words, and is what the fitted
    Zipf-Mandelbrot offset would measure.
    """
    ordered = corpus_stats.sort("document_frequency", descending=True)
    freqs = [float(v) for v in ordered.get_column("document_frequency").to_list()]
    ranks = [float(r) for r in range(1, len(freqs) + 1)]
    result: dict[str, float] = {"corpus_slope": 0.0, "reference_slope": 0.0}
    if not freqs:
        ax.set_title(title)
        return result

    corpus_slope = fit_zipf_slope(ranks, freqs)
    result["corpus_slope"] = corpus_slope
    relative = _relative_to_top(freqs)

    ax.plot(
        ranks,
        relative,
        color="#2E5EAA",
        linewidth=1.5,
        label=f"corpus (fitted slope={corpus_slope:.2f})",
    )

    if wordfreq_curve is not None and wordfreq_curve.height > 0:
        wf_ranks = [float(r) for r in wordfreq_curve.get_column("rank").to_list()]
        wf_zipf = wordfreq_curve.get_column("zipf_frequency").to_list()
        wf_linear = [10.0**value for value in wf_zipf]
        reference_slope = fit_zipf_slope(wf_ranks, wf_linear)
        result["reference_slope"] = reference_slope
        ax.plot(
            wf_ranks,
            _relative_to_top(wf_linear),
            color="#999999",
            linestyle="--",
            linewidth=1.5,
            label=f"general language (fitted slope={reference_slope:.2f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Relative frequency (log, normalized to rank 1)")
    ax.set_title(title)
    ax.legend(fontsize=_LEGEND_FONTSIZE)
    return result


def render_zipf_plot(
    corpus_stats: pl.DataFrame,
    wordfreq_curve: pl.DataFrame | None,
    output_path: Path | None = None,
    *,
    title: str,
    render: bool = True,
) -> dict[str, float]:
    """Render the log-log rank-vs-relative-frequency Zipf plot to a file.

    `render=False`, or no `output_path`, returns the same fitted slopes
    without writing anything. Rasterising a figure is where a plotting call's
    cost actually sits, and a caller asserting the numbers this returns is
    asserting its own output rather than matplotlib's renderer.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    result = draw_zipf_curve(ax, corpus_stats, wordfreq_curve, title=title)
    if render and output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return result


def draw_head_words(
    ax: plt.Axes,
    corpus_stats: pl.DataFrame,
    *,
    title: str,
    top_n: int = HEAD_WORDS_TOP_N,
) -> list[str]:
    """Draw the head of the distribution as named words onto `ax`.

    The `top_n` most frequent words in rank order on a linear rank axis, each
    named on the axis, against the share of names containing it on a log
    axis (or the bare count when `corpus_stats` carries no
    `document_frequency_pct`). Where the rank-frequency curve compresses the
    first decade into a corner, this is the corner read out: the legal-form
    spike, the shelf of descriptor words under it, and the rank at which the
    shelf gives way to the tail, with the words themselves saying what each
    is made of. Returns the words drawn, in rank order, so a caller can
    assert or persist what the picture shows without reading the picture.
    """
    ordered = corpus_stats.sort("document_frequency", descending=True).head(top_n)
    words = [str(word) for word in ordered.get_column("token").to_list()]
    ax.set_title(title)
    if not words:
        return words

    if "document_frequency_pct" in ordered.columns:
        values = [
            float(v) * 100
            for v in ordered.get_column("document_frequency_pct").to_list()
        ]
        ax.set_ylabel("share of names containing the word (%)")
    else:
        values = [float(v) for v in ordered.get_column("document_frequency").to_list()]
        ax.set_ylabel("names containing the word")
    ranks = list(range(1, len(words) + 1))

    ax.plot(ranks, values, marker="o", color="#2E5EAA", linewidth=1.5)
    ax.set_yscale("log")
    # Each word runs up and to the right from its own point, so the eye reads
    # the word and the share together and the rank axis stays numeric.
    for rank, value, word in zip(ranks, values, words):
        ax.annotate(
            word,
            (rank, value),
            rotation=45,
            fontsize=_LEGEND_FONTSIZE - 1,
            textcoords="offset points",
            xytext=(3, 4),
            ha="left",
            va="bottom",
            rotation_mode="anchor",
        )
    ax.set_xticks(ranks)
    ax.set_xlabel("rank")
    ax.set_ylim(min(values) * 0.5, max(values) * 3)
    return words


def render_head_words_plot(
    corpus_stats: pl.DataFrame,
    output_path: Path | None = None,
    *,
    title: str,
    top_n: int = HEAD_WORDS_TOP_N,
    render: bool = True,
) -> list[str]:
    """Render `draw_head_words` to a file, returning the words it drew.

    `render=False`, or no `output_path`, returns the words without writing,
    for the same reason `render_zipf_plot` offers it.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    words = draw_head_words(ax, corpus_stats, title=title, top_n=top_n)
    if render and output_path is not None:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return words


# `compare_power_law_and_lognormal`'s return-dict keys that get persisted
# into `token_zipf_summary.parquet` as `power_law_<field>` columns (all but
# `n_tokens`, redundant with `unique_tokens` already in the same row). One
# list drives both directions --
# `run_token_zipf_analysis`'s write and `load_persisted_power_law_fit`'s
# read -- so they can't drift out of step with each other or with the
# function's actual return shape.
_POWER_LAW_SUMMARY_FIELDS = (
    "alpha",
    "xmin",
    "n_fitted",
    "ks_statistic",
    "loglikelihood_ratio",
    "mean_loglikelihood_diff",
    "p_value",
)


def compare_power_law_and_lognormal(
    corpus_stats: pl.DataFrame,
    *,
    skip_top_n: int = 0,
) -> dict[str, float]:
    """Clauset-Shalizi-Newman-style test for whether a corpus's token
    document-frequency distribution is better explained by a power law or a
    log-normal, via the `powerlaw` package.

    Returns a Kolmogorov-Smirnov statistic for the power-law fit's own
    goodness of fit (`ks_statistic`), and a log-likelihood-ratio test
    (Vuong's test) comparing power-law against log-normal as competing
    models (`loglikelihood_ratio` positive favors power-law, negative favors
    log-normal; `p_value` is that preference's significance -- above ~0.1
    the test can't distinguish the two).

    `loglikelihood_ratio` is `powerlaw`'s normalized Vuong statistic
    (R / sqrt(n*variance), via `normalized_ratio=True`) -- correctly
    normalized *for significance testing*, but as any z-statistic does, it
    still grows with sqrt(n) for a genuine non-zero effect, so its
    magnitude isn't comparable across corpora of different sizes.
    `mean_loglikelihood_diff` is: the per-token average log-likelihood
    advantage of power-law over log-normal within the fitted range,
    independent of n, and the number worth comparing across systems/tiers
    of different sizes. `n_fitted` is that range's token count (after
    `skip_top_n` and the auto-estimated `xmin`, so usually smaller than
    `n_tokens`).

    `skip_top_n` drops that many of the highest-document-frequency tokens
    before fitting, for a caller that wants the legal-form spike out of the
    test by hand; the sweep passes nothing, since `powerlaw` auto-estimates
    `xmin` over the whole vocabulary and a spike of one or two tokens in a
    million lies far above any `xmin` it picks. The remaining tokens' `xmin`
    is still auto-estimated in the usual way; this only removes tokens the
    caller already knows to be outliers, it doesn't hand-pick a cutoff for
    the statistical test itself.

    Runtime warning: discrete `xmin` estimation is expensive -- on
    offeneregister's ~1M-token cleansed-tier vocabulary this took ~70s even
    with `estimate_discrete=True` (the fast approximation). Call this
    deliberately, not from a cell that reruns automatically.
    """
    ordered = corpus_stats.sort("document_frequency", descending=True)
    document_frequencies = [
        int(v) for v in ordered.get_column("document_frequency").to_list()[skip_top_n:]
    ]
    # Not just `len(document_frequencies) < 2`: `powerlaw`'s discrete xmin
    # search needs at least 2 *distinct* values to fit anything, and hard
    # crashes (`ValueError: No data points in defined range`) rather than
    # returning nan when it doesn't -- found running this for real against a
    # small corpus/tier. A degenerate distribution with one repeated value
    # has nothing meaningful to compare anyway.
    if len(set(document_frequencies)) < 2:
        return {
            "n_tokens": float(len(document_frequencies)),
            "n_fitted": 0.0,
            "alpha": 0.0,
            "xmin": 0.0,
            "ks_statistic": 0.0,
            "loglikelihood_ratio": 0.0,
            "mean_loglikelihood_diff": 0.0,
            "p_value": 1.0,
        }

    try:
        fit = powerlaw.Fit(
            document_frequencies, discrete=True, estimate_discrete=True, verbose=False
        )
        loglikelihood_ratio, p_value = fit.distribution_compare(
            "power_law", "lognormal", normalized_ratio=True
        )
        # Not `fit.power_law.KS()`: powerlaw 2.0.0's back-compat shim for
        # that method calls `compute_distance_metrics(data)` as a bare name
        # instead of `self.compute_distance_metrics(data)`, an upstream
        # NameError bug. Calling the real method directly and reading `.D`
        # sidesteps it.
        fit.power_law.compute_distance_metrics()
        alpha = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        ks_statistic = float(fit.power_law.D)
        # The same in-range data `distribution_compare` just scored both
        # distributions against, reused here to compute a size-independent
        # effect size (a mean) rather than the size-dependent test
        # statistic (a sum) `loglikelihood_ratio` already reports.
        power_law_loglikelihoods = fit.power_law.loglikelihoods()
        lognormal_loglikelihoods = fit.lognormal.loglikelihoods()
        n_fitted = float(len(power_law_loglikelihoods))
        mean_loglikelihood_diff = float(
            (sum(power_law_loglikelihoods) - sum(lognormal_loglikelihoods)) / n_fitted
        )
    except ValueError:
        # `powerlaw`'s discrete xmin search hard-crashes rather than
        # returning nan when a small corpus/tier leaves too few points in a
        # candidate xmin's range ("No data points in defined range of the
        # distribution") -- a distribution this thin has nothing meaningful
        # to compare anyway, same as the too-few-distinct-values case above.
        return {
            "n_tokens": float(len(document_frequencies)),
            "n_fitted": 0.0,
            "alpha": 0.0,
            "xmin": 0.0,
            "ks_statistic": 0.0,
            "loglikelihood_ratio": 0.0,
            "mean_loglikelihood_diff": 0.0,
            "p_value": 1.0,
        }
    return {
        "n_tokens": float(len(document_frequencies)),
        "n_fitted": n_fitted,
        "alpha": alpha,
        "xmin": xmin,
        "ks_statistic": ks_statistic,
        "loglikelihood_ratio": float(loglikelihood_ratio),
        "mean_loglikelihood_diff": mean_loglikelihood_diff,
        "p_value": float(p_value),
    }


def render_zipf_overview_grid(
    entries: list[tuple[str, pl.DataFrame, pl.DataFrame | None]],
    output_path: Path,
    *,
    suptitle: str,
    render: bool = True,
) -> None:
    """Render a small-multiples grid of Zipf curves (one panel per entry).

    Intended as a single "drop in the directory" overview image -- e.g. one
    panel per system for a fixed name-tier -- so cross-system comparison
    doesn't require opening a dozen separate per-system-tier PNGs. Not
    referenced from the markdown report; it's a standalone visual artefact
    in `viz_dir` alongside the per-system-tier files.
    """
    if not entries:
        return

    ncols = min(3, len(entries))
    nrows = math.ceil(len(entries) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]

    for ax, (label, corpus_stats, wordfreq_curve) in zip(flat_axes, entries):
        draw_zipf_curve(ax, corpus_stats, wordfreq_curve, title=label)

    for ax in flat_axes[len(entries) :]:
        ax.set_visible(False)

    fig.suptitle(suptitle)
    if render:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)


_TIER_BAR_COLORS: dict[str, str] = {
    "raw": "#999999",
    "basic": "#2E5EAA",
    "cleansed": "#3F9142",
}
_REGRESSION_COLOR = "#C0392B"
_REGRESSION_BASELINE_TIER = "raw"
_REGRESSION_FLAGGED_TIER = "cleansed"


def render_tier_metric_bar_chart(
    summary_rows: list[dict[str, object]],
    metric_key: str,
    output_path: Path,
    *,
    ylabel: str,
    title: str,
    highlight_regressions: bool = False,
    render: bool = True,
) -> bool:
    """Grouped bar chart of one summary metric, systems on the x-axis with one
    bar per name tier, in place of reading the equivalent column out of the
    markdown table by eye.

    `raw` is included as a plain, labeled bar/reference only -- it is not a
    trustworthy "cleaning should only improve on this" baseline. Per
    docs/findings/token-zipf.md's `singlechar`-collapse
    confound, raw's isolated single-letter fragments (e.g. the "I"/"B"/"M" in
    "I B M") coincidentally score as ordinary high-frequency English to
    `wordfreq`, so raw's OOV% reads artificially low; basic/cleansed
    legitimately score *higher* once those fragments collapse into real
    tokens ("ibm"). A cleansed-worse-than-raw bar is therefore an expected
    consequence of that confound in the common case, not evidence cleansing
    regressed -- so it is off by default.

    When `highlight_regressions=True`, a system's `cleansed`-tier bar is
    still highlighted (`_REGRESSION_COLOR`) when worse than that same
    system's `raw`-tier bar, for anyone who wants that comparison anyway
    (opt-in via `run_token_zipf_analysis(highlight_regressions=...)` /
    `--highlight-regressions` on the CLI).

    Rows with a `None` metric value (systems with no resolved reference
    language) are dropped rather than plotted as zero. Returns whether a
    chart was actually written (nothing to plot is not an error -- e.g.
    `names_with_unseen_in_wordfreq_pct` is `None` for every system when none
    of them resolve to a wordfreq reference language).
    """
    tiers = [tier for tier in NAME_TIER_COLUMNS if tier in _TIER_BAR_COLORS]
    by_system: dict[str, dict[str, float]] = {}
    systems: list[str] = []
    for row in summary_rows:
        value = row.get(metric_key)
        if not isinstance(value, (int, float)):
            continue
        system = str(row["system"])
        tier = str(row["tier"])
        if tier not in tiers:
            continue
        if system not in by_system:
            by_system[system] = {}
            systems.append(system)
        by_system[system][tier] = float(value)

    systems = [system for system in systems if len(by_system[system]) == len(tiers)]
    if not systems:
        return False

    x = list(range(len(systems)))
    bar_width = 0.8 / len(tiers)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(systems)), 5))

    for tier_index, tier in enumerate(tiers):
        offsets = [pos + (tier_index - (len(tiers) - 1) / 2) * bar_width for pos in x]
        heights = [by_system[system][tier] * 100 for system in systems]
        colors = [_TIER_BAR_COLORS[tier]] * len(systems)
        if highlight_regressions and tier == _REGRESSION_FLAGGED_TIER:
            for i, system in enumerate(systems):
                baseline = by_system[system].get(_REGRESSION_BASELINE_TIER)
                if baseline is not None and by_system[system][tier] > baseline:
                    colors[i] = _REGRESSION_COLOR
        ax.bar(offsets, heights, width=bar_width, color=colors, label=tier)

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_TIER_BAR_COLORS[tier]) for tier in tiers
    ]
    labels = list(tiers)
    if highlight_regressions:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_REGRESSION_COLOR))
        labels.append(
            f"{_REGRESSION_FLAGGED_TIER} worse than {_REGRESSION_BASELINE_TIER}"
        )
    ax.legend(handles, labels, fontsize=8)
    if render:
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def compute_name_level_incidence(
    files: list[Path],
    name_col: str,
    rarity_lookup: pl.DataFrame,
    *,
    noise_words: frozenset[str] | None = None,
) -> dict[str, float]:
    """`TokenizedCorpus.name_level_incidence` from a fresh read.

    For a caller with no stats pass of its own; the two runners in this module
    hold the `tokenize_corpus` result their stats came from and ask it instead.
    """
    return tokenize_corpus(
        files, name_col, noise_words=noise_words
    ).name_level_incidence(rarity_lookup)


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(f"[token_zipf] {message}")


def run_token_zipf_analysis(
    roots: WorkspaceRoots,
    *,
    systems: Iterable[str] | None = None,
    run_date: str | None = None,
    wordfreq_top_n: int = 5000,
    highlight_regressions: bool = False,
    compare_power_law: bool = False,
    render_plots: bool = True,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> TokenZipfPaths:
    effective_run_date = run_date or datetime.now(UTC).date().isoformat()
    paths = resolve_token_zipf_paths(roots, effective_run_date, force=force)

    if systems is not None:
        systems_list = list(systems)
    else:
        data_root_dir = data_root(roots)
        discovered_systems = (
            sorted(
                child.name
                for child in data_root_dir.iterdir()
                if child.is_dir()
                and system_layer_dir(
                    roots, child.name, layer=CLEANSED_LAYER_NAME
                ).exists()
            )
            if data_root_dir.exists()
            else []
        )
        systems_list = filter_runnable_systems(discovered_systems)

    wordfreq_curve_cache: dict[str, pl.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    tier_entries: dict[str, list[tuple[str, pl.DataFrame, pl.DataFrame | None]]] = {
        tier_name: [] for tier_name in NAME_TIER_COLUMNS
    }

    for system in systems_list:
        files = discover_cleansed_files(roots, system)
        if not files:
            _emit_progress(progress, f"system={system} no cleansed files, skipping")
            continue

        language = resolve_system_language(system)
        wordfreq_curve: pl.DataFrame | None = None
        if language:
            if language not in wordfreq_curve_cache:
                wordfreq_curve_cache[language] = build_wordfreq_reference_curve(
                    language, wordfreq_top_n
                )
            wordfreq_curve = wordfreq_curve_cache[language]

        for tier_name, name_col in NAME_TIER_COLUMNS.items():
            if not _files_have_column(files, name_col):
                _emit_progress(
                    progress,
                    f"system={system} tier={tier_name} column={name_col} not present in this system's cleansed schema, skipping",
                )
                continue

            corpus = tokenize_corpus(files, name_col)
            stats, total_docs = corpus.word_token_stats(), corpus.total_docs
            if stats.height == 0:
                _emit_progress(
                    progress,
                    f"system={system} tier={tier_name} column={name_col} no tokens, skipping",
                )
                continue

            stats = stats.with_columns(
                (pl.col("document_frequency") / pl.lit(max(total_docs, 1))).alias(
                    "document_frequency_pct"
                )
            )
            if language:
                stats = attach_language_frequency(stats, language=language)

            stats.write_parquet(
                paths.stats_dir / f"{system}_{tier_name}_token_stats.parquet"
            )

            # Two pictures per system/tier: the rank-frequency curve, and the
            # head of the same distribution as named words. Both are archived
            # into the consolidated per-system report, so each has a column
            # of its own -- that report reads a persisted summary from an
            # earlier run and can only find a file it has a recorded path for.
            plot_path = paths.viz_dir / f"{system}_{tier_name}_zipf.png"
            head_words_path = paths.viz_dir / f"{system}_{tier_name}_head_words.png"
            slopes = render_zipf_plot(
                stats,
                wordfreq_curve,
                plot_path,
                title=f"{system.upper()} {tier_name} ({name_col}) -- token rank-frequency",
                render=render_plots,
            )
            render_head_words_plot(
                stats,
                head_words_path,
                title=(
                    f"{system.upper()} {tier_name} ({name_col}) -- "
                    f"{HEAD_WORDS_TOP_N} most frequent words"
                ),
                render=render_plots,
            )
            tier_entries[tier_name].append((system.upper(), stats, wordfreq_curve))

            # Opt-in (`compare_power_law`): the discrete xmin search is
            # expensive (~70s on a ~1M-token vocabulary -- see
            # `compare_power_law_and_lognormal`'s own docstring), so this
            # sweep doesn't pay it by default. When it is on, every system
            # the caller named is fitted; the fit needs no reference
            # language, only the corpus's own document frequencies.
            power_law_fit = (
                compare_power_law_and_lognormal(stats) if compare_power_law else None
            )

            total_types = stats.height
            type_level_unseen_pct = (
                float(stats.get_column("unseen_in_wordfreq").sum()) / total_types
                if language and total_types > 0
                else None
            )

            incidence = corpus.name_level_incidence(stats)

            summary_rows.append(
                {
                    "system": system,
                    "tier": tier_name,
                    "name_col": name_col,
                    "language": language,
                    "total_docs": total_docs,
                    "unique_tokens": total_types,
                    "corpus_zipf_slope": slopes["corpus_slope"],
                    "reference_zipf_slope": slopes["reference_slope"],
                    **{
                        f"power_law_{field}": (
                            power_law_fit[field] if power_law_fit else None
                        )
                        for field in _POWER_LAW_SUMMARY_FIELDS
                    },
                    "type_level_unseen_in_wordfreq_pct": type_level_unseen_pct,
                    "occurrence_weighted_unseen_in_wordfreq_pct": incidence[
                        "occurrence_weighted_unseen_pct"
                    ]
                    if language
                    else None,
                    "names_with_hapax_pct": incidence["names_with_hapax_pct"],
                    "names_with_unseen_in_wordfreq_pct": (
                        incidence["names_with_unseen_in_wordfreq_pct"]
                        if language
                        else None
                    ),
                    "plot_path": to_portable_path_str(
                        plot_path, project_root=roots.checkout
                    ),
                    "plot_relative_link": _relative_viz_link(plot_path),
                    "head_words_plot_path": to_portable_path_str(
                        head_words_path, project_root=roots.checkout
                    ),
                    "head_words_plot_relative_link": _relative_viz_link(
                        head_words_path
                    ),
                }
            )
            _emit_progress(
                progress,
                f"system={system} tier={tier_name} names={total_docs:,} "
                f"types={total_types:,} slope={slopes['corpus_slope']:.2f}",
            )

    overview_images: list[tuple[str, str]] = []
    for tier_name, entries in tier_entries.items():
        if not entries:
            continue
        overview_path = paths.viz_dir / f"_zipf_overview_{tier_name}.png"
        render_zipf_overview_grid(
            entries,
            overview_path,
            suptitle=f"Token rank-frequency by system -- {tier_name} tier",
            render=render_plots,
        )
        overview_images.append(
            (
                f"Overview -- {tier_name} tier (all systems)",
                _relative_viz_link(overview_path),
            )
        )
        _emit_progress(progress, f"tier={tier_name} overview -> {overview_path}")

    bar_chart_specs = [
        (
            "type_level_unseen_in_wordfreq_pct",
            "Type OOV % (vs wordfreq)",
            "Type-level OOV%, by system and name tier",
            "_oov_type_by_system.png",
        ),
        (
            "occurrence_weighted_unseen_in_wordfreq_pct",
            "Occurrence-weighted OOV % (vs wordfreq)",
            "Occurrence-weighted OOV%, by system and name tier",
            "_oov_occurrence_weighted_by_system.png",
        ),
        (
            "names_with_hapax_pct",
            "Names with a hapax token %",
            "Name-level hapax incidence, by system and name tier",
            "_hapax_incidence_by_system.png",
        ),
        (
            "names_with_unseen_in_wordfreq_pct",
            "Names with an OOV token %",
            "Name-level OOV incidence, by system and name tier",
            "_oov_incidence_by_system.png",
        ),
    ]
    for metric_key, ylabel, title, filename in bar_chart_specs:
        bar_path = paths.viz_dir / filename
        written = render_tier_metric_bar_chart(
            summary_rows,
            metric_key,
            bar_path,
            ylabel=ylabel,
            title=title,
            highlight_regressions=highlight_regressions,
            render=render_plots,
        )
        if written:
            _emit_progress(progress, f"metric={metric_key} bar chart -> {bar_path}")
            overview_images.append((title, _relative_viz_link(bar_path)))

    summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    summary_df.write_parquet(paths.summary_path)
    _write_report(
        paths.report_path,
        summary_rows,
        run_date=effective_run_date,
        overview_images=overview_images,
    )
    return paths


def build_trim_settings(
    system: str, roots: WorkspaceRoots
) -> list[tuple[str, frozenset[str]]]:
    """Resolve the tokenize-stage noise-word trim settings swept for a
    system: no trim, the packaged default, and -- when a per-corpus
    `noise_words.json` has actually been generated for this system (see
    `generate_noise_words.py`'s train-iterate-promote workflow) --
    that per-corpus file at each of its strict/balanced/aggressive profiles.

    Uses `resolve_noise_words()` for token-set resolution/precedence
    (grounded by `src/tests/company_tokenize/scripts/test_noise_reduction_layers.py`), not the
    row-level trim (see `trim_text_expr`).
    """
    settings: list[tuple[str, frozenset[str]]] = [
        ("none", frozenset()),
        (
            "packaged_default",
            frozenset(resolve_noise_words(noise_words_profile="aggressive")),
        ),
    ]

    per_corpus_path = scope_directory_files(
        tokenizer_scope_dir(roots, system=system)
    ).noise_words
    if per_corpus_path.exists():
        for profile in ("strict", "balanced", "aggressive"):
            settings.append(
                (
                    f"per_corpus_{profile}",
                    frozenset(
                        resolve_noise_words(
                            noise_words_path=per_corpus_path,
                            noise_words_profile=profile,
                        )
                    ),
                )
            )
    return settings


def run_trim_profile_sweep(
    roots: WorkspaceRoots,
    *,
    systems: Iterable[str],
    name_col: str = "name_cleansed",
    run_date: str | None = None,
    wordfreq_top_n: int = 5000,
    render_plots: bool = True,
    progress: Callable[[str], None] | None = None,
    force: bool = False,
) -> TokenZipfPaths:
    """The trim-profile axis: holds the name-column tier fixed (default
    `name_cleansed`, the production tier) and sweeps tokenize-stage
    noise-word trim settings instead, to make visible how much of a
    system's wordfreq-divergence signal is attributable to noise-reduction
    choices versus structural (irreducible) vocabulary difference -- the
    methodological risk this sweep exists to expose: aggressive
    per-corpus trimming disproportionately removes ordinary high-DF words
    first, which can inflate apparent divergence as a side effect of the
    trim setting rather than a real property of the corpus.

    Scoped deliberately to whichever systems the caller passes (`ie` by
    default is the cheap pilot; this does not auto-discover every system the
    way `run_token_zipf_analysis` does, since the trim sweep multiplies run
    count by up to 5 settings per system).
    """
    effective_run_date = run_date or datetime.now(UTC).date().isoformat()
    paths = resolve_token_zipf_paths(
        roots, f"{effective_run_date}-trim-sweep", force=force
    )

    wordfreq_curve_cache: dict[str, pl.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for system in systems:
        files = discover_cleansed_files(roots, system)
        if not files or not _files_have_column(files, name_col):
            _emit_progress(
                progress, f"system={system} column={name_col} unavailable, skipping"
            )
            continue

        language = resolve_system_language(system)
        wordfreq_curve: pl.DataFrame | None = None
        if language:
            if language not in wordfreq_curve_cache:
                wordfreq_curve_cache[language] = build_wordfreq_reference_curve(
                    language, wordfreq_top_n
                )
            wordfreq_curve = wordfreq_curve_cache[language]

        for setting_name, noise_words in build_trim_settings(system, roots):
            corpus = tokenize_corpus(files, name_col, noise_words=noise_words)
            stats, total_docs = corpus.word_token_stats(), corpus.total_docs
            if stats.height == 0:
                _emit_progress(
                    progress,
                    f"system={system} trim={setting_name} no tokens, skipping",
                )
                continue

            stats = stats.with_columns(
                (pl.col("document_frequency") / pl.lit(max(total_docs, 1))).alias(
                    "document_frequency_pct"
                )
            )
            if language:
                stats = attach_language_frequency(stats, language=language)

            # The same pair the tier runner writes: the trim setting's effect
            # on the head is which words it removed, and the head-words
            # picture names them.
            plot_path = paths.viz_dir / f"{system}_{setting_name}_zipf.png"
            head_words_path = paths.viz_dir / f"{system}_{setting_name}_head_words.png"
            slopes = render_zipf_plot(
                stats,
                wordfreq_curve,
                plot_path,
                title=f"{system.upper()} trim={setting_name} ({name_col}) -- token rank-frequency",
                render=render_plots,
            )
            render_head_words_plot(
                stats,
                head_words_path,
                title=(
                    f"{system.upper()} trim={setting_name} ({name_col}) -- "
                    f"{HEAD_WORDS_TOP_N} most frequent words"
                ),
                render=render_plots,
            )

            total_types = stats.height
            type_level_unseen_pct = (
                float(stats.get_column("unseen_in_wordfreq").sum()) / total_types
                if language and total_types > 0
                else None
            )
            incidence = corpus.name_level_incidence(stats)

            summary_rows.append(
                {
                    "system": system,
                    "trim_setting": setting_name,
                    "noise_word_count": len(noise_words),
                    "total_docs": total_docs,
                    "unique_tokens": total_types,
                    "corpus_zipf_slope": slopes["corpus_slope"],
                    "reference_zipf_slope": slopes["reference_slope"],
                    "type_level_unseen_in_wordfreq_pct": type_level_unseen_pct,
                    "occurrence_weighted_unseen_in_wordfreq_pct": (
                        incidence["occurrence_weighted_unseen_pct"]
                        if language
                        else None
                    ),
                    "names_with_hapax_pct": incidence["names_with_hapax_pct"],
                    "names_with_unseen_in_wordfreq_pct": (
                        incidence["names_with_unseen_in_wordfreq_pct"]
                        if language
                        else None
                    ),
                    "plot_path": to_portable_path_str(
                        plot_path, project_root=roots.checkout
                    ),
                    "plot_relative_link": _relative_viz_link(plot_path),
                    "head_words_plot_path": to_portable_path_str(
                        head_words_path, project_root=roots.checkout
                    ),
                    "head_words_plot_relative_link": _relative_viz_link(
                        head_words_path
                    ),
                }
            )
            _emit_progress(
                progress,
                f"system={system} trim={setting_name} noise_words={len(noise_words):,} "
                f"types={total_types:,} slope={slopes['corpus_slope']:.2f}",
            )

    summary_df = pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()
    summary_df.write_parquet(paths.summary_path)
    _write_trim_sweep_report(
        paths.report_path, summary_rows, run_date=effective_run_date
    )
    return paths


def _relative_viz_link(image_path: Path) -> str:
    """Markdown-relative link from a report in `reports/` to an image in the
    sibling `viz/` directory `resolve_token_zipf_paths` always creates --
    both are fixed children of the same run roots, so this is just
    `../viz/<filename>`, not a general-purpose path-relativization helper.
    """
    return f"../viz/{image_path.name}"


def _markdown_image_block(title: str, *relative_links: str) -> list[str]:
    """One heading, then every picture given under it.

    A system and tier's curve and its head-words picture share a heading:
    they are two views of one distribution, and the second's title already
    says what it shows.
    """
    lines = [f"### {title}", ""]
    for relative_link in relative_links:
        lines += [f"![{title}]({relative_link})", ""]
    return lines


def _write_trim_sweep_report(
    report_path: Path, summary_rows: list[dict[str, object]], *, run_date: str
) -> None:
    lines = [
        "# Token Zipf Trim-Profile Sweep",
        "",
        f"- Run date: {run_date}",
        "- Name-column tier held fixed (see each row's system); tokenize-stage",
        "  noise-word trim setting swept instead: `none`, `packaged_default`",
        "  (company_cleanse's static profile), `per_corpus_<profile>` (this",
        "  system's own generated noise_words.json, strict/balanced/aggressive).",
        "- Watch for divergence numbers *rising* with more aggressive trim --",
        "  that's the known bias where trimming removes ordinary high-DF words",
        "  first, inflating apparent wordfreq-divergence as a trim artifact",
        "  rather than a real corpus property.",
        "",
        "| system | trim setting | noise words | names | unique tokens | corpus slope | type OOV% | occurrence-weighted OOV% | names w/ hapax% | names w/ OOV% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['trim_setting']} | {row['noise_word_count']:,} | "
            f"{row['total_docs']:,} | {row['unique_tokens']:,} | "
            f"{_fmt_slope(row['corpus_zipf_slope'])} | "
            f"{_fmt_pct(row['type_level_unseen_in_wordfreq_pct'])} | "
            f"{_fmt_pct(row['occurrence_weighted_unseen_in_wordfreq_pct'])} | "
            f"{_fmt_pct(row['names_with_hapax_pct'])} | "
            f"{_fmt_pct(row['names_with_unseen_in_wordfreq_pct'])} |"
        )

    plotted_rows = [row for row in summary_rows if row.get("plot_relative_link")]
    if plotted_rows:
        lines += ["", "## Rank-Frequency Plots", ""]
        for row in plotted_rows:
            lines += _markdown_image_block(
                f"{row['system']} -- trim={row['trim_setting']}",
                str(row["plot_relative_link"]),
                str(row["head_words_plot_relative_link"]),
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_pct(value: object) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "n/a"


def _fmt_slope(value: object) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "n/a"


def _write_report(
    report_path: Path,
    summary_rows: list[dict[str, object]],
    *,
    run_date: str,
    overview_images: list[tuple[str, str]] | None = None,
) -> None:
    """Write the markdown summary, embedding its own `viz/` PNGs by
    reference (per-system-tier rank-frequency and head-words plots, plus --
    when given -- the cross-system overview grid and OOV/hapax bar charts)
    instead of leaving them as "drop in the directory" artefacts nobody
    links to.
    """
    lines = [
        "# Token Zipf / Divergence Summary",
        "",
        f"- Run date: {run_date}",
        "- Name tiers: raw (`name`), basic (`name_cleansed_basic`), cleansed (`name_cleansed`).",
        "- `raw` carries a known confound: spaced single-character acronym runs",
        '  (e.g. "I B M") score as ordinary high-frequency English at the',
        "  per-fragment level, which can make raw look *less* divergent from",
        "  general language than it structurally is. See",
        "  docs/findings/token-zipf.md.",
        "",
        "| system | tier | names | unique tokens | corpus slope | reference slope | type OOV% | occurrence-weighted OOV% | names w/ hapax% | names w/ OOV% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['system']} | {row['tier']} | {row['total_docs']:,} | "
            f"{row['unique_tokens']:,} | {_fmt_slope(row['corpus_zipf_slope'])} | "
            f"{_fmt_slope(row['reference_zipf_slope'])} | "
            f"{_fmt_pct(row['type_level_unseen_in_wordfreq_pct'])} | "
            f"{_fmt_pct(row['occurrence_weighted_unseen_in_wordfreq_pct'])} | "
            f"{_fmt_pct(row['names_with_hapax_pct'])} | "
            f"{_fmt_pct(row['names_with_unseen_in_wordfreq_pct'])} |"
        )

    plotted_rows = [row for row in summary_rows if row.get("plot_relative_link")]
    if plotted_rows:
        lines += ["", "## Rank-Frequency Plots", ""]
        for row in plotted_rows:
            lines += _markdown_image_block(
                f"{row['system']} -- {row['tier']} tier",
                str(row["plot_relative_link"]),
                str(row["head_words_plot_relative_link"]),
            )

    if overview_images:
        lines += ["", "## Cross-System Overview", ""]
        for title, relative_link in overview_images:
            lines += _markdown_image_block(title, relative_link)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
