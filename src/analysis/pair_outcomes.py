"""Which truth pairs each blocking run found, read across a pairing's runs.

A strategy comparison reports each run's recall on a population; it says how
many pairs a run found, not which. `blocking.comparison` writes the two tables
that do: `pair_outcomes`, one row per truth pair per run with the run's
verdict, and `pair_outcome_runs`, one row per run with the settings it was
made under. This module reads those two frames and nothing else, so it
imports nothing from `blocking`; a caller finds the files through
`blocking.run_layout.resolve_comparison_location`.

The reading has three steps, one function each. `select_runs` and
`latest_runs` pick the runs to compare from the runs table.
`pair_found_matrix` turns their outcomes into one row per pair with a
found-or-missed column per run, filtered to a name-equality level and a
country. `drop_unanimous_pairs` leaves the remainder the runs disagree on.
`count_found_by` and `pairs_found_by` then read that matrix: how many pairs
each combination of runs found, and the named pairs behind one combination.

Runs made at one threshold do not spend the same comparisons, since cosine is
on a different scale per representation, so a run that keeps more candidates
finds more pairs for that reason alone. `pair_found_matrix` therefore takes a
cutoff per run, read against the similarity each pair was found at, and
`cutoffs_for_candidates` turns one budget of candidates into each run's
cutoff from the third table, `pair_outcome_candidates`. `candidates_at` says
what each run keeps at a cutoff. A cap on candidates per source is the other
way to hold cost, and the one that means the same on every corpus, so the
matrix and both of those take a `max_rank` too. A run is only ever read more
tightly than it was made: the pairs below its threshold or beyond its `top_k`
were never kept.

Two more readings say how far a difference between runs can be trusted.
`random_pair_level` scores the pairs' names against each other's at random, so
`with_name_similarity` can mark the pairs whose own names are no more alike
than two unrelated names are: those are out of any name-based method's reach,
and the rest is the population a method can be held to. `count_found_by_k` and
`run_disagreement` read runs of one method against each other, where every
disagreement is the backend's and none the representation's: a difference
between two methods smaller than that is not a finding.

`draw_found_by_upset` draws the matrix as an UpSet plot: a bar per
combination of runs, counting the pairs found by exactly those runs, by at
least those runs, or both, the matrix of dots under it saying which runs the
combination is, and each run's own total beside its row. `run_overlap` and
`draw_run_overlap` read the same matrix a couple of runs at a time, as Jaccard
and as containment, and `count_by_reach` and `draw_name_similarity` split its
pairs by the random-pair level.

`level_metrics` gives each run's precision, recall, reduction ratio and F2 on
one level's own sources, from a fourth table, `pair_outcome_source_candidates`,
which keeps each run's candidates per truth source. Read on `never`, those are
the figures a method is judged by: pairs equal by name at any level are found
by an exact join before any method runs, and counting them lifts every
method alike. Given the pairing's `pair_outcome_recall_curves` and
`pair_outcome_audits`, it also gives each run's own recall area over the
comparisons it spent and what its backend's routing lost against the exact
scan.
`cluster_sizes` and `draw_cluster_sizes` count the pairs by how
many candidates each run gave their source, a pair given none being a miss
that cost nothing.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import matplotlib
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from upsetplot import UpSet, from_indicators

from validation.contracts import POPULATION_UNIVERSE
from validation.recall_curve import recall_curve_area
from validation.runner import NAME_EQUALITY_NEVER

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# What identifies one truth pair inside a pairing's outcomes, and the columns
# a pair's row carries beside its verdicts so it reads without a rejoin.
PAIR_KEY: tuple[str, ...] = ("country", "source_id", "target_id")
PAIR_NAMES: tuple[str, str] = ("source_name", "target_name")
CLEANSED_NAMES: tuple[str, str] = ("source_name_cleansed", "target_name_cleansed")
NAME_SIMILARITY = "name_similarity"
ABOVE_RANDOM = "above_random"

# `pair_audit.verdict` for a pair the run missed and the exact scan at the
# run's own settings would have kept, `blocking.audit.VERDICT_LOST_TO_BACKEND`.
AUDIT_LOST_TO_BACKEND = "lost_to_backend"

FOUND_BY = "found_by"
FOUND_BY_SEPARATOR = " + "
FOUND_BY_NONE = "(none)"

# The plots' colours, the light tokens of docs/diagrams/house-style.css (named
# in each comment) and the style guide's C4 component blue, kept to the lighter
# tones: a bar counting the pairs found by at least its runs, one counting
# those found by exactly its runs, drawn downward when both are shown, the dot
# matrix and its run totals, and label text.
_AT_LEAST_COLOUR = "#7FA3CC"  # C4 component blue, docs/diagrams/style-guide.md
_EXACT_COLOUR = "#DCE8F7"  # --accent-soft
_DOTS = "#5A6B85"  # --muted
_INK = "#13213A"  # --ink
_LINE = "#D3DCE8"  # --line
_SHADING = "#F3F6FB"  # --surface-2
# The overlap heatmap's scale, light throughout so every label reads in ink:
# --surface-2, --accent-soft, then the C4 component blue from the style guide.
_OVERLAP_SCALE = LinearSegmentedColormap.from_list(
    "overlap", ["#F3F6FB", "#DCE8F7", "#7FA3CC"]
)


def select_runs(
    runs: pl.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    representation: Sequence[str] | None = None,
    tokenizer: Sequence[str] | None = None,
    similarity_backend: Sequence[str] | None = None,
) -> pl.DataFrame:
    """The rows of a `pair_outcome_runs` frame matching every filter given;
    a filter left `None` admits every run.

    Raises `ValueError` naming any label asked for that the frame lacks, since
    a mistyped run key would otherwise read as a run that found nothing.
    """
    if labels is not None:
        unknown = sorted(set(labels) - set(runs.get_column("label").to_list()))
        if unknown:
            raise ValueError(f"no run is labelled {unknown!r}")
    selected = runs
    for column, values in (
        ("label", labels),
        ("representation", representation),
        ("tokenizer", tokenizer),
        ("similarity_backend", similarity_backend),
    ):
        if values is not None:
            selected = selected.filter(pl.col(column).is_in(list(values)))
    return selected


def latest_runs(runs: pl.DataFrame, *, by: str = "representation") -> pl.DataFrame:
    """The most recently finished run of each `by` value, by the finish time
    its production record carries. A run with no finish time, one made in
    process rather than read from disk, is never the latest while another of
    its `by` value has one.
    """
    return (
        runs.sort("finished_at", nulls_last=False)
        .group_by(by, maintain_order=True)
        .last()
        .select(runs.columns)
        .sort(by)
    )


def pair_found_matrix(
    outcomes: pl.DataFrame,
    runs: pl.DataFrame,
    *,
    name_equality: str | None = NAME_EQUALITY_NEVER,
    country: str | None = None,
    run_names: Mapping[str, str] | None = None,
    min_similarity: Mapping[str, float] | None = None,
    max_rank: int | None = None,
) -> pl.DataFrame:
    """One row per truth pair, with a found-or-missed column per run in `runs`.

    `name_equality` keeps the pairs at one level, `never` by default, and
    `None` keeps every level. A pair is kept when any selected run gave it
    that level, and then carries every selected run's verdict: a run under
    another cleanse profile may place the pair elsewhere, and its verdict on
    the pair counts all the same. `country` keeps one country's pairs.

    A run's column is named by `run_names[label]` where given, else its label.
    A run with no outcome for a pair leaves a null there.

    `min_similarity` cuts runs after the fact, by label: a pair a run found
    counts as found only where the similarity it was found at reaches that
    run's cutoff, and a run not named is read as it was made. A found pair
    carrying no similarity has nothing to be cut by and stays found.
    `max_rank` caps every run at that many candidates per source: a found pair
    counts only where its rank among its source's candidates is within it.

    Where the outcomes carry them, each pair also takes the two cleansed names
    of the first selected run, by label, that scored it.

    Raises `ValueError` if `runs` is empty, two runs share a column name,
    `min_similarity` names a run not among `runs`, or `max_rank` is below one.
    """
    labels = runs.get_column("label").to_list()
    if not labels:
        raise ValueError("no run selected")
    names = {label: (run_names or {}).get(label, label) for label in labels}
    if len(set(names.values())) != len(names):
        raise ValueError(f"two runs share a column name: {sorted(names.values())!r}")

    unknown = sorted(set(min_similarity or {}) - set(labels))
    if unknown:
        raise ValueError(f"a cutoff names runs not selected: {unknown!r}")
    if max_rank is not None and max_rank < 1:
        raise ValueError(f"a cap per source is at least one, got {max_rank}")

    selected = outcomes.filter(pl.col("label").is_in(labels))
    if max_rank is not None:
        selected = selected.with_columns(
            (
                pl.col("found")
                & (pl.col("rank").is_null() | (pl.col("rank") <= max_rank))
            ).alias("found")
        )
    if min_similarity:
        cutoff = pl.col("label").replace_strict(
            dict(min_similarity), default=None, return_dtype=pl.Float64
        )
        selected = selected.with_columns(
            (
                pl.col("found")
                & (
                    cutoff.is_null()
                    | pl.col("similarity").is_null()
                    | (pl.col("similarity") >= cutoff)
                )
            ).alias("found")
        )
    if country is not None:
        selected = selected.filter(pl.col("country") == country)
    if name_equality is not None:
        at_level = (
            selected.filter(pl.col("name_equality") == name_equality)
            .select(PAIR_KEY)
            .unique()
        )
        selected = selected.join(at_level, on=list(PAIR_KEY), how="semi")

    index = [*PAIR_KEY, *PAIR_NAMES]
    if selected.height == 0:
        return pl.DataFrame(
            schema={
                **{column: pl.Utf8 for column in index},
                **{name: pl.Boolean for name in names.values()},
            }
        )
    matrix = selected.pivot(on="label", index=index, values="found")
    # A selected run with no outcome among the kept pairs has no pivot column.
    matrix = matrix.with_columns(
        pl.lit(None, dtype=pl.Boolean).alias(label)
        for label in labels
        if label not in matrix.columns
    )
    matrix = matrix.select(*index, *labels).rename(names)
    if set(CLEANSED_NAMES) <= set(selected.columns):
        cleansed = (
            selected.sort("label")
            .group_by(PAIR_KEY, maintain_order=True)
            .agg(pl.col(column).first() for column in CLEANSED_NAMES)
        )
        matrix = matrix.join(cleansed, on=list(PAIR_KEY), how="left")
    return matrix.sort(list(PAIR_KEY))


def _kept_by_cutoff(
    candidates: pl.DataFrame, label: str, max_rank: int | None
) -> pl.DataFrame:
    """One run's candidate pairs at or above each cutoff it holds a candidate
    at, within `max_rank` per source where given, lowest cutoff first."""
    cells = candidates.filter(pl.col("label") == label)
    if max_rank is not None:
        cells = cells.filter(pl.col("rank") <= max_rank)
    return (
        cells.group_by("min_similarity")
        .agg(pl.col("candidates").sum())
        .sort("min_similarity", descending=True)
        .with_columns(pl.col("candidates").cum_sum())
        .sort("min_similarity")
    )


def cutoffs_for_candidates(
    candidates: pl.DataFrame,
    runs: pl.DataFrame,
    *,
    budget: int,
    max_rank: int | None = None,
) -> dict[str, float]:
    """The cutoff that holds each run in `runs` to at most `budget` candidate
    pairs, from a `pair_outcome_candidates` frame: for each run, the lowest
    `min_similarity` at which the pairs it keeps, within `max_rank` per source
    where given, fit.

    A run already within the budget keeps its lowest cutoff, the threshold it
    was made at. A run that exceeds the budget even at its highest cutoff,
    which a run holding many pairs at a similarity of one does, takes that
    highest cutoff and stays over; `candidates_at` shows what each run really
    keeps. A run with no rows in `candidates` is left out.

    Raises `ValueError` for a budget below one.
    """
    if budget < 1:
        raise ValueError(f"a candidate budget is at least one, got {budget}")
    cutoffs: dict[str, float] = {}
    for label in runs.get_column("label").to_list():
        curve = _kept_by_cutoff(candidates, label, max_rank)
        if curve.height == 0:
            continue
        fitting = curve.filter(pl.col("candidates") <= budget)
        chosen = fitting if fitting.height else curve.tail(1)
        cutoffs[label] = float(chosen.get_column("min_similarity")[0])
    return cutoffs


def candidates_at(
    candidates: pl.DataFrame,
    min_similarity: Mapping[str, float],
    *,
    max_rank: int | None = None,
) -> dict[str, int]:
    """How many candidate pairs each run keeps at its cutoff in
    `min_similarity`, within `max_rank` per source where given, from a
    `pair_outcome_candidates` frame: the count at the lowest recorded cutoff
    at or above the one asked for, and none above a run's highest."""
    kept: dict[str, int] = {}
    for label, cutoff in min_similarity.items():
        at_or_above = _kept_by_cutoff(candidates, label, max_rank).filter(
            pl.col("min_similarity") >= cutoff - 1e-9
        )
        kept[label] = (
            int(at_or_above.get_column("candidates")[0]) if at_or_above.height else 0
        )
    return kept


def char_ngram_jaccard(left: str, right: str, *, n: int = 3) -> float:
    """The Jaccard similarity of two names' character n-grams, each name padded
    with a space either side so a word's first and last letters count: the
    share of all the n-grams either name has that both have. Two empty names
    share nothing."""

    def grams(name: str) -> set[str]:
        padded = f" {name} "
        return {padded[i : i + n] for i in range(len(padded) - n + 1)}

    left_grams, right_grams = grams(left), grams(right)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _name_columns(matrix: pl.DataFrame) -> tuple[str, str]:
    """The cleansed names where `matrix` carries them, else the names as given."""
    return CLEANSED_NAMES if set(CLEANSED_NAMES) <= set(matrix.columns) else PAIR_NAMES


def random_pair_level(
    matrix: pl.DataFrame, *, quantile: float = 0.99, draws: int = 20, seed: int = 0
) -> float:
    """The name similarity two unrelated names reach, from `matrix`'s own
    names: each source name scored against another pair's target name, over
    `draws` seeded shuffles, and the `quantile` of those scores.

    A truth pair whose own names score no higher is, by name, no different
    from a non-match, so no name-based method can be expected to find it.
    Several shuffles are drawn because a high quantile of one is set by a
    handful of scores and moves with the seed. The level is a property of
    these names, not a constant.

    Raises `ValueError` for fewer than two pairs, which have no other pair to
    be scored against, for fewer than one draw, or when every draw left each
    name paired with its own, so nothing was scored.
    """
    if draws < 1:
        raise ValueError(f"a random-pair level needs at least one draw, got {draws}")
    source_column, target_column = _name_columns(matrix)
    sources = matrix.get_column(source_column).fill_null("").to_list()
    targets = matrix.get_column(target_column).fill_null("").to_list()
    if len(sources) < 2:
        raise ValueError("a random-pair level needs at least two pairs")
    shuffler = random.Random(seed)  # nosec B311 - seeded for reproducibility, not security
    order = list(range(len(targets)))
    scores: list[float] = []
    for _ in range(draws):
        shuffler.shuffle(order)
        scores.extend(
            char_ngram_jaccard(sources[i], targets[j])
            for i, j in enumerate(order)
            if i != j
        )
    level = pl.Series(scores, dtype=pl.Float64).quantile(
        quantile, interpolation="linear"
    )
    if level is None:
        raise ValueError("no draw scored a source name against another pair's target")
    return float(level)


def with_name_similarity(matrix: pl.DataFrame, *, random_level: float) -> pl.DataFrame:
    """`matrix` with each pair's `name_similarity`, the character 3-gram
    Jaccard of its two names, and `above_random`, whether that exceeds
    `random_level`."""
    source_column, target_column = _name_columns(matrix)
    scores = [
        char_ngram_jaccard(source or "", target or "")
        for source, target in matrix.select(source_column, target_column).iter_rows()
    ]
    return matrix.with_columns(
        pl.Series(NAME_SIMILARITY, scores, dtype=pl.Float64)
    ).with_columns((pl.col(NAME_SIMILARITY) > random_level).alias(ABOVE_RANDOM))


def count_found_by_k(matrix: pl.DataFrame) -> pl.DataFrame:
    """How many pairs were found by exactly k of the matrix's runs, one row
    per k from none to all of them. Read over runs of one method, the pairs
    between the two ends are the ones the backend finds on one setting or
    seed and loses on another."""
    columns = run_columns(matrix)
    found = pl.sum_horizontal(
        pl.col(column).fill_null(False).cast(pl.Int64) for column in columns
    )
    counted = (
        matrix.select(found.alias("found_by_runs")).group_by("found_by_runs").len()
    )
    return (
        pl.DataFrame({"found_by_runs": list(range(len(columns) + 1))})
        .join(counted, on="found_by_runs", how="left")
        .select(
            "found_by_runs", pl.col("len").fill_null(0).cast(pl.Int64).alias("pairs")
        )
    )


def run_disagreement(matrix: pl.DataFrame) -> pl.DataFrame:
    """Every two runs of the matrix against each other: the pairs the first
    found and the second missed, the reverse, and their sum, one row per
    unordered couple of runs."""
    columns = run_columns(matrix)
    rows = []
    for index, first in enumerate(columns):
        for second in columns[index + 1 :]:
            a = pl.col(first).fill_null(False)
            b = pl.col(second).fill_null(False)
            only_first = matrix.filter(a & ~b).height
            only_second = matrix.filter(~a & b).height
            rows.append(
                {
                    "run": first,
                    "other_run": second,
                    "only_run": only_first,
                    "only_other_run": only_second,
                    "disagree": only_first + only_second,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "run": pl.Utf8,
            "other_run": pl.Utf8,
            "only_run": pl.Int64,
            "only_other_run": pl.Int64,
            "disagree": pl.Int64,
        },
    )


def run_overlap(matrix: pl.DataFrame) -> pl.DataFrame:
    """Every run of the matrix against every run, itself included, one row
    per ordered couple: the pairs both found, each one's own total, their
    `jaccard`, both over either, and `containment`, both over the first run's
    total, the share of what the first run found that the other found too.

    Jaccard falls when two runs find different numbers of pairs even where one
    finds a subset of the other; containment near one says the first run adds
    nothing the other lacks. A run that found nothing has null for both."""
    columns = run_columns(matrix)
    found = {column: matrix.get_column(column).fill_null(False) for column in columns}
    rows = []
    for first in columns:
        for second in columns:
            both = int((found[first] & found[second]).sum())
            either = int((found[first] | found[second]).sum())
            first_total = int(found[first].sum())
            rows.append(
                {
                    "run": first,
                    "other_run": second,
                    "both": both,
                    "run_found": first_total,
                    "other_run_found": int(found[second].sum()),
                    "jaccard": both / either if either else None,
                    "containment": both / first_total if first_total else None,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "run": pl.Utf8,
            "other_run": pl.Utf8,
            "both": pl.Int64,
            "run_found": pl.Int64,
            "other_run_found": pl.Int64,
            "jaccard": pl.Float64,
            "containment": pl.Float64,
        },
    )


def count_by_reach(matrix: pl.DataFrame) -> pl.DataFrame:
    """A `with_name_similarity` matrix's pairs counted by whether any run
    found them and whether their names are above the random-pair level: the
    pairs no run found split into those a name-based method could have found
    and those it could not."""
    found_by_any = pl.any_horizontal(
        pl.col(column).fill_null(False) for column in run_columns(matrix)
    )
    return (
        matrix.select(found_by_any.alias("found_by_any_run"), pl.col(ABOVE_RANDOM))
        .group_by("found_by_any_run", ABOVE_RANDOM)
        .len("pairs")
        .sort("found_by_any_run", ABOVE_RANDOM, descending=True)
    )


# The cluster sizes `cluster_sizes` counts sources in, as inclusive bounds and
# a label; the last is open above.
CLUSTER_SIZE_BUCKETS: tuple[tuple[int, int | None, str], ...] = (
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 5, "4-5"),
    (6, 10, "6-10"),
    (11, 19, "11-19"),
    (20, None, "20+"),
)

# One colour per run in a drawing, the cool ramp of
# `docs/diagrams/style-guide.md`, repeated past its fourth run.
RUN_COLOURS: tuple[str, ...] = ("#0b5fa5", "#1f7a8c", "#2a9d8f", "#84a59d")

# What became of a pair in `cluster_sizes`, and the shade each is drawn at.
OUTCOME_FOUND = "found"
OUTCOME_FINDABLE = "findable"
OUTCOME_RANDOM = "random"
OUTCOME_SHADES: tuple[tuple[str, float], ...] = (
    (OUTCOME_FOUND, 1.0),
    (OUTCOME_FINDABLE, 0.45),
    (OUTCOME_RANDOM, 0.15),
)


def cluster_sizes(
    matrix: pl.DataFrame,
    source_candidates: pl.DataFrame,
    runs: pl.DataFrame,
    *,
    run_names: Mapping[str, str] | None = None,
    min_similarity: Mapping[str, float] | None = None,
    max_rank: int | None = None,
    random_level: float | None = None,
) -> pl.DataFrame:
    """How many of the matrix's pairs sit in each size of cluster, per run in
    `runs`: a pair takes the number of candidates the run gave its source, by
    `CLUSTER_SIZE_BUCKETS`, from a `pair_outcome_source_candidates` frame read
    at the same cutoffs and cap as `pair_found_matrix`. One row per run per
    bucket and per `outcome`, in bucket order: `found`; `findable`, missed by
    the run; and, with `random_level`, `random`, missed and no more alike by
    name than two unrelated names, which no name-based method could be held
    to. Without `random_level` every missed pair is `findable`.

    A pair whose source the run gave no candidates is in the `0` bucket: a
    miss for that run, and one that cost it nothing."""
    pairs = matrix.select("country", "source_id")
    below_random = (
        with_name_similarity(matrix, random_level=random_level)
        .get_column(ABOVE_RANDOM)
        .not_()
        if random_level is not None
        else pl.Series([False] * matrix.height)
    )
    sources = pairs.unique()
    kept = source_candidates.join(sources, on=["country", "source_id"], how="semi")
    if max_rank is not None:
        kept = kept.filter(pl.col("rank") <= max_rank)
    rows = []
    for label in runs.get_column("label").to_list():
        cells = kept.filter(pl.col("label") == label)
        cutoff = (min_similarity or {}).get(label)
        if cutoff is not None:
            cells = cells.filter(pl.col("min_similarity") >= cutoff - 1e-9)
        per_source = cells.group_by("country", "source_id").agg(
            pl.col("candidates").sum()
        )
        column = (run_names or {}).get(label, label)
        sizes = (
            pairs.join(per_source, on=["country", "source_id"], how="left")
            .get_column("candidates")
            .fill_null(0)
        )
        found = matrix.get_column(column).fill_null(False)
        outcomes = [(OUTCOME_FOUND, found), (OUTCOME_FINDABLE, ~found & ~below_random)]
        if random_level is not None:
            outcomes.append((OUTCOME_RANDOM, ~found & below_random))
        for low, high, bucket in CLUSTER_SIZE_BUCKETS:
            inside = sizes >= low if high is None else (sizes >= low) & (sizes <= high)
            for outcome, held in outcomes:
                rows.append(
                    {
                        "run": column,
                        "cluster_size": bucket,
                        "outcome": outcome,
                        "pairs": int((inside & held).sum()),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "run": pl.Utf8,
            "cluster_size": pl.Utf8,
            "outcome": pl.Utf8,
            "pairs": pl.Int64,
        },
    )


def level_metrics(
    outcomes: pl.DataFrame,
    runs: pl.DataFrame,
    source_candidates: pl.DataFrame,
    *,
    name_equality: str | None = NAME_EQUALITY_NEVER,
    country: str | None = None,
    run_names: Mapping[str, str] | None = None,
    min_similarity: Mapping[str, float] | None = None,
    max_rank: int | None = None,
    random_level: float | None = None,
    above_random_only: bool = False,
    recall_curves: pl.DataFrame | None = None,
    audits: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Precision, recall, reduction ratio and F2 of each run in `runs` on one
    name-equality level's own sources, read at the same cutoffs and cap as
    `pair_found_matrix`.

    A level is a set of truth pairs and so of sources. Recall is the level's
    pairs found over its pairs. The candidates are what each run proposed for
    those sources, from a `pair_outcome_source_candidates` frame, and
    precision is the truth pairs found for those sources, at any level, over
    those candidates: a source's candidates serve every truth pair it holds,
    so each is counted once. The reduction ratio is one less those candidates
    over the sources times the run's `target_rows`, and F2 weights recall four
    times precision. Precision is a floor, since a candidate outside the truth
    set may be an unlabelled match.

    With `random_level`, `recall_above_random` is the recall over the pairs
    whose names are above it, the ones a name-based method can be held to,
    and each run's misses are split as `cluster_sizes` splits them:
    `missed_findable`, above that level, and `missed_random`, no more alike
    by name than two unrelated names. `above_random_only` keeps only the
    pairs above it, and so only their sources, for every figure: what a
    name-based method is measured against once the pairs no such method
    could find are set aside.

    With `recall_curves`, a `pair_outcome_recall_curves` frame,
    `recall_area` is each run's mean recall over the comparisons it spent on
    the level (the universe when `name_equality` is `None`), over its own
    budget. It describes one run and does not rank runs: read to the budget
    every run reached, a run whose first rank already spends more than that
    scores nothing, as `sbert` did on `gleif -> ie`, so runs are compared at
    equal cost through the cutoffs and cap above. The curves pool every
    country and ignore the cutoffs, cap and `above_random_only`, being each
    run as it was made.

    With `audits`, a `pair_audit` frame, `lost_to_backend` counts the level's
    pairs the run missed that the exact scan at the run's own settings would
    have kept, and `exact_scan_recall` is that scan's recall over the pairs
    audited; both are null for a run with no audit. An audit reads the run as
    it was made, not at the cutoffs or cap.

    Raises `ValueError` for `above_random_only` without a `random_level`, and
    what `pair_found_matrix` raises.
    """
    if above_random_only and random_level is None:
        raise ValueError("keeping the pairs above random needs a random_level")
    level = pair_found_matrix(
        outcomes,
        runs,
        name_equality=name_equality,
        country=country,
        run_names=run_names,
        min_similarity=min_similarity,
        max_rank=max_rank,
    )
    if above_random_only:
        level = (
            with_name_similarity(level, random_level=random_level)  # type: ignore[arg-type]
            .filter(ABOVE_RANDOM)
            .drop(NAME_SIMILARITY, ABOVE_RANDOM)
        )
    sources = level.select("country", "source_id").unique()
    every_level = pair_found_matrix(
        outcomes,
        runs,
        name_equality=None,
        country=country,
        run_names=run_names,
        min_similarity=min_similarity,
        max_rank=max_rank,
    ).join(sources, on=["country", "source_id"], how="semi")
    above = (
        with_name_similarity(level, random_level=random_level).filter(ABOVE_RANDOM)
        if random_level is not None
        else None
    )
    kept = source_candidates.join(sources, on=["country", "source_id"], how="semi")
    if max_rank is not None:
        kept = kept.filter(pl.col("rank") <= max_rank)
    areas = (
        _recall_areas(
            recall_curves,
            runs,
            population=name_equality or POPULATION_UNIVERSE,
        )
        if recall_curves is not None
        else None
    )
    level_audits = (
        audits.join(level.select(PAIR_KEY), on=list(PAIR_KEY), how="semi")
        if audits is not None
        else None
    )

    rows = []
    for label, target_rows in runs.select("label", "target_rows").iter_rows():
        column = (run_names or {}).get(label, label)
        cells = kept.filter(pl.col("label") == label)
        cutoff = (min_similarity or {}).get(label)
        if cutoff is not None:
            cells = cells.filter(pl.col("min_similarity") >= cutoff - 1e-9)
        candidates = int(cells.get_column("candidates").sum())
        found = int(level.get_column(column).fill_null(False).sum())
        true_found = int(every_level.get_column(column).fill_null(False).sum())
        recall = found / level.height if level.height else None
        precision = true_found / candidates if candidates else None
        space = sources.height * target_rows if target_rows else 0
        row = {
            "run": column,
            "sources": sources.height,
            "pairs": level.height,
            "found": found,
            "recall": recall,
            "candidates": candidates,
            "truth_found": true_found,
            "precision": precision,
            "reduction_ratio": 1 - candidates / space if space else None,
            "f2": (
                5 * precision * recall / (4 * precision + recall)
                if precision and recall
                else None
            ),
        }
        if above is not None:
            above_found = int(above.get_column(column).fill_null(False).sum())
            row["recall_above_random"] = (
                above_found / above.height if above.height else None
            )
            row["missed_findable"] = above.height - above_found
            row["missed_random"] = level.height - above.height - (found - above_found)
        if areas is not None:
            row["recall_area"] = areas.get(label)
        if level_audits is not None:
            audited = level_audits.filter(pl.col("label") == label)
            row["lost_to_backend"] = (
                audited.filter(pl.col("verdict") == AUDIT_LOST_TO_BACKEND).height
                if audited.height
                else None
            )
            row["exact_scan_recall"] = (
                int(audited.get_column("exact_kept").sum()) / audited.height
                if audited.height
                else None
            )
        rows.append(row)
    schema = {
        "run": pl.Utf8,
        "sources": pl.Int64,
        "pairs": pl.Int64,
        "found": pl.Int64,
        "recall": pl.Float64,
        "candidates": pl.Int64,
        "truth_found": pl.Int64,
        "precision": pl.Float64,
        "reduction_ratio": pl.Float64,
        "f2": pl.Float64,
    }
    if above is not None:
        schema["recall_above_random"] = pl.Float64
        schema["missed_findable"] = pl.Int64
        schema["missed_random"] = pl.Int64
    if areas is not None:
        schema["recall_area"] = pl.Float64
    if level_audits is not None:
        schema["lost_to_backend"] = pl.Int64
        schema["exact_scan_recall"] = pl.Float64
    return pl.DataFrame(rows, schema=schema)


def _recall_areas(
    recall_curves: pl.DataFrame, runs: pl.DataFrame, *, population: str
) -> dict[str, float | None]:
    """Each of `runs`' own `recall_area` on `population`, by label
    (`validation.recall_curve.recall_curve_area`); a run with no curve is
    left out."""
    labels = runs.get_column("label").to_list()
    return {
        label: recall_curve_area(curve, population=population)
        for (label,), curve in recall_curves.filter(
            pl.col("label").is_in(labels)
        ).group_by("label")
    }


def run_columns(matrix: pl.DataFrame) -> list[str]:
    """The run columns of a `pair_found_matrix` frame, in its own order."""
    fixed = {
        *PAIR_KEY,
        *PAIR_NAMES,
        *CLEANSED_NAMES,
        FOUND_BY,
        NAME_SIMILARITY,
        ABOVE_RANDOM,
    }
    return [column for column in matrix.columns if column not in fixed]


def drop_unanimous_pairs(
    matrix: pl.DataFrame, *, found_by_none: bool = True, found_by_all: bool = False
) -> pl.DataFrame:
    """`matrix` without the pairs no run found, the ones every run found, or
    both. The pairs no run found say nothing about how the runs differ and are
    usually most of a `never` population, so they go by default.
    """
    columns = run_columns(matrix)
    found = [pl.col(column).fill_null(False) for column in columns]
    keep = pl.lit(True)
    if found_by_none:
        keep = keep & pl.any_horizontal(found)
    if found_by_all:
        keep = keep & ~pl.all_horizontal(found)
    return matrix.filter(keep)


def with_found_by(matrix: pl.DataFrame) -> pl.DataFrame:
    """`matrix` with a `found_by` column naming the runs that found each pair,
    joined in column order, or `(none)`."""
    columns = run_columns(matrix)
    found_by = pl.concat_str(
        [
            pl.when(pl.col(column).fill_null(False))
            .then(pl.lit(column + FOUND_BY_SEPARATOR))
            .otherwise(pl.lit(""))
            for column in columns
        ]
    ).str.strip_suffix(FOUND_BY_SEPARATOR)
    return matrix.with_columns(
        pl.when(found_by == "")
        .then(pl.lit(FOUND_BY_NONE))
        .otherwise(found_by)
        .alias(FOUND_BY)
    )


def count_found_by(matrix: pl.DataFrame) -> pl.DataFrame:
    """How many pairs each combination of runs found: one row per `found_by`
    value with its `pairs` and its share of the matrix, largest first."""
    total = matrix.height
    return (
        with_found_by(matrix)
        .group_by(FOUND_BY)
        .agg(pl.len().alias("pairs"))
        .with_columns((pl.col("pairs") / total).alias("share"))
        .sort(["pairs", FOUND_BY], descending=[True, False])
    )


def pairs_found_by(matrix: pl.DataFrame, found_by: Sequence[str]) -> pl.DataFrame:
    """The pairs exactly the runs in `found_by` found and no other run did;
    an empty `found_by` gives the pairs no run found.

    Raises `ValueError` naming any run asked for that `matrix` lacks.
    """
    columns = run_columns(matrix)
    unknown = sorted(set(found_by) - set(columns))
    if unknown:
        raise ValueError(f"the matrix has no run column {unknown!r}")
    return matrix.filter(
        pl.all_horizontal(
            pl.col(column).fill_null(False) == (column in found_by)
            for column in columns
        )
    )


def draw_found_by_upset(
    fig: plt.Figure, matrix: pl.DataFrame, *, title: str, counts: str = "exact"
) -> None:
    """Draw `matrix` onto `fig` as an UpSet plot: a bar per combination of
    runs with its pair count on it, the single runs first, so each method's
    own pairs lead, then the combinations of two runs, and so on.

    `counts` is the library's: `"exact"` counts the pairs found by exactly
    that combination of runs, `"at_least"` the pairs found by at least those
    runs, so a pair three runs found also counts under every two of them, and
    `"both"` draws the at-least bars above the axis and the exact ones below.

    Takes a `Figure`, as `noise_layers.draw_overlap_diagram` does, since an
    UpSet plot is a grid of axes, and restores the figure's size because
    `UpSet.plot` resizes it.

    Raises `ValueError` for a matrix with fewer than two run columns or no
    pairs, neither of which has a combination to draw, and for a `counts`
    the library does not know.
    """
    columns = run_columns(matrix)
    if len(columns) < 2:
        raise ValueError(f"an UpSet plot needs at least two runs, got {columns!r}")
    if matrix.height == 0:
        raise ValueError("no pairs to draw")

    # The library sizes the label column from the label text alone, without
    # the tick-label padding, so the longest name touches the totals bars; two
    # leading spaces are that padding, measured with the text.
    labels = [f"  {column}" for column in columns]
    indicators = (
        matrix.select(pl.col(column).fill_null(False) for column in columns)
        .rename(dict(zip(columns, labels)))
        .to_pandas()
    )
    original_size = fig.get_size_inches()
    # `element_size=None` fits the grid to the figure it is given, sizing the
    # label column to the measured text; with a fixed element size the
    # library resizes the figure and the label column stretches when the
    # size is restored below.
    axes = UpSet(
        from_indicators(labels, data=indicators),
        subset_size="count",
        sort_by="degree",
        show_counts=True,
        counts=counts,
        element_size=None,
        facecolor=_DOTS,
        other_dots_color=_LINE,
        shading_color=_SHADING,
    ).plot(fig=fig)
    intersections = axes["intersections"]
    for container in intersections.containers:
        exact = counts == "exact" or min(bar.get_height() for bar in container) < 0
        for bar in container:
            bar.set_facecolor(_EXACT_COLOUR if exact else _AT_LEAST_COLOUR)
            bar.set_edgecolor(_AT_LEAST_COLOUR)
            bar.set_alpha(1.0)
    handles = []
    if counts != "exact":
        handles.append(Patch(color=_AT_LEAST_COLOUR, label="Inclusive (at least)"))
    if counts != "at_least":
        handles.append(
            Patch(
                facecolor=_EXACT_COLOUR,
                edgecolor=_AT_LEAST_COLOUR,
                label="Exclusive (exactly)",
            )
        )
    intersections.legend(handles=handles, loc="upper right", fontsize=8)
    if counts == "both":
        intersections.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    intersections.set_ylabel("Pairs found")
    intersections.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{abs(value):.0f}")
    )
    fig.set_size_inches(*original_size)
    fig.suptitle(title)


def draw_run_overlap(
    ax: plt.Axes, overlap: pl.DataFrame, *, value: str = "containment", title: str
) -> None:
    """Draw a `run_overlap` frame onto `ax` as a heatmap of `value`, a row per
    run and a column per other run, each cell labelled with its value.

    Raises `ValueError` for a `value` that is neither `jaccard` nor
    `containment`, or an empty frame."""
    if value not in ("jaccard", "containment"):
        raise ValueError(f"value is jaccard or containment, got {value!r}")
    if overlap.height == 0:
        raise ValueError("no runs to draw")
    runs = list(dict.fromkeys(overlap.get_column("run").to_list()))
    cells = {
        (run, other): score
        for run, other, score in overlap.select("run", "other_run", value).iter_rows()
    }
    grid = [[cells.get((run, other)) for other in runs] for run in runs]
    shown = [
        [float("nan") if score is None else score for score in row] for row in grid
    ]
    image = ax.imshow(shown, vmin=0.0, vmax=1.0, cmap=_OVERLAP_SCALE)
    ax.set_xticks(range(len(runs)), runs, rotation=45, ha="right")
    ax.set_yticks(range(len(runs)), runs)
    for row, scores in enumerate(grid):
        for column, score in enumerate(scores):
            if score is not None:
                ax.text(
                    column,
                    row,
                    f"{score:.2f}",
                    ha="center",
                    va="center",
                    color=_INK,
                )
    ax.set_xlabel("other run")
    ax.set_ylabel("run")
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label=value)


def draw_cluster_sizes(ax: plt.Axes, sizes: pl.DataFrame, *, title: str) -> None:
    """Draw a `cluster_sizes` frame onto `ax` as grouped, stacked bars: cluster
    size along the bottom, a bar per run in each group in that run's
    `RUN_COLOURS` colour, stacked by `OUTCOME_SHADES`: found solid, findable
    paler above it, and random palest on top, each bar labelled with its
    total. The legend gives each run its shades side by side.

    Raises `ValueError` for an empty frame."""
    if sizes.height == 0:
        raise ValueError("no cluster sizes to draw")
    buckets = list(dict.fromkeys(sizes.get_column("cluster_size").to_list()))
    runs = list(dict.fromkeys(sizes.get_column("run").to_list()))
    present = set(sizes.get_column("outcome").to_list())
    shades = [
        (outcome, alpha) for outcome, alpha in OUTCOME_SHADES if outcome in present
    ]
    width = 0.8 / len(runs)
    for index, run in enumerate(runs):
        colour = RUN_COLOURS[index % len(RUN_COLOURS)]
        counts = {
            (bucket, outcome): pairs
            for bucket, outcome, pairs in sizes.filter(pl.col("run") == run)
            .select("cluster_size", "outcome", "pairs")
            .iter_rows()
        }
        positions = [
            position + (index - (len(runs) - 1) / 2) * width
            for position in range(len(buckets))
        ]
        bottom = [0] * len(buckets)
        bars = None
        for outcome, alpha in shades:
            heights = [counts.get((bucket, outcome), 0) for bucket in buckets]
            bars = ax.bar(
                positions,
                heights,
                width=width,
                bottom=bottom,
                color=colour,
                alpha=alpha,
            )
            bottom = [a + b for a, b in zip(bottom, heights, strict=True)]
        assert bars is not None  # nosec B101 - type narrowing; shades is never empty
        ax.bar_label(
            bars, labels=[str(total) for total in bottom], padding=1, fontsize=7
        )
    ax.set_xticks(range(len(buckets)), buckets)
    ax.set_xlabel("cluster size")
    ax.set_ylabel("pairs")
    ax.set_title(title)
    ax.legend(
        [
            tuple(
                Patch(color=RUN_COLOURS[index % len(RUN_COLOURS)], alpha=alpha)
                for _, alpha in shades
            )
            for index in range(len(runs))
        ],
        runs,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        title=" / ".join(outcome for outcome, _ in shades),
        loc="upper right",
    )


def draw_name_similarity(
    ax: plt.Axes, matrix: pl.DataFrame, *, random_level: float, title: str
) -> None:
    """Draw a `with_name_similarity` matrix's pairs onto `ax` as two
    histograms of their names' similarity, the pairs some run found and the
    pairs none did, with the random-pair level as a vertical line: a pair left
    of it is no more alike by name than two unrelated names.

    Raises `ValueError` for a matrix with no pairs."""
    if matrix.height == 0:
        raise ValueError("no pairs to draw")
    found_by_any = pl.any_horizontal(
        pl.col(column).fill_null(False) for column in run_columns(matrix)
    )
    bins = [step / 40 for step in range(41)]
    for found, label in ((True, "found by some run"), (False, "found by no run")):
        scores = (
            matrix.filter(found_by_any == found).get_column(NAME_SIMILARITY).to_list()
        )
        ax.hist(scores, bins=bins, alpha=0.6, label=f"{label} ({len(scores):,})")
    ax.axvline(random_level, color="black", linestyle="--", label="random-pair level")
    ax.set_xlabel("name similarity (character 3-gram Jaccard)")
    ax.set_ylabel("pairs")
    ax.set_title(title)
    ax.legend()


__all__ = [
    "ABOVE_RANDOM",
    "CLEANSED_NAMES",
    "CLUSTER_SIZE_BUCKETS",
    "OUTCOME_FINDABLE",
    "OUTCOME_FOUND",
    "OUTCOME_RANDOM",
    "OUTCOME_SHADES",
    "RUN_COLOURS",
    "FOUND_BY",
    "FOUND_BY_NONE",
    "NAME_SIMILARITY",
    "PAIR_KEY",
    "PAIR_NAMES",
    "candidates_at",
    "char_ngram_jaccard",
    "cluster_sizes",
    "count_by_reach",
    "count_found_by",
    "count_found_by_k",
    "cutoffs_for_candidates",
    "draw_cluster_sizes",
    "draw_found_by_upset",
    "draw_name_similarity",
    "draw_run_overlap",
    "drop_unanimous_pairs",
    "latest_runs",
    "level_metrics",
    "pair_found_matrix",
    "pairs_found_by",
    "random_pair_level",
    "run_columns",
    "run_disagreement",
    "run_overlap",
    "select_runs",
    "with_found_by",
    "with_name_similarity",
]
