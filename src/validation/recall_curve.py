"""Recall against comparisons spent, and its normalised area.

`compute_recall_curve` reads a completed run's `matched_edges` and `pair_truth_eval_detail` and gives one row per population per budget step, where a step is a uniform per-source rank threshold and `comparisons_spent` the costed candidate edges at or below it. A pair the exact-name fast path resolved costs nothing and counts from step zero. The populations are the universe and the name-equality levels (`RECALL_CURVE_POPULATIONS`), each source placed at its own truth pair's level as `pair_truth_eval` places it.

`recall_curve_area` reduces one population's curve to its mean recall over comparisons spent; `summarize_recall_curve` gives a run's one-row summary with `recall_area_<level>` per level, `recall_area_never` the headline; `rank_by_recall_area` ranks runs on one population's area over the smallest budget any of them reached. Comparisons spent is the axis representations share: one cosine cut sits at a different point on each representation's curve. The blocking area writes the two artefacts (`blocking.workflow.compute_recall_curve_for_run`).
"""

from __future__ import annotations

import polars as pl

from .contracts import ARTIFACT_SCHEMAS as _ARTIFACT_SCHEMAS
from .contracts import POPULATION_UNIVERSE, PolarsDType
from .runner import NAME_EQUALITY_LEVELS

# The populations a recall curve is built for: the universe (every labelled
# source, any level, `unknown` included) and the four name-equality levels a
# truth pair can cascade through. `unknown` gets no curve of its own, matching
# `NAME_EQUALITY_LEVELS`' own exclusion -- a source with no verdict has
# nothing to report a curve against, and stays folded into the universe.
RECALL_CURVE_POPULATIONS: tuple[str, ...] = (POPULATION_UNIVERSE, *NAME_EQUALITY_LEVELS)

# One row per population per budget step. `budget_step` is the uniform
# per-source rank threshold applied to reach that row (0 before any scored
# candidate is admitted); `comparisons_spent` is the actual cost that
# threshold bought -- the count of *costed* candidate edges at or below it,
# summed across every source in the population. A pair the exact-name fast
# path resolved (`similarity == 1.0`, the same convention
# `blocking`'s diagnostics already read matched_edges by) costs
# nothing and is counted at every step including `budget_step == 0`, so a
# run dominated by the fast path shows a curve that starts high rather than
# climbing from zero for pairs that were never scored at all.
#
# The contract is `validation.contracts.ARTIFACT_SCHEMAS`; restating it here
# is how the artefact check and this module's own output drifted apart
# before (`runner._result_schemas()`'s `pair_truth_eval` entry is the
# precedent this follows).
RECALL_CURVE_COLUMNS: dict[str, PolarsDType] = dict(_ARTIFACT_SCHEMAS["recall_curve"])

# One row per completed run: the normalised area (mean recall over
# comparisons spent, `recall_curve_area()`) per name-equality level, plus
# what a reader needs to place the figure -- the backend the run retrieved
# with, whether that backend is exact or approximate
# (`blocking.comparison._is_exact_backend`), and the target population size.
# `recall_area_never` is the headline: the still-unequal residual is the
# non-trivial population a representation actually has to earn, unlike the
# blended universe figure, which trivial exact-name matches dominate.
RECALL_CURVE_SUMMARY_COLUMNS: dict[str, PolarsDType] = dict(
    _ARTIFACT_SCHEMAS["recall_curve_summary"]
)


def _population_curve(
    *,
    population: str,
    truth_status: pl.DataFrame,
    edges: pl.DataFrame,
) -> pl.DataFrame:
    """One population's step curve: `truth_status` is one row per labelled
    source in this population carrying that source's own truth pair's
    `found`/`similarity`/`rank` (`similarity`/`rank` null when never
    predicted), and `edges` is every candidate `matched_edges` row for those
    same sources, truth pair or not -- so a source's wrong candidates spend
    its budget exactly as its right one does."""
    truth_pairs = truth_status.height
    if truth_pairs == 0:
        return pl.DataFrame(
            {
                "population": [population],
                "budget_step": [0],
                "comparisons_spent": [0],
                "truth_pairs": [0],
                "pairs_found": [0],
                "recall": [None],
            },
            schema=RECALL_CURVE_COLUMNS,
        )

    free_found = truth_status.filter(
        pl.col("found") & (pl.col("similarity") == 1.0)
    ).height
    costed_found = truth_status.filter(
        pl.col("found") & (pl.col("similarity") != 1.0)
    ).select("rank")

    costed_edges = edges.filter(pl.col("similarity") != 1.0)

    if costed_edges.height == 0:
        # Every candidate in this population was free (or there were none at
        # all): the curve never leaves budget_step 0, covering the
        # degenerate "every truth pair retrieved before any non-truth pair"
        # case when that retrieval was entirely the exact-name fast path.
        return pl.DataFrame(
            {
                "population": [population],
                "budget_step": [0],
                "comparisons_spent": [0],
                "truth_pairs": [truth_pairs],
                "pairs_found": [free_found],
                "recall": [free_found / truth_pairs],
            },
            schema=RECALL_CURVE_COLUMNS,
        )

    per_rank_cost = (
        costed_edges.group_by("rank")
        .agg(pl.len().alias("edge_count"))
        .sort("rank")
        .with_columns(pl.col("edge_count").cum_sum().alias("comparisons_spent"))
    )
    per_rank_found = (
        costed_found.group_by("rank").agg(pl.len().alias("found_count")).sort("rank")
    )
    steps = (
        per_rank_cost.join(per_rank_found, on="rank", how="left")
        .with_columns(pl.col("found_count").fill_null(0))
        .sort("rank")
        .with_columns(
            (pl.col("found_count").cum_sum() + free_found).alias("pairs_found")
        )
        .with_columns(
            (pl.col("pairs_found") / truth_pairs).alias("recall"),
            pl.col("rank").alias("budget_step"),
        )
        .select("budget_step", "comparisons_spent", "pairs_found", "recall")
    )

    base_row = pl.DataFrame(
        {
            "budget_step": [0],
            "comparisons_spent": [0],
            "pairs_found": [free_found],
            "recall": [free_found / truth_pairs],
        }
    )
    combined = pl.concat([base_row, steps], how="vertical_relaxed")
    return (
        combined.with_columns(
            pl.lit(population).alias("population"),
            pl.lit(truth_pairs).alias("truth_pairs"),
        )
        .select(list(RECALL_CURVE_COLUMNS.keys()))
        .cast(pl.Schema(RECALL_CURVE_COLUMNS))
    )


def compute_recall_curve(
    *,
    matched_edges: pl.DataFrame,
    pair_truth_eval_detail: pl.DataFrame,
) -> pl.DataFrame:
    """A recall-against-comparisons-spent curve for a completed run, read
    from artefacts already on disk -- no re-running of candidate generation.

    `matched_edges` is the run's own per-pair frame (`similarity`/`rank` per
    candidate); `pair_truth_eval_detail` is `validation.runner.
    compute_pair_truth_eval_detail()`'s per-pair companion, which already
    classifies every truth pair by the source's own name-equality level (the
    first name form at which the source and its true target become the same
    string). Every candidate `matched_edges` emits for a source is attributed
    to that source's own level here -- never to a candidate row's own
    `name_equality`, which (for a wrong candidate) describes the pair it was
    actually paired with, not the source's truth level.

    One row per population (`RECALL_CURVE_POPULATIONS`) per budget step
    (`RECALL_CURVE_COLUMNS`). `recall_curve_area()` reduces one population's
    curve to a single normalised figure.
    """
    truth = pair_truth_eval_detail.filter(pl.col("is_truth_pair"))
    truth_status = truth.select(
        "source_id",
        pl.col("name_equality").alias("level"),
        "found",
        "similarity",
        "rank",
    ).unique(subset=["source_id"], keep="first")

    rows = []
    for population in RECALL_CURVE_POPULATIONS:
        pop_sources = (
            truth_status
            if population == POPULATION_UNIVERSE
            else truth_status.filter(pl.col("level") == population)
        )
        pop_edges = matched_edges.join(
            pop_sources.select("source_id"), on="source_id", how="inner"
        )
        rows.append(
            _population_curve(
                population=population,
                truth_status=pop_sources,
                edges=pop_edges,
            )
        )
    return pl.concat(rows, how="vertical_relaxed")


def recall_curve_area(
    curve: pl.DataFrame,
    *,
    population: str,
    max_comparisons: int | None = None,
) -> float | None:
    """The normalised area under one population's step curve: the mean
    recall over comparisons spent, read only as far as `max_comparisons`
    when given, or this curve's own full reached budget otherwise.

    `None` when the population has no truth pairs at all (`recall` is null
    throughout, per `_population_curve`'s empty-population row) -- there is
    nothing to average. A curve whose every truth pair was free (never left
    `comparisons_spent == 0`) reduces to that single point's recall, the
    mean of a zero-width budget.

    Reads the curve as a right-continuous step function: recall holds at a
    step's value until the next step (or `max_comparisons`) is reached, the
    same reading `_population_curve`'s cumulative construction produces.
    """
    pop = (
        curve.filter(pl.col("population") == population)
        .sort("comparisons_spent")
        .select("comparisons_spent", "recall")
    )
    if pop.height == 0:
        return None
    rows = pop.rows()
    if rows[0][1] is None:
        return None

    own_budget = rows[-1][0]
    budget = own_budget if max_comparisons is None else min(own_budget, max_comparisons)
    if budget <= 0:
        return rows[0][1]

    area = 0.0
    covered = 0
    for index, (comparisons_spent, recall) in enumerate(rows):
        if comparisons_spent >= budget:
            break
        upper = rows[index + 1][0] if index + 1 < len(rows) else budget
        upper = min(upper, budget)
        if upper <= comparisons_spent:
            continue
        area += recall * (upper - comparisons_spent)
        covered = upper
    if covered < budget:
        area += rows[-1][1] * (budget - covered)
    return area / budget


def summarize_recall_curve(
    curve: pl.DataFrame,
    *,
    similarity_backend: str,
    is_exact_backend: bool | None,
    target_rows: int | None,
    min_similarity: float | None = None,
    top_k: int | None = None,
    max_candidates_per_source: int | None = None,
) -> pl.DataFrame:
    """One summary row: `recall_area_<level>` per `NAME_EQUALITY_LEVELS`,
    `recall_area_never` the headline, beside what a reader needs to place the
    figures against -- `similarity_backend`, whether it is exact or
    approximate, the target population it retrieved against, and the
    `min_similarity`/`top_k`/`max_candidates_per_source` the run retrieved
    under, since each of those three truncates the curve where that run's
    own candidates stop, the same way `max_comparisons` truncates a shared
    read in `recall_curve_area()`. Each area is this run's own, over its own
    reached budget; ranking two runs against each other's shared budget is
    `rank_by_recall_area()`'s job, not this one's."""
    areas = {
        f"recall_area_{level}": recall_curve_area(curve, population=level)
        for level in NAME_EQUALITY_LEVELS
    }
    return pl.DataFrame(
        [
            {
                "similarity_backend": similarity_backend,
                "is_exact_backend": is_exact_backend,
                "target_rows": target_rows,
                "min_similarity": min_similarity,
                "top_k": top_k,
                "max_candidates_per_source": max_candidates_per_source,
                **areas,
            }
        ],
        schema=RECALL_CURVE_SUMMARY_COLUMNS,
    )


RECALL_RANKING_COLUMNS: dict[str, PolarsDType] = {
    "label": pl.Utf8,
    "similarity_backend": pl.Utf8,
    "population": pl.Utf8,
    "shared_budget": pl.Int64,
    "recall_area": pl.Float64,
}


def rank_by_recall_area(
    entries: list[tuple[str, str, pl.DataFrame]],
    *,
    population: str,
) -> pl.DataFrame:
    """Rank two or more completed runs' curves on one population's area,
    each read only as far as the smallest budget any entry reached
    (`shared_budget`), so a run with a larger candidate window is not
    credited for territory another run never covered. `entries` is
    `(label, similarity_backend, curve)` per run; the returned frame is
    sorted by `recall_area` descending, the strongest strategy first.
    """
    max_budgets = []
    for _, _, curve in entries:
        pop = curve.filter(pl.col("population") == population)
        if pop.height == 0:
            continue
        pop_max = pop.select(pl.col("comparisons_spent").max()).item()
        if pop_max is not None:
            max_budgets.append(int(pop_max))
    shared_budget = min(max_budgets) if max_budgets else 0

    rows = [
        {
            "label": label,
            "similarity_backend": similarity_backend,
            "population": population,
            "shared_budget": shared_budget,
            "recall_area": recall_curve_area(
                curve, population=population, max_comparisons=shared_budget
            ),
        }
        for label, similarity_backend, curve in entries
    ]
    return pl.DataFrame(rows, schema=RECALL_RANKING_COLUMNS).sort(
        "recall_area", descending=True
    )
