"""Which vocabulary sizes the optimize expansion phase gives more seeds, as a named, swappable strategy.

`ExpansionStrategy` bundles the frontier selection, prune reason and result recording that `run_optimize_expansion_phase` calls. One strategy is built in, `"heuristic"` (`HEURISTIC_EXPANSION_STRATEGY`, looked up by `resolve_expansion_strategy`), which keeps the vocabulary sizes nearest the fertility target by median fertility distance, plus any within tolerance of the nearest.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from company_tokenize import OptimizeCandidatePolicy, OptimizeCandidateResult


def _score_vocab_candidates(
    pilot_results: list[OptimizeCandidateResult],
    min_frequency: int,
    vocab_schedule: list[int],
) -> list[tuple[int, float, int]]:
    per_vocab_scores: list[tuple[int, float, int]] = []
    for vocab_requested in vocab_schedule:
        subset = [
            result
            for result in pilot_results
            if result.min_frequency == min_frequency
            and result.vocab_requested == vocab_requested
        ]
        if not subset:
            continue

        passing = [result for result in subset if not result.rejected]
        scored = passing if passing else subset
        distances = [
            float(result.validation_metrics["fertility_distance"]) for result in scored
        ]
        median_distance = statistics.median(distances) if distances else float("inf")
        pass_count = len(passing)
        per_vocab_scores.append((vocab_requested, median_distance, pass_count))
    return per_vocab_scores


def select_vocab_frontier(
    *,
    pilot_results: list[OptimizeCandidateResult],
    vocab_schedule: list[int],
    min_frequencies: list[int],
    top_k: int,
    distance_tolerance: float,
    vocab_sort_key: Callable[[int], tuple[int, int]],
    max_distance: float | None = None,
) -> set[tuple[int, int]]:
    """Per `min_frequency`, the vocab points worth more seeds: the `top_k`
    nearest the fertility target, plus any within `distance_tolerance` of the
    nearest.

    `max_distance` keeps the `top_k` fill to points no further than that from
    the target, so a short pilot line does not fill it with a point far out
    of range. When no point is within `max_distance` the fill is
    unrestricted, so every `min_frequency` still gets a frontier.
    """
    frontier: set[tuple[int, int]] = set()
    vocab_rank = {vocab: idx for idx, vocab in enumerate(vocab_schedule)}

    for min_frequency in min_frequencies:
        per_vocab_scores = _score_vocab_candidates(
            pilot_results, min_frequency, vocab_schedule
        )

        if not per_vocab_scores:
            continue

        per_vocab_scores.sort(
            key=lambda item: (item[1], -item[2], vocab_rank.get(item[0], 10**9))
        )
        best_distance = per_vocab_scores[0][1]
        fill_limit = (
            max_distance
            if max_distance is not None and best_distance <= max_distance
            else float("inf")
        )
        selected: set[int] = set()
        for index, (vocab_requested, distance, _pass_count) in enumerate(
            per_vocab_scores
        ):
            if (index < top_k and distance <= fill_limit) or (
                distance <= best_distance + distance_tolerance
            ):
                selected.add(vocab_requested)

        if not selected:
            selected.add(per_vocab_scores[0][0])

        for vocab_requested in selected:
            frontier.add((min_frequency, vocab_requested))

    return frontier


def select_refining_bracket(
    *,
    frontier: set[tuple[int, int]],
    winning_min_frequency: int,
    winning_vocab_requested: int,
) -> tuple[int, int, int] | None:
    """Pick the three vocab points a refining phase interpolates fertility
    between: the winning vocab as the interior point and the numeric vocab
    points `select_vocab_frontier` kept for `winning_min_frequency`
    immediately below and above it (`optimize_search_methodology.md` Part 8).

    Returns `None` when there's no bracket to search inside: the winner's
    vocab is `-1` (auto-sized -- no numeric bracket), its frontier has fewer
    than three distinct numeric points, or the winner sits at one edge of
    its own frontier rather than strictly between the other two (so no
    three *distinct* points are available to bracket it).
    """
    if winning_vocab_requested == -1:
        return None
    frontier_vocabs = sorted(
        {
            vocab
            for min_frequency, vocab in frontier
            if min_frequency == winning_min_frequency and vocab != -1
        }
    )
    if winning_vocab_requested not in frontier_vocabs or len(frontier_vocabs) < 3:
        return None
    winner_index = frontier_vocabs.index(winning_vocab_requested)
    if winner_index in (0, len(frontier_vocabs) - 1):
        return None
    return (
        frontier_vocabs[winner_index - 1],
        winning_vocab_requested,
        frontier_vocabs[winner_index + 1],
    )


def build_crossing_vocab_points(
    *, crossing: float, bracket_low: int, bracket_high: int, step: int
) -> list[int]:
    """The predicted crossing and one step either side, strictly inside the bracket.

    The distance is a V whose point is the crossing
    (`optimize_search_methodology.md` Part 8), so these three points are
    what a refining round needs: the crossing itself, and a neighbour each
    side to confirm the V's slope there.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if bracket_high <= bracket_low:
        raise ValueError("bracket_high must be greater than bracket_low.")
    centre = round(crossing)
    return sorted(
        {
            point
            for point in (centre - step, centre, centre + step)
            if bracket_low < point < bracket_high
        }
    )


def combo_prune_reason(
    *,
    runs: int,
    passes: int,
    fertilities: list[float],
    total_seeds: int,
    policy: OptimizeCandidatePolicy,
) -> str | None:
    required_passes = math.ceil(float(total_seeds) * policy.eligibility_pass_rate)
    remaining = total_seeds - runs
    if passes + remaining < required_passes:
        return "ineligible_pass_rate_ceiling"

    if runs >= 2 and passes == 0 and fertilities:
        below_min = all(fertility < policy.fertility_min for fertility in fertilities)
        above_max = all(fertility > policy.fertility_max for fertility in fertilities)
        if below_min or above_max:
            return "fertility_outside_range_consistent"

    return None


def vocab_sort_key(value: int) -> tuple[int, int]:
    """Sort explicit vocab sizes ascending and place auto (-1) last."""
    if value == -1:
        return (1, 0)
    return (0, value)


def default_record_result(
    *,
    combo: tuple[int, int],
    result: OptimizeCandidateResult,
    expansion_round: int,
) -> None:
    """No-op extension point.

    Called once per completed expansion-phase candidate result (not pilot
    results). Exists so a future stateful strategy (e.g. one that needs to
    persist a posterior across expansion rounds) has a stable place to hook
    in without a second edit to run_optimize_expansion_phase's call sites.
    Intentionally inert here -- the heuristic strategy's elimination state
    lives entirely in combo_progress/prune_reason.
    """
    return


def _default_select_vocab_frontier(
    *,
    pilot_results: list[OptimizeCandidateResult],
    vocab_schedule: list[int],
    min_frequencies: list[int],
    top_k: int,
    distance_tolerance: float,
    max_distance: float | None = None,
) -> set[tuple[int, int]]:
    return select_vocab_frontier(
        pilot_results=pilot_results,
        vocab_schedule=vocab_schedule,
        min_frequencies=min_frequencies,
        top_k=top_k,
        distance_tolerance=distance_tolerance,
        vocab_sort_key=vocab_sort_key,
        max_distance=max_distance,
    )


@dataclass(frozen=True)
class ExpansionStrategy:
    """Names and bundles the arm allocation/elimination decisions made
    during run_optimize_expansion_phase behind one injectable value,
    mirroring the *_func parameter convention already used by
    run_optimize_target and the dataclass-of-callables shape of
    OptimizeWorkloadServices (src/training/workloads.py), rather than a
    Protocol/class-hierarchy strategy object.

    Allocation (which combos run in a given round) is deliberately NOT part
    of this interface yet -- today it's unconditionally "every un-pruned
    frontier combo, once per remaining seed" with no extraction precedent,
    and its correct shape depends on which future strategy is eventually
    chosen. See run_optimize_expansion_phase for where it would be injected
    once that decision is made.
    """

    name: str
    select_frontier: Callable[..., set[tuple[int, int]]]
    prune_reason: Callable[..., str | None]
    record_result: Callable[..., None] = default_record_result


HEURISTIC_STRATEGY_NAME = "heuristic"

HEURISTIC_EXPANSION_STRATEGY = ExpansionStrategy(
    name=HEURISTIC_STRATEGY_NAME,
    select_frontier=_default_select_vocab_frontier,
    prune_reason=combo_prune_reason,
)

# One-line registration point for future strategies (deferred pending a
# separate trust/noise-calibration validation session -- see
# optimize_search_methodology.md).
EXPANSION_STRATEGIES: dict[str, ExpansionStrategy] = {
    HEURISTIC_STRATEGY_NAME: HEURISTIC_EXPANSION_STRATEGY,
}


def resolve_expansion_strategy(name: str) -> ExpansionStrategy:
    try:
        return EXPANSION_STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(EXPANSION_STRATEGIES))
        raise ValueError(
            f"Unknown --expansion-strategy {name!r}; known strategies: {known}."
        ) from exc
