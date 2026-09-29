"""Binary-classification metrics for one slice, including the ranking and sizing figures blocking uses.

`schema.MetricBundle` defines each figure.
"""

from __future__ import annotations

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from .schema import MetricBundle


def _recall_at_k(labels: list[int], scores: list[float], k: int) -> float:
    """Fraction of true positives captured in the top-`k` scored examples.

    Ranks all `(label, score)` pairs by `score` descending and checks how
    many of the `label == 1` examples fall within the top `k` positions.
    Returns `0.0` if `k <= 0` or there are no true positives to find.
    """
    total_positives = sum(1 for label in labels if label == 1)
    if k <= 0 or total_positives == 0:
        return 0.0

    ranked_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )
    top_k_indices = ranked_indices[:k]
    captured = sum(1 for index in top_k_indices if labels[index] == 1)

    return float(captured) / float(total_positives)


def _precision_at_k(labels: list[int], scores: list[float], k: int) -> float:
    """Fraction of the top-`k` scored examples that are true positives.

    Ranks all `(label, score)` pairs by `score` descending, same window
    `_recall_at_k()` uses, and checks how many of the top `k` positions hold
    a `label == 1` example. Complements `_recall_at_k()`, which measures
    coverage of the positive set, by measuring how clean that window is.
    Returns `0.0` if `k <= 0` or the slice is empty.
    """
    if k <= 0 or not scores:
        return 0.0

    ranked_indices = sorted(
        range(len(scores)), key=lambda index: scores[index], reverse=True
    )
    top_k_indices = ranked_indices[:k]
    if not top_k_indices:
        return 0.0
    captured = sum(1 for index in top_k_indices if labels[index] == 1)

    return float(captured) / float(len(top_k_indices))


def _candidate_set_size_ratio(predicted: list[int]) -> float:
    """Fraction of examples kept as candidates (`predicted == 1`)."""
    if not predicted:
        return 0.0
    return float(sum(predicted)) / float(len(predicted))


def _duplication_rate(labels: list[int], predicted: list[int]) -> float:
    """Fraction of truly negative examples the thresholded predictions still flag as a match.

    Restricted to `label == 0` examples (truly distinct entities), the
    fraction where `predicted == 1` -- a false-positive rate, and a
    blocking-oriented over-merge risk indicator. Returns `0.0` if the slice
    has no negative examples to measure it against.
    """
    negative_indices = [index for index, label in enumerate(labels) if label == 0]
    if not negative_indices:
        return 0.0
    false_positives = sum(1 for index in negative_indices if predicted[index] == 1)

    return float(false_positives) / float(len(negative_indices))


def _latency_per_1k_rows(elapsed_seconds: float, row_count: int) -> float:
    """Scoring wall-clock time, in seconds, normalized to a 1,000-row unit.

    `elapsed_seconds` should time exactly the model's `predict_scores()`
    call that produced `scores`, not the whole evaluation step; callers
    that cannot supply a real timing (e.g. an empty-slice placeholder) pass
    `0.0`. Returns `0.0` if `row_count <= 0`.
    """
    if row_count <= 0:
        return 0.0
    return float(elapsed_seconds) * 1000.0 / float(row_count)


def compute_binary_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float = 0.5,
    k: int | None = None,
    elapsed_seconds: float = 0.0,
) -> MetricBundle:
    """Compute a `MetricBundle` for one evaluation slice.

    Args:
        labels: Ground-truth binary labels, one per example.
        scores: Positive-class probability scores, one per example, aligned
            with `labels`.
        threshold: Score cutoff (inclusive) for the thresholded
            `predicted` decision that `precision`/`recall`/`f1`/
            `candidate_set_size_ratio`/`duplication_rate` are computed from.
        k: Window size for `recall_at_k`/`precision_at_k`. Defaults to the
            number of true positives in `labels` when omitted.
        elapsed_seconds: Wall-clock time the caller spent scoring this
            slice (timing the `predict_scores()` call, not this function),
            used to compute `latency_per_1k_rows`. Defaults to `0.0` for
            callers that do not measure timing.
    """
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be equal length")

    predicted = [1 if score >= threshold else 0 for score in scores]

    precision = precision_score(labels, predicted, zero_division=0)
    recall = recall_score(labels, predicted, zero_division=0)
    f1 = f1_score(labels, predicted, zero_division=0)
    pr_auc = average_precision_score(labels, scores)

    effective_k = k if k is not None else sum(1 for label in labels if label == 1)

    return MetricBundle(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        pr_auc=float(pr_auc),
        recall_at_k=_recall_at_k(labels, scores, effective_k),
        candidate_set_size_ratio=_candidate_set_size_ratio(predicted),
        precision_at_k=_precision_at_k(labels, scores, effective_k),
        duplication_rate=_duplication_rate(labels, predicted),
        latency_per_1k_rows=_latency_per_1k_rows(elapsed_seconds, len(labels)),
    )
