import pytest
from company_classify.evaluate import compute_binary_metrics


def test_recall_at_k_defaults_to_true_positive_count():
    # 3 true positives (idx 0, 1, 2), 2 negatives (idx 3, 4).
    # Ranked by score descending: idx0(0.9,pos), idx3(0.8,neg), idx1(0.4,pos),
    # idx2(0.3,pos), idx4(0.1,neg).
    # Default k = count(true positives) = 3, so top-3 by rank = idx0, idx3, idx1.
    # Captured true positives within that window: idx0, idx1 -> 2 of 3.
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.8, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.recall_at_k == 2.0 / 3.0


def test_recall_at_k_respects_explicit_k_override():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.8, 0.1]

    # Top-1 by score is idx0 (label 1) -> 1 of 3 true positives captured.
    bundle = compute_binary_metrics(labels, scores, threshold=0.5, k=1)
    assert bundle.recall_at_k == 1.0 / 3.0

    # Top-5 (the whole slice) captures all 3 true positives.
    bundle_full = compute_binary_metrics(labels, scores, threshold=0.5, k=5)
    assert bundle_full.recall_at_k == 1.0


def test_recall_at_k_is_zero_when_no_true_positives():
    labels = [0, 0, 0]
    scores = [0.9, 0.5, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.recall_at_k == 0.0


def test_candidate_set_size_ratio_matches_thresholded_fraction():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.8, 0.1]

    # Two of five scores (0.9, 0.8) are >= 0.5 -> ratio 2/5.
    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.candidate_set_size_ratio == 2.0 / 5.0


def test_candidate_set_size_ratio_at_extremes():
    labels = [1, 0, 1]
    scores = [0.9, 0.8, 0.7]

    all_kept = compute_binary_metrics(labels, scores, threshold=0.0)
    assert all_kept.candidate_set_size_ratio == 1.0

    none_kept = compute_binary_metrics(labels, scores, threshold=1.1)
    assert none_kept.candidate_set_size_ratio == 0.0


def test_metric_bundle_values_stay_in_unit_interval():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.95, 0.2, 0.6, 0.55, 0.4, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert 0.0 <= bundle.recall_at_k <= 1.0
    assert 0.0 <= bundle.candidate_set_size_ratio <= 1.0
    assert 0.0 <= bundle.precision_at_k <= 1.0
    assert 0.0 <= bundle.duplication_rate <= 1.0
    assert bundle.latency_per_1k_rows >= 0.0


def test_precision_at_k_defaults_to_true_positive_count():
    # Same worked example as test_recall_at_k_defaults_to_true_positive_count:
    # ranked by score descending -> idx0(0.9,pos), idx3(0.8,neg), idx1(0.4,pos),
    # idx2(0.3,pos), idx4(0.1,neg). Default k = 3 true positives, so the top-3
    # window is idx0, idx3, idx1 -> 2 of those 3 slots are true positives.
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.8, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.precision_at_k == 2.0 / 3.0


def test_precision_at_k_respects_explicit_k_override():
    labels = [1, 1, 1, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.8, 0.1]

    # Top-1 by score is idx0 (label 1) -> 1 of 1 slot is a true positive.
    bundle = compute_binary_metrics(labels, scores, threshold=0.5, k=1)
    assert bundle.precision_at_k == 1.0

    # Top-5 (the whole slice) has 3 true positives out of 5 slots.
    bundle_full = compute_binary_metrics(labels, scores, threshold=0.5, k=5)
    assert bundle_full.precision_at_k == 3.0 / 5.0


def test_precision_at_k_is_zero_when_k_is_zero():
    labels = [1, 1, 0]
    scores = [0.9, 0.8, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5, k=0)

    assert bundle.precision_at_k == 0.0


def test_duplication_rate_matches_false_positive_fraction_of_negatives():
    # 3 negatives (idx 1, 3, 4), 2 positives (idx 0, 2).
    # Threshold 0.5 predicts idx0(0.9), idx3(0.6) as positive; the rest negative.
    # Of the 3 negatives (idx1, idx3, idx4), only idx3 is wrongly predicted positive.
    labels = [1, 0, 1, 0, 0]
    scores = [0.9, 0.2, 0.8, 0.6, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.duplication_rate == 1.0 / 3.0


def test_duplication_rate_is_zero_when_no_negatives():
    labels = [1, 1, 1]
    scores = [0.9, 0.8, 0.7]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.duplication_rate == 0.0


def test_duplication_rate_at_extremes():
    labels = [1, 0, 0]
    scores = [0.9, 0.8, 0.1]

    all_flagged = compute_binary_metrics(labels, scores, threshold=0.0)
    assert all_flagged.duplication_rate == 1.0

    none_flagged = compute_binary_metrics(labels, scores, threshold=1.1)
    assert none_flagged.duplication_rate == 0.0


def test_latency_per_1k_rows_scales_elapsed_seconds_by_row_count():
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.2, 0.6, 0.4]

    # 4 rows scored in 0.002s -> 0.002 * 1000 / 4 = 0.5s per 1k rows.
    bundle = compute_binary_metrics(
        labels, scores, threshold=0.5, elapsed_seconds=0.002
    )

    assert bundle.latency_per_1k_rows == pytest.approx(0.5)


def test_latency_per_1k_rows_defaults_to_zero_without_timing():
    labels = [1, 0]
    scores = [0.9, 0.1]

    bundle = compute_binary_metrics(labels, scores, threshold=0.5)

    assert bundle.latency_per_1k_rows == 0.0
