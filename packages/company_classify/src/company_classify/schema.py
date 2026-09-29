"""The shared record types: labelled examples, predictions, metric bundles and splits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabeledExample:
    """A single training/evaluation example for classifier models.

    Attributes:
        text: The raw text to classify (candidate company name or noise).
        label: Binary ground-truth label consumed as-is by
            `ClassifierModel.fit()` and `evaluate.compute_binary_metrics()`.
            `1` means `text` is a valid company name, `0` means it is not.
        system: Optional caller-supplied provenance identifier (e.g. source
            system the example came from). Not read by anything in this
            package; carried through purely as metadata for callers.
        language: Optional caller-supplied language tag for `text`. Not
            read by anything in this package; carried through purely as
            metadata for callers.
    """

    text: str
    label: int
    system: str | None = None
    language: str | None = None


@dataclass(frozen=True)
class DatasetSplit:
    """A deterministic train/validation/test partition.

    Returned by `split.split_examples()`.

    Attributes:
        train: Examples used to fit a `ClassifierModel`.
        validation: Examples used for model-selection/threshold checks
            during `train.cross_train()`.
        test: Examples held out for final `MetricBundle` reporting.
    """

    train: list[LabeledExample]
    validation: list[LabeledExample]
    test: list[LabeledExample]


@dataclass(frozen=True)
class Prediction:
    """A single scored inference result.

    Returned by `inference.predict_batch()`.

    Attributes:
        text: The input text that was scored.
        score: Positive-class probability from the underlying
            `ClassifierModel`, in `[0.0, 1.0]`.
        label: Thresholded binary decision (`1` if `score` was at or above
            `predict_batch()`'s `threshold`, else `0`).
    """

    text: str
    score: float
    label: int


@dataclass(frozen=True)
class MetricBundle:
    """Binary classification metrics for one evaluation slice.

    Returned by `evaluate.compute_binary_metrics()`.

    Attributes:
        precision: Precision of the thresholded predictions
            (`sklearn.metrics.precision_score`).
        recall: Recall of the thresholded predictions
            (`sklearn.metrics.recall_score`).
        f1: F1 score of the thresholded predictions
            (`sklearn.metrics.f1_score`).
        pr_auc: Area under the precision-recall curve, computed from the
            raw (unthresholded) scores via
            `sklearn.metrics.average_precision_score`.
        recall_at_k: Ranking-quality recall computed over the `k`
            highest-scored examples (independent of `threshold`): the
            fraction of all true-positive (`label == 1`) examples that fall
            within the top `k` examples when ranked by score descending.
            Unless `compute_binary_metrics()` is called with an explicit
            `k`, `k` defaults to the number of true-positive examples in
            the slice, so this measures how well the model ranks true
            company names ahead of everything else, decoupled from any
            particular threshold choice.
        candidate_set_size_ratio: Blocking-oriented sizing metric: the
            fraction of all examples in the slice that the thresholded
            predictions keep as candidates (`predicted == 1`), i.e.
            `count(predicted == 1) / count(examples)`. Reports how much a
            downstream blocking/dedup stage would need to process if it
            consumed this model's threshold decision as-is.
        precision_at_k: Ranking-quality precision computed over the `k`
            highest-scored examples (the same window `recall_at_k` uses,
            independent of `threshold`): the fraction of the top `k`
            examples, ranked by score descending, that are true positives
            (`label == 1`). Complements `recall_at_k` by measuring how
            "clean" that top-`k` window is rather than how much of the
            positive set it captures.
        duplication_rate: Blocking-oriented over-merge risk indicator:
            restricted to the truly negative (`label == 0`) examples in the
            slice, the fraction the thresholded predictions still flag as a
            match (`predicted == 1`), i.e. a false-positive rate. Reports
            how often a downstream blocking/dedup stage would wrongly treat
            distinct entities as duplicates if it consumed this model's
            threshold decision as-is.
        latency_per_1k_rows: Scoring wall-clock time, in seconds, for the
            underlying model's `predict_scores()` call over this slice,
            normalized to a 1,000-row unit so it is comparable across
            slices/runs of different sizes. `0.0` if the caller did not
            supply a timed `elapsed_seconds` to
            `compute_binary_metrics()` (e.g. `zeroed()`'s empty-slice
            placeholder).
    """

    precision: float
    recall: float
    f1: float
    pr_auc: float
    recall_at_k: float
    candidate_set_size_ratio: float
    precision_at_k: float
    duplication_rate: float
    latency_per_1k_rows: float

    @classmethod
    def zeroed(cls) -> MetricBundle:
        """Every metric at `0.0`, for a slice with no examples to score.

        An empty slice has nothing to compute from, so the evaluation
        helpers return this rather than calling `compute_binary_metrics()`
        on an empty input. It lives here so a field added to this dataclass
        is zeroed in one place instead of in each caller.
        """
        return cls(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            pr_auc=0.0,
            recall_at_k=0.0,
            candidate_set_size_ratio=0.0,
            precision_at_k=0.0,
            duplication_rate=0.0,
            latency_per_1k_rows=0.0,
        )
