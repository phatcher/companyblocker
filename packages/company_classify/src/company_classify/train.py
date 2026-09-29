"""Fit and evaluate several `ClassifierModel`s over one shared split, so they compare fairly.

Labels are binary (`0`/`1`) and there is no class-imbalance reweighting. Each model's
`predict_scores` call is timed here and passed to `evaluate.compute_binary_metrics` as
`elapsed_seconds`, since that function never calls the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .evaluate import compute_binary_metrics
from .models import ClassifierModel
from .schema import DatasetSplit, LabeledExample, MetricBundle


@dataclass(frozen=True)
class TrainedModelResult:
    """One model's fitted state and evaluation results from `cross_train()`.

    Attributes:
        name: The key this model was registered under in the `models` dict
            passed to `cross_train()`.
        model: The fitted `ClassifierModel` instance (mutated in place by
            `fit()` during `cross_train()`).
        validation_metrics: `MetricBundle` computed on `DatasetSplit.validation`.
        test_metrics: `MetricBundle` computed on `DatasetSplit.test`.
    """

    name: str
    model: ClassifierModel
    validation_metrics: MetricBundle
    test_metrics: MetricBundle


def _evaluate_slice(
    model: ClassifierModel, examples: list[LabeledExample]
) -> MetricBundle:
    if not examples:
        return MetricBundle.zeroed()

    start = time.perf_counter()
    scores = model.predict_scores([example.text for example in examples])
    elapsed_seconds = time.perf_counter() - start
    labels = [example.label for example in examples]
    return compute_binary_metrics(labels, scores, elapsed_seconds=elapsed_seconds)


def _evaluate(
    model: ClassifierModel, split: DatasetSplit
) -> tuple[MetricBundle, MetricBundle]:
    validation_metrics = _evaluate_slice(model, split.validation)
    test_metrics = _evaluate_slice(model, split.test)

    return validation_metrics, test_metrics


def cross_train(
    split: DatasetSplit,
    models: dict[str, ClassifierModel],
) -> dict[str, TrainedModelResult]:
    if not models:
        raise ValueError("models must not be empty")

    results: dict[str, TrainedModelResult] = {}

    for name, model in models.items():
        model.fit(split.train)
        validation_metrics, test_metrics = _evaluate(model, split)
        results[name] = TrainedModelResult(
            name=name,
            model=model,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

    return results
