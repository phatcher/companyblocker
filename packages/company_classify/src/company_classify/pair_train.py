"""Training/evaluation hooks for the pair-classifier training mode (`pair_models.py`).

`split_pairs()` partitions an already-tagged pair set by the `split` each `PairRecord` carries
(the shared `EntitySplitter` tag, set once by whichever `pair_producers.py` producer generated
it) rather than re-shuffling, so pair-level and entity-level split membership can never diverge.
`cross_train_pairs()` mirrors `train.cross_train()`'s shape for `pair_models.PairClassifierModel`
implementations.

The baseline half (`pairs_to_labeled_examples()`, `cross_train_pair_baseline()`) adapts pairs
into the single-text `LabeledExample` shape `models.py`'s classifiers already consume
(`f"{left}{separator}{right}"`, pair label), so `TfidfLogRegClassifier`/`TfidfMlpClassifier` and
`train.cross_train()` run, unmodified, over the identical pairs and split as the dedicated pair
classifier -- the same `MetricBundle` shape, directly comparable, without a second evaluation
harness.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from .entity_split import SplitName
from .evaluate import compute_binary_metrics
from .models import ClassifierModel
from .pair_models import PairClassifierModel
from .pairs import PairRecord
from .schema import DatasetSplit, LabeledExample, MetricBundle
from .train import TrainedModelResult, cross_train

PAIR_TEXT_SEPARATOR = " ||| "


@dataclass(frozen=True)
class PairDatasetSplit:
    """A pair set partitioned by each pair's already-assigned `split` tag.

    Attributes:
        train: Pairs used to fit a `PairClassifierModel`.
        validation: Pairs used for model-selection/threshold checks during
            `cross_train_pairs()`.
        test: Pairs held out for final `MetricBundle` reporting.
    """

    train: list[PairRecord]
    validation: list[PairRecord]
    test: list[PairRecord]


@dataclass(frozen=True)
class TrainedPairModelResult:
    """One pair model's fitted state and evaluation results from `cross_train_pairs()`.

    Attributes:
        name: The key this model was registered under in the `models` dict passed to
            `cross_train_pairs()`.
        model: The fitted `PairClassifierModel` instance (mutated in place by `fit()` during
            `cross_train_pairs()`).
        validation_metrics: `MetricBundle` computed on `PairDatasetSplit.validation`.
        test_metrics: `MetricBundle` computed on `PairDatasetSplit.test`.
    """

    name: str
    model: PairClassifierModel
    validation_metrics: MetricBundle
    test_metrics: MetricBundle


def split_pairs(pairs: Sequence[PairRecord]) -> PairDatasetSplit:
    """Partition `pairs` by their existing `split` tag, one bucket per `SplitName`."""
    buckets: dict[SplitName, list[PairRecord]] = {name: [] for name in SplitName}
    for pair in pairs:
        buckets[pair.split].append(pair)

    return PairDatasetSplit(
        train=buckets[SplitName.TRAIN],
        validation=buckets[SplitName.VALIDATION],
        test=buckets[SplitName.TEST],
    )


def _evaluate_pair_slice(
    model: PairClassifierModel, pairs: list[PairRecord]
) -> MetricBundle:
    if not pairs:
        return MetricBundle.zeroed()

    start = time.perf_counter()
    scores = model.predict_scores([(pair.left_name, pair.right_name) for pair in pairs])
    elapsed_seconds = time.perf_counter() - start
    labels = [pair.label for pair in pairs]
    return compute_binary_metrics(labels, scores, elapsed_seconds=elapsed_seconds)


def cross_train_pairs(
    split: PairDatasetSplit,
    models: dict[str, PairClassifierModel],
) -> dict[str, TrainedPairModelResult]:
    if not models:
        raise ValueError("models must not be empty")

    results: dict[str, TrainedPairModelResult] = {}

    for name, model in models.items():
        model.fit(split.train)
        validation_metrics = _evaluate_pair_slice(model, split.validation)
        test_metrics = _evaluate_pair_slice(model, split.test)
        results[name] = TrainedPairModelResult(
            name=name,
            model=model,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )

    return results


def pairs_to_labeled_examples(
    pairs: Sequence[PairRecord], separator: str = PAIR_TEXT_SEPARATOR
) -> list[LabeledExample]:
    """Adapt pairs into the single-text `LabeledExample` shape: `f"{left}{separator}{right}"`, pair label."""
    return [
        LabeledExample(
            text=f"{pair.left_name}{separator}{pair.right_name}", label=pair.label
        )
        for pair in pairs
    ]


def pair_split_to_labeled_split(
    split: PairDatasetSplit, separator: str = PAIR_TEXT_SEPARATOR
) -> DatasetSplit:
    """Adapt a `PairDatasetSplit` into `DatasetSplit`'s single-text shape, slice by slice."""
    return DatasetSplit(
        train=pairs_to_labeled_examples(split.train, separator),
        validation=pairs_to_labeled_examples(split.validation, separator),
        test=pairs_to_labeled_examples(split.test, separator),
    )


def cross_train_pair_baseline(
    split: PairDatasetSplit,
    models: dict[str, ClassifierModel],
    separator: str = PAIR_TEXT_SEPARATOR,
) -> dict[str, TrainedModelResult]:
    """Benchmark the single-text classifier models against the same pairs and split.

    Runs `models` (typically `TfidfLogRegClassifier`/`TfidfMlpClassifier`) through
    `train.cross_train()`, unmodified, over `split` adapted via `pair_split_to_labeled_split()`.
    The resulting `TrainedModelResult.validation_metrics`/`test_metrics` are the pair
    classifier's baseline: same pairs, same split, same `MetricBundle` shape as
    `cross_train_pairs()`'s output.
    """
    return cross_train(pair_split_to_labeled_split(split, separator), models)
