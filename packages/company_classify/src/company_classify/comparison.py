"""Shared supervised-vs-self-supervised comparison harness over one fixed split.

`compare_pair_models()` is the one seam every model-family comparison in this package goes
through: both groups run through `pair_train.cross_train_pairs()` over the identical
`PairDatasetSplit`, so a supervised model (fit against labelled pairs, e.g.
`pair_models.TfidfPairMlpClassifier`) and a self-supervised one (a representation shaped by its
own training signal rather than a label, e.g. `encoder_models.PooledSubwordContrastiveEncoder`)
are always scored on the same pairs and the same partition. That single shared `PairDatasetSplit`
argument is the split protocol this harness fixes: there is no code path here that builds a
second split for either side, so a supervised/self-supervised difference in the result is never
explainable by a different sample either side happened to see. `ModelCategory` and
`ComparisonReport`'s two dicts are bookkeeping this harness adds on top of
`cross_train_pairs()`'s flat model dict, not a second evaluation path -- fitting, scoring and
`MetricBundle` computation are unchanged.

`check_repeatability()` is the other half: refitting the same model spec twice, from a fresh
unfitted instance each time, over the identical split, and reporting how far the two runs'
`MetricBundle`s land from each other. Every model already shipped in this package seeds its own
randomness (`TfidfPairMlpClassifier.random_state`, `PooledSubwordContrastiveEncoder.random_state`),
so two clean fits are expected to land within a caller's tolerance of each other; a spec that does
not surfaces here as `reproducible=False` rather than being silently trusted.

A runtime-budget/timeout policy is deliberately not built into this module: a budget policy wraps
`compare_pair_models()` from outside it, a separate concern from the comparison and repeatability
primitives here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum

from .pair_models import PairClassifierModel
from .pair_train import PairDatasetSplit, TrainedPairModelResult, cross_train_pairs
from .schema import MetricBundle

_REPEATABILITY_FIELDS = tuple(
    field.name for field in fields(MetricBundle) if field.name != "latency_per_1k_rows"
)
"""Every `MetricBundle` field `check_repeatability()` compares, except `latency_per_1k_rows`:
wall-clock timing genuinely varies between two otherwise-identical fits (scheduling noise, cache
state), so including it would flag a perfectly deterministic model as unreproducible on nothing
but measurement jitter."""


class ModelCategory(StrEnum):
    """Which family a `compare_pair_models()` model belongs to."""

    SUPERVISED = "supervised"
    SELF_SUPERVISED = "self_supervised"


@dataclass(frozen=True)
class ComparisonReport:
    """Supervised and self-supervised `TrainedPairModelResult`s over one shared `PairDatasetSplit`.

    Both dicts have the shape `cross_train_pairs()` already returns (name ->
    `TrainedPairModelResult`); this is that same output grouped by category rather than merged
    into one dict, so a caller never has to know which family a name belongs to from its string
    alone.

    Attributes:
        supervised: name -> `TrainedPairModelResult`, one per model `compare_pair_models()` was
            given as `supervised`.
        self_supervised: name -> `TrainedPairModelResult`, one per model given as
            `self_supervised`.
    """

    supervised: dict[str, TrainedPairModelResult]
    self_supervised: dict[str, TrainedPairModelResult]

    def _pool(self, category: ModelCategory) -> dict[str, TrainedPairModelResult]:
        return (
            self.supervised
            if category is ModelCategory.SUPERVISED
            else self.self_supervised
        )

    def best(
        self, category: ModelCategory, *, metric: str = "f1"
    ) -> tuple[str, TrainedPairModelResult]:
        """The highest-`test_metrics.<metric>` model in `category`; ties keep dict order (first wins).

        Raises `ValueError` if `category` has no models in this report.
        """
        pool = self._pool(category)
        if not pool:
            raise ValueError(f"no {category.value} models in this report")
        name = max(
            pool, key=lambda candidate: getattr(pool[candidate].test_metrics, metric)
        )
        return name, pool[name]

    def delta(self, *, metric: str = "f1") -> float:
        """Best self-supervised minus best supervised `test_metrics.<metric>`; positive favours self-supervised."""
        _, supervised_best = self.best(ModelCategory.SUPERVISED, metric=metric)
        _, self_supervised_best = self.best(
            ModelCategory.SELF_SUPERVISED, metric=metric
        )
        return getattr(self_supervised_best.test_metrics, metric) - getattr(
            supervised_best.test_metrics, metric
        )


def compare_pair_models(
    split: PairDatasetSplit,
    *,
    supervised: Mapping[str, PairClassifierModel],
    self_supervised: Mapping[str, PairClassifierModel],
) -> ComparisonReport:
    """Fit and evaluate `supervised` and `self_supervised` pair models over the identical `split`.

    Raises `ValueError` if either group is empty: a comparison needs both sides present, since an
    empty side would make `ComparisonReport.best()`/`delta()` meaningless.
    """
    if not supervised:
        raise ValueError("supervised must not be empty")
    if not self_supervised:
        raise ValueError("self_supervised must not be empty")

    supervised_results = cross_train_pairs(split, dict(supervised))
    self_supervised_results = cross_train_pairs(split, dict(self_supervised))
    return ComparisonReport(
        supervised=supervised_results, self_supervised=self_supervised_results
    )


def _max_abs_diff(first: MetricBundle, second: MetricBundle) -> float:
    return max(
        abs(getattr(first, name) - getattr(second, name))
        for name in _REPEATABILITY_FIELDS
    )


@dataclass(frozen=True)
class RepeatabilityResult:
    """Whether two independent fits of the same model spec reproduce the same `test_metrics`.

    Attributes:
        name: The model's key, as passed to `check_repeatability()`.
        first: `test_metrics` from the first fresh fit.
        second: `test_metrics` from the second fresh fit.
        max_abs_diff: Largest absolute difference between `first` and `second` across every
            `MetricBundle` field except `latency_per_1k_rows` (wall-clock timing, expected to
            vary between two fits even of a fully deterministic model).
        reproducible: Whether `max_abs_diff` is within the caller's `tolerance`.
    """

    name: str
    first: MetricBundle
    second: MetricBundle
    max_abs_diff: float
    reproducible: bool


def check_repeatability(
    split: PairDatasetSplit,
    model_factories: Mapping[str, Callable[[], PairClassifierModel]],
    *,
    tolerance: float = 1e-9,
) -> dict[str, RepeatabilityResult]:
    """Fit two fresh instances of each model spec over the identical `split` and compare `test_metrics`.

    `model_factories` builds a fresh, unfitted model per call rather than reusing one mutable
    instance: `cross_train_pairs()` fits in place (`PairClassifierModel.fit()` mutates the model
    it is given), so replaying the same instance for a "second" run would keep fitting on top of
    the first run's state instead of proving anything about the model spec's own repeatability.

    Raises `ValueError` if `model_factories` is empty.
    """
    if not model_factories:
        raise ValueError("model_factories must not be empty")

    results: dict[str, RepeatabilityResult] = {}
    for name, factory in model_factories.items():
        first = cross_train_pairs(split, {name: factory()})[name].test_metrics
        second = cross_train_pairs(split, {name: factory()})[name].test_metrics
        diff = _max_abs_diff(first, second)
        results[name] = RepeatabilityResult(
            name=name,
            first=first,
            second=second,
            max_abs_diff=diff,
            reproducible=diff <= tolerance,
        )
    return results
