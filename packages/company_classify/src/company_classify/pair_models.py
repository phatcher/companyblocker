"""A TF-IDF pair rescorer over generated training pairs, kept as a baseline.

`TfidfPairMlpClassifier` reuses the cheapest representation already in this repo (a
character-n-gram TF-IDF vectorizer, the same one `models.py`'s single-text classifiers build) and
adds nothing new: no pretrained embedding, no differentiable-training framework. A pair's two
names are vectorized independently through one shared vectorizer, combined into a single feature
row (`PairFeatureMode`), and classified match/non-match by an `MLPClassifier`.

It is modelled on DeepBlocker's CTT scheme but is not that scheme. CTT trains a Siamese summarizer
and classifies the absolute difference of two summarized tuple vectors; here the shared vectorizer
carries no trained parameters and the only trained component sits after the pair is merged, so
there is no trainable shared branch and no extractable embedding, and the model can only rescore
pairs another stage generated. It is kept as the baseline the trained encoders are measured
against.

`PairClassifierModel.predict_scores()` takes raw `(left, right)` name tuples rather than
`PairRecord` values: scoring a candidate pair needs only the two texts, not the entity/split
metadata `fit()`'s labelled training pairs carry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import scipy.sparse as sp
from sklearn.neural_network import MLPClassifier

from .models import build_char_ngram_vectorizer
from .pairs import PairRecord


class PairFeatureMode(StrEnum):
    """How a pair's two vectorized names are combined into one classifier feature row.

    DIFFERENCE: the signed element-wise difference `left - right`. Cheapest, but it collapses each
        side's own magnitude into a single delta. DeepBlocker's CTT takes the absolute difference
        of two trained summaries instead, a different feature over different vectors.
    CONCAT: `left` and `right` side by side. Costs twice the input width, but keeps each name's
        own representation available to the classifier, not just their difference.
    """

    DIFFERENCE = "difference"
    CONCAT = "concat"


def _pair_features(
    left_matrix: sp.spmatrix, right_matrix: sp.spmatrix, mode: PairFeatureMode
) -> sp.spmatrix:
    if mode is PairFeatureMode.DIFFERENCE:
        return left_matrix - right_matrix
    if mode is PairFeatureMode.CONCAT:
        return sp.hstack([left_matrix, right_matrix]).tocsr()
    raise ValueError(f"unsupported feature mode: {mode!r}")


class PairClassifierModel(Protocol):
    """What every Tier 1 pair classifier exposes, mirroring `models.ClassifierModel`'s shape."""

    def fit(self, pairs: Sequence[PairRecord]) -> None: ...

    def predict_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


@dataclass
class TfidfPairMlpClassifier:
    """TF-IDF character-n-gram + MLP pair rescorer, a baseline with no trainable shared branch.

    Implements `PairClassifierModel`: `fit()` builds one shared character-n-gram
    `TfidfVectorizer` (`analyzer="char_wb"`) over both sides of every training pair, combines
    each pair's two vectors per `feature_mode`, and fits an `MLPClassifier` on the result;
    `predict_scores()` returns the match-class probability for each `(left, right)` name tuple.

    Attributes:
        max_features: Maximum vocabulary size for the underlying `TfidfVectorizer`.
        ngram_min: Minimum character n-gram length (inclusive) used by the vectorizer's
            `ngram_range`.
        ngram_max: Maximum character n-gram length (inclusive) used by the vectorizer's
            `ngram_range`.
        hidden_layer_sizes: Sizes of each hidden layer passed to
            `sklearn.neural_network.MLPClassifier`.
        random_state: Seed passed to `MLPClassifier` for reproducible fitting.
        feature_mode: How a pair's two vectors are combined; see `PairFeatureMode`.
    """

    max_features: int = 20000
    ngram_min: int = 2
    ngram_max: int = 5
    hidden_layer_sizes: tuple[int, ...] = (128,)
    random_state: int = 42
    feature_mode: PairFeatureMode = PairFeatureMode.CONCAT

    def __post_init__(self) -> None:
        self._vectorizer = build_char_ngram_vectorizer(
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            max_features=self.max_features,
        )
        self._classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=300,
            random_state=self.random_state,
        )

    def fit(self, pairs: Sequence[PairRecord]) -> None:
        left_texts = [pair.left_name for pair in pairs]
        right_texts = [pair.right_name for pair in pairs]
        labels = [pair.label for pair in pairs]

        self._vectorizer.fit(left_texts + right_texts)
        features = _pair_features(
            self._vectorizer.transform(left_texts),
            self._vectorizer.transform(right_texts),
            self.feature_mode,
        )
        self._classifier.fit(features, labels)

    def predict_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        left_texts = [left for left, _ in pairs]
        right_texts = [right for _, right in pairs]
        features = _pair_features(
            self._vectorizer.transform(left_texts),
            self._vectorizer.transform(right_texts),
            self.feature_mode,
        )
        probabilities = self._classifier.predict_proba(features)
        return [float(row[1]) for row in probabilities]
