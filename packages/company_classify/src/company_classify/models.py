"""Single-text validity classifiers behind one `ClassifierModel` protocol.

The neural baseline is scikit-learn's `MLPClassifier`, a baseline for comparison rather than a
production model. No model here is calibrated, so a threshold is checked on a holdout slice of
the system it will score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from .schema import LabeledExample


class ClassifierModel(Protocol):
    def fit(self, examples: list[LabeledExample]) -> None: ...

    def predict_scores(self, texts: list[str]) -> list[float]: ...


def build_char_ngram_vectorizer(
    *, ngram_min: int, ngram_max: int, max_features: int
) -> TfidfVectorizer:
    """The character n-gram TF-IDF vectorizer every classifier here builds on.

    Word-bounded (`char_wb`) and lowercased, over `ngram_min` to `ngram_max`
    characters, keeping at most `max_features` n-grams.
    """
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        lowercase=True,
        max_features=max_features,
    )


@dataclass
class TfidfLogRegClassifier:
    """TF-IDF character-n-gram + logistic-regression baseline classifier.

    Implements `ClassifierModel`: `fit()` builds a character-n-gram
    `TfidfVectorizer` (`analyzer="char_wb"`) from the training texts and
    fits a `LogisticRegression` classifier on top of it; `predict_scores()`
    returns the positive-class probability for each input text.

    Attributes:
        max_features: Maximum vocabulary size for the underlying
            `TfidfVectorizer`. Larger values capture more distinct
            character n-grams at the cost of memory and fit time.
        ngram_min: Minimum character n-gram length (inclusive) used by the
            vectorizer's `ngram_range`.
        ngram_max: Maximum character n-gram length (inclusive) used by the
            vectorizer's `ngram_range`.
        random_state: Seed passed to `LogisticRegression` for reproducible
            fitting.
    """

    max_features: int = 20000
    ngram_min: int = 2
    ngram_max: int = 5
    random_state: int = 42

    def __post_init__(self) -> None:
        self._vectorizer = build_char_ngram_vectorizer(
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            max_features=self.max_features,
        )
        self._classifier = LogisticRegression(
            max_iter=1000, random_state=self.random_state
        )

    def fit(self, examples: list[LabeledExample]) -> None:
        texts = [example.text for example in examples]
        labels = [example.label for example in examples]
        matrix = self._vectorizer.fit_transform(texts)
        self._classifier.fit(matrix, labels)

    def predict_scores(self, texts: list[str]) -> list[float]:
        matrix = self._vectorizer.transform(texts)
        probabilities = self._classifier.predict_proba(matrix)
        return [float(row[1]) for row in probabilities]


@dataclass
class TfidfMlpClassifier:
    """TF-IDF character-n-gram + MLP neural baseline classifier.

    Implements `ClassifierModel`: `fit()` builds a character-n-gram
    `TfidfVectorizer` (`analyzer="char_wb"`) from the training texts and
    fits an `MLPClassifier` on top of it; `predict_scores()` returns the
    positive-class probability for each input text.

    Attributes:
        max_features: Maximum vocabulary size for the underlying
            `TfidfVectorizer`. Larger values capture more distinct
            character n-grams at the cost of memory and fit time.
        ngram_min: Minimum character n-gram length (inclusive) used by the
            vectorizer's `ngram_range`.
        ngram_max: Maximum character n-gram length (inclusive) used by the
            vectorizer's `ngram_range`.
        hidden_layer_sizes: Sizes of each hidden layer passed to
            `sklearn.neural_network.MLPClassifier`, e.g. `(128,)` for a
            single hidden layer of 128 units, or `(128, 64)` for two.
        random_state: Seed passed to `MLPClassifier` for reproducible
            fitting.
    """

    max_features: int = 20000
    ngram_min: int = 2
    ngram_max: int = 5
    hidden_layer_sizes: tuple[int, ...] = (128,)
    random_state: int = 42

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

    def fit(self, examples: list[LabeledExample]) -> None:
        texts = [example.text for example in examples]
        labels = [example.label for example in examples]
        matrix = self._vectorizer.fit_transform(texts)
        self._classifier.fit(matrix, labels)

    def predict_scores(self, texts: list[str]) -> list[float]:
        matrix = self._vectorizer.transform(texts)
        probabilities = self._classifier.predict_proba(matrix)
        return [float(row[1]) for row in probabilities]
