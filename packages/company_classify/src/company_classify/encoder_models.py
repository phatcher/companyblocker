"""A pooled-subword contrastive encoder, trained over an already-trained tokenizer's own tokens.

`PooledSubwordContrastiveEncoder` trains a genuine embedding table over an already-trained
tokenizer's own subword vocabulary (`company_tokenize.TokenizerVocabulary`, WordPiece or
SentencePiece) with a cosine contrastive loss. Training over already-tokenized subwords, rather
than a pretrained checkpoint's fixed embedding matrix, is what makes the tokenizer a real,
independently-choosable control on the resulting embedding space.

Unlike `pair_models.TfidfPairMlpClassifier` (a TF-IDF pair rescorer: character n-grams into a
downstream `MLPClassifier` head), this model has no classifier head at all. `embed()` returns a
name's raw pooled vector directly: mean-pooled over the embeddings of its tokenizer subwords, and
that pooled vector is what training shapes, not a match/non-match decision layered on top. The
result is directly usable as an indexable embedding space (a candidate-generation or ANN index
consuming raw vectors), which a classifier's scalar score is not.

It still implements `pair_models.PairClassifierModel` (`fit()`/`predict_scores()`), so it slots
into the existing `pair_train.cross_train_pairs()` harness unmodified and is directly comparable,
`MetricBundle` for `MetricBundle`, against another pair classifier's measured metrics:
`predict_scores()` maps a pair's cosine similarity into `[0, 1]` (`(cosine + 1) / 2`), so
`evaluate.compute_binary_metrics`'s default 0.5 threshold reads as "orthogonal is the
match/non-match boundary" -- the same boundary training's `margin` decision uses when
`margin=0.0`.

Training is plain, hand-rolled SGD over `numpy` (already a transitive dependency through
`scikit-learn`/`scipy`, not a new one): no differentiable-training framework is added here.
Pooling and contrastive-training a shallow embedding table over already-tokenized subwords is a
handful of vectorized operations and an analytic gradient; it does not need one. A fine-tuned
transformer encoder is a separate, larger step with its own dependency and its own trigger.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .pairs import PairRecord

_EPSILON = 1e-8


def _cosine_similarity(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, float, float]:
    """Return `(cosine, ||left||, ||right||)`, norms floored by `_EPSILON` to avoid division by zero."""
    norm_left = float(np.linalg.norm(left)) + _EPSILON
    norm_right = float(np.linalg.norm(right)) + _EPSILON
    cosine = float(np.dot(left, right) / (norm_left * norm_right))
    return cosine, norm_left, norm_right


@dataclass
class PooledSubwordContrastiveEncoder:
    """Pooled-subword contrastive encoder, trained over an already-trained tokenizer vocabulary.

    Implements `pair_models.PairClassifierModel`. `embed()` is what this model exists
    for: a name's pooled vector, usable as an indexable embedding on its own, not just an input
    to `predict_scores()`.

    Attributes:
        vocab: `token -> id` mapping from the trained tokenizer artifact this encoder pools
            over (`company_tokenize.TokenizerVocabulary.vocab`). Loading that artifact is the
            caller's job -- which tokenizer to train against is a training-pipeline decision,
            not this model's.
        encode: `text -> token list` callable for the same trained artifact
            (`TokenizerVocabulary.encode`).
        unk_token: The artifact's unknown-token marker (`TokenizerVocabulary.unk_token`),
            substituted for any token `encode()` returns that `vocab` does not contain.
        embed_dim: Width of the trained embedding vectors.
        margin: Contrastive-loss margin: a non-match pair's cosine similarity is only
            penalised once it exceeds this value.
        learning_rate: Plain SGD step size.
        epochs: Passes over the training pairs.
        batch_size: Pairs per gradient step (embeddings updated once per batch).
        random_state: Seed for embedding initialization and per-epoch shuffling.
    """

    vocab: dict[str, int]
    encode: Callable[[str], list[str]]
    unk_token: str
    embed_dim: int = 32
    margin: float = 0.0
    learning_rate: float = 0.1
    epochs: int = 20
    batch_size: int = 32
    random_state: int = 42
    _embeddings: np.ndarray = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_state)
        vocab_size = max(self.vocab.values(), default=-1) + 1
        self._embeddings = self._rng.normal(
            scale=1.0 / np.sqrt(self.embed_dim),
            size=(max(vocab_size, 1), self.embed_dim),
        )

    def _token_ids(self, name: str) -> list[int]:
        ids: list[int] = []
        for token in self.encode(name):
            token_id = self.vocab.get(token, self.vocab.get(self.unk_token))
            if token_id is not None:
                ids.append(token_id)
        return ids

    def embed(self, name: str) -> np.ndarray:
        """Return `name`'s pooled embedding: the mean of its subword embeddings.

        Returns an all-zero vector for a name whose tokens are all unrecognized (no token id
        resolves, including no `unk_token` entry in `vocab`) rather than raising -- the same
        degenerate case `predict_scores()` and training both already tolerate.
        """
        token_ids = self._token_ids(name)
        if not token_ids:
            return np.zeros(self.embed_dim)
        return self._embeddings[token_ids].mean(axis=0)

    def fit(self, pairs: Sequence[PairRecord]) -> None:
        pair_list = list(pairs)
        if not pair_list:
            return

        indices = np.arange(len(pair_list))
        for _ in range(self.epochs):
            self._rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                self._train_batch([pair_list[index] for index in batch_indices])

    def _train_batch(self, batch: Sequence[PairRecord]) -> None:
        gradients: dict[int, np.ndarray] = {}

        def accumulate(token_ids: list[int], gradient: np.ndarray) -> None:
            if not token_ids:
                return
            per_token_gradient = gradient / len(token_ids)
            for token_id in token_ids:
                existing = gradients.get(token_id)
                gradients[token_id] = (
                    per_token_gradient
                    if existing is None
                    else existing + per_token_gradient
                )

        for pair in batch:
            left_ids = self._token_ids(pair.left_name)
            right_ids = self._token_ids(pair.right_name)
            if not left_ids or not right_ids:
                continue

            left_vector = self._embeddings[left_ids].mean(axis=0)
            right_vector = self._embeddings[right_ids].mean(axis=0)
            cosine, norm_left, norm_right = _cosine_similarity(
                left_vector, right_vector
            )

            if pair.is_match:
                d_loss_d_cosine = -2.0 * (1.0 - cosine)
            else:
                excess = cosine - self.margin
                d_loss_d_cosine = 2.0 * excess if excess > 0.0 else 0.0
            if d_loss_d_cosine == 0.0:
                continue

            d_cosine_d_left = right_vector / (
                norm_left * norm_right
            ) - cosine * left_vector / (norm_left**2)
            d_cosine_d_right = left_vector / (
                norm_left * norm_right
            ) - cosine * right_vector / (norm_right**2)

            accumulate(left_ids, d_loss_d_cosine * d_cosine_d_left)
            accumulate(right_ids, d_loss_d_cosine * d_cosine_d_right)

        for token_id, gradient in gradients.items():
            self._embeddings[token_id] -= self.learning_rate * gradient

    def predict_scores(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for left_name, right_name in pairs:
            cosine, _, _ = _cosine_similarity(
                self.embed(left_name), self.embed(right_name)
            )
            scores.append((cosine + 1.0) / 2.0)
        return scores
