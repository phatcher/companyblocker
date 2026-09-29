from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Protocol


class Vectorizer(Protocol):
    """Interface for vectorization backends."""

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddedRow:
    entity_id: str
    text: str
    vector: list[float]


@dataclass(frozen=True)
class MultiEmbeddedRow:
    entity_id: str
    texts: dict[str, str]
    vectors: dict[str, list[float]]


def l2_normalize(vector: list[float]) -> list[float]:
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector.copy()
    return [value / magnitude for value in vector]
