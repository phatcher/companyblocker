from __future__ import annotations

import hashlib

from .base import l2_normalize


class HashVectorizer:
    """Deterministic lightweight vectorizer for testing and local scaffolding."""

    def __init__(self, *, dimension: int = 128, normalize: bool = True):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        values: list[float] = []
        seed = text.encode("utf-8")
        counter = 0

        while len(values) < self._dimension:
            digest = hashlib.blake2b(
                seed + counter.to_bytes(4, "little"), digest_size=32
            ).digest()
            for i in range(0, len(digest), 4):
                if len(values) >= self._dimension:
                    break
                chunk = digest[i : i + 4]
                # Map uint32-ish bytes into roughly [-1, 1].
                number = int.from_bytes(chunk, "little") / 2_147_483_648.0 - 1.0
                values.append(number)
            counter += 1

        if self._normalize:
            return l2_normalize(values)
        return values
