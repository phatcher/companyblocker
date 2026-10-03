"""A token-to-vector lookup, and the pretrained fastText checkpoint that serves the first one.

`TokenVectorLookup` is the interface both the pretrained ingredient here and a domain-trained
ingredient added later implement, so a consumer -- a frequency-weighted aggregation over
static word vectors, the alias-pair probe below -- reads a name's tokens through one shape and
cannot tell which ingredient it was given except by the `TokenVectorProvenance` it carries. A
vector, a raw occurrence count, and the checkpoint's total are exposed together per token
because a frequency-weighted aggregation needs a token probability, `count / total`, beside
the vector it weights -- an ingredient that only served vectors would push every consumer back
to deriving its own word-frequency table.

`PretrainedFastTextVectors` wraps a `gensim.models.fasttext.FastTextKeyedVectors` loaded
through `load_facebook_vectors` -- never the raw `fasttext` package, which publishes no
prebuilt Windows wheel. That loader keeps the checkpoint's subword vectors, so a word it never
saw during training still embeds through its character n-grams rather than falling back to a
zero vector; `mean_pool_name_vector()` pools that per-word behaviour up to a whole name by
splitting on whitespace and averaging, the same unweighted pooling
`encoder_models.PooledSubwordContrastiveEncoder.embed()` uses over subword tokens, ahead of a
frequency-weighted version of the same aggregation added later.

`load_pretrained_fasttext_vectors()` takes a plain local file path, never a URL or a gzip
archive: `gensim`'s docs advertise `smart_open`-transparent gzip reading, but the real
`en-cc-300` checkpoint (Facebook's Common Crawl English release, ~6.7 GiB decompressed --
past gzip's 4 GiB `ISIZE` trailer field, per this item's dispatch) read short through that
path (`gensim._fasttext_bin._load_matrix`'s vector-count assertion failed, short by tens of
millions of floats) even though the downloaded archive's own gzip integrity check passes. The
caller (this item's download script) decompresses the downloaded archive to a plain `.bin`
once and points this function at that file, rather than this module carrying a compressed-read
path that is untrustworthy at this file size.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class TokenVectorProvenance:
    """What a `TokenVectorLookup`'s vectors were served from.

    Attributes:
        slug: The registry slug the checkpoint was resolved from (`FastTextCheckpointEntry.slug`
            for the pretrained ingredient), or a caller-chosen identifier for an ingredient with
            no registry entry.
        source: Where the checkpoint came from -- `FastTextCheckpointEntry.source_url` for the
            pretrained ingredient, or a corpus/training description for a domain-trained one.
        checksum: sha256 hex digest of the checkpoint file actually loaded, or `None` when the
            checkpoint's checksum was not verified before loading.
    """

    slug: str
    source: str
    checksum: str | None


class TokenVectorLookup(Protocol):
    """A token-to-vector lookup any static-vector ingredient implements.

    A consumer reads a name's tokens through this shape alone; which ingredient produced it is
    recoverable only from `provenance`, never from the interface.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def total(self) -> int: ...

    @property
    def provenance(self) -> TokenVectorProvenance: ...

    def vector(self, token: str) -> np.ndarray: ...

    def count(self, token: str) -> int: ...


class _KeyedVectorsLike(Protocol):
    """The `FastTextKeyedVectors` surface `PretrainedFastTextVectors` reads.

    Narrowed to what this module actually calls, so a test can inject a fake without
    constructing a real `gensim` model.
    """

    vector_size: int
    key_to_index: dict[str, int]

    def __getitem__(self, key: str) -> np.ndarray: ...

    def get_vecattr(self, key: str, attr: str) -> Any: ...

    def has_index_for(self, key: str) -> bool: ...


@dataclass(frozen=True)
class PretrainedFastTextVectors:
    """`TokenVectorLookup` over a loaded fastText checkpoint.

    Attributes:
        keyed_vectors: The loaded `FastTextKeyedVectors` (or a test fake matching
            `_KeyedVectorsLike`). `vector()` indexes it directly, which is what gives an
            out-of-vocabulary word its subword-composed vector rather than a zero one --
            `FastTextKeyedVectors.__getitem__` falls back to the checkpoint's character
            n-gram buckets for a word not in `key_to_index`.
        provenance: Where this checkpoint came from.
        total: Sum of `count(token)` over every in-vocabulary token, computed once at
            construction. The checkpoint's `.bin` format does not carry a corpus-token total
            directly reachable through `load_facebook_vectors`'s public return value (only the
            per-word vocabulary counts survive into it), so the total used here is the sum of
            those counts, consistent with how a consumer would derive `p(w)` from them anyway.
    """

    keyed_vectors: _KeyedVectorsLike
    provenance: TokenVectorProvenance
    total: int = field(init=False)

    def __post_init__(self) -> None:
        computed_total = sum(
            self.count(token) for token in self.keyed_vectors.key_to_index
        )
        object.__setattr__(self, "total", computed_total)

    @property
    def dimension(self) -> int:
        return int(self.keyed_vectors.vector_size)

    def vector(self, token: str) -> np.ndarray:
        """`token`'s vector, subword-composed when `token` is out of vocabulary."""
        return np.asarray(self.keyed_vectors[token], dtype=np.float64)

    def count(self, token: str) -> int:
        """`token`'s occurrence count in the checkpoint's training corpus, or 0 out of vocabulary.

        A subword-composed vector still exists for an out-of-vocabulary token (`vector()`
        above), but it has no observed count of its own -- 0 keeps `count / total` a valid
        (zero) probability rather than raising, so a frequency-based weighting scheme can give
        an unseen token its ceiling weight instead of dropping it.
        """
        if not self.keyed_vectors.has_index_for(token):
            return 0
        value = self.keyed_vectors.get_vecattr(token, "count")
        return int(value) if value is not None else 0


def _load_facebook_vectors(path: Path) -> Any:
    from gensim.models.fasttext import load_facebook_vectors

    return load_facebook_vectors(str(path))


def load_pretrained_fasttext_vectors(
    path: Path,
    *,
    provenance: TokenVectorProvenance,
    loader: Callable[[Path], _KeyedVectorsLike] = _load_facebook_vectors,
) -> PretrainedFastTextVectors:
    """Load a downloaded fastText checkpoint file and serve it through `TokenVectorLookup`.

    `path` is an already-downloaded, plain (not gzip-compressed) `.bin` checkpoint -- this
    function does not download or decompress anything itself; see the module docstring for why
    a caller should not hand this a `.bin.gz` archive directly on a checkpoint of real size.

    `loader` defaults to the real `gensim.models.fasttext.load_facebook_vectors`, imported
    lazily so importing this module never triggers a multi-gigabyte checkpoint load; a test
    injects a fake `_KeyedVectorsLike` here instead of constructing a real checkpoint file.
    """
    return PretrainedFastTextVectors(keyed_vectors=loader(path), provenance=provenance)


def default_name_tokenizer(name: str) -> list[str]:
    """Lowercased whitespace split -- the token granularity `TokenVectorLookup.vector()` embeds.

    A fastText checkpoint's own subword mechanism already makes an unseen *word* embed through
    its character n-grams (`PretrainedFastTextVectors.vector()`'s docstring), so pooling at
    word granularity, rather than through a trained tokenizer's own subword vocabulary,
    already gets robustness to unseen words for free and needs no tokenizer artifact of its
    own -- unlike `PooledSubwordContrastiveEncoder`, which pools over a *trained* subword
    vocabulary because it has no per-word subword fallback to rely on.
    """
    return [token for token in name.strip().lower().split() if token]


def mean_pool_name_vector(
    name: str,
    lookup: TokenVectorLookup,
    *,
    tokenize: Callable[[str], Iterable[str]] = default_name_tokenizer,
) -> np.ndarray:
    """`name`'s pooled embedding: the mean of its tokens' vectors under `lookup`.

    Returns an all-zero vector of `lookup.dimension` width for a name with no tokens (empty or
    whitespace-only), the same degenerate-input handling
    `PooledSubwordContrastiveEncoder.embed()` gives an all-unrecognized name.
    """
    tokens = list(tokenize(name))
    if not tokens:
        return np.zeros(lookup.dimension)
    vectors = np.stack([lookup.vector(token) for token in tokens])
    return vectors.mean(axis=0)


@dataclass(frozen=True)
class MeanPooledTokenVectorEncoder:
    """An encoder over a `TokenVectorLookup`: one name's vector is the mean of its tokens' vectors.

    Has the one-name-at-a-time `embed()` a blocking run's `"encoder"` representation scores
    through, so a static-vector ingredient (the pretrained fastText one here, a domain-trained
    one later) is a row in a blocking comparison without that area importing this package.
    `embed()` is `mean_pool_name_vector()` unchanged.

    Attributes:
        lookup: The token-vector ingredient the names are embedded under.
        tokenize: How a name is split into the tokens `lookup` is asked for; lowercased
            whitespace split by default.
    """

    lookup: TokenVectorLookup
    tokenize: Callable[[str], Iterable[str]] = default_name_tokenizer

    def embed(self, name: str) -> np.ndarray:
        """`name`'s mean-pooled vector, all zeros for a name with no tokens."""
        return mean_pool_name_vector(name, self.lookup, tokenize=self.tokenize)
