import numpy as np
import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntitySplitter,
    PairRecord,
    PairSource,
    PooledSubwordContrastiveEncoder,
)

SPLITTER = EntitySplitter(seed=42)

_BASE_NAMES = [
    "acme systems",
    "blue river technologies",
    "northshore trading",
    "westfield logistics",
    "meridian group",
    "summit capital partners",
    "cedar valley holdings",
    "ironbridge freight",
    "lighthouse consulting",
    "harbor point ventures",
]
_LEGAL_SUFFIXES = ["limited", "ltd", "gmbh", "llc", "plc", "inc"]

UNK_TOKEN = "[UNK]"


def _whitespace_vocab() -> dict[str, int]:
    """A word-level "subword" vocabulary: real enough to pool/train over without pulling in a
    real WordPiece/SentencePiece artifact, which the direct test has no business training."""
    words = {word for name in _BASE_NAMES for word in name.split()}
    words.update(_LEGAL_SUFFIXES)
    vocab = {word: index for index, word in enumerate(sorted(words))}
    vocab[UNK_TOKEN] = len(vocab)
    return vocab


def _encoder(
    *,
    vocab: dict[str, int] | None = None,
    embed_dim: int = 8,
    learning_rate: float = 0.1,
    epochs: int = 20,
    random_state: int = 3,
) -> PooledSubwordContrastiveEncoder:
    return PooledSubwordContrastiveEncoder(
        vocab=_whitespace_vocab() if vocab is None else vocab,
        encode=str.split,
        unk_token=UNK_TOKEN,
        embed_dim=embed_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        random_state=random_state,
    )


def _pairs() -> list[PairRecord]:
    """Positives: a base name against a different legal-suffix variant of itself. Negatives:
    that same base name against a different company's name entirely."""
    pairs: list[PairRecord] = []
    for index, base in enumerate(_BASE_NAMES):
        uri = f"co://{index:04d}"
        other_uri = f"co://{(index + 1) % len(_BASE_NAMES):04d}"
        other_base = _BASE_NAMES[(index + 1) % len(_BASE_NAMES)]
        anchor = f"{base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}"
        variant = f"{base} {_LEGAL_SUFFIXES[(index + 1) % len(_LEGAL_SUFFIXES)]}"
        split = SPLITTER.assign(uri)

        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=variant,
                label=MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=uri,
            )
        )
        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=f"{other_base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}",
                label=NON_MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=other_uri,
            )
        )
    return pairs


def test_embed_returns_zero_vector_for_all_unknown_tokens() -> None:
    encoder = _encoder(vocab={})
    assert np.array_equal(encoder.embed("nothing recognized here"), np.zeros(8))


def test_embed_returns_mean_pooled_vector_over_known_tokens() -> None:
    encoder = _encoder()
    vocab = encoder.vocab
    expected = encoder._embeddings[[vocab["acme"], vocab["systems"]]].mean(axis=0)
    assert np.allclose(encoder.embed("acme systems"), expected)


def test_fit_with_empty_pairs_is_a_noop() -> None:
    encoder = _encoder()
    before = encoder._embeddings.copy()
    encoder.fit([])
    assert np.array_equal(before, encoder._embeddings)


def test_predict_scores_returns_one_value_per_pair_in_unit_interval() -> None:
    encoder = _encoder()
    scores = encoder.predict_scores(
        [
            ("acme systems limited", "acme systems ltd"),
            ("acme systems limited", "harbor point ventures inc"),
        ]
    )
    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0


@pytest.mark.integration
def test_encoder_separates_match_from_non_match_pairs_after_training() -> None:
    """Not a benchmark -- confirms the contrastive-loss wiring actually pulls a match pair's
    pooled vectors together and pushes a non-match pair's apart, on clearly separable data."""
    encoder = _encoder(embed_dim=8, learning_rate=0.2, epochs=40, random_state=1)
    encoder.fit(_pairs())

    match_score, non_match_score = encoder.predict_scores(
        [
            ("acme systems limited", "acme systems ltd"),
            ("acme systems limited", "harbor point ventures inc"),
        ]
    )

    assert match_score > non_match_score


def test_fit_tolerates_pairs_with_no_recognized_tokens() -> None:
    encoder = _encoder(vocab={})
    encoder.fit(_pairs())  # must not raise, even though every token is unrecognized
    scores = encoder.predict_scores([("acme systems limited", "acme systems ltd")])
    assert scores == [0.5]
