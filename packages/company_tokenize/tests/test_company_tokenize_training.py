from pathlib import Path

import polars as pl
import pytest
from company_tokenize.training import (
    estimate_wordpiece_vocab_size,
    load_tokenizer_encoder,
    train_sentencepiece,
    train_wordpiece,
)
from tokenizers import Tokenizer


def test_train_wordpiece_raises_when_corpus_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Training corpus parquet not found"):
        train_wordpiece(tmp_path / "missing.parquet", tmp_path / "tokenizer.json")


def test_estimate_wordpiece_vocab_size_uses_unique_tokens_and_clamps(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "alpha holdings"]}
    ).write_parquet(corpus_path)

    vocab_size = estimate_wordpiece_vocab_size(
        corpus_path,
        multiplier=1.5,
        min_vocab_size=1,
        max_vocab_size=10,
    )

    assert vocab_size == 5


def test_estimate_wordpiece_vocab_size_raises_when_corpus_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Training corpus parquet not found"):
        estimate_wordpiece_vocab_size(tmp_path / "missing.parquet")


def test_train_wordpiece_writes_tokenizer_file(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"

    resolved_vocab_size = train_wordpiece(
        corpus_path, tokenizer_path, vocab_size=50, show_progress=False
    )

    assert resolved_vocab_size == 50
    assert tokenizer_path.exists()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = tokenizer.encode("betacorp")
    assert encoded.tokens


def test_train_wordpiece_auto_computes_vocab_size(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "beta holdings"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"

    resolved_vocab_size = train_wordpiece(
        corpus_path, tokenizer_path, vocab_size=None, show_progress=False
    )

    assert resolved_vocab_size == estimate_wordpiece_vocab_size(corpus_path)
    assert tokenizer_path.exists()


def test_load_tokenizer_encoder_for_wordpiece(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "beta holdings"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"
    train_wordpiece(corpus_path, tokenizer_path, vocab_size=64, show_progress=False)

    encode_tokens, unk_token = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer="wordpiece"
    )
    tokens = encode_tokens("beta holdings")

    assert unk_token == "[UNK]"
    assert tokens


def test_train_sentencepiece_and_load_encoder(tmp_path: Path):
    pytest.importorskip("sentencepiece")
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "beta holdings"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "sentencepiece.model"

    resolved_vocab_size = train_sentencepiece(
        corpus_path,
        tokenizer_path,
        vocab_size=64,
        model_type="unigram",
    )

    encode_tokens, unk_token = load_tokenizer_encoder(
        tokenizer_path=tokenizer_path, trainer="sentencepiece"
    )
    tokens = encode_tokens("beta holdings")

    assert resolved_vocab_size == 64
    assert tokenizer_path.exists()
    assert isinstance(unk_token, str)
    assert tokens
