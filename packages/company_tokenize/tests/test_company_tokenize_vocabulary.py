from pathlib import Path

import polars as pl
import pytest
from company_tokenize.tokenization import tokenize_name_dataframe
from company_tokenize.training import train_sentencepiece, train_wordpiece
from company_tokenize.vocabulary import TokenizerVocabulary, load_tokenizer_vocabulary
from tokenizers import Tokenizer


def _write_corpus(path: Path) -> None:
    pl.DataFrame(
        {
            "system_uri": ["s:1", "s:2", "s:3"],
            "name": [
                "alpha limited",
                "beta holdings incorporated",
                "gamma trading company",
            ],
        }
    ).write_parquet(path)


def test_load_tokenizer_vocabulary_raises_when_tokenizer_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
        load_tokenizer_vocabulary(tmp_path / "missing.json", trainer="wordpiece")


def test_load_tokenizer_vocabulary_for_wordpiece(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    _write_corpus(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"
    train_wordpiece(corpus_path, tokenizer_path, vocab_size=64, show_progress=False)

    result = load_tokenizer_vocabulary(tokenizer_path, trainer="wordpiece")

    assert isinstance(result, TokenizerVocabulary)
    assert result.trainer == "wordpiece"
    assert result.tokenizer_path == tokenizer_path
    assert result.unk_token == "[UNK]"
    assert result.vocab == Tokenizer.from_file(str(tokenizer_path)).get_vocab()
    assert isinstance(result.noise_words, frozenset)
    assert result.noise_words  # packaged default profile is non-empty
    assert result.encode("beta holdings")


def test_load_tokenizer_vocabulary_for_sentencepiece(tmp_path: Path):
    pytest.importorskip("sentencepiece")
    corpus_path = tmp_path / "corpus.parquet"
    _write_corpus(corpus_path)
    tokenizer_path = tmp_path / "sentencepiece.model"
    train_sentencepiece(
        corpus_path, tokenizer_path, vocab_size=64, model_type="unigram"
    )

    result = load_tokenizer_vocabulary(tokenizer_path, trainer="sentencepiece")

    assert result.trainer == "sentencepiece"
    assert result.vocab
    assert isinstance(result.unk_token, str)
    assert result.encode("gamma trading")


def test_load_tokenizer_vocabulary_respects_explicit_noise_words(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    _write_corpus(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"
    train_wordpiece(corpus_path, tokenizer_path, vocab_size=64, show_progress=False)

    result = load_tokenizer_vocabulary(
        tokenizer_path, trainer="wordpiece", noise_words=["limited", "holdings"]
    )

    assert result.noise_words == frozenset({"limited", "holdings"})


def test_load_tokenizer_vocabulary_matches_tokenize_stage_encoding(tmp_path: Path):
    """Fixture proving the same trained artifact drives the tokenize stage's
    dataframe path and this non-pipeline loading API identically: a caller
    outside the pipeline gets the same encoding, not merely a similar one."""
    corpus_path = tmp_path / "corpus.parquet"
    _write_corpus(corpus_path)
    tokenizer_path = tmp_path / "wordpiece.json"
    train_wordpiece(corpus_path, tokenizer_path, vocab_size=64, show_progress=False)

    explicit_noise_words = {"limited", "incorporated"}
    names = ["alpha limited", "beta holdings incorporated", "gamma trading company"]
    df = pl.DataFrame({"name_cleansed": names})

    tokenize_stage_result = tokenize_name_dataframe(
        df,
        tokenizer_path=tokenizer_path,
        trainer="wordpiece",
        noise_words=explicit_noise_words,
    )
    tokenize_stage_tokens = tokenize_stage_result["name_tokens"].to_list()

    vocabulary = load_tokenizer_vocabulary(
        tokenizer_path, trainer="wordpiece", noise_words=explicit_noise_words
    )
    assert vocabulary.noise_words == frozenset(explicit_noise_words)

    from company_tokenize.tfidf import trim_text_before_tokenization

    trimmed_names = trim_text_before_tokenization(
        names, noise_words=vocabulary.noise_words
    )
    non_pipeline_tokens = [vocabulary.encode(name) for name in trimmed_names]

    assert non_pipeline_tokens == tokenize_stage_tokens

    # And the vocabulary loaded through the non-pipeline path is exactly
    # what backs the tokenize stage's own encoder for the same artifact.
    assert vocabulary.vocab == Tokenizer.from_file(str(tokenizer_path)).get_vocab()
