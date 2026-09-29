import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from company_tokenize import tokenize_name_dataframe
from company_tokenize.manifest import build_run_manifest
from company_tokenize.tokenization import (
    tokenize_name_dataframe_dual,
    tokenize_name_dataframe_multi,
)
from company_tokenize.training import (
    LoadedTokenizerBackend,
    train_sentencepiece,
    train_wordpiece,
)


def test_tokenize_name_dataframe_uses_packaged_tokenizer_when_present():
    df = pl.DataFrame(
        {
            "name_cleansed": ["alpha ltd", "betacorp", None],
            "company_type": ["ltd", "ltd", "ltd"],
        }
    )

    actual = tokenize_name_dataframe(df, noise_words_profile="aggressive")

    assert actual["name_tokens"].to_list()[0] == ["alpha"]
    assert isinstance(actual["name_tokens"].to_list()[1], list)
    assert actual["name_tokens"].to_list()[1]
    assert all(" " not in token for token in actual["name_tokens"].to_list()[1])
    assert actual["name_tokens"].to_list()[2] == []
    assert actual["company_type"].to_list() == ["ltd", "ltd", "ltd"]


def test_tokenize_name_dataframe_raises_when_column_missing():
    df = pl.DataFrame({"other": ["x"]})

    with pytest.raises(ValueError, match="name_cleansed"):
        tokenize_name_dataframe(df)


def test_tokenize_name_dataframe_writes_custom_token_column():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})

    actual = tokenize_name_dataframe(
        df, token_col="country_tokens", noise_words_profile="aggressive"
    )

    assert "country_tokens" in actual.columns
    assert "name_tokens" not in actual.columns
    assert actual["country_tokens"].to_list() == [["alpha"]]


def test_tokenize_name_dataframe_dual_writes_country_and_global_columns():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]})

    actual = tokenize_name_dataframe_dual(df, noise_words_profile="aggressive")

    assert "country_tokens" in actual.columns
    assert "global_tokens" in actual.columns
    assert "name_tokens" not in actual.columns
    assert actual["country_tokens"].to_list()[0] == ["alpha"]
    assert actual["global_tokens"].to_list()[0] == ["alpha"]


def test_tokenize_name_dataframe_preserves_existing_non_token_columns():
    df = pl.DataFrame(
        {
            "name_cleansed": ["alpha ltd"],
            "source_record_uri": ["ie://4"],
            "custom_payload": ["keep-me"],
        }
    )

    actual = tokenize_name_dataframe(df)

    assert actual["source_record_uri"].to_list() == ["ie://4"]
    assert actual["custom_payload"].to_list() == ["keep-me"]
    assert "name_tokens" in actual.columns


def test_tokenize_name_dataframe_uses_explicit_tokenizer_path():
    tokenizer_path = Path(
        "c:/Devel/Brunel/blocking/packages/company_tokenize/src/company_tokenize/resources/company_wordpiece_tokenizer.json"
    )
    assert tokenizer_path.exists()

    df = pl.DataFrame({"name_cleansed": ["betacorp"]})

    actual = tokenize_name_dataframe(df, tokenizer_path=tokenizer_path)

    assert isinstance(actual["name_tokens"].to_list()[0], list)
    assert actual["name_tokens"].to_list()[0]


def test_tokenize_name_dataframe_raises_for_missing_explicit_tokenizer_path():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})

    with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
        tokenize_name_dataframe(df, tokenizer_path="does-not-exist-tokenizer.json")


def test_tokenize_name_dataframe_dual_raises_when_column_missing():
    df = pl.DataFrame({"other": ["x"]})

    with pytest.raises(ValueError, match="name_cleansed"):
        tokenize_name_dataframe_dual(df)


def _train_dual_wordpiece_pair(tmp_path: Path) -> tuple[Path, Path]:
    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    country_tokenizer = tmp_path / "ie" / "model.json"
    global_tokenizer = tmp_path / "global" / "model.json"
    country_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    global_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    train_wordpiece(corpus_file, country_tokenizer, vocab_size=50, show_progress=False)
    train_wordpiece(corpus_file, global_tokenizer, vocab_size=50, show_progress=False)
    return country_tokenizer, global_tokenizer


def _write_manifest_sidecar(tokenizer_path: Path, **overrides: object) -> None:
    kwargs: dict[str, Any] = {
        "created_utc": "2026-08-29T00:00:00Z",
        "mode": "train",
        "scope": "country",
        "systems": ["ie"],
        "profile": None,
        "trainer": "wordpiece",
        "all_rows": True,
        "vocab_size": 50,
        "min_frequency": 1,
        "name_col": "name_cleansed",
        "preprocess_profile": "default|-company_type",
        "corpus_path": "artifacts/tokenizers/ie/training_corpus.parquet",
        "corpus_content_hash": "0123456789abcdef0123456789abcdef",
        "tokenizer_path": str(tokenizer_path),
        "line_count": 2,
        "trainer_params": {"tokenizer_encoding": None},
        "token_score_artifact": {"status": "ok", "invoked": True},
    }
    kwargs.update(overrides)
    manifest = build_run_manifest(**kwargs)
    tokenizer_path.with_name("metadata.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_tokenize_name_dataframe_dual_passes_when_manifests_coherent(
    tmp_path: Path,
):
    country_tokenizer, global_tokenizer = _train_dual_wordpiece_pair(tmp_path)
    _write_manifest_sidecar(country_tokenizer, scope="country")
    _write_manifest_sidecar(global_tokenizer, scope="global", systems=["ie", "gb"])

    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})
    actual = tokenize_name_dataframe_dual(
        df,
        country_tokenizer_path=country_tokenizer,
        global_tokenizer_path=global_tokenizer,
    )

    assert "country_tokens" in actual.columns
    assert "global_tokens" in actual.columns


def test_tokenize_name_dataframe_dual_raises_on_incoherent_manifests(
    tmp_path: Path,
):
    country_tokenizer, global_tokenizer = _train_dual_wordpiece_pair(tmp_path)
    _write_manifest_sidecar(
        country_tokenizer, scope="country", name_col="name_cleansed"
    )
    _write_manifest_sidecar(global_tokenizer, scope="global", name_col="name_raw")

    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})
    with pytest.raises(ValueError, match="disagree on name_col"):
        tokenize_name_dataframe_dual(
            df,
            country_tokenizer_path=country_tokenizer,
            global_tokenizer_path=global_tokenizer,
        )


def test_tokenize_name_dataframe_uses_sentencepiece_model_when_requested(
    tmp_path: Path,
):
    pytest.importorskip("sentencepiece")

    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "sentencepiece.model"
    train_sentencepiece(
        corpus_path, tokenizer_path, vocab_size=64, model_type="unigram"
    )

    df = pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]})
    actual = tokenize_name_dataframe(
        df, tokenizer_path=tokenizer_path, trainer="sentencepiece"
    )

    assert isinstance(actual["name_tokens"].to_list()[0], list)
    assert actual["name_tokens"].to_list()[0]
    assert isinstance(actual["name_tokens"].to_list()[1], list)
    assert actual["name_tokens"].to_list()[1]


def test_tokenize_name_dataframe_sentencepiece_requires_explicit_path():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})

    with pytest.raises(FileNotFoundError, match="No bundled tokenizer is available"):
        tokenize_name_dataframe(df, trainer="sentencepiece")


def test_tokenize_name_dataframe_can_override_packaged_noise_words_with_path(
    tmp_path: Path,
):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text('["alpha"]', encoding="utf-8")

    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})
    actual = tokenize_name_dataframe(
        df, noise_words_path=override_path, noise_words_profile="aggressive"
    )

    assert actual["name_tokens"].to_list() == [["ltd"]]


def test_tokenize_name_dataframe_can_override_packaged_noise_words_in_memory():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})
    actual = tokenize_name_dataframe(df, noise_words={"alpha"})

    assert actual["name_tokens"].to_list() == [["ltd"]]


def test_tokenize_name_dataframe_trims_long_german_name_before_encoding():
    df = pl.DataFrame(
        {
            "name_cleansed": [
                "neunte grundstucksverwaltung ahg beteiligungs & handelsgesellschaft mbh & co kg"
            ]
        }
    )

    actual = tokenize_name_dataframe(
        df, trimmed_name_col="trimmed_name", noise_words_profile="aggressive"
    )

    assert actual["trimmed_name"].to_list() == [
        "neunte grundstucksverwaltung ahg beteiligungs & handelsgesellschaft &"
    ]
    assert isinstance(actual["name_tokens"].to_list()[0], list)
    assert actual["name_tokens"].to_list()[0]


def test_tokenize_name_dataframe_multi_writes_multiple_columns(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_path)
    tokenizer_path = tmp_path / "tok.json"
    from company_tokenize.training import train_wordpiece

    train_wordpiece(corpus_path, tokenizer_path, vocab_size=64, show_progress=False)

    df = pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]})
    specs: list[Mapping[str, object]] = [
        {
            "label": "wp_a",
            "trainer": "wordpiece",
            "tokenizer_path": tokenizer_path,
            "token_col": "tokens_wp_a",
        },
        {
            "label": "wp_b",
            "trainer": "wordpiece",
            "tokenizer_path": tokenizer_path,
            "token_col": "tokens_wp_b",
        },
    ]
    actual = tokenize_name_dataframe_multi(df, tokenizer_specs=specs)

    assert "tokens_wp_a" in actual.columns
    assert "tokens_wp_b" in actual.columns
    assert actual["tokens_wp_a"].to_list()[0]
    assert actual["tokens_wp_b"].to_list()[0]


def test_tokenize_name_dataframe_multi_reuses_encoder_for_shared_spec(mocker):
    load_calls: list[tuple[str, str]] = []

    def fake_load_tokenizer_backend(*, tokenizer_path: Path, trainer: str):
        load_calls.append((str(tokenizer_path), trainer))
        return LoadedTokenizerBackend(
            trainer=trainer,
            encode=lambda text: [text],
            encode_batch=lambda texts: [[text] for text in texts],
            unk_token="[UNK]",
            vocab={},
        )

    mocker.patch(
        "company_tokenize.tokenization.load_tokenizer_backend",
        side_effect=fake_load_tokenizer_backend,
    )

    df = pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]})
    actual = tokenize_name_dataframe_multi(
        df,
        tokenizer_specs=[
            {
                "token_col": "tokens_a",
                "trainer": "wordpiece",
            },
            {
                "token_col": "tokens_b",
                "trainer": "wordpiece",
            },
        ],
    )

    assert "tokens_a" in actual.columns
    assert "tokens_b" in actual.columns
    assert len(load_calls) == 1


def test_a_default_tokenize_run_removes_no_words():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd"]})

    actual = tokenize_name_dataframe(df, trimmed_name_col="trimmed")

    assert actual["trimmed"].to_list() == ["alpha ltd"]


def test_a_preprocess_profile_tokenizes_the_text_a_run_would():
    df = pl.DataFrame({"name_cleansed": ["Alpha LTD", "Beta Holdings Ltd"]})

    removed = tokenize_name_dataframe(
        df, trimmed_name_col="trimmed", preprocess_profile="default"
    )
    kept = tokenize_name_dataframe(
        df, trimmed_name_col="trimmed", preprocess_profile="default|-company_type"
    )

    assert removed["trimmed"].to_list() == ["alpha", "beta holdings"]
    assert kept["trimmed"].to_list() == ["alpha ltd", "beta holdings ltd"]
    assert df["name_cleansed"].to_list() == ["Alpha LTD", "Beta Holdings Ltd"]
