import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from company_tokenize.manifest import build_run_manifest
from company_tokenize.ops import (
    _run_tokenization_for_files,
    tokenize_name,
    tokenize_name_dual,
    tokenize_name_multi,
)
from company_tokenize.training import train_wordpiece


def test_tokenize_name_raises_when_tokenizer_missing(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
        tokenize_name(cleansed_dir, tokenized_dir, tmp_path / "missing.json")


def test_tokenize_name_writes_tokenized_parquet(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    tokenizer_file = tmp_path / "tok.json"
    train_wordpiece(corpus_file, tokenizer_file, vocab_size=50, show_progress=False)

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    rows = tokenize_name(
        cleansed_dir,
        tokenized_dir,
        tokenizer_file,
        source_files=(cleansed_dir / "fr-001.parquet",),
        noise_words_profile="aggressive",
    )

    assert rows == 2
    written = pl.read_parquet(tokenized_dir / "fr-001.parquet")
    assert "name_tokens" in written.columns
    assert written["name_tokens"].to_list()[0] == ["alpha"]


def test_tokenize_name_dual_writes_both_token_columns(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    country_tokenizer = tmp_path / "country.json"
    global_tokenizer = tmp_path / "global.json"
    train_wordpiece(corpus_file, country_tokenizer, vocab_size=50, show_progress=False)
    train_wordpiece(corpus_file, global_tokenizer, vocab_size=50, show_progress=False)

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    rows = tokenize_name_dual(
        cleansed_dir,
        tokenized_dir,
        country_tokenizer_path=country_tokenizer,
        global_tokenizer_path=global_tokenizer,
        source_files=(cleansed_dir / "fr-001.parquet",),
    )

    assert rows == 2
    written = pl.read_parquet(tokenized_dir / "fr-001.parquet")
    assert "country_tokens" in written.columns
    assert "global_tokens" in written.columns


def _write_manifest_sidecar(tokenizer_path: Path, **overrides: object) -> None:
    kwargs: dict[str, Any] = {
        "created_utc": "2026-08-29T00:00:00Z",
        "mode": "train",
        "scope": "country",
        "systems": ["fr"],
        "profile": None,
        "trainer": "wordpiece",
        "all_rows": True,
        "vocab_size": 50,
        "min_frequency": 1,
        "name_col": "name_cleansed",
        "preprocess_profile": "default|-company_type",
        "corpus_path": "artifacts/tokenizers/fr/training_corpus.parquet",
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


def test_tokenize_name_dual_raises_on_incoherent_manifests(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    country_tokenizer = tmp_path / "fr" / "model.json"
    global_tokenizer = tmp_path / "global" / "model.json"
    country_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    global_tokenizer.parent.mkdir(parents=True, exist_ok=True)
    train_wordpiece(corpus_file, country_tokenizer, vocab_size=50, show_progress=False)
    train_wordpiece(corpus_file, global_tokenizer, vocab_size=50, show_progress=False)
    _write_manifest_sidecar(country_tokenizer, scope="country")
    # Wrong scope on the global side: this is the "path mix-up" case the
    # contract exists to catch.
    _write_manifest_sidecar(global_tokenizer, scope="country", systems=["fr", "gb"])

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    with pytest.raises(ValueError, match="expected 'global'"):
        tokenize_name_dual(
            cleansed_dir,
            tokenized_dir,
            country_tokenizer_path=country_tokenizer,
            global_tokenizer_path=global_tokenizer,
        )


def test_tokenize_name_honors_input_file_selection(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    tokenizer_file = tmp_path / "tok.json"
    train_wordpiece(corpus_file, tokenizer_file, vocab_size=50, show_progress=False)

    pl.DataFrame({"name_cleansed": ["alpha ltd"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )
    pl.DataFrame({"name_cleansed": ["betacorp"]}).write_parquet(
        cleansed_dir / "fr-002.parquet"
    )

    rows = tokenize_name(
        cleansed_dir, tokenized_dir, tokenizer_file, input_file="fr-002.parquet"
    )

    assert rows == 1
    assert not (tokenized_dir / "fr-001.parquet").exists()
    assert (tokenized_dir / "fr-002.parquet").exists()


def test_tokenize_name_multi_writes_requested_token_columns(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    tokenizer_file = tmp_path / "tok.json"
    train_wordpiece(corpus_file, tokenizer_file, vocab_size=50, show_progress=False)

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    tokenizer_specs: list[Mapping[str, object]] = [
        {
            "label": "wp_a",
            "trainer": "wordpiece",
            "tokenizer_path": tokenizer_file,
            "token_col": "tokens_wp_a",
        },
        {
            "label": "wp_b",
            "trainer": "wordpiece",
            "tokenizer_path": tokenizer_file,
            "token_col": "tokens_wp_b",
        },
    ]
    rows = tokenize_name_multi(
        cleansed_dir,
        tokenized_dir,
        tokenizer_specs=tokenizer_specs,
        source_files=(cleansed_dir / "fr-001.parquet",),
    )

    assert rows == 2
    written = pl.read_parquet(tokenized_dir / "fr-001.parquet")
    assert "tokens_wp_a" in written.columns
    assert "tokens_wp_b" in written.columns


def test_tokenize_name_removes_chunks_scratch_dir_after_run(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    tokenizer_file = tmp_path / "tok.json"
    train_wordpiece(corpus_file, tokenizer_file, vocab_size=50, show_progress=False)

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    tokenize_name(
        cleansed_dir,
        tokenized_dir,
        tokenizer_file,
        source_files=(cleansed_dir / "fr-001.parquet",),
    )

    assert not (tokenized_dir / "chunks").exists()
    assert (tokenized_dir / "fr-001.parquet").exists()


def test_run_tokenization_for_files_stages_before_promoting_to_final_dir(
    tmp_path: Path,
):
    source_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    source_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"name_cleansed": ["alpha"]}).write_parquet(
        source_dir / "fr-001.parquet"
    )
    pl.DataFrame({"name_cleansed": ["beta"]}).write_parquet(
        source_dir / "fr-002.parquet"
    )

    calls: list[str] = []

    def transform_fn(df: pl.DataFrame) -> pl.DataFrame:
        if calls:
            # Second file's turn: the first file's output should already be
            # staged on disk under chunks/ (progressive, visible durability),
            # but not yet promoted to its final location -- promotion only
            # happens once every file in this run has been processed.
            calls.append(
                "first_staged="
                + str((tokenized_dir / "chunks" / "fr-001.parquet").exists())
            )
            calls.append(
                "first_not_promoted="
                + str(not (tokenized_dir / "fr-001.parquet").exists())
            )
        else:
            calls.append("first_call")
        return df

    total_rows = _run_tokenization_for_files(
        eligible_files=[source_dir / "fr-001.parquet", source_dir / "fr-002.parquet"],
        source_dir=source_dir,
        tokenized_dir=tokenized_dir,
        name_col="name_cleansed",
        skipped_files=0,
        transform_fn=transform_fn,
    )

    assert total_rows == 2
    assert calls == ["first_call", "first_staged=True", "first_not_promoted=True"]
    assert (tokenized_dir / "fr-001.parquet").exists()
    assert (tokenized_dir / "fr-002.parquet").exists()
    assert not (tokenized_dir / "chunks").exists()


def test_tokenize_name_can_use_noise_word_override_path(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"system_uri": ["s:1", "s:2"], "name": ["alpha ltd", "betacorp"]}
    ).write_parquet(corpus_file)
    tokenizer_file = tmp_path / "tok.json"
    train_wordpiece(corpus_file, tokenizer_file, vocab_size=50, show_progress=False)

    noise_words_path = tmp_path / "noise_words.json"
    noise_words_path.write_text('["alpha"]', encoding="utf-8")

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    rows = tokenize_name(
        cleansed_dir,
        tokenized_dir,
        tokenizer_file,
        noise_words_path=noise_words_path,
        noise_words_profile="aggressive",
        source_files=(cleansed_dir / "fr-001.parquet",),
    )

    assert rows == 2
    written = pl.read_parquet(tokenized_dir / "fr-001.parquet")
    assert written["name_tokens"].to_list()[0] == ["ltd"]
