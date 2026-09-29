from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from company_tokenize import optimize as opt
from company_tokenize import tfidf as tf
from company_tokenize import tokenization as tok
from company_tokenize import training as tr


def _write_small_corpus(path: Path) -> None:
    pl.DataFrame(
        {
            "system_uri": ["ie://1", "gb://2", "fr/3"],
            "name": ["alpha ltd", "beta holdings", "gamma"],
        }
    ).write_parquet(path)


def test_training_helpers_and_validation_branches(tmp_path: Path):
    assert tr.resolve_tokenizer_path_for_trainer(
        trainer="sentencepiece", scope_directory=tmp_path, tokenizer_encoding="unigram"
    ) == (tmp_path / "sentencepiece_unigram" / "model.model")
    assert tr.resolve_metadata_path_for_trainer(
        trainer="sentencepiece", scope_directory=tmp_path
    ) == (tmp_path / "sentencepiece_bpe" / "metadata.json")
    assert (
        tr.resolve_candidate_model_suffix_for_trainer(trainer="sentencepiece")
        == "model"
    )

    with pytest.raises(ValueError, match="Unsupported trainer"):
        tr.resolve_tokenizer_path_for_trainer(
            trainer="invalid", scope_directory=tmp_path
        )

    with pytest.raises(ValueError, match="sp-character-coverage"):
        tr.validate_trainer_options(
            trainer="sentencepiece",
            options=tr.TrainerOptions(sp_character_coverage=0.0),
        )

    with pytest.raises(ValueError, match="--min-frequencies 1"):
        tr.normalize_optimize_min_frequencies_for_trainer(
            trainer="sentencepiece", min_frequencies=[1, 2]
        )

    assert tr.normalize_optimize_min_frequencies_for_trainer(
        trainer="sentencepiece", min_frequencies=[1]
    ) == [1]
    assert tr.trainer_params_payload(
        trainer="wordpiece", options=tr.TrainerOptions()
    ) == {
        "tokenizer_encoding": None,
        "sp_character_coverage": None,
        "sp_byte_fallback": None,
    }


def test_training_value_error_paths_and_dispatch(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "corpus.parquet"
    _write_small_corpus(corpus_path)

    with pytest.raises(ValueError, match="multiplier"):
        tr.estimate_wordpiece_vocab_size(corpus_path, multiplier=0)
    with pytest.raises(ValueError, match="min_vocab_size"):
        tr.estimate_wordpiece_vocab_size(corpus_path, min_vocab_size=0)
    with pytest.raises(ValueError, match="max_vocab_size"):
        tr.estimate_wordpiece_vocab_size(
            corpus_path, min_vocab_size=10, max_vocab_size=5
        )

    with pytest.raises(ValueError, match="vocab_size"):
        tr.train_wordpiece(
            corpus_path, tmp_path / "tok.json", vocab_size=0, show_progress=False
        )
    with pytest.raises(ValueError, match="min_frequency"):
        tr.train_wordpiece(
            corpus_path,
            tmp_path / "tok.json",
            vocab_size=50,
            min_frequency=0,
            show_progress=False,
        )

    with pytest.raises(ValueError, match="min_frequency=1"):
        tr.train_sentencepiece(corpus_path, tmp_path / "sp.model", min_frequency=2)
    with pytest.raises(ValueError, match="model_type"):
        tr.train_sentencepiece(corpus_path, tmp_path / "sp.model", model_type="char")
    with pytest.raises(ValueError, match="character_coverage"):
        tr.train_sentencepiece(
            corpus_path, tmp_path / "sp.model", character_coverage=2.0
        )

    called: dict[str, object] = {}

    def _fake_train_sp(**kwargs):
        called["sp"] = kwargs
        return 123

    def _fake_train_wp(**kwargs):
        called["wp"] = kwargs
        return 456

    monkeypatch.setattr(tr, "train_sentencepiece", _fake_train_sp)
    monkeypatch.setattr(tr, "train_wordpiece", _fake_train_wp)

    assert (
        tr.train_tokenizer_with_trainer(
            trainer="sentencepiece",
            corpus_path=corpus_path,
            tokenizer_path=tmp_path / "sp.model",
            options=tr.TrainerOptions(
                tokenizer_encoding="unigram",
                sp_character_coverage=1.0,
                sp_byte_fallback=True,
            ),
        )
        == 123
    )
    assert (
        tr.train_tokenizer_with_trainer(
            trainer="wordpiece",
            corpus_path=corpus_path,
            tokenizer_path=tmp_path / "wp.json",
            show_progress=False,
        )
        == 456
    )


def test_tfidf_file_format_and_helper_error_paths(tmp_path: Path):
    names = ["Acme Ltd", "Beta GmbH", "Gamma LLC"]

    # trim_token_column branch: no noise words and alias output.
    frame = pl.DataFrame({"name_tokens": [["Acme", "Ltd"]]})
    same = tf.trim_token_column(frame, noise_words=set())
    aliased = tf.trim_token_column(frame, noise_words=set(), output_col="trimmed")
    assert same.equals(frame)
    assert aliased.get_column("trimmed").to_list() == [["Acme", "Ltd"]]

    txt_path = tmp_path / "stop.txt"
    txt_path.write_text("alpha\nBeta\n", encoding="utf-8")
    assert tf.resolve_noise_words(noise_words_path=txt_path) == {"alpha", "beta"}

    csv_path = tmp_path / "stop.csv"
    pl.DataFrame({"token": ["x", "y"]}).write_csv(csv_path)
    assert tf.resolve_noise_words(noise_words_path=csv_path) == {"x", "y"}

    parquet_path = tmp_path / "stop.parquet"
    pl.DataFrame({"first": ["m", "n"]}).write_parquet(parquet_path)
    assert tf.resolve_noise_words(noise_words_path=parquet_path) == {"m", "n"}

    empty_csv = tmp_path / "empty.csv"
    pl.DataFrame({"token": []}).write_csv(empty_csv)
    assert tf.resolve_noise_words(noise_words_path=empty_csv) == set()

    with pytest.raises(FileNotFoundError, match="Noise-word list not found"):
        tf.resolve_noise_words(noise_words_path=tmp_path / "missing.txt")

    bad_ext = tmp_path / "stop.bin"
    bad_ext.write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported noise-word file format"):
        tf.resolve_noise_words(noise_words_path=bad_ext)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(
        '{"profiles": {"balanced": {"seed_legal": ["ltd"]}}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="set kind"):
        tf.resolve_noise_words(
            noise_words_path=bad_json,
            noise_words_profile="balanced",
            noise_words_set_kind="unknown",
        )

    non_string_json = tmp_path / "non_string.json"
    non_string_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="entries must be strings"):
        tf.resolve_noise_words(noise_words_path=non_string_json)

    missing_profile_json = tmp_path / "missing_profile.json"
    missing_profile_json.write_text(
        '{"profiles": {"strict": {"tokens": ["a"]}}}', encoding="utf-8"
    )
    with pytest.raises(TypeError, match="profile 'aggressive' not found"):
        tf.resolve_noise_words(noise_words_path=missing_profile_json)

    legacy_json = tmp_path / "legacy.json"
    legacy_json.write_text(
        '{"profiles": {"balanced": {"tfidf_descriptor": ["systems"], "seed_legal": ["ltd"]}}}',
        encoding="utf-8",
    )
    assert tf.resolve_noise_words(
        noise_words_path=legacy_json,
        noise_words_profile="balanced",
        noise_words_set_kind="tokens",
    ) == {"systems"}

    custom_profile_json = tmp_path / "custom_profile.json"
    custom_profile_json.write_text(
        '{"profiles": {"custom": {"seed_legal": ["ltd"], "tokens": ["holdings"]}}}',
        encoding="utf-8",
    )
    assert tf.resolve_noise_words(
        noise_words_path=custom_profile_json,
        noise_words_profile="custom",
        noise_words_set_kind="combined",
    ) == {"ltd", "holdings"}

    assert (
        tf._estimate_document_frequency_from_idf(n_docs=0, idf=1.0, smooth_idf=True)
        == 0
    )
    assert (
        tf._estimate_document_frequency_from_idf(
            n_docs=10, idf=float("inf"), smooth_idf=True
        )
        == 0
    )
    assert (
        tf._estimate_document_frequency_from_idf(n_docs=10, idf=1.0, smooth_idf=False)
        >= 1
    )

    # Smoke call so docs variable is used and tokenization path still exercised.
    trimmed = tf.trim_text_before_tokenization(names, noise_words={"ltd"})
    assert trimmed[0] == "Acme"


def test_tokenization_multi_validation_and_spec_noise_word_paths(tmp_path: Path):
    df = pl.DataFrame({"name_cleansed": ["alpha ltd", ""]})

    with pytest.raises(ValueError, match="at least one tokenizer specification"):
        tok.tokenize_name_dataframe_multi(df, tokenizer_specs=[])

    with pytest.raises(ValueError, match="non-empty 'token_col'"):
        tok.tokenize_name_dataframe_multi(df, tokenizer_specs=[{"token_col": "  "}])

    with pytest.raises(ValueError, match="Duplicate token column"):
        tok.tokenize_name_dataframe_multi(
            df,
            tokenizer_specs=[
                {"token_col": "dup"},
                {"token_col": "dup"},
            ],
        )

    with pytest.raises(ValueError, match="noise_words"):
        tok.tokenize_name_dataframe_multi(
            df,
            tokenizer_specs=[
                {"token_col": "a", "noise_words": "invalid-type"},
            ],
        )

    stop_path = tmp_path / "stop.json"
    stop_path.write_text('["alpha"]', encoding="utf-8")
    actual = tok.tokenize_name_dataframe_multi(
        df,
        tokenizer_specs=[
            {
                "token_col": "tokens_a",
                "noise_words_path": stop_path,
                "trainer": "wordpiece",
            }
        ],
        trimmed_name_col="trimmed",
        noise_words_profile="aggressive",
    )
    assert "tokens_a" in actual.columns
    assert "trimmed" in actual.columns
    assert actual.get_column("trimmed").to_list() == ["alpha", ""]

    dual = tok.tokenize_name_dataframe_dual(df, trimmed_name_col="trimmed_name")
    assert "trimmed_name" in dual.columns


def test_optimize_helper_branches(tmp_path: Path, monkeypatch):
    assert opt.split_system_from_uri(None) == "unknown"
    assert opt.split_system_from_uri("fr/company/1") == "fr"

    rejected, reason = opt.classify_candidate_rejection(
        metrics={
            "unk_rate": 0.0,
            "fertility": 0.5,
            "fertility_distance": 0.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        },
        unk_rate_threshold=1.0,
        fertility_tolerance=1.0,
        fertility_min=1.0,
        fertility_max=2.0,
        token_count_median_max=10.0,
        token_count_p95_max=10.0,
    )
    assert rejected is True
    assert reason == "fertility_too_low"

    corpus_path = tmp_path / "corpus.parquet"
    # 40 rows (not 2) so a 50/50 split reliably lands on both sides regardless
    # of seed/hash outcome -- a couple of rows risks both landing on the same
    # side by chance with a real hash function.
    pl.DataFrame(
        {
            "system_uri": [f"ie://{i}" for i in range(40)],
            "name": [f"name{i}" for i in range(40)],
        }
    ).write_parquet(corpus_path)

    train_path, val_path, train_rows, val_rows = opt.write_seed_split_artifacts(
        corpus_path=corpus_path,
        optimize_dir=tmp_path,
        seed=7,
        validation_fraction=0.5,
    )
    assert train_path.exists() and val_path.exists()
    assert train_rows > 0 and val_rows > 0

    with pytest.raises(ValueError, match="Validation split is empty"):
        opt.write_seed_split_artifacts(
            corpus_path=corpus_path,
            optimize_dir=tmp_path,
            seed=7,
            validation_fraction=0.0,
        )
    with pytest.raises(ValueError, match="Train split is empty"):
        opt.write_seed_split_artifacts(
            corpus_path=corpus_path,
            optimize_dir=tmp_path,
            seed=7,
            validation_fraction=1.0,
        )

    def fake_encoder(*, tokenizer_path, trainer):
        def encode_tokens(name: str):
            if name == "skip-me":
                return []
            if name == "alpha":
                return ["##a", "[UNK]"]
            if name == "beta":
                return ["▁b", "x", "y"]
            return ["token", "token", "token", "token"]

        return encode_tokens, "[UNK]"

    monkeypatch.setattr(opt, "load_tokenizer_encoder", fake_encoder)

    eval_path = tmp_path / "eval.parquet"
    pl.DataFrame(
        {
            "system_uri": ["ie://1", "", "fr/path", "fr/path", None],
            "name": ["alpha", "skip-me", "beta", "omega", "   "],
        }
    ).write_parquet(eval_path)

    metrics, per_system_rows, high_fertility_rows, unk_rows = (
        opt.evaluate_tokenizer_metrics(
            corpus_path=eval_path,
            tokenizer_path=tmp_path / "tok.json",
            trainer="wordpiece",
            fertility_target=1.0,
            diagnostic_top_n=1,
        )
    )
    assert metrics["rows"] > 0
    assert per_system_rows
    assert len(high_fertility_rows) == 1
    assert len(unk_rows) <= 1

    empty_eval = tmp_path / "empty_eval.parquet"
    pl.DataFrame({"system_uri": ["ie://1"], "name": [""]}).write_parquet(empty_eval)
    with pytest.raises(ValueError, match="No evaluable validation rows"):
        opt.evaluate_tokenizer_metrics(
            corpus_path=empty_eval,
            tokenizer_path=tmp_path / "tok.json",
            trainer="wordpiece",
            fertility_target=1.0,
            diagnostic_top_n=1,
        )

    log_path = tmp_path / "runs.parquet"
    incoming_rows = [{"seed": 1, "score": 0.5}]

    combined = opt.append_run_log(run_log_path=log_path, run_rows=incoming_rows)
    assert combined.height == 1

    existing = pl.DataFrame({"seed": [0], "score": [0.1]})
    combined_existing = opt.append_run_log(
        run_log_path=log_path, run_rows=incoming_rows, existing_df=existing
    )
    assert combined_existing.height == 2

    empty_existing_path = tmp_path / "empty_existing.parquet"
    pl.DataFrame().write_parquet(empty_existing_path)
    combined_empty_existing = opt.append_run_log(
        run_log_path=empty_existing_path, run_rows=incoming_rows
    )
    assert combined_empty_existing.height == 1

    merged_empty = opt.merge_run_logs(
        run_log_dir=tmp_path / "missing_dir",
        merged_run_log_path=tmp_path / "merged_empty.parquet",
    )
    assert merged_empty.height == 0

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame().write_parquet(shard_dir / "a.parquet")
    merged_no_frames = opt.merge_run_logs(
        run_log_dir=shard_dir, merged_run_log_path=tmp_path / "merged_none.parquet"
    )
    assert merged_no_frames.height == 0

    pl.DataFrame({"seed": [1], "score": [0.2]}).write_parquet(shard_dir / "b.parquet")
    merged = opt.merge_run_logs(
        run_log_dir=shard_dir, merged_run_log_path=tmp_path / "merged.parquet"
    )
    assert merged.height == 1

    with pytest.raises(ValueError, match="greater than or equal to zero"):
        opt.select_best_pair(
            run_log_df=pl.DataFrame(
                {
                    "vocab_size_requested": [1000],
                    "vocab_size_resolved": [1000],
                    "min_frequency": [1],
                    "selection_score": [1.0],
                    "rejected": [False],
                    "fertility_distance": [0.1],
                    "unk_rate": [0.01],
                }
            ),
            eligibility_pass_rate=0.1,
            vocab_size_distance_tolerance=-0.1,
        )

    assert (
        opt.select_best_pair(
            run_log_df=pl.DataFrame(
                {
                    "vocab_size_requested": [1000],
                    "vocab_size_resolved": [1000],
                    "min_frequency": [1],
                    "selection_score": [1.0],
                    "rejected": [True],
                    "fertility_distance": [0.1],
                    "unk_rate": [0.01],
                }
            ),
            eligibility_pass_rate=0.9,
            vocab_size_distance_tolerance=0.0,
        )
        is None
    )
