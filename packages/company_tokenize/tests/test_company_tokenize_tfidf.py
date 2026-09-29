from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    compute_corpus_token_tfidf_stats,
    compute_tokenizer_token_tfidf_stats,
    count_tokens_above_document_frequency_pct,
    cumulative_document_frequency_mass_cutoff,
    expected_tfidf_namespace,
    infer_document_count_from_tfidf_stats,
    pool_token_tfidf_stats,
    resolve_noise_words,
    select_rare_token_candidates,
    select_stopword_candidates,
    train_wordpiece,
    trim_name_before_tokenization,
    trim_pooled_corpus_by_system,
    trim_token_column,
    validate_tfidf_namespace,
)
from company_tokenize.tfidf import trim_text_before_tokenization


def test_compute_corpus_token_tfidf_stats_raises_when_corpus_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Training corpus parquet not found"):
        compute_corpus_token_tfidf_stats(tmp_path / "missing.parquet")


def test_compute_corpus_token_tfidf_stats_returns_expected_columns(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "system_uri": ["s:1", "s:2", "s:3"],
            "name": ["cisco systems", "cisco ltd", "acme systems"],
        }
    ).write_parquet(corpus_path)

    stats = compute_corpus_token_tfidf_stats(corpus_path)

    assert stats.columns == [
        "token",
        "document_frequency",
        "document_frequency_pct",
        "idf",
    ]
    assert "cisco" in stats.get_column("token").to_list()
    assert "systems" in stats.get_column("token").to_list()

    cisco_row = stats.filter(pl.col("token") == "cisco")
    assert cisco_row.height == 1
    assert cisco_row.get_column("document_frequency").item() == 2
    assert cisco_row.get_column("document_frequency_pct").item() == pytest.approx(2 / 3)


def test_compute_corpus_token_tfidf_stats_returns_empty_for_empty_documents(
    tmp_path: Path,
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "system_uri": ["s:1", "s:2"],
            "name": ["", None],
        }
    ).write_parquet(corpus_path)

    stats = compute_corpus_token_tfidf_stats(corpus_path)
    assert stats.height == 0


def test_compute_tokenizer_token_tfidf_stats_returns_expected_columns(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    tokenizer_path = tmp_path / "wordpiece.json"
    pl.DataFrame(
        {
            "system_uri": ["s:1", "s:2", "s:3"],
            "name": ["alpha beta", "alpha gamma", "beta gamma"],
        }
    ).write_parquet(corpus_path)

    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=64,
        min_frequency=1,
        show_progress=False,
    )

    stats = compute_tokenizer_token_tfidf_stats(
        corpus_path,
        tokenizer_path=tokenizer_path,
        trainer="wordpiece",
    )

    assert stats.columns == [
        "token",
        "document_frequency",
        "document_frequency_pct",
        "idf",
    ]
    assert stats.height > 0
    assert stats.get_column("document_frequency").min() >= 1
    assert stats.get_column("document_frequency").max() <= 3
    assert stats.get_column("document_frequency_pct").max() <= 1.0


def test_tfidf_namespace_contract_rejects_mixed_usage():
    assert expected_tfidf_namespace(use_case="corpus") == "corpus_tfidf"
    assert (
        expected_tfidf_namespace(use_case="tokenizer", trainer="wordpiece")
        == "wordpiece_tfidf"
    )

    validate_tfidf_namespace(use_case="corpus", namespace="corpus_tfidf")
    validate_tfidf_namespace(
        use_case="tokenizer", trainer="wordpiece", namespace="wordpiece_tfidf"
    )

    with pytest.raises(ValueError, match="namespace mismatch"):
        validate_tfidf_namespace(use_case="corpus", namespace="wordpiece_tfidf")

    with pytest.raises(ValueError, match="namespace mismatch"):
        validate_tfidf_namespace(
            use_case="tokenizer", trainer="wordpiece", namespace="corpus_tfidf"
        )


def test_cumulative_document_frequency_mass_cutoff_stops_once_limit_reached():
    # 0.5 alone clears a 0.25 limit -- the common real case (a single
    # dominant legal-form token) shouldn't require a second item.
    assert (
        cumulative_document_frequency_mass_cutoff(
            [0.5, 0.1, 0.05], cumulative_df_mass_limit=0.25
        )
        == 1
    )
    assert (
        cumulative_document_frequency_mass_cutoff(
            [0.1, 0.1, 0.1], cumulative_df_mass_limit=0.25
        )
        == 3
    )


def test_cumulative_document_frequency_mass_cutoff_returns_full_length_when_never_reached():
    assert (
        cumulative_document_frequency_mass_cutoff(
            [0.1, 0.05], cumulative_df_mass_limit=0.9
        )
        == 2
    )


def test_cumulative_document_frequency_mass_cutoff_returns_zero_for_non_positive_limit():
    assert (
        cumulative_document_frequency_mass_cutoff(
            [0.5, 0.5], cumulative_df_mass_limit=0.0
        )
        == 0
    )


def test_count_tokens_above_document_frequency_pct_stops_at_first_below_floor():
    assert (
        count_tokens_above_document_frequency_pct(
            [0.30, 0.10, 0.08, 0.01, 0.009], min_document_frequency_pct=0.05
        )
        == 3
    )


def test_count_tokens_above_document_frequency_pct_returns_zero_when_none_qualify():
    assert (
        count_tokens_above_document_frequency_pct(
            [0.04, 0.03, 0.02], min_document_frequency_pct=0.05
        )
        == 0
    )


def test_count_tokens_above_document_frequency_pct_returns_full_length_when_all_qualify():
    assert (
        count_tokens_above_document_frequency_pct(
            [0.9, 0.8, 0.7], min_document_frequency_pct=0.05
        )
        == 3
    )


def test_select_stopword_candidates_filters_by_df_pct_and_idf():
    token_stats = pl.DataFrame(
        {
            "token": ["cisco", "systems", "inc", "acme"],
            "document_frequency": [9, 9, 8, 1],
            "document_frequency_pct": [0.9, 0.9, 0.8, 0.1],
            "idf": [1.1, 1.2, 1.3, 3.5],
        }
    )

    candidates = select_stopword_candidates(
        token_stats,
        min_document_frequency_pct=0.7,
        max_idf=1.25,
    )

    assert candidates == {"cisco", "systems"}


def test_select_stopword_candidates_validates_input():
    with pytest.raises(ValueError, match="missing required columns"):
        select_stopword_candidates(pl.DataFrame({"token": ["x"]}))

    token_stats = pl.DataFrame(
        {
            "token": ["a"],
            "document_frequency_pct": [0.5],
            "idf": [1.2],
        }
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_stopword_candidates(token_stats, min_document_frequency_pct=1.5)


def test_select_rare_token_candidates_defaults_to_hapax_legomena():
    token_stats = pl.DataFrame(
        {
            "token": ["cisco", "systems", "inc", "acme", "zyx"],
            "document_frequency": [9, 9, 8, 2, 1],
            "document_frequency_pct": [0.9, 0.9, 0.8, 0.2, 0.1],
            "idf": [1.1, 1.2, 1.3, 2.5, 3.5],
        }
    )

    rare = select_rare_token_candidates(token_stats)

    assert rare.get_column("token").to_list() == ["zyx"]


def test_select_rare_token_candidates_filters_by_count_and_pct():
    token_stats = pl.DataFrame(
        {
            "token": ["cisco", "acme", "zyx", "qq"],
            "document_frequency": [9, 2, 1, 1],
            "document_frequency_pct": [0.9, 0.2, 0.1, 0.01],
            "idf": [1.1, 2.5, 3.5, 4.0],
        }
    )

    rare = select_rare_token_candidates(
        token_stats,
        max_document_frequency=2,
        max_document_frequency_pct=0.15,
    )

    assert rare.get_column("token").to_list() == ["qq", "zyx"]


def test_select_rare_token_candidates_respects_min_token_length():
    token_stats = pl.DataFrame(
        {
            "token": ["a", "zyx"],
            "document_frequency": [1, 1],
            "document_frequency_pct": [0.1, 0.1],
            "idf": [3.5, 3.5],
        }
    )

    rare = select_rare_token_candidates(token_stats, min_token_length=2)

    assert rare.get_column("token").to_list() == ["zyx"]


def test_select_rare_token_candidates_validates_input():
    with pytest.raises(ValueError, match="missing required columns"):
        select_rare_token_candidates(pl.DataFrame({"token": ["x"]}))

    token_stats = pl.DataFrame(
        {
            "token": ["a"],
            "document_frequency": [1],
            "document_frequency_pct": [0.5],
        }
    )
    with pytest.raises(ValueError, match="max_document_frequency must be"):
        select_rare_token_candidates(token_stats, max_document_frequency=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        select_rare_token_candidates(token_stats, max_document_frequency_pct=1.5)


def test_trim_token_column_removes_noise_words_case_insensitive():
    df = pl.DataFrame(
        {
            "name_tokens": [["Cisco", "Systems", "Inc"], ["Acme", "Ltd"]],
        }
    )

    actual = trim_token_column(df, noise_words={"cisco", "inc", "ltd"})

    assert actual.get_column("name_tokens").to_list() == [["Systems"], ["Acme"]]


def test_trim_token_column_can_write_to_different_output_column():
    df = pl.DataFrame(
        {
            "name_tokens": [["cisco", "systems", "inc"]],
        }
    )

    actual = trim_token_column(df, noise_words={"inc"}, output_col="trimmed_tokens")

    assert actual.get_column("name_tokens").to_list() == [["cisco", "systems", "inc"]]
    assert actual.get_column("trimmed_tokens").to_list() == [["cisco", "systems"]]


def test_trim_token_column_validates_required_column():
    df = pl.DataFrame({"other_tokens": [["cisco"]]})

    with pytest.raises(ValueError, match="required column"):
        trim_token_column(df, noise_words={"cisco"}, token_col="name_tokens")


def test_resolve_noise_words_uses_packaged_global_default():
    tokens = resolve_noise_words()
    assert "ltd" in tokens
    assert "mbh" in tokens


def test_resolve_noise_words_prefers_explicit_set_over_path(tmp_path: Path):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text('["foo", "bar"]', encoding="utf-8")

    tokens = resolve_noise_words(noise_words={"alpha"}, noise_words_path=override_path)

    assert tokens == {"alpha"}


def test_resolve_noise_words_reads_override_path(tmp_path: Path):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text('{"tokens": ["custom", "tokens"]}', encoding="utf-8")

    tokens = resolve_noise_words(noise_words_path=override_path)

    assert tokens == {"custom", "tokens"}


def test_resolve_noise_words_reads_profiled_noise_word_artifact(tmp_path: Path):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text(
        '{"profiles": {"strict": {"seed_legal": ["ltd"], "tokens": []}, "balanced": {"seed_legal": ["ltd"], "tokens": ["systems"]}}}',
        encoding="utf-8",
    )

    tokens = resolve_noise_words(
        noise_words_path=override_path,
        noise_words_profile="balanced",
        noise_words_set_kind="combined",
    )

    assert tokens == {"ltd", "systems"}


def test_resolve_noise_words_reads_profiled_set_kind(tmp_path: Path):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text(
        '{"profiles": {"balanced": {"seed_legal": ["ltd"], "tokens": ["systems"]}}}',
        encoding="utf-8",
    )

    tokens = resolve_noise_words(
        noise_words_path=override_path,
        noise_words_profile="balanced",
        noise_words_set_kind="seed_legal",
    )

    assert tokens == {"ltd"}


def test_resolve_noise_words_reads_profiled_tokens_set_kind(tmp_path: Path):
    override_path = tmp_path / "noise_words.json"
    override_path.write_text(
        '{"profiles": {"balanced": {"seed_legal": ["ltd"], "tokens": ["systems"]}}}',
        encoding="utf-8",
    )

    tokens = resolve_noise_words(
        noise_words_path=override_path,
        noise_words_profile="balanced",
        noise_words_set_kind="tokens",
    )

    assert tokens == {"systems"}


def test_trim_name_before_tokenization_uses_packaged_defaults():
    actual = trim_name_before_tokenization("Acme Ltd")
    assert actual == "Acme"


def test_trim_name_before_tokenization_respects_explicit_noise_words():
    actual = trim_name_before_tokenization(
        "Acme Systems Ltd",
        noise_words={"systems"},
    )
    assert actual == "Acme Ltd"


def test_trim_name_before_tokenization_matches_list_trim_for_same_tokens():
    noise_words = {"ltd", "co"}
    name = "Acme & Co Ltd"

    single = trim_name_before_tokenization(
        name,
        noise_words=noise_words,
    )
    list_trimmed = trim_text_before_tokenization(
        [name],
        noise_words=noise_words,
    )[0]

    assert single == list_trimmed


def test_trim_name_before_tokenization_matches_list_trim_for_none_input():
    noise_words = {"ltd"}

    single = trim_name_before_tokenization(
        None,
        noise_words=noise_words,
    )
    list_trimmed = trim_text_before_tokenization(
        [None],
        noise_words=noise_words,
    )[0]

    assert single == list_trimmed


def test_infer_document_count_from_tfidf_stats_recovers_n_docs():
    stats = pl.DataFrame(
        {
            "token": ["y", "common"],
            "document_frequency": [10, 5],
            "document_frequency_pct": [1.0, 0.5],
            "idf": [1.0, 1.5],
        }
    )

    assert infer_document_count_from_tfidf_stats(stats) == 10


def test_infer_document_count_from_tfidf_stats_raises_on_empty_frame():
    empty = pl.DataFrame(
        {
            "token": [],
            "document_frequency": [],
            "document_frequency_pct": [],
            "idf": [],
        },
        schema={
            "token": pl.Utf8,
            "document_frequency": pl.Int64,
            "document_frequency_pct": pl.Float64,
            "idf": pl.Float64,
        },
    )

    with pytest.raises(ValueError, match="empty TF-IDF stats"):
        infer_document_count_from_tfidf_stats(empty)


def _small_and_big_system_stats() -> dict[str, pl.DataFrame]:
    # "small" has 10 docs (all containing "y"); "big" has 1000 docs (none
    # containing "y"). Both share "common" at a different share each.
    small_stats = pl.DataFrame(
        {
            "token": ["y", "common"],
            "document_frequency": [10, 5],
            "document_frequency_pct": [1.0, 0.5],
            "idf": [1.0, 1.5],
        }
    )
    big_stats = pl.DataFrame(
        {
            "token": ["common"],
            "document_frequency": [200],
            "document_frequency_pct": [0.2],
            "idf": [2.0],
        }
    )
    return {"small": small_stats, "big": big_stats}


def test_pool_token_tfidf_stats_equal_weights_each_member_the_same_regardless_of_size():
    pooled, provenance = pool_token_tfidf_stats(
        _small_and_big_system_stats(), pooling="equal"
    )

    assert provenance["pooling_rule"] == "equal"
    assert provenance["systems"] == ["big", "small"]
    assert provenance["weights"] == {"big": 0.5, "small": 0.5}
    assert provenance["document_counts"] == {"big": 1000, "small": 10}

    metrics = {
        row["token"]: row["document_frequency_pct"]
        for row in pooled.iter_rows(named=True)
    }
    # "y" is present in only the 10-doc member at pct=1.0 and absent (0.0)
    # from the 1000-doc member: the equal-weighted mean is 0.5 either way,
    # unmoved by the 100x size difference between the two members.
    assert metrics["y"] == pytest.approx(0.5)
    assert metrics["common"] == pytest.approx((0.5 + 0.2) / 2)


def test_pool_token_tfidf_stats_count_reproduces_pooled_corpus_percentages():
    stats_by_system = _small_and_big_system_stats()
    pooled, provenance = pool_token_tfidf_stats(stats_by_system, pooling="count")

    assert provenance["pooling_rule"] == "count"
    total_docs = 1000 + 10
    assert provenance["weights"]["small"] == pytest.approx(10 / total_docs)
    assert provenance["weights"]["big"] == pytest.approx(1000 / total_docs)

    metrics = {
        row["token"]: row["document_frequency_pct"]
        for row in pooled.iter_rows(named=True)
    }
    # Row-weighted ("count") pooling reproduces the percentage a genuinely
    # concatenated 1010-document corpus would report for each token: "y"'s
    # 10 hits and "common"'s 5+200 hits, each over the combined 1010 docs.
    assert metrics["y"] == pytest.approx(10 / total_docs)
    assert metrics["common"] == pytest.approx((5 + 200) / total_docs)


def test_pool_token_tfidf_stats_count_matches_a_real_combined_corpus(tmp_path: Path):
    system_a_docs = ["ltd apple", "ltd banana", "ltd cherry"]
    system_b_docs = ["ltd date", "ltd date", "ltd date", "ltd elderberry", "fig"]

    def _write_and_compute(name: str, docs: list[str]) -> pl.DataFrame:
        corpus_path = tmp_path / f"{name}.parquet"
        pl.DataFrame(
            {
                "system_uri": [f"{name}:{i}" for i in range(len(docs))],
                "name": docs,
            }
        ).write_parquet(corpus_path)
        return compute_corpus_token_tfidf_stats(corpus_path)

    stats_by_system = {
        "a": _write_and_compute("a", system_a_docs),
        "b": _write_and_compute("b", system_b_docs),
    }
    combined_stats = _write_and_compute("combined", system_a_docs + system_b_docs)

    pooled, _provenance = pool_token_tfidf_stats(stats_by_system, pooling="count")

    pooled_metrics = {
        row["token"]: row["document_frequency_pct"]
        for row in pooled.iter_rows(named=True)
    }
    combined_metrics = {
        row["token"]: row["document_frequency_pct"]
        for row in combined_stats.iter_rows(named=True)
    }

    assert pooled_metrics.keys() == combined_metrics.keys()
    for token, pct in combined_metrics.items():
        assert pooled_metrics[token] == pytest.approx(pct)


def test_pool_token_tfidf_stats_rejects_unsupported_pooling_rule():
    with pytest.raises(ValueError, match="Unsupported pooling rule"):
        pool_token_tfidf_stats(_small_and_big_system_stats(), pooling="median")


def test_pool_token_tfidf_stats_rejects_empty_mapping():
    with pytest.raises(ValueError, match="at least one system"):
        pool_token_tfidf_stats({}, pooling="equal")


def test_trim_pooled_corpus_by_system_uses_each_row_s_own_system_list(
    tmp_path: Path,
):
    ie_noise_words_path = tmp_path / "ie_noise_words.json"
    ie_noise_words_path.write_text('["ltd"]', encoding="utf-8")
    gb_noise_words_path = tmp_path / "gb_noise_words.json"
    gb_noise_words_path.write_text('["limited"]', encoding="utf-8")

    corpus = pl.DataFrame(
        {
            "system_uri": ["ie://companies/1", "gb://companies/2"],
            "name": ["Acme Ltd", "Acme Limited"],
        }
    )

    trimmed = trim_pooled_corpus_by_system(
        corpus,
        noise_words_path_by_system={
            "ie": ie_noise_words_path,
            "gb": gb_noise_words_path,
        },
    )

    trimmed_by_system_uri = dict(
        zip(
            trimmed.get_column("system_uri").to_list(),
            trimmed.get_column("name").to_list(),
            strict=True,
        )
    )
    assert trimmed_by_system_uri["ie://companies/1"] == "Acme"
    assert trimmed_by_system_uri["gb://companies/2"] == "Acme"
    assert trimmed.height == corpus.height


def test_trim_pooled_corpus_by_system_raises_on_missing_system_noise_words(
    tmp_path: Path,
):
    ie_noise_words_path = tmp_path / "ie_noise_words.json"
    ie_noise_words_path.write_text('["ltd"]', encoding="utf-8")

    corpus = pl.DataFrame(
        {
            "system_uri": ["ie://companies/1", "gb://companies/2"],
            "name": ["Acme Ltd", "Acme Limited"],
        }
    )

    with pytest.raises(ValueError, match="no noise-word file"):
        trim_pooled_corpus_by_system(
            corpus,
            noise_words_path_by_system={"ie": ie_noise_words_path},
        )


def test_trim_pooled_corpus_by_system_validates_required_columns():
    corpus = pl.DataFrame({"name": ["Acme Ltd"]})

    with pytest.raises(ValueError, match="system_uri"):
        trim_pooled_corpus_by_system(corpus, noise_words_path_by_system={})


def test_trim_pooled_corpus_by_system_returns_empty_corpus_unchanged():
    corpus = pl.DataFrame(
        {"system_uri": [], "name": []},
        schema={"system_uri": pl.Utf8, "name": pl.Utf8},
    )

    trimmed = trim_pooled_corpus_by_system(
        corpus, noise_words_path_by_system={"ie": Path("unused.json")}
    )

    assert trimmed.height == 0


def test_the_none_profile_removes_no_words_even_given_a_path(tmp_path: Path):
    noise_words_path = tmp_path / "noise_words.json"
    noise_words_path.write_text('["ltd"]', encoding="utf-8")

    assert resolve_noise_words(noise_words_profile="none") == set()
    assert (
        resolve_noise_words(
            noise_words_path=noise_words_path, noise_words_profile="none"
        )
        == set()
    )
