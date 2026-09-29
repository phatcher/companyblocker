from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import pool_token_tfidf_stats

from scripts import generate_noise_words
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def test_run_generation_persists_token_stats_and_sets(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "training_corpus.parquet"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2", "ie:3"],
            "name": ["cisco systems", "cisco ltd", "acme systems"],
        }
    ).write_parquet(corpus_path)

    exit_code = generate_noise_words.run_generation(
        system="ie",
        profile="default",
        root=tmp_path,
        corpus_path=None,
        name_col="name",
        min_document_frequency_pct=0.60,
        max_idf=1.35,
        min_token_length=2,
        token_stats_out=None,
        token_set_out=None,
        noise_words_out=None,
        seed_noise_words_path=None,
        strict_token_count=1,
        balanced_token_count=10,
        aggressive_token_count=25,
        strict_min_document_frequency_pct=None,
        strict_max_idf=None,
        balanced_min_document_frequency_pct=None,
        balanced_max_idf=None,
        aggressive_min_document_frequency_pct=None,
        aggressive_max_idf=None,
    )

    assert exit_code == 0

    base = tokenizer_scope_dir(workspace_roots, system="ie")
    stats_parquet = base / "token_tfidf_stats.parquet"
    stats_csv = base / "token_tfidf_stats.csv"
    token_set_json = base / "token_set.json"
    noise_words_json = base / "noise_words.json"

    assert stats_parquet.exists()
    assert stats_csv.exists()
    assert token_set_json.exists()
    assert noise_words_json.exists()

    stats = pl.read_parquet(stats_parquet)
    assert "token" in stats.columns
    assert "idf" in stats.columns

    token_set = json.loads(token_set_json.read_text(encoding="utf-8"))
    noise_words = json.loads(noise_words_json.read_text(encoding="utf-8"))
    assert "cisco" in token_set
    assert noise_words["tfidf_use_case"] == "corpus"
    assert noise_words["tfidf_namespace"] == "corpus_tfidf"
    assert "profiles" in noise_words
    assert "balanced" in noise_words["profiles"]
    assert "tokens" in noise_words["profiles"]["balanced"]
    assert noise_words["profiles"]["strict"]["selection"]["token_count_actual"] >= 1


def test_run_generation_merges_external_seed_noise_words(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "training_corpus.parquet"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2", "ie:3"],
            "name": ["acme innovation", "zenith analytics", "northwind logistics"],
        }
    ).write_parquet(corpus_path)

    seed_path = tmp_path / "seed_noise_words.json"
    seed_path.write_text(json.dumps(["ltd", "clg", "LTD"]), encoding="utf-8")

    exit_code = generate_noise_words.run_generation(
        system="ie",
        profile="default",
        root=tmp_path,
        corpus_path=None,
        name_col="name",
        min_document_frequency_pct=0.99,
        max_idf=1.01,
        min_token_length=2,
        token_stats_out=None,
        token_set_out=None,
        noise_words_out=None,
        seed_noise_words_path=str(seed_path),
        strict_token_count=1,
        balanced_token_count=10,
        aggressive_token_count=25,
        strict_min_document_frequency_pct=None,
        strict_max_idf=None,
        balanced_min_document_frequency_pct=None,
        balanced_max_idf=None,
        aggressive_min_document_frequency_pct=None,
        aggressive_max_idf=None,
    )

    assert exit_code == 0

    noise_words_json = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    noise_words = json.loads(noise_words_json.read_text(encoding="utf-8"))
    assert sorted(noise_words["profiles"]["balanced"]["seed_legal"]) == ["clg", "ltd"]


def _profile_tokens_in_written_order(stats: pl.DataFrame) -> list[str]:
    """Every profile's tokens, strict then balanced then aggressive, as the
    generator writes them for `stats` under its own defaults."""
    defaults = generate_noise_words.build_parser().parse_args([])
    payloads = generate_noise_words._build_noise_word_profile_payloads(
        stats,
        seed_noise_words=set(),
        min_document_frequency_pct=0.001,
        max_idf=10.0,
        min_token_length=2,
        strict_token_count=defaults.strict_token_count,
        balanced_token_count=defaults.balanced_token_count,
        aggressive_token_count=defaults.aggressive_token_count,
        strict_cumulative_df_mass=defaults.strict_cumulative_df_mass,
        balanced_cumulative_df_mass=defaults.balanced_cumulative_df_mass,
        aggressive_cumulative_df_mass=defaults.aggressive_cumulative_df_mass,
        strict_min_document_frequency_pct=None,
        strict_max_idf=None,
        balanced_min_document_frequency_pct=None,
        balanced_max_idf=None,
        aggressive_min_document_frequency_pct=None,
        aggressive_max_idf=None,
    )
    return [
        token
        for profile in ("strict", "balanced", "aggressive")
        for token in payloads[profile]["tokens"]
    ]


def _stats_frame(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "token": [token for token, _, _ in rows],
            "document_frequency": [int(pct * 100_000) for _, pct, _ in rows],
            "document_frequency_pct": [pct for _, pct, _ in rows],
            "idf": [idf for _, _, idf in rows],
        }
    )


def test_profile_tokens_are_written_in_descending_document_frequency():
    """A cutoff placed along the list only means something if the list runs
    from the most frequent word down, across the three profiles as one list."""
    rows = [(f"w{rank:02d}", 0.30 / (rank + 1), 1.0 + 0.2 * rank) for rank in range(40)]
    stats = _stats_frame(rows)
    frequency = {token: pct for token, pct, _ in rows}

    written = _profile_tokens_in_written_order(stats)

    assert len(written) == len(set(written)) > 25
    shares = [frequency[token] for token in written]
    assert shares == sorted(shares, reverse=True)


@pytest.mark.parametrize("pooling", ["count", "equal"])
def test_pooled_profile_tokens_are_written_in_descending_document_frequency(
    pooling: str,
):
    """Pooling recomputes IDF from the pooled frequency, so the pooled list
    keeps the order a cutoff relies on, under either rule and with members of
    very different size whose frequent words differ."""
    large = _stats_frame(
        [(f"w{rank:02d}", 0.30 / (rank + 1), 1.0 + 0.2 * rank) for rank in range(30)]
    ).with_columns((pl.col("document_frequency") * 50).alias("document_frequency"))
    small = _stats_frame(
        [
            (f"w{39 - rank:02d}", 0.20 / (rank + 1), 1.0 + 0.2 * rank)
            for rank in range(30)
        ]
    )
    pooled, _provenance = pool_token_tfidf_stats(
        {"large": large, "small": small}, pooling=pooling
    )
    frequency = dict(
        zip(
            pooled.get_column("token").to_list(),
            pooled.get_column("document_frequency_pct").to_list(),
            strict=True,
        )
    )

    written = _profile_tokens_in_written_order(pooled)

    assert len(written) > 25
    shares = [frequency[token] for token in written]
    assert shares == sorted(shares, reverse=True)


def _write_system_tfidf_stats(
    roots: WorkspaceRoots, system: str, stats: pl.DataFrame
) -> Path:
    base = tokenizer_scope_dir(roots, system=system)
    base.mkdir(parents=True, exist_ok=True)
    stats_path = base / "token_tfidf_stats.parquet"
    stats.write_parquet(stats_path)
    return stats_path


def test_run_pooled_generation_writes_rule_named_sibling_of_global_noise_words(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    _write_system_tfidf_stats(
        workspace_roots,
        "ie",
        pl.DataFrame(
            {
                "token": ["ltd", "ireland"],
                "document_frequency": [9, 1],
                "document_frequency_pct": [0.9, 0.1],
                "idf": [1.1, 3.0],
            }
        ),
    )
    _write_system_tfidf_stats(
        workspace_roots,
        "gb",
        pl.DataFrame(
            {
                "token": ["ltd", "limited"],
                "document_frequency": [80, 20],
                "document_frequency_pct": [0.8, 0.2],
                "idf": [1.2, 2.0],
            }
        ),
    )

    exit_code = generate_noise_words.run_pooled_generation(
        systems=["ie", "gb"],
        pooling="equal",
        root=tmp_path,
        min_document_frequency_pct=0.05,
        max_idf=5.0,
        min_token_length=2,
        noise_words_out=None,
        seed_noise_words_path=None,
        strict_token_count=1,
        balanced_token_count=2,
        aggressive_token_count=4,
        strict_min_document_frequency_pct=None,
        strict_max_idf=None,
        balanced_min_document_frequency_pct=None,
        balanced_max_idf=None,
        aggressive_min_document_frequency_pct=None,
        aggressive_max_idf=None,
    )

    assert exit_code == 0

    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system=None) / "noise_words.equal.json"
    )
    assert noise_words_path.exists()

    payload = json.loads(noise_words_path.read_text(encoding="utf-8"))
    assert payload["pooling_rule"] == "equal"
    assert payload["pooling_systems"] == ["gb", "ie"]
    assert payload["pooling_weights"] == {"gb": 0.5, "ie": 0.5}
    assert payload["tfidf_use_case"] == "corpus"
    all_profile_tokens = {
        token
        for profile_payload in payload["profiles"].values()
        for token in profile_payload["tokens"]
    }
    assert "ltd" in all_profile_tokens

    # The un-pooled per-system noise_words.json is untouched by this run.
    assert not (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    ).exists()
    assert not (
        tokenizer_scope_dir(workspace_roots, system=None) / "noise_words.json"
    ).exists()


def test_run_pooled_generation_raises_when_a_system_stats_file_is_missing(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    _write_system_tfidf_stats(
        workspace_roots,
        "ie",
        pl.DataFrame(
            {
                "token": ["ltd"],
                "document_frequency": [9],
                "document_frequency_pct": [0.9],
                "idf": [1.1],
            }
        ),
    )

    with pytest.raises(FileNotFoundError, match="gb"):
        generate_noise_words.run_pooled_generation(
            systems=["ie", "gb"],
            pooling="count",
            root=tmp_path,
            min_document_frequency_pct=0.05,
            max_idf=5.0,
            min_token_length=2,
            noise_words_out=None,
            seed_noise_words_path=None,
            strict_token_count=1,
            balanced_token_count=2,
            aggressive_token_count=4,
            strict_min_document_frequency_pct=None,
            strict_max_idf=None,
            balanced_min_document_frequency_pct=None,
            balanced_max_idf=None,
            aggressive_min_document_frequency_pct=None,
            aggressive_max_idf=None,
        )


def test_main_requires_pooling_with_systems(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_noise_words.py",
            "--systems",
            "ie",
            "gb",
            "--root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit):
        generate_noise_words.main()


def test_main_pooled_path_end_to_end(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    _write_system_tfidf_stats(
        workspace_roots,
        "ie",
        pl.DataFrame(
            {
                "token": ["ltd", "ireland"],
                "document_frequency": [9, 1],
                "document_frequency_pct": [0.9, 0.1],
                "idf": [1.1, 3.0],
            }
        ),
    )
    _write_system_tfidf_stats(
        workspace_roots,
        "gb",
        pl.DataFrame(
            {
                "token": ["ltd", "limited"],
                "document_frequency": [80, 20],
                "document_frequency_pct": [0.8, 0.2],
                "idf": [1.2, 2.0],
            }
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_noise_words.py",
            "--systems",
            "ie",
            "gb",
            "--pooling",
            "count",
            "--root",
            str(tmp_path),
            "--min-document-frequency-pct",
            "0.05",
            "--max-idf",
            "5.0",
        ],
    )

    exit_code = generate_noise_words.main()

    assert exit_code == 0
    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system=None) / "noise_words.count.json"
    )
    assert noise_words_path.exists()
    payload = json.loads(noise_words_path.read_text(encoding="utf-8"))
    assert payload["pooling_rule"] == "count"


def test_main_honors_explicit_output_paths(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2"],
            "name": ["alpha systems", "beta systems"],
        }
    ).write_parquet(corpus_path)

    stats_out = tmp_path / "out" / "stats.parquet"
    token_set_out = tmp_path / "out" / "token_set.json"
    noise_words_out = tmp_path / "out" / "noise_words.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_noise_words.py",
            "--system",
            "ie",
            "--root",
            str(tmp_path),
            "--corpus-path",
            str(corpus_path),
            "--token-stats-out",
            str(stats_out),
            "--token-set-out",
            str(token_set_out),
            "--noise-words-out",
            str(noise_words_out),
            "--min-document-frequency-pct",
            "0.5",
            "--max-idf",
            "2.0",
        ],
    )

    exit_code = generate_noise_words.main()

    assert exit_code == 0
    assert stats_out.exists()
    assert stats_out.with_suffix(".csv").exists()
    assert token_set_out.exists()
    assert noise_words_out.exists()
