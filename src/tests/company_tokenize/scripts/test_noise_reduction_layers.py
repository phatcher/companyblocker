from __future__ import annotations

from pathlib import Path

import polars as pl
from company_cleanse.config import get_profiled_noise_words
from company_tokenize import (
    compute_corpus_token_tfidf_stats,
    resolve_noise_words,
    tokenize_name_dataframe,
)

from scripts import generate_noise_words
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots

# Grounds the claim that noise reduction happens in three layers:
#   1. cleanse-stage static profile (company_cleanse.noise_words.json)
#   2. tokenize-stage trim, which falls back to layer 1's own list unless a
#      per-corpus noise_words_path is given
#   3. analysis-stage advisory-only stoplist candidates (not covered here --
#      it is never consumed as a live filter, so there is nothing to prove)


def test_tokenize_stage_packaged_default_is_sourced_from_cleanse_stage_profile():
    """Layer 2's fallback (no noise_words_path) is literally layer 1's list, not an
    independently-defined one -- so the two layers are NOT independent by default;
    they only diverge once a per-corpus noise_words_path override is supplied."""
    tokenize_default = resolve_noise_words(noise_words_profile="aggressive")
    cleanse_default = {
        token.lower()
        for token in get_profiled_noise_words(noise_words_profile="aggressive")
    }

    assert tokenize_default == cleanse_default
    assert "ltd" in tokenize_default


def test_precomputed_corpus_stats_are_pre_trim_while_tokenizer_output_is_post_trim(
    tmp_path: Path,
):
    """The confound between the two stages: compute_corpus_token_tfidf_stats
    (what analysis.token_rarity and generate_noise_words.py read) sees 'ltd'
    because it runs directly on text, but tokenize_name_dataframe's default
    trim removes 'ltd' before the tokenizer ever sees it. A Zipf/rarity
    analysis built on the tokenizer's output columns would therefore already
    have lost 'ltd' -- not because it's rare or OOV, but because the default
    trim removed it first."""
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {
            "system_uri": ["s:1", "s:2"],
            "name": ["alpha ltd", "beta ltd"],
        }
    ).write_parquet(corpus_path)

    stats = compute_corpus_token_tfidf_stats(corpus_path)
    assert "ltd" in stats.get_column("token").to_list()

    tokenized = tokenize_name_dataframe(
        pl.DataFrame({"name_cleansed": ["alpha ltd", "beta ltd"]}),
        noise_words_profile="aggressive",
    )
    all_tokens = {token for row in tokenized["name_tokens"].to_list() for token in row}
    assert "ltd" not in all_tokens


def test_per_corpus_computed_noise_words_diverge_from_the_packaged_default(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """Proves 'per corpus noise' is a real, distinct mechanism: a domain word
    ('acme') that dominates one corpus's document frequency gets selected as a
    noise word by generate_noise_words.py even though it is nowhere in the
    packaged static noise-word list -- and once pointed at via noise_words_path,
    it changes tokenizer output in a way the packaged default alone would not."""
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "training_corpus.parquet"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie:1", "ie:2", "ie:3"],
            "name": ["acme systems", "acme holdings", "acme trading"],
        }
    ).write_parquet(corpus_path)

    exit_code = generate_noise_words.run_generation(
        system="ie",
        profile="default",
        root=tmp_path,
        corpus_path=None,
        name_col="name",
        min_document_frequency_pct=0.5,
        max_idf=2.0,
        min_token_length=2,
        token_stats_out=None,
        token_set_out=None,
        noise_words_out=None,
        seed_noise_words_path=None,
        strict_token_count=1,
        balanced_token_count=5,
        aggressive_token_count=5,
        strict_min_document_frequency_pct=None,
        strict_max_idf=None,
        balanced_min_document_frequency_pct=None,
        balanced_max_idf=None,
        aggressive_min_document_frequency_pct=None,
        aggressive_max_idf=None,
    )
    assert exit_code == 0

    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )

    # "acme" dominates document frequency in this tiny corpus, so a per-corpus
    # sweep selects it -- but it is not a legal-form/noise word, so the packaged
    # static default never would.
    packaged_default = resolve_noise_words(noise_words_profile="aggressive")
    assert "acme" not in packaged_default

    per_corpus = resolve_noise_words(
        noise_words_path=noise_words_path, noise_words_profile="aggressive"
    )
    assert "acme" in per_corpus

    df = pl.DataFrame(
        {"name_cleansed": ["acme systems", "acme holdings", "acme trading"]}
    )

    # The bundled WordPiece vocab has no whole-word "acme" entry, so when the
    # word survives trimming it gets split into subword pieces ("ac", "##me")
    # rather than appearing as a single "acme" token -- that fragmentation is
    # itself part of the tokenizer-fertility story, but here it's just the
    # signal that trimming did *not* remove the word before encoding.
    default_tokens = tokenize_name_dataframe(df, noise_words_profile="aggressive")
    assert any(
        "ac" in row and "##me" in row for row in default_tokens["name_tokens"].to_list()
    )

    per_corpus_trimmed = tokenize_name_dataframe(
        df, noise_words_path=noise_words_path, noise_words_profile="aggressive"
    )
    assert all(
        "ac" not in row and "##me" not in row
        for row in per_corpus_trimmed["name_tokens"].to_list()
    )
