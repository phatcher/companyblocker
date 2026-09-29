from pathlib import Path

import pytest
from company_tokenize import (
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
)
from company_tokenize.paths import (
    TOKENIZER_IDS,
    TRAINING_CORPUS_FILENAME,
    promoted_candidate_key,
    resolve_country_tokenizer_paths,
    resolve_global_tokenizer_paths,
    resolve_tokenizer_paths,
    scope_directory_files,
    tokenizer_directory,
    tokenizer_directory_files,
    tokenizer_id,
    write_candidate_directory,
)


def test_resolve_country_tokenizer_paths(tmp_path: Path):
    paths = resolve_country_tokenizer_paths(tmp_path, "GB")

    assert paths.scope == "system"
    assert paths.directory == tmp_path / "gb"
    assert paths.corpus_path == tmp_path / "gb" / "training_corpus.parquet"


def test_resolve_global_tokenizer_paths(tmp_path: Path):
    paths = resolve_global_tokenizer_paths(tmp_path, "baseline")

    assert paths.scope == "global"
    assert paths.directory == tmp_path / "global"


def test_resolve_tokenizer_paths_supports_custom_corpus_filename(tmp_path: Path):
    paths = resolve_tokenizer_paths(
        tokenizer_root=tmp_path,
        scope="country",
        system="GB",
        corpus_filename="bpe_training_corpus.txt",
    )

    assert paths.corpus_path == tmp_path / "gb" / "bpe_training_corpus.txt"


def test_training_corpus_filename_default_constant():
    assert TRAINING_CORPUS_FILENAME == "training_corpus.parquet"


def test_resolve_tokenizer_paths_country_requires_system(tmp_path: Path):
    with pytest.raises(ValueError, match="system code is required"):
        resolve_tokenizer_paths(tokenizer_root=tmp_path, scope="country", system=None)


def test_resolve_tokenizer_paths_supports_global_scope_case_insensitive(tmp_path: Path):
    paths = resolve_tokenizer_paths(tokenizer_root=tmp_path, scope=" GLOBAL ")

    assert paths.scope == "global"
    assert paths.directory == tmp_path / "global"


def test_resolve_tokenizer_paths_rejects_unknown_scope(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported tokenizer scope"):
        resolve_tokenizer_paths(tokenizer_root=tmp_path, scope="team")


@pytest.mark.parametrize(
    ("trainer", "model"),
    [("wordpiece", "model.json"), ("sentencepiece", "model.model")],
)
def test_tokenizer_directory_files_names_are_fixed_but_for_the_model_extension(
    tmp_path: Path, trainer: str, model: str
):
    files = tokenizer_directory_files(tmp_path, trainer=trainer)

    assert files.directory == tmp_path
    assert files.model == tmp_path / model
    assert files.metadata == tmp_path / "metadata.json"
    assert files.token_scores == tmp_path / "token_scores.json"
    assert files.metrics == tmp_path / "metrics.json"
    assert files.optimize_summary == tmp_path / "optimize_summary.json"


def test_scope_directory_files_name_the_corpus_and_what_is_derived_from_it(
    tmp_path: Path,
):
    """A scope's statistics belong to its corpus, not to any tokenizer, so
    they are named with no trainer and sit beside the corpus."""
    files = scope_directory_files(tmp_path)

    assert files.directory == tmp_path
    assert files.corpus == tmp_path / "training_corpus.parquet"
    assert files.metadata == tmp_path / "metadata.json"
    assert files.tfidf_stats == tmp_path / "token_tfidf_stats.parquet"
    assert files.tfidf_stats_csv == tmp_path / "token_tfidf_stats.csv"
    assert files.token_set == tmp_path / "token_set.json"
    assert files.noise_words == tmp_path / "noise_words.json"
    assert files.noise_word_candidates == tmp_path / "noise_word_candidates.json"
    assert files.pooled_noise_words("equal") == tmp_path / "noise_words.equal.json"


def test_tokenizer_directory_gives_each_id_its_own_folder(tmp_path: Path):
    folders = {
        tokenizer_directory(tmp_path, trainer=trainer, tokenizer_encoding=encoding)
        for trainer, encoding in [
            ("wordpiece", None),
            ("sentencepiece", "bpe"),
            ("sentencepiece", "unigram"),
        ]
    }

    assert folders == {tmp_path / identifier for identifier in TOKENIZER_IDS}


def test_tokenizers_in_one_scope_share_no_file_and_no_name_says_which_it_is(
    tmp_path: Path,
):
    """Training `bpe` and then `unigram` for one scope cannot overwrite either:
    every file a tokenizer owns, its sweeps included, sits in its own folder."""
    owned: dict[str, set[Path]] = {}
    for identifier, fields in TOKENIZER_IDS.items():
        folder = tokenizer_directory(
            tmp_path,
            trainer=fields.tokenizer,
            tokenizer_encoding=fields.tokenizer_encoding,
        )
        files = tokenizer_directory_files(folder, trainer=fields.tokenizer)
        optimize_dir = resolve_optimize_dir(
            scope_directory=tmp_path,
            trainer=fields.tokenizer,
            tokenizer_encoding=fields.tokenizer_encoding,
        )
        sweep = resolve_optimize_sweep_paths(
            optimize_dir=optimize_dir,
            corpus_content_hash="0a1b2c3d4e5f",
            grid_hash="9r1d",
        )
        owned[identifier] = {
            files.model,
            files.metadata,
            files.token_scores,
            resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir),
            sweep.run_log_path,
            sweep.summary_path,
            sweep.canary_sidecar_path,
        }
        for path in owned[identifier]:
            assert folder in path.parents
            for word in ("wordpiece", "sentencepiece", "bpe", "unigram"):
                assert word not in path.name

    assert not owned["sentencepiece_bpe"] & owned["sentencepiece_unigram"]
    assert not owned["wordpiece"] & owned["sentencepiece_bpe"]


def test_tokenizer_directory_files_answer_alike_wherever_the_directory_sits(
    tmp_path: Path,
):
    """A promoted candidate in an artifact store and a tokenizer shipped as a
    package resource are both just a directory, so moving one is a plain copy."""
    store = tmp_path / "artifacts" / "tokenizers" / "data" / "ie" / "wordpiece" / "k1"
    resource = tmp_path / "company_tokenize" / "resources" / "ie"

    in_store = tokenizer_directory_files(store, trainer="wordpiece")
    shipped = tokenizer_directory_files(resource, trainer="wordpiece")

    for name in ("model", "metadata", "token_scores"):
        assert getattr(in_store, name).relative_to(store) == getattr(
            shipped, name
        ).relative_to(resource)


def test_tokenizer_directory_files_rejects_an_unknown_trainer(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported trainer"):
        tokenizer_directory_files(tmp_path, trainer="bytepair")


def test_promoted_candidate_key_reads_by_parameters_and_closes_on_the_model(
    tmp_path: Path,
):
    model = tmp_path / "wordpiece.json"
    model.write_text("one", encoding="utf-8")
    key = promoted_candidate_key(
        corpus_content_hash="0a1b2c3d4e5f",
        vocab_size=24000,
        min_frequency=2,
        model_path=model,
    )

    assert key.startswith("0a1b2c3d_v24000_mf2_")
    assert key == promoted_candidate_key(
        corpus_content_hash="0a1b2c3d4e5f",
        vocab_size=24000,
        min_frequency=2,
        model_path=model,
    )
    model.write_text("two", encoding="utf-8")
    assert key != promoted_candidate_key(
        corpus_content_hash="0a1b2c3d4e5f",
        vocab_size=24000,
        min_frequency=2,
        model_path=model,
    )


def test_write_candidate_directory_files_sources_under_the_package_names(
    tmp_path: Path,
):
    model = tmp_path / "out" / "custom-name.json"
    metadata = tmp_path / "out" / "metadata.json"
    model.parent.mkdir()
    model.write_text("model", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")

    written = write_candidate_directory(
        tmp_path / "candidate",
        model=model,
        metadata=metadata,
        token_scores=tmp_path / "out" / "absent.json",
        trainer="wordpiece",
    )

    assert written.model.name == "model.json"
    assert written.model.read_text(encoding="utf-8") == "model"
    assert written.metadata.name == "metadata.json"
    assert not written.token_scores.exists()


def test_write_candidate_directory_refuses_a_missing_model(tmp_path: Path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="candidate file not found"):
        write_candidate_directory(
            tmp_path / "candidate",
            model=tmp_path / "absent.json",
            metadata=metadata,
            trainer="wordpiece",
        )


def test_tokenizer_id_carries_the_sentencepiece_model_type():
    assert tokenizer_id(" WordPiece ") == "wordpiece"
    assert tokenizer_id("sentencepiece") == "sentencepiece_bpe"
    assert tokenizer_id("sentencepiece", "Unigram") == ("sentencepiece_unigram")


def test_tokenizer_id_is_a_lookup_in_the_declared_table():
    for identifier, fields in TOKENIZER_IDS.items():
        assert tokenizer_id(fields.tokenizer, fields.tokenizer_encoding) == identifier

    with pytest.raises(ValueError, match="No tokenizer id"):
        tokenizer_id("sentencepiece", "char")
