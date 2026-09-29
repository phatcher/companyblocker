from pathlib import Path

import polars as pl
import pytest

from acquisition import tokenizer_ops
from acquisition.tokenizer_ops import (
    sample_training_corpus,
    sample_training_corpus_for_global,
    tokenize_name,
    tokenize_name_dual,
    train_wordpiece,
)
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def test_sample_training_corpus_raises_when_no_matching_chunks(
    tmp_path: Path, layer_fixture_dir
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        FileNotFoundError, match="No cleansed parquet files under .*jurisdiction_code="
    ):
        sample_training_corpus(cleansed_dir, tmp_path / "corpus.parquet")


def test_sample_training_corpus_raises_when_name_column_missing(
    tmp_path: Path, layer_fixture_dir
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"other": ["x"]}).write_parquet(partition_dir / "part-001.parquet")

    with pytest.raises(
        ValueError, match="contain required columns 'system_uri' and 'name_cleansed'"
    ):
        sample_training_corpus(cleansed_dir, tmp_path / "corpus.parquet")


def test_sample_training_corpus_writes_lines_and_skips_empty_names(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="gb") / "training_corpus.parquet"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://1", "gb://2", "gb://3", "gb://4", "gb://5", "gb://6"],
            "name_cleansed": ["alpha ltd", "", "   ", None, "\t", "beta ltd"],
        }
    ).write_parquet(partition_dir / "part-001.parquet")

    count = sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    assert count == 2
    out = pl.read_parquet(corpus_path)
    assert out.columns == ["system_uri", "name"]
    assert out["name"].to_list() == ["alpha ltd", "beta ltd"]


def test_sample_training_corpus_reads_family_split_primary_directory(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    """Real cleansed/ output lives under a `primary/` family directory, not
    directly at the layer's own top level -- the old hand-built
    PARTITION_PREFIX glob only found the pre-split shape the other test
    above exercises.
    """
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "primary" / "jurisdiction_code=gb"
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="gb") / "training_corpus.parquet"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://1", "gb://2"],
            "name_cleansed": ["alpha ltd", "beta ltd"],
        }
    ).write_parquet(partition_dir / "part-001.parquet")

    count = sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    assert count == 2
    out = pl.read_parquet(corpus_path)
    assert out["name"].to_list() == ["alpha ltd", "beta ltd"]


def _write_partition(partition_dir: Path, *, uris: list[str], names: list[str]) -> None:
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": uris, "name_cleansed": names}).write_parquet(
        partition_dir / "part-001.parquet"
    )


def test_sample_training_corpus_leaves_unchanged_content_untouched(
    tmp_path: Path, layer_fixture_dir
):
    # A resample that reproduces byte-identical content must not rewrite the
    # file -- an unnecessary rewrite would bump its mtime and could fool
    # is_final_result_fresh() into treating a still-valid promoted result as
    # stale relative to a corpus that never actually changed.
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    _write_partition(
        cleansed_dir / "jurisdiction_code=gb",
        uris=["gb://1"],
        names=["alpha ltd"],
    )
    corpus_path = tmp_path / "corpus.parquet"

    sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)
    first_mtime_ns = corpus_path.stat().st_mtime_ns

    sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    assert corpus_path.stat().st_mtime_ns == first_mtime_ns


def test_sample_training_corpus_changed_content_proceeds_when_no_sweep_exists(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    _write_partition(partition_dir, uris=["gb://1"], names=["alpha ltd"])
    corpus_path = tokenizer_scope_dir(workspace_roots, system="gb") / "corpus.parquet"

    sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)
    _write_partition(
        partition_dir, uris=["gb://1", "gb://2"], names=["alpha ltd", "beta ltd"]
    )

    count = sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    assert count == 2
    assert pl.read_parquet(corpus_path)["name"].to_list() == ["alpha ltd", "beta ltd"]


def test_sample_training_corpus_refuses_when_sweep_at_risk_without_flag(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    _write_partition(partition_dir, uris=["gb://1"], names=["alpha ltd"])
    corpus_path = tokenizer_scope_dir(workspace_roots, system="gb") / "corpus.parquet"
    sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)
    original_bytes = corpus_path.read_bytes()

    scope_dir = corpus_path.parent
    wordpiece_leaf = scope_dir / "wordpiece" / "optimize" / "corpushash" / "gridhash"
    sentencepiece_leaf = (
        scope_dir / "sentencepiece_bpe" / "optimize" / "corpushash" / "gridhash"
    )
    wordpiece_leaf.mkdir(parents=True)
    sentencepiece_leaf.mkdir(parents=True)
    pl.DataFrame({"seed": [42, 43]}).write_parquet(
        wordpiece_leaf / "optimize_runs.parquet"
    )
    pl.DataFrame({"seed": [42]}).write_parquet(
        sentencepiece_leaf / "optimize_runs.parquet"
    )

    _write_partition(
        partition_dir, uris=["gb://1", "gb://2"], names=["alpha ltd", "beta ltd"]
    )

    with pytest.raises(RuntimeError, match="wordpiece \\(2 candidates\\)"):
        sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    assert corpus_path.read_bytes() == original_bytes


def test_sample_training_corpus_proceeds_when_sweep_at_risk_with_flag(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    _write_partition(partition_dir, uris=["gb://1"], names=["alpha ltd"])
    corpus_path = tokenizer_scope_dir(workspace_roots, system="gb") / "corpus.parquet"
    sample_training_corpus(cleansed_dir, corpus_path, sample_fraction=1.0)

    wordpiece_leaf = (
        corpus_path.parent / "wordpiece" / "optimize" / "corpushash" / "gridhash"
    )
    wordpiece_leaf.mkdir(parents=True)
    pl.DataFrame({"seed": [42]}).write_parquet(wordpiece_leaf / "optimize_runs.parquet")

    _write_partition(
        partition_dir, uris=["gb://1", "gb://2"], names=["alpha ltd", "beta ltd"]
    )

    count = sample_training_corpus(
        cleansed_dir, corpus_path, sample_fraction=1.0, allow_optimize_wipe=True
    )

    assert count == 2
    assert pl.read_parquet(corpus_path)["name"].to_list() == ["alpha ltd", "beta ltd"]


def test_sample_training_corpus_for_systems_skips_empty_names(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    gb_partition = gb_cleansed / "jurisdiction_code=gb"
    gb_partition.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gb://1", "gb://2", "gb://3", "gb://4", "gb://5"],
            "name_cleansed": ["alpha ltd", "", "   ", None, "beta ltd"],
        }
    ).write_parquet(gb_partition / "part-001.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed],
        corpus_path=corpus_path,
        sample_fraction=1.0,
    )

    assert count == 2
    out = pl.read_parquet(corpus_path)
    assert out.columns == ["system_uri", "name"]
    assert out["name"].to_list() == ["alpha ltd", "beta ltd"]


def test_sample_training_corpus_for_systems_rejects_invalid_sample_fraction(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="sample_fraction must be in the interval"):
        sample_training_corpus_for_global(
            cleansed_dirs=[],
            corpus_path=tmp_path / "corpus.parquet",
            sample_fraction=1.5,
        )


def test_sample_training_corpus_for_systems_raises_when_expected_prefix_files_missing(
    tmp_path: Path,
    layer_fixture_dir,
):
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": ["alpha"]}).write_parquet(
        cleansed_dir / "sample.parquet"
    )

    with pytest.raises(
        FileNotFoundError, match="No cleansed parquet files under .*jurisdiction_code="
    ):
        sample_training_corpus_for_global(
            cleansed_dirs=[cleansed_dir], corpus_path=tmp_path / "corpus.parquet"
        )


def test_train_wordpiece_delegates_to_runtime(tmp_path: Path, mocker):
    runtime = mocker.patch.object(
        tokenizer_ops, "_train_wordpiece_runtime", return_value=123
    )

    resolved = train_wordpiece(
        tmp_path / "corpus.parquet",
        tmp_path / "tokenizer.json",
        vocab_size=50,
        show_progress=False,
    )

    assert resolved == 123
    runtime.assert_called_once_with(
        corpus_path=tmp_path / "corpus.parquet",
        tokenizer_path=tmp_path / "tokenizer.json",
        vocab_size=50,
        show_progress=False,
    )


def test_tokenize_name_delegates_to_runtime(tmp_path: Path, layer_fixture_dir, mocker):
    runtime = mocker.patch.object(
        tokenizer_ops, "_tokenize_name_runtime", return_value=7
    )

    rows = tokenize_name(
        cleansed_dir=layer_fixture_dir("gb", layer="cleansed"),
        tokenized_dir=layer_fixture_dir("gb", layer="tokenized"),
        tokenizer_path=tmp_path / "tokenizer.json",
        name_col="name_cleansed",
        token_col="name_tokens",
        input_file="gb-001.parquet",
    )

    assert rows == 7
    runtime.assert_called_once()
    assert runtime.call_args.kwargs["input_file"] == "gb-001.parquet"
    assert runtime.call_args.kwargs["token_col"] == "name_tokens"


def test_tokenize_name_dual_delegates_to_runtime(
    tmp_path: Path, layer_fixture_dir, mocker
):
    runtime = mocker.patch.object(
        tokenizer_ops, "_tokenize_name_dual_runtime", return_value=11
    )

    rows = tokenize_name_dual(
        cleansed_dir=layer_fixture_dir("gb", layer="cleansed"),
        tokenized_dir=layer_fixture_dir("gb", layer="tokenized"),
        country_tokenizer_path=tmp_path / "country.json",
        global_tokenizer_path=tmp_path / "global.json",
        country_token_col="country_tokens",
        global_token_col="global_tokens",
    )

    assert rows == 11
    runtime.assert_called_once()
    assert runtime.call_args.kwargs["country_token_col"] == "country_tokens"
    assert runtime.call_args.kwargs["global_token_col"] == "global_tokens"


def test_sample_training_corpus_for_systems_combines_outputs(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    ie_cleansed = layer_fixture_dir("ie", layer="cleansed")
    gb_partition = gb_cleansed / "jurisdiction_code=gb"
    ie_partition = ie_cleansed / "jurisdiction_code=ie"
    gb_partition.mkdir(parents=True, exist_ok=True)
    ie_partition.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {"system_uri": ["gb://1", "gb://2"], "name_cleansed": ["alpha ltd", "beta ltd"]}
    ).write_parquet(gb_partition / "part-001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["ie://1", "ie://2"],
            "name_cleansed": ["gamma ltd", "delta ltd"],
        }
    ).write_parquet(ie_partition / "part-001.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed, ie_cleansed],
        corpus_path=corpus_path,
        sample_fraction=1.0,
    )

    assert count == 4
    out = pl.read_parquet(corpus_path)
    assert out["name"].to_list() == ["alpha ltd", "beta ltd", "gamma ltd", "delta ltd"]


def test_sample_training_corpus_for_global_reads_family_split_primary_directories(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """Real cleansed/ output lives under a `primary/` family directory, not
    directly at the layer's own top level.
    """
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    gb_partition = gb_cleansed / "primary" / "jurisdiction_code=gb"
    gb_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"system_uri": ["gb://1", "gb://2"], "name_cleansed": ["alpha ltd", "beta ltd"]}
    ).write_parquet(gb_partition / "part-001.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed],
        corpus_path=corpus_path,
        sample_fraction=1.0,
    )

    assert count == 2
    out = pl.read_parquet(corpus_path)
    assert out["name"].to_list() == ["alpha ltd", "beta ltd"]


def test_sample_training_corpus_for_systems_uses_all_files_per_folder(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    ie_cleansed = layer_fixture_dir("ie", layer="cleansed")
    gb_partition = gb_cleansed / "jurisdiction_code=gb"
    ie_partition = ie_cleansed / "jurisdiction_code=ie"
    gb_partition.mkdir(parents=True, exist_ok=True)
    ie_partition.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 5):
        pl.DataFrame(
            {
                "system_uri": [
                    f"gb://{idx}/a",
                    f"gb://{idx}/b",
                    f"gb://{idx}/c",
                    f"gb://{idx}/d",
                    f"gb://{idx}/e",
                ],
                "name_cleansed": [
                    f"gb-{idx}-a",
                    f"gb-{idx}-b",
                    f"gb-{idx}-c",
                    f"gb-{idx}-d",
                    f"gb-{idx}-e",
                ],
            }
        ).write_parquet(gb_partition / f"part-{idx:03d}.parquet")

    for idx in range(1, 3):
        pl.DataFrame(
            {
                "system_uri": [
                    f"ie://{idx}/a",
                    f"ie://{idx}/b",
                    f"ie://{idx}/c",
                    f"ie://{idx}/d",
                    f"ie://{idx}/e",
                ],
                "name_cleansed": [
                    f"ie-{idx}-a",
                    f"ie-{idx}-b",
                    f"ie-{idx}-c",
                    f"ie-{idx}-d",
                    f"ie-{idx}-e",
                ],
            }
        ).write_parquet(ie_partition / f"part-{idx:03d}.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed, ie_cleansed],
        corpus_path=corpus_path,
        sample_fraction=1.0,
    )

    assert count == 30
    assert pl.read_parquet(corpus_path).height == 30


def test_sample_training_corpus_for_systems_equal_mode_caps_large_systems(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    ie_cleansed = layer_fixture_dir("ie", layer="cleansed")
    gb_partition = gb_cleansed / "jurisdiction_code=gb"
    ie_partition = ie_cleansed / "jurisdiction_code=ie"
    gb_partition.mkdir(parents=True, exist_ok=True)
    ie_partition.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 5):
        pl.DataFrame(
            {
                "system_uri": [f"gb://{idx}/{n}" for n in range(100)],
                "name_cleansed": [f"gb-{idx}-{n}" for n in range(100)],
            }
        ).write_parquet(gb_partition / f"part-{idx:03d}.parquet")
    pl.DataFrame(
        {
            "system_uri": [f"ie://{n}" for n in range(100)],
            "name_cleansed": [f"ie-{n}" for n in range(100)],
        }
    ).write_parquet(ie_partition / "part-001.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed, ie_cleansed],
        corpus_path=corpus_path,
        sample_fraction=1.0,
        balance_mode="equal",
        system_target_rows=50,
        system_min_rows=0,
        system_max_rows=None,
        file_min_rows=0,
        balance_seed=123,
    )

    assert count == 100
    lines = pl.read_parquet(corpus_path)["name"].to_list()
    assert sum(1 for line in lines if line.startswith("gb ")) == 50
    assert sum(1 for line in lines if line.startswith("ie ")) == 50


def test_sample_training_corpus_for_systems_respects_global_file_min_rows(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    gb_cleansed = layer_fixture_dir("gb", layer="cleansed")
    gb_partition = gb_cleansed / "jurisdiction_code=gb"
    gb_partition.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 3):
        pl.DataFrame(
            {
                "system_uri": [f"gb://{idx}/{n}" for n in range(10)],
                "name_cleansed": [f"gb-{idx}-{n}" for n in range(10)],
            }
        ).write_parquet(gb_partition / f"part-{idx:03d}.parquet")

    corpus_path = tokenizer_scope_dir(workspace_roots) / "training_corpus.parquet"
    count = sample_training_corpus_for_global(
        cleansed_dirs=[gb_cleansed],
        corpus_path=corpus_path,
        sample_fraction=0.01,
        balance_mode="equal",
        system_target_rows=100,
        system_min_rows=0,
        system_max_rows=None,
        file_min_rows=3,
        balance_seed=42,
    )

    assert count >= 6


def test_a_column_that_keeps_capitals_gives_the_corpus_of_its_lowercased_twin(
    tmp_path: Path, layer_fixture_dir
):
    # The corpus is what a tokenizer learns from. Training itself is not
    # repeatable, so two trainings on one corpus are not compared here.
    names = ["Acme Holdings LTD", "Beta Trading Co.", "Gamma Café Services"]
    corpora = {}
    for system, column in (
        ("gb", names),
        ("ie", [name.lower() for name in names]),
    ):
        cleansed_dir = layer_fixture_dir(system, layer="cleansed")
        _write_partition(
            cleansed_dir / f"jurisdiction_code={system}",
            uris=[f"{system}://{n}" for n in range(len(column))],
            names=column,
        )
        corpora[system] = tmp_path / f"{system}_corpus.parquet"
        sample_training_corpus(cleansed_dir, corpora[system], sample_fraction=1.0)

    assert pl.read_parquet(corpora["gb"])["name"].to_list() == [
        "acme holdings ltd",
        "beta trading co",
        "gamma cafe services",
    ]
    assert (
        pl.read_parquet(corpora["gb"])["name"].to_list()
        == pl.read_parquet(corpora["ie"])["name"].to_list()
    )
