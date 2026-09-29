from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest

from tests.promoted_tokenizers import promoted_tokenizer_files
from validation import runner
from validation.config import ValidationRunConfig
from validation.prepared_dataset import PreparedSystemStats
from workspace.roots import WorkspaceRoots


def _config(roots: WorkspaceRoots, **overrides: Any) -> ValidationRunConfig:
    """A stub standing in for `ValidationRunConfig`.

    The functions under test read only the attributes set below, so a
    `SimpleNamespace` avoids constructing (and keeping in step with) the full
    dataclass; the cast records that substitution.
    """
    base: dict[str, Any] = {
        "roots": roots,
        "representation": "wordpiece",
        "text_view": "tokens",
        "tokenizer_scope": "country",
        "tokenizer_profile": "promoted",
        "tokenizer_path": None,
        "tokenizer": "wordpiece",
        "noise_words_profile": "default",
        "noise_words_set_kind": "standard",
    }
    base.update(overrides)
    return cast("ValidationRunConfig", SimpleNamespace(**base))


def _stats(system_dir: Path) -> PreparedSystemStats:
    return PreparedSystemStats(
        system="gleif",
        system_dir=system_dir,
        total_rows_by_country={},
        filtered_out_rows_by_country={},
        kept_rows_by_country={},
    )


def test_prepared_country_fingerprint_changes_with_metadata_and_partition_contents(
    tmp_path: Path,
):
    baseline = runner._prepared_country_fingerprint(system_dir=tmp_path, country="fr")

    (tmp_path / "_prepared_metadata.json").write_text("{}", encoding="utf-8")
    with_metadata = runner._prepared_country_fingerprint(
        system_dir=tmp_path, country="fr"
    )
    assert with_metadata != baseline

    partition_dir = tmp_path / "primary" / "jurisdiction_code=fr"
    partition_dir.mkdir(parents=True)
    (partition_dir / "part-0.parquet").write_text("x", encoding="utf-8")
    with_partition = runner._prepared_country_fingerprint(
        system_dir=tmp_path, country="fr"
    )
    assert with_partition != with_metadata

    assert len(baseline) == 16


def test_tokenizer_artifact_path_resolves_country_and_global_scope(
    workspace_roots: WorkspaceRoots,
):
    country_model = promoted_tokenizer_files(workspace_roots, system="fr").model
    global_model = promoted_tokenizer_files(workspace_roots, system=None).model

    country_config = _config(workspace_roots, tokenizer_scope="country")
    assert (
        runner._tokenizer_artifact_path(config=country_config, system="fr")
        == country_model
    )

    global_config = _config(workspace_roots, tokenizer_scope="global")
    assert (
        runner._tokenizer_artifact_path(config=global_config, system="fr")
        == global_model
    )


def _promoted_tokenizer(roots: WorkspaceRoots) -> Path:
    """The `ie` wordpiece model file the pointer names as `promoted`."""
    promoted_path = promoted_tokenizer_files(roots, system="ie").model
    promoted_path.write_text("promoted-model", encoding="utf-8")
    return promoted_path


def test_tokenizer_artifact_path_defaults_to_the_promoted_tokenizer(
    workspace_roots: WorkspaceRoots,
):
    promoted_path = _promoted_tokenizer(workspace_roots)
    config = _config(workspace_roots)

    assert runner._tokenizer_artifact_path(config=config, system="ie") == promoted_path


def test_tokenizer_artifact_path_reads_the_tokenizer_the_profile_names(
    workspace_roots: WorkspaceRoots,
):
    promoted_path = _promoted_tokenizer(workspace_roots)
    naive_path = promoted_tokenizer_files(
        workspace_roots, system="ie", key="first_guess", profile="naive"
    ).model
    config = _config(workspace_roots, tokenizer_profile="naive")

    assert runner._tokenizer_artifact_path(config=config, system="ie") == naive_path
    assert naive_path != promoted_path


def test_tokenizer_artifact_path_uses_a_supplied_file_as_it_is(
    workspace_roots: WorkspaceRoots, tmp_path: Path
):
    candidate = tmp_path / "v21000_mf8.json"
    candidate.write_text("{}", encoding="utf-8")

    config = _config(workspace_roots, tokenizer_path=str(candidate))
    assert runner._tokenizer_artifact_path(config=config, system="ie") == candidate

    absent = _config(workspace_roots, tokenizer_path=str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="names no file"):
        runner._tokenizer_artifact_path(config=absent, system="ie")


def test_tokenizer_artifact_fingerprint_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        runner._tokenizer_artifact_fingerprint(tokenizer_path=tmp_path / "missing.json")


def test_tokenizer_artifact_fingerprint_returns_stable_digest(tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("model", encoding="utf-8")

    digest = runner._tokenizer_artifact_fingerprint(tokenizer_path=tokenizer_path)

    assert len(digest) == 16
    assert digest == runner._tokenizer_artifact_fingerprint(
        tokenizer_path=tokenizer_path
    )


def test_token_cache_manifest_builds_expected_payload():
    manifest = runner._token_cache_manifest(
        system="gleif",
        country="fr",
        prepared_fingerprint="pf",
        tokenizer_path=Path("tokenizer.json"),
        tokenizer_fingerprint="tf",
        tokenizer="wordpiece",
        tokenizer_scope="country",
        tokenizer_profile="default",
        noise_words_profile="default",
        noise_words_set_kind="standard",
    )

    assert manifest == {
        "system": "gleif",
        "country": "fr",
        "prepared_fingerprint": "pf",
        "tokenizer_path": "tokenizer.json",
        "tokenizer_fingerprint": "tf",
        "tokenizer": "wordpiece",
        "tokenizer_scope": "country",
        "tokenizer_profile": "default",
        "noise_words_profile": "default",
        "noise_words_set_kind": "standard",
    }


def test_load_cached_token_frame_misses_when_files_absent(tmp_path: Path):
    frame, hit = runner._load_cached_token_frame(
        cache_dir=tmp_path / "cache", expected_manifest={"a": 1}
    )
    assert frame is None
    assert hit is False


def test_write_then_load_token_cache_round_trips_on_matching_manifest(tmp_path: Path):
    manifest: dict[str, object] = {"a": 1}
    token_frame = pl.DataFrame({"validation_tokens": [["a", "b"]]})

    paths = runner._write_token_cache(
        cache_dir=tmp_path / "cache", manifest=manifest, token_frame=token_frame
    )
    assert paths["manifest"].exists()
    assert paths["tokenized"].exists()

    loaded_frame, hit = runner._load_cached_token_frame(
        cache_dir=tmp_path / "cache", expected_manifest=manifest
    )
    assert hit is True
    assert loaded_frame is not None
    assert loaded_frame.equals(token_frame)


def test_load_cached_token_frame_misses_on_manifest_mismatch(tmp_path: Path):
    token_frame = pl.DataFrame({"validation_tokens": [["a"]]})
    runner._write_token_cache(
        cache_dir=tmp_path / "cache", manifest={"a": 1}, token_frame=token_frame
    )

    frame, hit = runner._load_cached_token_frame(
        cache_dir=tmp_path / "cache", expected_manifest={"a": 2}
    )
    assert frame is None
    assert hit is False


def test_load_cached_token_frame_misses_on_corrupt_manifest_json(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    token_path, manifest_path = runner._token_cache_paths(cache_dir)
    token_path.write_text("not-parquet", encoding="utf-8")
    manifest_path.write_text("{not valid json", encoding="utf-8")

    frame, hit = runner._load_cached_token_frame(
        cache_dir=cache_dir, expected_manifest={"a": 1}
    )
    assert frame is None
    assert hit is False


def test_load_cached_token_frame_misses_on_corrupt_parquet(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    token_path, manifest_path = runner._token_cache_paths(cache_dir)
    # `"complete": True` so the manifest itself reads as a hit and the code
    # under test actually reaches the corrupt parquet file, not an earlier
    # incomplete-candidate exit.
    manifest_path.write_text(json.dumps({"a": 1, "complete": True}), encoding="utf-8")
    token_path.write_text("not-a-real-parquet-file", encoding="utf-8")

    frame, hit = runner._load_cached_token_frame(
        cache_dir=cache_dir, expected_manifest={"a": 1}
    )
    assert frame is None
    assert hit is False


def test_build_validation_text_frame_computes_then_reuses_cache(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    config = _config(workspace_roots)
    stats = _stats(system_dir=tmp_path / "gleif")
    frame = pl.DataFrame({"name": ["acme limited", "acme ltd"]})

    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("model", encoding="utf-8")
    resolve = mocker.patch.object(
        runner, "_tokenizer_artifact_path", return_value=tokenizer_path
    )

    def fake_tokenize(frame, **kwargs):
        return frame.with_columns(
            pl.col("name").str.split(" ").alias(kwargs["token_col"])
        )

    tokenize = mocker.patch.object(
        runner, "tokenize_name_dataframe", side_effect=fake_tokenize
    )

    first_frame, first_key, first_hit = runner._build_validation_text_frame(
        config=config,
        frame=frame,
        stats=stats,
        system="gleif",
        target_system="fr",
        country="fr",
    )
    resolve.assert_called_once_with(config=config, system="fr")
    assert first_hit is False
    assert first_key is not None
    assert "validation_tokens" in first_frame.columns
    tokenize.assert_called_once()

    second_frame, second_key, second_hit = runner._build_validation_text_frame(
        config=config,
        frame=frame,
        stats=stats,
        system="gleif",
        target_system="fr",
        country="fr",
    )
    assert second_hit is True
    assert second_key == first_key
    tokenize.assert_called_once()  # not called again; served from cache
    assert second_frame.equals(first_frame)
