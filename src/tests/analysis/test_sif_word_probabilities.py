from __future__ import annotations

import polars as pl
import pytest

from analysis.sif_word_probabilities import (
    DEFAULT_PREPARATION,
    GLOBAL_LIST_NAME,
    attach_token_probabilities,
    count_word_document_frequencies,
    resolve_word_probabilities,
)
from workspace.artifact_archive import MANIFEST_FILENAME, read_artifact_manifest
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_cleansed_fixture(
    roots: WorkspaceRoots, system: str, names: list[str]
) -> None:
    """One system's cleansed shard, `system_uri` plus `name_cleansed` only --
    the two columns `resolve_word_probabilities` reads under the default
    preparation."""
    directory = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}://{index}" for index in range(len(names))],
            "name_cleansed": names,
        }
    ).write_parquet(directory / f"{system}-001.parquet")


# Three names, six word occurrences total (two words each): mean tokens per
# name is exactly 2.0, known by construction rather than measured, satisfying
# the requirement for a fixture list whose mean tokens per name is known.
_IE_NAMES = ["acme ltd", "acme group", "zenith ltd"]
_GB_NAMES = ["acme ltd"]


def test_count_word_document_frequencies_counts_names_and_tokens():
    frame = pl.DataFrame({"name_cleansed": _IE_NAMES})

    stats, num_names, total_tokens = count_word_document_frequencies(
        frame, name_col="name_cleansed"
    )

    assert num_names == 3
    assert total_tokens == 6
    by_token = dict(stats.iter_rows())
    assert by_token == {"acme": 2, "ltd": 2, "group": 1, "zenith": 1}


def test_count_word_document_frequencies_drops_null_and_blank_names():
    frame = pl.DataFrame({"name_cleansed": ["acme ltd", None, "  ", ""]})

    stats, num_names, total_tokens = count_word_document_frequencies(
        frame, name_col="name_cleansed"
    )

    assert num_names == 1
    assert total_tokens == 2
    assert dict(stats.iter_rows()) == {"acme": 1, "ltd": 1}


def test_count_word_document_frequencies_handles_empty_frame():
    frame = pl.DataFrame({"name_cleansed": []}, schema={"name_cleansed": pl.Utf8})

    stats, num_names, total_tokens = count_word_document_frequencies(
        frame, name_col="name_cleansed"
    )

    assert num_names == 0
    assert total_tokens == 0
    assert stats.height == 0


def test_attach_token_probabilities_converts_document_frequency_by_mean_tokens():
    # Document frequency alone (df / num_names) would give acme=2/3, ltd=2/3,
    # group=1/3, zenith=1/3 -- the point is that this is not the token
    # probability. The conversion divides by num_names * mean_tokens_per_name
    # (3 * 2.0 = 6) instead.
    stats = pl.DataFrame(
        {
            "token": ["acme", "ltd", "group", "zenith"],
            "document_frequency": [2, 2, 1, 1],
        }
    )

    converted = attach_token_probabilities(stats, num_names=3, mean_tokens_per_name=2.0)

    by_token = {
        row["token"]: row["probability"] for row in converted.iter_rows(named=True)
    }
    assert by_token["acme"] == pytest.approx(1 / 3)
    assert by_token["ltd"] == pytest.approx(1 / 3)
    assert by_token["group"] == pytest.approx(1 / 6)
    assert by_token["zenith"] == pytest.approx(1 / 6)


def test_attach_token_probabilities_rejects_non_positive_inputs():
    stats = pl.DataFrame({"token": ["acme"], "document_frequency": [1]})

    with pytest.raises(ValueError):
        attach_token_probabilities(stats, num_names=0, mean_tokens_per_name=2.0)
    with pytest.raises(ValueError):
        attach_token_probabilities(stats, num_names=3, mean_tokens_per_name=0.0)


def test_resolve_word_probabilities_records_list_preparation_and_mean(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(workspace_roots, "ie", _IE_NAMES)

    entry = resolve_word_probabilities(
        workspace_roots, system="ie", preparation=DEFAULT_PREPARATION
    )

    manifest = read_artifact_manifest(entry)
    assert manifest is not None
    assert manifest["list"] == "ie"
    assert manifest["systems"] == ["ie"]
    assert manifest["preparation"] == DEFAULT_PREPARATION
    assert manifest["num_names"] == 3
    assert manifest["mean_tokens_per_name"] == pytest.approx(2.0)

    written = pl.read_parquet(entry / "word_probabilities.parquet")
    by_token = {
        row["token"]: row["probability"] for row in written.iter_rows(named=True)
    }
    assert by_token["acme"] == pytest.approx(1 / 3)
    assert by_token["group"] == pytest.approx(1 / 6)


def test_resolve_word_probabilities_reuses_existing_key_without_rewriting(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(workspace_roots, "ie", _IE_NAMES)

    first = resolve_word_probabilities(workspace_roots, system="ie")
    manifest_path = first / MANIFEST_FILENAME
    first_mtime = manifest_path.stat().st_mtime_ns

    second = resolve_word_probabilities(workspace_roots, system="ie")

    assert second == first
    assert manifest_path.stat().st_mtime_ns == first_mtime


def test_resolve_word_probabilities_global_list_pools_every_system(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(workspace_roots, "ie", _IE_NAMES)
    _write_cleansed_fixture(workspace_roots, "gb", _GB_NAMES)

    entry = resolve_word_probabilities(workspace_roots, system=None)

    manifest = read_artifact_manifest(entry)
    assert manifest is not None
    assert manifest["list"] == GLOBAL_LIST_NAME
    assert manifest["systems"] == ["gb", "ie"]
    assert manifest["num_names"] == 4
    assert manifest["mean_tokens_per_name"] == pytest.approx(2.0)

    written = pl.read_parquet(entry / "word_probabilities.parquet")
    by_token = dict(
        zip(
            written.get_column("token").to_list(),
            written.get_column("document_frequency").to_list(),
        )
    )
    assert by_token["acme"] == 3
    assert by_token["ltd"] == 3
    assert by_token["group"] == 1
    assert by_token["zenith"] == 1


def test_resolve_word_probabilities_raises_for_unknown_preparation(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(workspace_roots, "ie", _IE_NAMES)

    with pytest.raises(ValueError):
        resolve_word_probabilities(workspace_roots, system="ie", preparation="bogus")


def test_resolve_word_probabilities_raises_when_system_has_no_names(
    workspace_roots: WorkspaceRoots,
):
    with pytest.raises(ValueError):
        resolve_word_probabilities(workspace_roots, system="nowhere")


def test_resolve_word_probabilities_skips_a_shard_missing_the_id_column(
    workspace_roots: WorkspaceRoots,
):
    # A shard carrying `name_cleansed` but not `system_uri` is not eligible --
    # not every source has migrated to the standardized columns
    # (`analysis.token_zipf`) -- so this system contributes no names.
    directory = system_layer_dir(workspace_roots, "ie", layer=CLEANSED_LAYER_NAME)
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": _IE_NAMES}).write_parquet(
        directory / "ie-001.parquet"
    )

    with pytest.raises(ValueError):
        resolve_word_probabilities(workspace_roots, system="ie")


def test_resolve_word_probabilities_raises_when_every_name_is_blank(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(workspace_roots, "ie", ["", "  ", ""])

    with pytest.raises(ValueError):
        resolve_word_probabilities(workspace_roots, system="ie")


def test_resolve_word_probabilities_global_list_raises_with_no_data_root(
    workspace_roots: WorkspaceRoots,
):
    # No system has ever written a cleansed layer under this root: `data/`
    # itself does not exist yet, distinct from a `data/` holding no eligible
    # system.
    with pytest.raises(ValueError):
        resolve_word_probabilities(workspace_roots, system=None)
