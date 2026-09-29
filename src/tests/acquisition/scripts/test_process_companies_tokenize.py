from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from scripts import process_companies
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.data_layout import CLEANSED_LAYER_NAME, layer_system, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_cleansed_parquet(roots: WorkspaceRoots, system: str) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:1"],
            "name": ["example ltd"],
            "name_cleansed_basic": ["example ltd"],
            "name_cleansed": ["example ltd"],
            "jurisdiction_code": [system],
        }
    ).write_parquet(cleansed_dir / f"{system}-001.parquet")


def test_process_companies_uses_explicit_tokenizer_path_for_tokenize(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "gb")

    def fake_tokenize_name(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_path: Path | None,
        name_col: str,
        token_col: str,
        **kwargs,
    ):
        observed["tokenizer_path"] = tokenizer_path
        observed["system"] = layer_system(cleansed_dir)
        observed["token_col"] = token_col
        return 5

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )
    explicit_tokenizer = tmp_path / "custom" / "wordpiece.json"

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
        tokenizer_path=explicit_tokenizer,
    )

    assert exit_code == 0
    assert observed == {
        "tokenizer_path": explicit_tokenizer.resolve(),
        "system": "gb",
        "token_col": "name_tokens",
    }
    tokenize_name.assert_called_once()


def test_process_companies_uses_bundled_tokenizer_when_global_artifact_missing(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        # Nothing is promoted, so the global shared tokenizer resolves to None.
        return None

    def fake_tokenize_name(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_path: Path | None,
        name_col: str,
        token_col: str,
        **kwargs,
    ):
        observed["tokenizer_path"] = tokenizer_path
        observed["system"] = layer_system(cleansed_dir)
        observed["token_col"] = token_col
        return 3

    resolve_tokenizer_paths = mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )
    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        tokenizer_scope="global",
        train=False,
    )

    assert exit_code == 0
    assert observed == {
        "tokenizer_path": None,
        "system": "fr",
        "token_col": "name_tokens",
    }
    assert resolve_tokenizer_paths.call_count == 3
    scopes = [
        call.kwargs.get("scope") for call in resolve_tokenizer_paths.call_args_list
    ]
    assert scopes.count("country") == 1
    assert scopes.count("global") == 2
    tokenize_name.assert_called_once()


def test_process_companies_clears_stale_tokenized_outputs_before_tokenize(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "gb")
    tokenized_dir = layer_fixture_dir("gb", layer="tokenized")
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    stale_path = tokenized_dir / "gb-999.parquet"
    stale_path.write_text("stale", encoding="utf-8")

    def fake_tokenize_name(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_path: Path | None,
        name_col: str,
        token_col: str,
        **kwargs,
    ):
        assert not stale_path.exists()
        (tokenized_dir / "gb-001.parquet").write_text("fresh", encoding="utf-8")
        assert token_col == "name_tokens"
        return 1

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
        tokenizer_path=tmp_path / "custom" / "wordpiece.json",
    )

    assert exit_code == 0
    assert not stale_path.exists()
    assert (tokenized_dir / "gb-001.parquet").exists()
    tokenize_name.assert_called_once()


def test_process_companies_tokenize_dual_mode_uses_dual_tokenizer(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        tokenizer_root = roots.artifacts / "fixture-tokenizers"
        profile = "default"
        token_file = tokenizer_root / scope / (system or profile) / "wordpiece.json"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("{}", encoding="utf-8")
        return token_file if (token_file).exists() else None

    def fake_tokenize_name_dual(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        country_tokenizer_path: Path | None,
        global_tokenizer_path: Path | None,
        name_col: str,
        country_token_col: str,
        global_token_col: str,
        **kwargs,
    ):
        observed["system"] = layer_system(cleansed_dir)
        observed["country_col"] = country_token_col
        observed["global_col"] = global_token_col
        observed["country_scope"] = (
            "country" if country_tokenizer_path is not None else None
        )
        observed["global_scope"] = (
            "global" if global_tokenizer_path is not None else None
        )
        return 2

    resolve_tokenizer_paths = mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )
    tokenize_name_dual = mocker.patch.object(
        process_companies, "tokenize_name_dual", side_effect=fake_tokenize_name_dual
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        token_mode="dual",
        country_token_col="country_tokens",
        global_token_col="global_tokens",
    )

    assert exit_code == 0
    assert observed == {
        "system": "fr",
        "country_col": "country_tokens",
        "global_col": "global_tokens",
        "country_scope": "country",
        "global_scope": "global",
    }
    assert resolve_tokenizer_paths.call_count == 2
    tokenize_name_dual.assert_called_once()


def test_process_companies_tokenize_single_mode_auto_dual_when_global_available(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        tokenizer_root = roots.artifacts / "fixture-tokenizers"
        profile = "default"
        token_file = tokenizer_root / scope / (system or profile) / "wordpiece.json"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("{}", encoding="utf-8")
        return token_file if (token_file).exists() else None

    def fake_tokenize_name_dual(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        country_tokenizer_path: Path | None,
        global_tokenizer_path: Path | None,
        name_col: str,
        country_token_col: str,
        global_token_col: str,
        **kwargs,
    ):
        observed["mode"] = "dual"
        observed["system"] = layer_system(cleansed_dir)
        observed["country_col"] = country_token_col
        observed["global_col"] = global_token_col
        observed["country_scope"] = (
            "country" if country_tokenizer_path is not None else None
        )
        observed["global_scope"] = (
            "global" if global_tokenizer_path is not None else None
        )
        return 2

    def fake_tokenize_name(**kwargs):
        observed["mode"] = "single"
        return 2

    resolve_tokenizer_paths = mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )
    tokenize_name_dual = mocker.patch.object(
        process_companies, "tokenize_name_dual", side_effect=fake_tokenize_name_dual
    )
    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        token_mode="single",
    )

    assert exit_code == 0
    assert observed == {
        "mode": "dual",
        "system": "fr",
        "country_col": "country_tokens",
        "global_col": "global_tokens",
        "country_scope": "country",
        "global_scope": "global",
    }
    assert resolve_tokenizer_paths.call_count == 2
    tokenize_name_dual.assert_called_once()
    tokenize_name.assert_not_called()


def test_process_companies_tokenize_single_mode_stays_single_when_global_unavailable(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        tokenizer_root = roots.artifacts / "fixture-tokenizers"
        profile = "default"
        token_file = tokenizer_root / scope / (system or profile) / "wordpiece.json"
        if scope == "country":
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text("{}", encoding="utf-8")
        return token_file if (token_file).exists() else None

    def fake_tokenize_name_dual(**kwargs):
        observed["mode"] = "dual"
        return 2

    def fake_tokenize_name(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_path: Path | None,
        name_col: str,
        token_col: str,
        **kwargs,
    ):
        observed["mode"] = "single"
        observed["system"] = layer_system(cleansed_dir)
        observed["token_col"] = token_col
        return 2

    resolve_tokenizer_paths = mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )
    tokenize_name_dual = mocker.patch.object(
        process_companies, "tokenize_name_dual", side_effect=fake_tokenize_name_dual
    )
    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        token_mode="single",
    )

    assert exit_code == 0
    assert observed == {
        "mode": "single",
        "system": "fr",
        "token_col": "name_tokens",
    }
    assert resolve_tokenizer_paths.call_count == 3
    tokenize_name_dual.assert_not_called()
    tokenize_name.assert_called_once()


def test_process_companies_tokenize_dual_mode_raises_when_global_tokenizer_unavailable(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        tokenizer_root = roots.artifacts / "fixture-tokenizers"
        profile = "default"
        token_file = tokenizer_root / scope / (system or profile) / "wordpiece.json"
        if scope == "country":
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text("{}", encoding="utf-8")
        return token_file if (token_file).exists() else None

    resolve_tokenizer_paths = mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        token_mode="dual",
        country_token_col="country_tokens",
        global_token_col="global_tokens",
    )

    assert exit_code == 1
    assert resolve_tokenizer_paths.call_count == 2


def test_process_companies_passes_noise_words_path_to_tokenize(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    def fake_tokenize_name(**kwargs):
        observed["noise_words_path"] = kwargs.get("noise_words_path")
        observed["noise_words_profile"] = kwargs.get("noise_words_profile")
        observed["noise_words_set_kind"] = kwargs.get("noise_words_set_kind")
        observed["trimmed_name_col"] = kwargs.get("trimmed_name_col")
        return 1

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    noise_words_path.parent.mkdir(parents=True, exist_ok=True)
    noise_words_path.write_text('["systems"]', encoding="utf-8")

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        noise_words_path=noise_words_path,
        noise_words_profile="balanced",
        noise_words_set_kind="combined",
        trimmed_name_col="companyname_trimmed",
    )

    assert exit_code == 0
    assert observed == {
        "noise_words_path": noise_words_path.resolve(),
        "noise_words_profile": "balanced",
        "noise_words_set_kind": "combined",
        "trimmed_name_col": "companyname_trimmed",
    }
    tokenize_name.assert_called_once()


def test_process_companies_tokenize_uses_multi_tokenizer_specs(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "fr")

    model_a = tmp_path / "models" / "tok_a.json"
    model_b = tmp_path / "models" / "tok_b.json"
    model_a.parent.mkdir(parents=True, exist_ok=True)
    model_a.write_text("{}", encoding="utf-8")
    model_b.write_text("{}", encoding="utf-8")

    def fake_tokenize_name_multi(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_specs: list[object],
        name_col: str,
        input_file: str | None = None,
        **kwargs,
    ):
        observed["system"] = layer_system(cleansed_dir)
        observed["token_cols"] = [
            str(getattr(spec, "token_col", "")) for spec in tokenizer_specs
        ]
        observed["labels"] = [
            str(getattr(spec, "label", "")) for spec in tokenizer_specs
        ]
        observed["trainers"] = [
            str(getattr(spec, "trainer", "")) for spec in tokenizer_specs
        ]
        (tokenized_dir / "fr-001.parquet").write_text("tokenized", encoding="utf-8")
        return 1

    tokenize_name_multi = mocker.patch.object(
        process_companies,
        "tokenize_name_multi",
        side_effect=fake_tokenize_name_multi,
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["tokenize"],
        root=tmp_path,
        tokenizer_specs=[
            f"label=tok_a,tokenizer=wordpiece,path={model_a},col=tokens_a",
            f"label=tok_b,tokenizer=wordpiece,path={model_b},col=tokens_b",
        ],
    )

    assert exit_code == 0
    assert observed == {
        "system": "fr",
        "token_cols": ["tokens_a", "tokens_b"],
        "labels": ["tok_a", "tok_b"],
        "trainers": ["wordpiece", "wordpiece"],
    }
    sidecar_path = layer_fixture_dir("fr", layer="tokenized") / "tokenizer_specs.json"
    assert sidecar_path.exists()
    sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_payload["schema_version"] == "2"
    assert sidecar_payload["request"]["name_col"] == "name_cleansed"
    # Regression guard: tokenizer paths must be written repo-relative, not
    # absolute, so the sidecar is portable across machines/checkouts.
    assert {spec["tokenizer_path"] for spec in sidecar_payload["tokenizer_specs"]} == {
        "models/tok_a.json",
        "models/tok_b.json",
    }
    tokenize_name_multi.assert_called_once()


def test_process_companies_tokenize_writes_tokenized_when_using_cleansed(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "gb")
    observed: dict[str, str] = {}

    def fake_tokenize_name(**kwargs):
        observed["input_layer"] = kwargs["cleansed_dir"]
        observed["output_layer"] = kwargs["tokenized_dir"]
        return 1

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
    )

    assert exit_code == 0
    assert observed == {
        "input_layer": system_layer_dir(workspace_roots, "gb", layer="cleansed"),
        "output_layer": system_layer_dir(workspace_roots, "gb", layer="tokenized"),
    }
    tokenize_name.assert_called_once()


def test_process_companies_tokenize_finds_family_split_partitioned_cleansed_input(
    tmp_path: Path, layer_fixture_dir, mocker
):
    """A real cleansed/ view partitioned under `primary/` must still be
    recognised as tokenize input, not reported as missing.
    """
    partition_dir = (
        layer_fixture_dir("gb", layer="cleansed") / "primary" / "jurisdiction_code=gb"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb:1"],
            "name": ["example ltd"],
            "name_cleansed": ["example ltd"],
            "jurisdiction_code": ["gb"],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", return_value=1
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
    )

    assert exit_code == 0
    tokenize_name.assert_called_once()


def test_process_companies_hands_the_tokenize_stage_its_text_choices(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    """The stage tokenizes the text a blocking run would: its name column, its
    preprocess profile, and no word removal unless a profile is named."""
    observed: dict[str, object] = {}
    _write_cleansed_parquet(workspace_roots, "gb")

    def fake_tokenize_name(*, name_col: str, **kwargs):
        observed["name_col"] = name_col
        observed["preprocess_profile"] = kwargs["preprocess_profile"]
        observed["noise_words_profile"] = kwargs["noise_words_profile"]
        return 1

    mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
        preprocess_profile="default|-company_type",
    )

    assert exit_code == 0
    assert observed == {
        "name_col": "name_cleansed",
        "preprocess_profile": "default|-company_type",
        "noise_words_profile": "none",
    }
