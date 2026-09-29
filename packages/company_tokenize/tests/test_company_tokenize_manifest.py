import json

import pytest
from company_tokenize.manifest import (
    RUN_MANIFEST_MANDATORY_FIELDS,
    RUN_MANIFEST_MODES,
    RUN_MANIFEST_OPTIONAL_FIELDS,
    build_run_manifest,
    load_run_manifest_if_present,
    validate_dual_tokenizer_artifacts,
    validate_dual_tokenizer_coherence,
    validate_run_manifest,
)


def _train_kwargs(**overrides):
    kwargs = {
        "created_utc": "2026-08-29T00:00:00Z",
        "mode": "train",
        "scope": "country",
        "systems": ["ie"],
        "profile": None,
        "trainer": "wordpiece",
        "all_rows": True,
        "vocab_size": 40000,
        "min_frequency": 1,
        "name_col": "name",
        "preprocess_profile": "default|-company_type",
        "corpus_path": "artifacts/tokenizers/ie/training_corpus.parquet",
        "corpus_content_hash": "0123456789abcdef0123456789abcdef",
        "tokenizer_path": "artifacts/tokenizers/ie/wordpiece/model.json",
        "line_count": 12345,
        "trainer_params": {"tokenizer_encoding": None},
        "token_score_artifact": {"status": "ok", "invoked": True},
    }
    kwargs.update(overrides)
    return kwargs


def test_build_run_manifest_train_mode_omits_optional_fields():
    manifest = build_run_manifest(**_train_kwargs())

    for field in RUN_MANIFEST_MANDATORY_FIELDS:
        assert field in manifest
    for field in RUN_MANIFEST_OPTIONAL_FIELDS:
        assert field not in manifest
    assert manifest["mode"] == "train"


def test_build_run_manifest_optimize_mode_includes_optional_fields():
    manifest = build_run_manifest(
        **_train_kwargs(
            mode="optimize",
            run_log_path="artifacts/tokenizers/ie/optimize/optimize_runs.parquet",
            summary_path="artifacts/tokenizers/ie/optimize/optimize_summary.json",
        )
    )

    assert manifest["run_log_path"].endswith("optimize_runs.parquet")
    assert manifest["summary_path"].endswith("optimize_summary.json")
    for field in RUN_MANIFEST_MANDATORY_FIELDS:
        assert field in manifest


def test_validate_run_manifest_accepts_both_modes():
    for mode in RUN_MANIFEST_MODES:
        validate_run_manifest(build_run_manifest(**_train_kwargs(mode=mode)))


def test_validate_run_manifest_rejects_non_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        validate_run_manifest(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_run_manifest_rejects_missing_mandatory_field():
    payload = build_run_manifest(**_train_kwargs())
    del payload["trainer"]

    with pytest.raises(ValueError, match="missing mandatory field"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_unrecognized_field():
    payload = build_run_manifest(**_train_kwargs())
    payload["unexpected_field"] = "surprise"

    with pytest.raises(ValueError, match="unrecognized field"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_unknown_mode():
    payload = build_run_manifest(**_train_kwargs())
    payload["mode"] = "predict"

    with pytest.raises(ValueError, match="'mode' must be one of"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_non_list_systems():
    payload = build_run_manifest(**_train_kwargs())
    payload["systems"] = "ie"

    with pytest.raises(TypeError, match="'systems' must be a list of strings"):
        validate_run_manifest(payload)


@pytest.mark.parametrize(
    "field",
    [
        "created_utc",
        "scope",
        "trainer",
        "name_col",
        "preprocess_profile",
        "corpus_path",
        "corpus_content_hash",
        "tokenizer_path",
    ],
)
def test_validate_run_manifest_rejects_non_str_fields(field):
    payload = build_run_manifest(**_train_kwargs())
    payload[field] = 123

    with pytest.raises(TypeError, match=f"'{field}' must be a str"):
        validate_run_manifest(payload)


def test_build_run_manifest_uses_caller_supplied_corpus_content_hash():
    # `corpus_content_hash` proves what corpus a tokenizer was actually
    # trained on, independent of `corpus_path`, which is routinely reused
    # across a resample; the manifest schema stores it verbatim rather than
    # computing it itself, since only the caller knows which file was read.
    manifest = build_run_manifest(
        **_train_kwargs(corpus_content_hash="deadbeefdeadbeef")
    )

    assert manifest["corpus_content_hash"] == "deadbeefdeadbeef"


@pytest.mark.parametrize("field", ["vocab_size", "min_frequency", "line_count"])
def test_validate_run_manifest_rejects_non_int_numeric_fields(field):
    payload = build_run_manifest(**_train_kwargs())
    payload[field] = "not-an-int"

    with pytest.raises(TypeError, match=f"'{field}' must be an int"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_bool_for_int_field():
    # bool is a subclass of int in Python; a manifest writer that
    # accidentally passes True/False instead of a real count should still
    # fail schema validation rather than silently pass as "1"/"0".
    payload = build_run_manifest(**_train_kwargs())
    payload["line_count"] = True

    with pytest.raises(TypeError, match="'line_count' must be an int"):
        validate_run_manifest(payload)


@pytest.mark.parametrize("field", ["trainer_params", "token_score_artifact"])
def test_validate_run_manifest_rejects_non_dict_fields(field):
    payload = build_run_manifest(**_train_kwargs())
    payload[field] = "not-a-dict"

    with pytest.raises(TypeError, match=f"'{field}' must be a dict"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_non_bool_all_rows():
    payload = build_run_manifest(**_train_kwargs())
    payload["all_rows"] = "yes"

    with pytest.raises(TypeError, match="'all_rows' must be a bool"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_non_str_profile():
    payload = build_run_manifest(**_train_kwargs())
    payload["profile"] = 123

    with pytest.raises(TypeError, match="'profile' must be a str or None"):
        validate_run_manifest(payload)


def test_validate_run_manifest_rejects_non_str_optional_field():
    payload = build_run_manifest(
        **_train_kwargs(
            mode="optimize",
            run_log_path="artifacts/tokenizers/ie/optimize/optimize_runs.parquet",
        )
    )
    payload["run_log_path"] = 123

    with pytest.raises(TypeError, match="'run_log_path' must be a str"):
        validate_run_manifest(payload)


def test_build_run_manifest_includes_optimize_preset_when_given():
    manifest = build_run_manifest(
        **_train_kwargs(
            mode="optimize",
            optimize_preset="sentencepiece",
            optimize_preset_overrides={"fertility_target": 1.45},
        )
    )

    assert manifest["optimize_preset"] == "sentencepiece"
    assert manifest["optimize_preset_overrides"] == {"fertility_target": 1.45}
    for field in RUN_MANIFEST_MANDATORY_FIELDS:
        assert field in manifest


def test_build_run_manifest_omits_optimize_preset_when_not_given():
    manifest = build_run_manifest(**_train_kwargs())

    assert "optimize_preset" not in manifest
    assert "optimize_preset_overrides" not in manifest


def test_validate_run_manifest_rejects_non_dict_optimize_preset_overrides():
    payload = build_run_manifest(
        **_train_kwargs(
            mode="optimize",
            optimize_preset="wordpiece",
            optimize_preset_overrides={},
        )
    )
    payload["optimize_preset_overrides"] = "not-a-dict"

    with pytest.raises(TypeError, match="'optimize_preset_overrides' must be a dict"):
        validate_run_manifest(payload)


def _country_manifest(**overrides):
    return build_run_manifest(**_train_kwargs(scope="country", **overrides))


def _global_manifest(**overrides):
    return build_run_manifest(
        **_train_kwargs(
            scope="global",
            systems=["ie", "gb", "fr"],
            corpus_path="artifacts/tokenizers/global/training_corpus.parquet",
            tokenizer_path="artifacts/tokenizers/global/wordpiece/model.json",
            **overrides,
        )
    )


def test_validate_dual_tokenizer_coherence_accepts_matching_pair():
    validate_dual_tokenizer_coherence(
        country_manifest=_country_manifest(),
        global_manifest=_global_manifest(),
        country_trainer="wordpiece",
        global_trainer="wordpiece",
    )


def test_validate_dual_tokenizer_coherence_allows_different_trainers_per_side():
    validate_dual_tokenizer_coherence(
        country_manifest=_country_manifest(trainer="wordpiece"),
        global_manifest=_global_manifest(trainer="sentencepiece"),
        country_trainer="wordpiece",
        global_trainer="sentencepiece",
    )


def test_validate_dual_tokenizer_coherence_rejects_country_manifest_with_wrong_scope():
    with pytest.raises(ValueError, match="expected 'country'"):
        validate_dual_tokenizer_coherence(
            country_manifest=_global_manifest(),
            global_manifest=_global_manifest(),
            country_trainer="wordpiece",
            global_trainer="wordpiece",
        )


def test_validate_dual_tokenizer_coherence_rejects_global_manifest_with_wrong_scope():
    with pytest.raises(ValueError, match="expected 'global'"):
        validate_dual_tokenizer_coherence(
            country_manifest=_country_manifest(),
            global_manifest=_country_manifest(),
            country_trainer="wordpiece",
            global_trainer="wordpiece",
        )


def test_validate_dual_tokenizer_coherence_rejects_country_trainer_mismatch():
    with pytest.raises(ValueError, match="does not match the requested"):
        validate_dual_tokenizer_coherence(
            country_manifest=_country_manifest(trainer="wordpiece"),
            global_manifest=_global_manifest(),
            country_trainer="sentencepiece",
            global_trainer="wordpiece",
        )


def test_validate_dual_tokenizer_coherence_rejects_global_trainer_mismatch():
    with pytest.raises(ValueError, match="does not match the requested"):
        validate_dual_tokenizer_coherence(
            country_manifest=_country_manifest(),
            global_manifest=_global_manifest(trainer="wordpiece"),
            country_trainer="wordpiece",
            global_trainer="sentencepiece",
        )


def test_validate_dual_tokenizer_coherence_rejects_name_col_mismatch():
    with pytest.raises(ValueError, match="disagree on name_col"):
        validate_dual_tokenizer_coherence(
            country_manifest=_country_manifest(name_col="name_cleansed"),
            global_manifest=_global_manifest(name_col="name_raw"),
            country_trainer="wordpiece",
            global_trainer="wordpiece",
        )


def test_validate_dual_tokenizer_coherence_reuses_manifest_schema_validation():
    payload = _country_manifest()
    del payload["trainer"]

    with pytest.raises(ValueError, match="missing mandatory field"):
        validate_dual_tokenizer_coherence(
            country_manifest=payload,
            global_manifest=_global_manifest(),
            country_trainer="wordpiece",
            global_trainer="wordpiece",
        )


def test_load_run_manifest_if_present_returns_none_when_sidecar_missing(tmp_path):
    tokenizer_path = tmp_path / "model.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    assert (
        load_run_manifest_if_present(tokenizer_path=tokenizer_path, trainer="wordpiece")
        is None
    )


def test_load_run_manifest_if_present_loads_trainer_scoped_sidecar(tmp_path):
    tokenizer_path = tmp_path / "model.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    manifest = _country_manifest()
    (tmp_path / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_run_manifest_if_present(
        tokenizer_path=tokenizer_path, trainer="wordpiece"
    )

    assert loaded == manifest


def test_validate_dual_tokenizer_artifacts_skips_when_either_sidecar_missing(tmp_path):
    country_dir = tmp_path / "ie"
    global_dir = tmp_path / "global"
    country_dir.mkdir()
    global_dir.mkdir()
    country_tokenizer_path = country_dir / "model.json"
    global_tokenizer_path = global_dir / "model.json"
    country_tokenizer_path.write_text("{}", encoding="utf-8")
    global_tokenizer_path.write_text("{}", encoding="utf-8")
    # Only the country side has a manifest; an incoherent global side (wrong
    # scope) would otherwise raise, proving this really did skip rather than
    # validate against a missing manifest.
    (country_dir / "metadata.json").write_text(
        json.dumps(_country_manifest()), encoding="utf-8"
    )

    validate_dual_tokenizer_artifacts(
        country_tokenizer_path=country_tokenizer_path,
        global_tokenizer_path=global_tokenizer_path,
        country_trainer="wordpiece",
        global_trainer="wordpiece",
    )


def test_validate_dual_tokenizer_artifacts_raises_on_incoherent_pair(tmp_path):
    country_dir = tmp_path / "ie"
    global_dir = tmp_path / "global"
    country_dir.mkdir()
    global_dir.mkdir()
    country_tokenizer_path = country_dir / "model.json"
    global_tokenizer_path = global_dir / "model.json"
    country_tokenizer_path.write_text("{}", encoding="utf-8")
    global_tokenizer_path.write_text("{}", encoding="utf-8")
    (country_dir / "metadata.json").write_text(
        json.dumps(_country_manifest(name_col="name_cleansed")), encoding="utf-8"
    )
    (global_dir / "metadata.json").write_text(
        json.dumps(_global_manifest(name_col="name_raw")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="disagree on name_col"):
        validate_dual_tokenizer_artifacts(
            country_tokenizer_path=country_tokenizer_path,
            global_tokenizer_path=global_tokenizer_path,
            country_trainer="wordpiece",
            global_trainer="wordpiece",
        )
