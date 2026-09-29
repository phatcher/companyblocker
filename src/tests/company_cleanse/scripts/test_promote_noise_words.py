from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import promote_noise_words
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_payload(extra_tokens: int = 0) -> dict:
    tokens = ["ltd", "gmbh"] + [f"tok{i}" for i in range(extra_tokens)]
    return {
        "generated_utc": "2026-08-29T00:00:00Z",
        "system": "gb",
        "source_corpus": "artifacts/tokenizers/gb/training_corpus.parquet",
        "profiles": {
            "strict": {"seed_legal": ["LTD"], "tokens": tokens},
            "balanced": {"tokens": ["services", "group"]},
        },
    }


def test_dry_run_does_not_write_and_reports_validation(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
):
    source = tokenizer_scope_dir(workspace_roots, system="gb") / "noise_words.json"
    dest = tmp_path / "resources" / "noise_words.json"
    _write_json(source, _candidate_payload())
    _write_json(dest, {"profiles": {"strict": {"tokens": ["ltd"]}}})

    exit_code = promote_noise_words.run_promotion(
        source=str(source), dest=str(dest), write=False
    )

    assert exit_code == 0
    assert dest.read_text(encoding="utf-8") == json.dumps(
        {"profiles": {"strict": {"tokens": ["ltd"]}}}
    )
    output = capsys.readouterr().out
    assert "dry run only" in output
    assert "profile=strict current=1 candidate=3 (+2)" in output


def test_write_promotes_candidate_over_dest(tmp_path: Path):
    source = tmp_path / "noise_words.json"
    dest = tmp_path / "resources" / "noise_words.json"
    payload = _candidate_payload()
    _write_json(source, payload)

    exit_code = promote_noise_words.run_promotion(
        source=str(source), dest=str(dest), write=True
    )

    assert exit_code == 0
    assert json.loads(dest.read_text(encoding="utf-8")) == payload


def test_write_creates_missing_dest_parent_directory(tmp_path: Path):
    source = tmp_path / "noise_words.json"
    dest = tmp_path / "nested" / "does" / "not" / "exist" / "noise_words.json"
    _write_json(source, _candidate_payload())

    exit_code = promote_noise_words.run_promotion(
        source=str(source), dest=str(dest), write=True
    )

    assert exit_code == 0
    assert dest.exists()


def test_dry_run_with_no_existing_dest_reports_no_diff_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source = tmp_path / "noise_words.json"
    dest = tmp_path / "resources" / "noise_words.json"
    _write_json(source, _candidate_payload())

    exit_code = promote_noise_words.run_promotion(
        source=str(source), dest=str(dest), write=False
    )

    assert exit_code == 0
    assert not dest.exists()
    output = capsys.readouterr().out
    assert "no matching profile in the current file to diff against" in output


def test_invalid_candidate_shape_raises_before_writing(tmp_path: Path):
    source = tmp_path / "noise_words.json"
    dest = tmp_path / "resources" / "noise_words.json"
    _write_json(source, {"profiles": {"strict": {"tokens": [1, 2]}}})

    with pytest.raises(ValueError, match="entries must be strings"):
        promote_noise_words.run_promotion(
            source=str(source), dest=str(dest), write=True
        )

    assert not dest.exists()


def test_missing_source_raises_file_not_found(tmp_path: Path):
    source = tmp_path / "does_not_exist.json"
    dest = tmp_path / "noise_words.json"

    with pytest.raises(FileNotFoundError, match="candidate noise-word file not found"):
        promote_noise_words.run_promotion(
            source=str(source), dest=str(dest), write=False
        )


def test_build_parser_defaults_to_packaged_noise_words_dest():
    parser = promote_noise_words.build_parser()
    args = parser.parse_args(["--source", "artifacts/tokenizers/gb/noise_words.json"])
    assert args.dest == str(promote_noise_words.DEFAULT_DEST)
    assert args.write is False
