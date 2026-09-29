from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import promote_noise_word_candidates
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_payload(system: str = "gb", extra: int = 0) -> dict:
    candidates = [
        {"token": "ltd", "document_frequency_pct": 0.9, "idf": 1.1},
        {"token": "services", "document_frequency_pct": 0.05, "idf": 4.1},
    ] + [
        {"token": f"tok{i}", "document_frequency_pct": 0.01, "idf": 6.0 + i}
        for i in range(extra)
    ]
    return {
        "generated_utc": "2026-08-30T00:00:00Z",
        "system": system,
        "source_stats_path": f"artifacts/tokenizers/{system}/token_tfidf_stats.parquet",
        "tfidf_use_case": "corpus",
        "tfidf_namespace": "corpus_tfidf",
        "selection": {
            "max_candidates": 200,
            "candidate_count": len(candidates),
            "min_token_length": 2,
            "source_token_count": 1000,
            "document_frequency_pct_cutoff": candidates[-1]["document_frequency_pct"],
            "idf_cutoff": candidates[-1]["idf"],
        },
        "candidates": candidates,
    }


def test_dry_run_does_not_write_and_reports_validation(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
):
    source = (
        tokenizer_scope_dir(workspace_roots, system="gb") / "noise_word_candidates.json"
    )
    dest = tmp_path / "resources" / "gb.json"
    _write_json(source, _candidate_payload(extra=1))
    _write_json(dest, _candidate_payload())

    exit_code = promote_noise_word_candidates.run_promotion(
        source=str(source), dest=str(dest), write=False
    )

    assert exit_code == 0
    assert json.loads(dest.read_text(encoding="utf-8")) == _candidate_payload()
    output = capsys.readouterr().out
    assert "dry run only" in output
    assert "candidates: current=2 candidate=3 (+1)" in output


def test_write_promotes_candidate_over_dest(tmp_path: Path):
    source = tmp_path / "noise_word_candidates.json"
    dest = tmp_path / "gb.json"
    payload = _candidate_payload()
    _write_json(source, payload)

    exit_code = promote_noise_word_candidates.run_promotion(
        source=str(source), dest=str(dest), write=True
    )

    assert exit_code == 0
    assert json.loads(dest.read_text(encoding="utf-8")) == payload


def test_write_creates_missing_dest_parent_directory(tmp_path: Path):
    source = tmp_path / "noise_word_candidates.json"
    dest = tmp_path / "nested" / "does" / "not" / "exist" / "gb.json"
    _write_json(source, _candidate_payload())

    exit_code = promote_noise_word_candidates.run_promotion(
        source=str(source), dest=str(dest), write=True
    )

    assert exit_code == 0
    assert dest.exists()


def test_default_dest_derived_from_system_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Point DEFAULT_DEST_DIR at a throwaway tmp_path for this test -- it must
    # not touch the real packaged resources directory (which holds real,
    # committed promoted artifacts).
    fake_dest_dir = tmp_path / "resources" / "noise_word_candidates"
    monkeypatch.setattr(
        promote_noise_word_candidates, "DEFAULT_DEST_DIR", fake_dest_dir
    )

    source = tmp_path / "noise_word_candidates.json"
    _write_json(source, _candidate_payload(system="fr"))

    exit_code = promote_noise_word_candidates.run_promotion(
        source=str(source), dest=None, write=True
    )

    assert exit_code == 0
    assert (fake_dest_dir / "fr.json").exists()


def test_invalid_candidate_shape_raises_before_writing(tmp_path: Path):
    source = tmp_path / "noise_word_candidates.json"
    dest = tmp_path / "gb.json"
    _write_json(source, {"system": "gb", "candidates": [{"token": "a"}]})

    with pytest.raises(TypeError, match="document_frequency_pct"):
        promote_noise_word_candidates.run_promotion(
            source=str(source), dest=str(dest), write=True
        )

    assert not dest.exists()


def test_missing_source_raises_file_not_found(tmp_path: Path):
    source = tmp_path / "does_not_exist.json"
    dest = tmp_path / "gb.json"

    with pytest.raises(
        FileNotFoundError, match="candidate noise-word export not found"
    ):
        promote_noise_word_candidates.run_promotion(
            source=str(source), dest=str(dest), write=False
        )


def test_missing_dest_and_missing_system_field_raises(tmp_path: Path):
    # Payload-shape validation (requiring a non-empty 'system' string) runs before
    # dest resolution, so an invalid/missing 'system' is caught there first --
    # confirmed here so the two failure paths don't silently diverge.
    source = tmp_path / "noise_word_candidates.json"
    _write_json(source, {"candidates": []})

    with pytest.raises(ValueError, match="'system' string"):
        promote_noise_word_candidates.run_promotion(
            source=str(source), dest=None, write=False
        )


def test_resolve_dest_path_raises_when_system_field_missing():
    # Direct unit coverage for the defensive branch in _resolve_dest_path --
    # unreachable via run_promotion today since payload-shape validation
    # already requires a non-empty 'system' string before dest resolution
    # runs, but kept as a guard for any future direct caller.
    with pytest.raises(ValueError, match="'system' field"):
        promote_noise_word_candidates._resolve_dest_path(
            dest=None, candidate_payload={"candidates": []}
        )


def test_build_parser_defaults():
    parser = promote_noise_word_candidates.build_parser()
    args = parser.parse_args(
        ["--source", "artifacts/tokenizers/gb/noise_word_candidates.json"]
    )
    assert args.dest is None
    assert args.write is False
