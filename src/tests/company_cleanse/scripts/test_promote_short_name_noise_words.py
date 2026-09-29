from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import promote_short_name_noise_words


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate_payload(system: str, extra: int = 0) -> dict:
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


def _write_source_dir(source_dir: Path, systems: dict[str, dict]) -> None:
    for system, payload in systems.items():
        _write_json(source_dir / f"{system}.json", payload)


def test_build_merged_payload_keys_by_system(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_source_dir(
        source_dir,
        {
            "gb": _candidate_payload("gb"),
            "global": _candidate_payload("global"),
        },
    )

    merged = promote_short_name_noise_words.build_merged_payload(
        source_dir=source_dir, generated_utc="2026-08-30T00:00:00Z"
    )

    assert set(merged["systems"]) == {"gb", "global"}
    assert merged["systems"]["gb"]["candidates"][0]["token"] == "ltd"


def test_build_merged_payload_raises_on_empty_source_dir(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No candidate"):
        promote_short_name_noise_words.build_merged_payload(
            source_dir=tmp_path, generated_utc="2026-08-30T00:00:00Z"
        )


def test_dry_run_does_not_write_and_reports_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    source_dir = tmp_path / "sources"
    _write_source_dir(
        source_dir,
        {
            "gb": _candidate_payload("gb", extra=1),
            "global": _candidate_payload("global"),
        },
    )
    dest = tmp_path / "dest.json"
    _write_json(
        dest,
        {
            "generated_utc": "2026-08-29T00:00:00Z",
            "source": "prior",
            "systems": {"gb": {"candidates": _candidate_payload("gb")["candidates"]}},
        },
    )

    exit_code = promote_short_name_noise_words.run_promotion(
        source_dir=str(source_dir), dest=str(dest), write=False
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "dry run only" in output
    assert "system=gb current=2 candidate=3 (+1)" in output
    assert "no matching system in the current file to diff against" in output
    # dest untouched
    assert json.loads(dest.read_text(encoding="utf-8"))["source"] == "prior"


def test_write_promotes_merged_candidates_over_dest(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_source_dir(
        source_dir,
        {"gb": _candidate_payload("gb"), "global": _candidate_payload("global")},
    )
    dest = tmp_path / "dest.json"

    exit_code = promote_short_name_noise_words.run_promotion(
        source_dir=str(source_dir), dest=str(dest), write=True
    )

    assert exit_code == 0
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert set(written["systems"]) == {"gb", "global"}
    assert written["systems"]["gb"]["candidates"][0]["token"] == "ltd"


def test_write_creates_missing_dest_parent_directory(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_source_dir(source_dir, {"global": _candidate_payload("global")})
    dest = tmp_path / "nested" / "does" / "not" / "exist" / "dest.json"

    exit_code = promote_short_name_noise_words.run_promotion(
        source_dir=str(source_dir), dest=str(dest), write=True
    )

    assert exit_code == 0
    assert dest.exists()


def test_invalid_candidate_shape_raises_before_writing(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_json(
        source_dir / "gb.json",
        {"system": "gb", "candidates": [{"token": "a"}]},
    )
    _write_json(source_dir / "global.json", _candidate_payload("global"))
    dest = tmp_path / "dest.json"

    with pytest.raises(TypeError, match="document_frequency_pct"):
        promote_short_name_noise_words.run_promotion(
            source_dir=str(source_dir), dest=str(dest), write=True
        )

    assert not dest.exists()


def test_missing_global_entry_raises_before_writing(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_source_dir(source_dir, {"gb": _candidate_payload("gb")})
    dest = tmp_path / "dest.json"

    with pytest.raises(ValueError, match="global"):
        promote_short_name_noise_words.run_promotion(
            source_dir=str(source_dir), dest=str(dest), write=True
        )

    assert not dest.exists()


def test_source_missing_system_field_raises(tmp_path: Path):
    source_dir = tmp_path / "sources"
    _write_json(source_dir / "gb.json", {"candidates": []})
    dest = tmp_path / "dest.json"

    with pytest.raises(ValueError, match="'system' field"):
        promote_short_name_noise_words.run_promotion(
            source_dir=str(source_dir), dest=str(dest), write=False
        )


def test_build_parser_defaults():
    parser = promote_short_name_noise_words.build_parser()
    args = parser.parse_args([])
    assert args.write is False
