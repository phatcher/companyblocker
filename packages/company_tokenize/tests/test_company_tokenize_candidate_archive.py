from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    archive_naive_tokenizer,
    archive_optimize_candidate,
    archive_promoted_tokenizer,
    archive_trained_tokenizer,
    candidate_archive,
    compare_archive_entries,
    compute_corpus_content_hash,
    read_archive_index,
    resolve_archive_dir,
    resolve_archive_label,
    resolve_archived_tokenizer_path,
    resolve_candidate_model_path,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
)
from company_tokenize.candidate_archive import (
    migrate_legacy_archive_labels,
    write_archive_index,
)
from company_tokenize.training import train_wordpiece


def test_resolve_candidate_model_path_uses_auto_tag_for_requested_minus_one(
    tmp_path: Path,
):
    path = resolve_candidate_model_path(
        optimize_models_dir=tmp_path,
        trainer="wordpiece",
        seed=42,
        vocab_requested=-1,
        min_frequency=12,
    )
    assert path == tmp_path / "wordpiece_seed_42_vauto_mf12.json"


def test_resolve_candidate_model_path_uses_numeric_vocab_otherwise(tmp_path: Path):
    path = resolve_candidate_model_path(
        optimize_models_dir=tmp_path,
        trainer="wordpiece",
        seed=42,
        vocab_requested=25000,
        min_frequency=8,
    )
    assert path == tmp_path / "wordpiece_seed_42_v25000_mf8.json"


def _write_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"ie:{i}" for i in range(43)],
            "name": ["acme systems ltd", "beta holdings ltd", "gamma trading co"] * 14
            + ["delta group"],
        }
    ).write_parquet(path)


def _root_with_corpus(tmp_path: Path) -> Path:
    corpus_path = (
        tmp_path / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    )
    _write_corpus(corpus_path)
    return tmp_path


def _write_pointer_and_models_dir(
    *, root: Path, trainer: str, tokenizer_encoding: str | None, grid_hash: str
) -> Path:
    """Simulate a real optimize sweep having already run: writes the pointer
    file `archive_optimize_candidate` reads to find its sweep leaf, and
    returns that leaf's `models_dir` for the test to populate.
    """
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    optimize_dir = resolve_optimize_dir(
        scope_directory=root / "artifacts" / "tokenizers" / "ie",
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"grid_hash": grid_hash}), encoding="utf-8")
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(corpus_path),
        grid_hash=grid_hash,
    )
    sweep_paths.models_dir.mkdir(parents=True, exist_ok=True)
    return sweep_paths.models_dir


def test_archive_optimize_candidate_copies_model_evaluates_metrics_and_indexes(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    models_dir = _write_pointer_and_models_dir(
        root=root, trainer="wordpiece", tokenizer_encoding=None, grid_hash="testgrid1"
    )
    source_path = models_dir / "wordpiece_seed_42_v64_mf1.json"
    train_wordpiece(
        corpus_path=root
        / "artifacts"
        / "tokenizers"
        / "ie"
        / "training_corpus.parquet",
        tokenizer_path=source_path,
        vocab_size=64,
        min_frequency=1,
    )

    dest = archive_optimize_candidate(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        trainer="wordpiece",
        seed=42,
        vocab_requested=64,
        min_frequency=1,
        label="candidate_v64",
        notes="test candidate",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    assert dest == archive_dir / "candidate_v64.json"
    assert dest.exists()

    entries = read_archive_index(archive_dir)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["label"] == "candidate_v64"
    assert entry["kind"] == "optimize_candidate"
    assert entry["seed"] == 42
    assert entry["vocab_size_requested"] == 64
    assert entry["notes"] == "test candidate"
    assert "fertility" in entry["metrics"]
    assert "unk_rate" in entry["metrics"]


def test_archive_optimize_candidate_raises_when_no_sweep_recorded(tmp_path: Path):
    root = _root_with_corpus(tmp_path)

    with pytest.raises(FileNotFoundError, match="No optimize sweep recorded"):
        archive_optimize_candidate(
            tokenizer_root=root / "artifacts" / "tokenizers",
            scope="country",
            system="ie",
            trainer="wordpiece",
            seed=42,
            vocab_requested=40000,
            min_frequency=1,
            label="naive_baseline",
        )


def test_archive_optimize_candidate_raises_when_candidate_not_in_recorded_sweep(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    # A real sweep is recorded (pointer + leaf exist), but the requested
    # seed/vocab/min_frequency was never trained in it.
    _write_pointer_and_models_dir(
        root=root, trainer="wordpiece", tokenizer_encoding=None, grid_hash="testgrid1"
    )

    with pytest.raises(FileNotFoundError, match="wasn't trained in it"):
        archive_optimize_candidate(
            tokenizer_root=root / "artifacts" / "tokenizers",
            scope="country",
            system="ie",
            trainer="wordpiece",
            seed=42,
            vocab_requested=40000,
            min_frequency=1,
            label="naive_baseline",
        )


def test_archive_promoted_tokenizer_records_the_stored_reference(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    model_path = root / "artifacts" / "tokenizers" / "ie" / "wordpiece" / "model.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=model_path,
        vocab_size=64,
        min_frequency=1,
    )

    dest = archive_promoted_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=model_path,
        label="promoted_first",
        reference="tokenizer://ie/wordpiece/k1",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    (entry,) = read_archive_index(archive_dir)
    assert dest.exists()
    assert entry["kind"] == "promoted"
    assert entry["reference"] == "tokenizer://ie/wordpiece/k1"
    assert "is_current_operational" not in entry


def test_archive_naive_tokenizer_records_the_stored_reference(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    model_path = root / "artifacts" / "tokenizers" / "ie" / "wordpiece" / "model.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=model_path,
        vocab_size=64,
        min_frequency=1,
    )

    archive_naive_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=model_path,
        label="trained_wordpiece_1",
        reference="tokenizer://ie/wordpiece/k2",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    (entry,) = read_archive_index(archive_dir)
    assert entry["kind"] == "naive"
    assert entry["reference"] == "tokenizer://ie/wordpiece/k2"


def test_archive_promoted_tokenizer_writes_summary_sibling_file(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    operational_path = (
        root / "artifacts" / "tokenizers" / "ie" / "wordpiece" / "model.json"
    )
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=operational_path,
        vocab_size=64,
        min_frequency=1,
    )
    summary = {"winner_candidate": {"vocab_size_requested": 64}, "candidate_stats": {}}

    archive_promoted_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=operational_path,
        label="promoted_with_summary",
        reference="tokenizer://ie/wordpiece/promoted_with_summary",
        summary=summary,
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    summary_path = archive_dir / "promoted_with_summary.summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_archive_promoted_tokenizer_omits_summary_sibling_when_none_given(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    operational_path = (
        root / "artifacts" / "tokenizers" / "ie" / "wordpiece" / "model.json"
    )
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=operational_path,
        vocab_size=64,
        min_frequency=1,
    )

    archive_promoted_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=operational_path,
        label="promoted_no_summary",
        reference="tokenizer://ie/wordpiece/promoted_no_summary",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    assert not (archive_dir / "promoted_no_summary.summary.json").exists()


def test_archive_tokenizer_file_reuses_precomputed_metrics_without_recomputing(
    tmp_path: Path, monkeypatch
):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    operational_path = (
        root / "artifacts" / "tokenizers" / "ie" / "wordpiece" / "model.json"
    )
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=operational_path,
        vocab_size=64,
        min_frequency=1,
    )

    def _fail_if_called(**kwargs):
        raise AssertionError(
            "evaluate_tokenizer_metrics should not be called when "
            "precomputed_metrics is given"
        )

    monkeypatch.setattr(
        candidate_archive, "evaluate_tokenizer_metrics", _fail_if_called
    )

    precomputed = {"fertility": 1.1, "unk_rate": 0.0, "fertility_distance": 0.15}
    archive_promoted_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=operational_path,
        label="promoted_precomputed",
        reference="tokenizer://ie/wordpiece/promoted_precomputed",
        precomputed_metrics=precomputed,
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    entries = {entry["label"]: entry for entry in read_archive_index(archive_dir)}
    assert entries["promoted_precomputed"]["metrics"] == precomputed


def test_migrate_legacy_archive_labels_renames_files_and_updates_index(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    old_label = "promoted_wordpiece_opt-20260826T165953Z-7b5e2741"
    model_bytes = b"not a real tokenizer, just bytes to prove nothing is lost"
    (archive_dir / f"{old_label}.json").write_bytes(model_bytes)
    entry = {
        "label": old_label,
        "kind": "promoted",
        "archived_utc": "2026-08-26T16:59:53Z",
        "metrics": {"fertility_distance": 0.0068},
    }
    (archive_dir / f"{old_label}.metadata.json").write_text(
        json.dumps(entry), encoding="utf-8"
    )
    write_archive_index(archive_dir, [entry])

    renamed = migrate_legacy_archive_labels(archive_dir)

    assert renamed == [(old_label, "promoted_wordpiece_7b5e2741")]
    new_model_path = archive_dir / "promoted_wordpiece_7b5e2741.json"
    assert new_model_path.exists()
    assert new_model_path.read_bytes() == model_bytes  # no data lost, just renamed
    assert not (archive_dir / f"{old_label}.json").exists()
    assert not (archive_dir / f"{old_label}.metadata.json").exists()

    entries = read_archive_index(archive_dir)
    assert entries[0]["label"] == "promoted_wordpiece_7b5e2741"
    assert entries[0]["metrics"] == {"fertility_distance": 0.0068}  # preserved

    new_metadata = json.loads(
        (archive_dir / "promoted_wordpiece_7b5e2741.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert new_metadata["label"] == "promoted_wordpiece_7b5e2741"

    # Idempotent: re-running finds nothing left to migrate.
    assert migrate_legacy_archive_labels(archive_dir) == []


def test_migrate_legacy_archive_labels_recovers_variant_from_trainer_field(
    tmp_path: Path,
):
    # Earliest label scheme (before the trainer/model_type variant was added
    # to the label at all) -- must fall back to the entry's own "trainer"
    # field rather than leave the migrated label variant-less.
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    old_label = "trained_20260826T165855Z-882590ed"
    (archive_dir / f"{old_label}.json").write_bytes(b"model bytes")
    entry = {"label": old_label, "trainer": "wordpiece", "kind": "promoted"}
    write_archive_index(archive_dir, [entry])

    renamed = migrate_legacy_archive_labels(archive_dir)

    assert renamed == [(old_label, "trained_wordpiece_882590ed")]
    assert (archive_dir / "trained_wordpiece_882590ed.json").exists()


def test_migrate_legacy_archive_labels_ignores_manual_and_already_short_labels(
    tmp_path: Path,
):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    write_archive_index(
        archive_dir,
        [
            {
                "label": "v25000_mf1_train_split_fix_winner",
                "kind": "promoted",
            },
            {"label": "promoted_wordpiece_7b5e2741", "kind": "promoted"},
        ],
    )

    assert migrate_legacy_archive_labels(archive_dir) == []


def test_archive_trained_tokenizer_for_a_naive_baseline(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    naive_path = tmp_path / "scratch_naive.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=naive_path,
        vocab_size=64,
        min_frequency=1,
    )

    dest = archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=naive_path,
        label="naive_v64_mf1",
        notes="naive baseline for test",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    entries = {entry["label"]: entry for entry in read_archive_index(archive_dir)}
    assert dest.exists()
    assert entries["naive_v64_mf1"]["kind"] == "manual"


def test_attach_report_to_archive_entry_writes_report_copies_images_and_indexes(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    source_path = tmp_path / "scratch.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=source_path,
        vocab_size=64,
        min_frequency=1,
    )
    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=source_path,
        label="candidate_v64",
    )

    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    image_source = tmp_path / "elbow.png"
    image_source.write_bytes(b"not a real png, just bytes for the test")

    report_path = candidate_archive.attach_report_to_archive_entry(
        archive_dir=archive_dir,
        label="candidate_v64",
        report_markdown="# Report\n\n![elbow](abcd1234_candidate_v64.elbow.wordpiece.png)\n",
        image_files={"abcd1234_candidate_v64.elbow.wordpiece.png": image_source},
    )

    assert report_path == archive_dir / "candidate_v64.report.md"
    assert report_path.read_text(encoding="utf-8").startswith("# Report")
    copied_image = archive_dir / "abcd1234_candidate_v64.elbow.wordpiece.png"
    assert copied_image.exists()
    assert copied_image.read_bytes() == image_source.read_bytes()

    entries = {entry["label"]: entry for entry in read_archive_index(archive_dir)}
    assert entries["candidate_v64"]["report_path"] == "candidate_v64.report.md"


def test_attach_report_to_archive_entry_raises_for_unknown_label(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    source_path = tmp_path / "scratch.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=source_path,
        vocab_size=64,
        min_frequency=1,
    )
    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=source_path,
        label="candidate_v64",
    )
    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )

    with pytest.raises(KeyError, match="no_such_label"):
        candidate_archive.attach_report_to_archive_entry(
            archive_dir=archive_dir,
            label="no_such_label",
            report_markdown="# Report\n",
        )


def test_attach_report_to_archive_entry_overwrites_report_on_repeat_call(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    source_path = tmp_path / "scratch.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=source_path,
        vocab_size=64,
        min_frequency=1,
    )
    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=source_path,
        label="candidate_v64",
    )
    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )

    candidate_archive.attach_report_to_archive_entry(
        archive_dir=archive_dir, label="candidate_v64", report_markdown="# v1\n"
    )
    report_path = candidate_archive.attach_report_to_archive_entry(
        archive_dir=archive_dir, label="candidate_v64", report_markdown="# v2\n"
    )

    assert report_path.read_text(encoding="utf-8") == "# v2\n"
    entries = read_archive_index(archive_dir)
    assert len(entries) == 1


def test_compare_archive_entries_reports_improvement_in_both_directions(
    tmp_path: Path,
):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    small_path = tmp_path / "small.json"
    large_path = tmp_path / "large.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=small_path,
        vocab_size=32,
        min_frequency=1,
    )
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=large_path,
        vocab_size=200,
        min_frequency=1,
    )

    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=small_path,
        label="small",
    )
    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=large_path,
        label="large",
    )

    result = compare_archive_entries(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        label_a="small",
        label_b="large",
    )

    assert result["label_a"] == "small"
    assert result["label_b"] == "large"
    assert set(result["metric_deltas"]) == {
        "fertility",
        "fertility_distance",
        "unk_rate",
        "single_char_token_pct",
        "token_count_mean",
    }
    assert result["better_count"] + result["worse_count"] > 0
    assert "small" in result["verdict"] or "large" in result["verdict"]


def test_compare_archive_entries_does_not_double_count_fertility_moving_toward_target(
    tmp_path: Path,
):
    """Regression: found running this for real on ie's naive-vs-optimized
    comparison. `fertility` is supposed to sit close to a target, not be
    minimized -- `fertility_distance` already is that deviation. Scoring
    both meant a candidate whose fertility moved *toward* the target (a
    real improvement, fertility_distance dropping) got counted as "worse"
    on the raw fertility number in the same breath as "better" on
    fertility_distance -- the same underlying change, double-counted with
    opposite verdicts."""
    archive_dir = resolve_archive_dir(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
    )
    write_archive_index(
        archive_dir,
        [
            {
                "label": "naive",
                "kind": "manual",
                "metrics": {
                    "fertility": 1.20,
                    "fertility_distance": 0.05,  # |1.20 - 1.25|
                    "unk_rate": 0.0,
                    "single_char_token_pct": 6.0,
                    "token_count_mean": 4.0,
                },
            },
            {
                "label": "optimized",
                "kind": "optimize_candidate",
                "metrics": {
                    "fertility": 1.24,
                    "fertility_distance": 0.01,  # |1.24 - 1.25|, closer to target
                    "unk_rate": 0.0,
                    "single_char_token_pct": 6.0,
                    "token_count_mean": 4.0,
                },
            },
        ],
    )

    result = compare_archive_entries(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        label_a="naive",
        label_b="optimized",
    )

    assert result["better_count"] == 1  # fertility_distance improved
    assert result["worse_count"] == 0  # fertility is context, not scored
    assert "improves" in result["verdict"]
    # still reported, just not scored
    assert result["metric_deltas"]["fertility"]["delta"] == pytest.approx(0.04)


def test_compare_archive_entries_raises_for_unknown_label(tmp_path: Path):
    root = _root_with_corpus(tmp_path)
    corpus_path = root / "artifacts" / "tokenizers" / "ie" / "training_corpus.parquet"
    tokenizer_path = tmp_path / "only.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=32,
        min_frequency=1,
    )
    archive_trained_tokenizer(
        tokenizer_root=root / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        source_tokenizer_path=tokenizer_path,
        label="only",
    )

    with pytest.raises(KeyError, match="missing_label"):
        compare_archive_entries(
            tokenizer_root=root / "artifacts" / "tokenizers",
            scope="country",
            system="ie",
            label_a="only",
            label_b="missing_label",
        )


def _write_two_trainer_entries(archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    write_archive_index(
        archive_dir,
        [
            {
                "label": "promoted_wordpiece_7b5e2741",
                "trainer": "wordpiece",
                "archived_utc": "2026-08-26T17:11:05Z",
            },
            {
                "label": "trained_wordpiece_882590ed",
                "trainer": "wordpiece",
                "archived_utc": "2026-08-26T16:59:12Z",
            },
            {
                "label": "promoted_sentencepiece_bpe_befe2461",
                "trainer": "sentencepiece",
                "archived_utc": "2026-08-28T07:56:51Z",
            },
        ],
    )


def test_resolve_archive_label_latest_across_all_trainers(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    label, trainer = resolve_archive_label(archive_dir=archive_dir, latest=True)

    assert (label, trainer) == ("promoted_sentencepiece_bpe_befe2461", "sentencepiece")


def test_resolve_archive_label_latest_scoped_to_one_trainer(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    label, trainer = resolve_archive_label(
        archive_dir=archive_dir, trainer="wordpiece", latest=True
    )

    assert (label, trainer) == ("promoted_wordpiece_7b5e2741", "wordpiece")


def test_resolve_archive_label_exact_match(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    label, trainer = resolve_archive_label(
        archive_dir=archive_dir, label="trained_wordpiece_882590ed"
    )

    assert (label, trainer) == ("trained_wordpiece_882590ed", "wordpiece")


def test_resolve_archive_label_unambiguous_trailing_slug(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    label, trainer = resolve_archive_label(archive_dir=archive_dir, label="7b5e274")

    assert (label, trainer) == ("promoted_wordpiece_7b5e2741", "wordpiece")


def test_resolve_archive_label_ambiguous_slug_across_trainers_hints_at_trainer(
    tmp_path: Path,
):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    with pytest.raises(ValueError, match="trainer=") as excinfo:
        resolve_archive_label(archive_dir=archive_dir, label="promoted")

    assert "promoted_wordpiece_7b5e2741" in str(excinfo.value)
    assert "promoted_sentencepiece_bpe_befe2461" in str(excinfo.value)


def test_resolve_archive_label_no_match_lists_available(tmp_path: Path):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    with pytest.raises(ValueError, match="No archived entry matches") as excinfo:
        resolve_archive_label(archive_dir=archive_dir, label="does_not_exist")

    assert "trained_wordpiece_882590ed" in str(excinfo.value)


def test_resolve_archive_label_requires_exactly_one_of_label_or_latest(
    tmp_path: Path,
):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    with pytest.raises(ValueError, match="exactly one"):
        resolve_archive_label(archive_dir=archive_dir)

    with pytest.raises(ValueError, match="exactly one"):
        resolve_archive_label(archive_dir=archive_dir, label="promoted", latest=True)


def test_resolve_archive_label_raises_when_trainer_filter_matches_nothing(
    tmp_path: Path,
):
    archive_dir = tmp_path / "archive"
    _write_two_trainer_entries(archive_dir)

    with pytest.raises(ValueError, match="No archived entries"):
        resolve_archive_label(archive_dir=archive_dir, trainer="bert", latest=True)


def _archive_with_two_trainer_models(root: Path) -> Path:
    """An `ie` country archive holding both a wordpiece and a sentencepiece
    entry, each with its model file actually present on disk.
    """
    archive_dir = resolve_archive_dir(
        tokenizer_root=root / "artifacts" / "tokenizers", scope="country", system="ie"
    )
    _write_two_trainer_entries(archive_dir)
    (archive_dir / "promoted_wordpiece_7b5e2741.json").write_text(
        "wordpiece-model", encoding="utf-8"
    )
    (archive_dir / "trained_wordpiece_882590ed.json").write_text(
        "older-wordpiece-model", encoding="utf-8"
    )
    (archive_dir / "promoted_sentencepiece_bpe_befe2461.model").write_bytes(
        b"sentencepiece-model"
    )
    return archive_dir


def test_resolve_archived_tokenizer_path_returns_named_candidate(tmp_path: Path):
    archive_dir = _archive_with_two_trainer_models(tmp_path)
    index_before = (archive_dir / "index.json").read_bytes()

    selection = resolve_archived_tokenizer_path(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        label="trained_wordpiece_882590ed",
    )

    assert selection.label == "trained_wordpiece_882590ed"
    assert selection.trainer == "wordpiece"
    assert selection.model_path == archive_dir / "trained_wordpiece_882590ed.json"
    assert selection.archive_dir == archive_dir
    # Resolution is read-only: it must not rewrite the index it just read.
    assert (archive_dir / "index.json").read_bytes() == index_before


def test_resolve_archived_tokenizer_path_uses_entry_trainer_for_suffix(tmp_path: Path):
    archive_dir = _archive_with_two_trainer_models(tmp_path)

    selection = resolve_archived_tokenizer_path(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        label="befe246",
    )

    assert selection.trainer == "sentencepiece"
    assert (
        selection.model_path
        == archive_dir / "promoted_sentencepiece_bpe_befe2461.model"
    )


def test_resolve_archived_tokenizer_path_narrows_by_trainer(tmp_path: Path):
    _archive_with_two_trainer_models(tmp_path)

    selection = resolve_archived_tokenizer_path(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
        label="promoted",
        trainer="wordpiece",
    )

    assert selection.label == "promoted_wordpiece_7b5e2741"


def test_resolve_archived_tokenizer_path_rejects_unknown_label(tmp_path: Path):
    _archive_with_two_trainer_models(tmp_path)

    with pytest.raises(ValueError, match="No archived entry matches") as excinfo:
        resolve_archived_tokenizer_path(
            tokenizer_root=tmp_path / "artifacts" / "tokenizers",
            scope="country",
            system="ie",
            label="does_not_exist",
        )

    # Names the real labels rather than silently falling back to operational.
    assert "promoted_wordpiece_7b5e2741" in str(excinfo.value)


def test_resolve_archived_tokenizer_path_rejects_indexed_entry_without_model_file(
    tmp_path: Path,
):
    archive_dir = resolve_archive_dir(
        tokenizer_root=tmp_path / "artifacts" / "tokenizers",
        scope="country",
        system="ie",
    )
    _write_two_trainer_entries(archive_dir)

    with pytest.raises(FileNotFoundError, match="model file is missing"):
        resolve_archived_tokenizer_path(
            tokenizer_root=tmp_path / "artifacts" / "tokenizers",
            scope="country",
            system="ie",
            label="promoted_wordpiece_7b5e2741",
        )
