from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts import measure_short_name_pattern_layers as module
from scripts.measure_short_name_pattern_layers import (
    entity_split,
    measure_pairs,
    normalize_for_comparison,
)
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_plan_without_reading_or_writing(
    tmp_path: Path, workspace_roots: WorkspaceRoots, repo_root: Path, capsys
) -> None:
    exit_code = module.main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--seed",
            "3",
            "--with-german-decompounding",
            "true",
            "--json-out",
            "layer-yield.json",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] measure_short_name_pattern_layers: would proceed with:" in out
    assert "short_name_layer_seed=3 (given)" in out
    assert "short_name_layer_german_decompounding=True (given)" in out
    assert (
        f"wikidata_canonical_dir={workspace_roots.data / 'wikidata' / 'canonical'!r}"
        in out
    )
    assert f"fr_canonical_dir={workspace_roots.data / 'fr' / 'canonical'!r}" in out
    # `--json-out` is not a workspace root: the checkout it resolves a
    # relative path against is this suite's own repository root, the one
    # `cli_common.resolve_workspace_roots_from_args` derives from its own
    # file location, never overridable by `--data-dir`/`--output-dir`.
    expected_json_out = repo_root / "layer-yield.json"
    assert f"would clear {expected_json_out}" in out
    assert not expected_json_out.exists()


def test_dry_run_resolves_a_relative_json_out_against_the_checkout_not_cwd(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
    repo_root: Path,
    monkeypatch,
    capsys,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    exit_code = module.main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--json-out",
            "layer-yield.json",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(repo_root / "layer-yield.json") in out
    assert not (elsewhere / "layer-yield.json").exists()


def test_dry_run_reports_no_planned_output_when_json_out_is_not_given(
    workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = module.main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "json_out=None" in out
    assert "would clear" not in out


def test_normalize_for_comparison_lowercases_and_strips_punctuation():
    assert normalize_for_comparison("I.B.M.") == "ibm"
    assert normalize_for_comparison("  SSG   Leipzig ") == "ssg leipzig"
    assert normalize_for_comparison(None) == ""
    assert normalize_for_comparison("") == ""


class TestEntitySplit:
    def test_every_entity_gets_exactly_one_split(self):
        entity_ids = [f"Q{i}" for i in range(200)]
        assignment = entity_split(entity_ids, seed=0)
        assert set(assignment) == set(entity_ids)
        assert set(assignment.values()) <= {"train", "test", "validate"}

    def test_deterministic_across_reruns(self):
        entity_ids = [f"Q{i}" for i in range(200)]
        first = entity_split(entity_ids, seed=7)
        second = entity_split(entity_ids, seed=7)
        assert first == second

    def test_different_seeds_can_move_an_entity(self):
        entity_ids = [f"Q{i}" for i in range(200)]
        seed_0 = entity_split(entity_ids, seed=0)
        seed_1 = entity_split(entity_ids, seed=1)
        assert seed_0 != seed_1

    def test_roughly_matches_requested_weights(self):
        entity_ids = [f"Q{i}" for i in range(5000)]
        assignment = entity_split(entity_ids, seed=0, weights=(0.7, 0.15, 0.15))
        counts = {"train": 0, "test": 0, "validate": 0}
        for split_name in assignment.values():
            counts[split_name] += 1
        assert 0.65 < counts["train"] / len(entity_ids) < 0.75
        assert 0.10 < counts["test"] / len(entity_ids) < 0.20
        assert 0.10 < counts["validate"] / len(entity_ids) < 0.20


def test_measure_pairs_scores_layers_and_default_order_independently():
    # "International Business Machines" -> "IBM" is exact_initials-shaped;
    # "National Aeronautics and Space Administration" -> "NASA" only
    # resolves once "and" is skipped (subsequence_initials).
    pairs = pl.DataFrame(
        {
            "entity_id": ["Q1", "Q2"],
            "language_code": ["en", "en"],
            "official_name": [
                "International Business Machines",
                "National Aeronautics and Space Administration",
            ],
            "short_name": ["IBM", "NASA"],
            "source": ["wikidata", "wikidata"],
        }
    )
    split_map = {"Q1": "train", "Q2": "train"}

    results = measure_pairs(pairs, split_map=split_map)
    by_layer = {entry.layer: entry for entry in results}

    assert by_layer["exact_initials"].n_correct == 1
    assert by_layer["exact_initials"].n_pairs == 2
    assert by_layer["subsequence_initials"].n_correct == 1
    # default_order tries subsequence_initials first, then exact_initials,
    # so both names resolve correctly through the combinator.
    assert by_layer["default_order"].n_correct == 2
    assert by_layer["oracle_any_layer"].n_correct == 2


def test_measure_pairs_never_credits_a_null_or_blank_target():
    pairs = pl.DataFrame(
        {
            "entity_id": ["Q1"],
            "language_code": ["en"],
            "official_name": ["Acme Widgets"],
            "short_name": [None],
            "source": ["wikidata"],
        }
    )
    results = measure_pairs(pairs, split_map={"Q1": "train"})
    assert all(entry.n_correct == 0 for entry in results)


def test_measure_pairs_splits_rows_into_their_assigned_split():
    pairs = pl.DataFrame(
        {
            "entity_id": ["Q1", "Q2"],
            "language_code": ["en", "en"],
            "official_name": ["Acme Widgets", "Beta Systems"],
            "short_name": ["AW", "BS"],
            "source": ["wikidata", "wikidata"],
        }
    )
    results = measure_pairs(pairs, split_map={"Q1": "train", "Q2": "test"})
    splits = {entry.split for entry in results}
    assert splits == {"train", "test"}
    train_default = next(
        e for e in results if e.split == "train" and e.layer == "default_order"
    )
    test_default = next(
        e for e in results if e.split == "test" and e.layer == "default_order"
    )
    assert train_default.n_pairs == 1
    assert test_default.n_pairs == 1
