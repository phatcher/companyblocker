from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    TOKENIZER_IDS,
    compute_corpus_content_hash,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
    tokenizer_directory_files,
)

from analysis.token_zipf import run_token_zipf_analysis
from training.promotion import point_profile_at_candidate, store_candidate
from training.tokenizer_corpus_report import (
    PUBLISHED_REPORT_FILENAME,
    TokenizerCorpusReportBuild,
    build_tokenizer_corpus_report,
    generate_tokenizer_corpus_report,
    publish_tokenizer_corpus_report,
)
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.data_layout import CLEANSED_LAYER_NAME
from workspace.kind_layout import Kind
from workspace.pointer import NothingPromotedError
from workspace.published_reports import tokenizer_reports_dir
from workspace.reference import Reference, Side, reference
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import tokenizer_selection

METRICS = {
    "fertility": 1.2,
    "fertility_distance": 0.0123,
    "unk_rate": 0.001,
    "single_char_token_pct": 4.5,
}


def _write_training_corpus(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"ie:{i}" for i in range(43)],
            "name": ["acme systems ltd", "beta holdings ltd", "gamma trading co"] * 14
            + ["delta group"],
        }
    ).write_parquet(path)


def _store(
    roots: WorkspaceRoots,
    *,
    system: str | None = "ie",
    identifier: str = "wordpiece",
    model_text: str = "model one",
    optimize_summary: dict[str, object] | None = None,
    profile: str | None = "promoted",
) -> Reference:
    """Store one candidate as training does, and point `profile` at it when given."""
    fields = TOKENIZER_IDS[identifier]
    corpus_path = tokenizer_scope_dir(roots, system=system) / "training_corpus.parquet"
    if not corpus_path.exists():
        _write_training_corpus(corpus_path)
    work = roots.checkout / "work" / identifier
    work.mkdir(parents=True, exist_ok=True)
    files = tokenizer_directory_files(work, trainer=fields.tokenizer)
    files.model.write_text(model_text, encoding="utf-8")
    files.metadata.write_text("{}", encoding="utf-8")
    candidate = store_candidate(
        roots=roots,
        system=system,
        trainer=fields.tokenizer,
        tokenizer_encoding=fields.tokenizer_encoding,
        model_path=files.model,
        metadata_path=files.metadata,
        token_scores_path=None,
        corpus_path=corpus_path,
        vocab_size=64,
        min_frequency=1,
        metrics=METRICS,
        optimize_summary=optimize_summary,
        parameters={"mode": "optimize" if optimize_summary else "train"},
        invocation=["train_tokenizer.py", "--systems", str(system)],
    )
    if profile is not None:
        point_profile_at_candidate(roots, candidate, profile=profile)
    return candidate


def _write_zipf_cleansed_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "name": ["I B M", "Acme Systems Ltd", "Acme Holdings Ltd", "Beta Ltd"],
            "name_cleansed_basic": [
                "ibm",
                "acme systems ltd",
                "acme holdings ltd",
                "beta ltd",
            ],
            "name_cleansed": ["ibm", "acme systems", "acme holdings", "beta"],
        }
    ).write_parquet(path)


def test_build_report_with_nothing_stored_and_no_zipf_run_is_still_valid_markdown(
    workspace_roots: WorkspaceRoots,
):
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")
    absent = reference(Kind.TOKENIZER, Side.DATA, **selection.fields, key="absent")

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=absent)

    assert build.tokenizers_included == []
    assert build.image_files == {}
    assert "No stored tokenizer candidates found" in build.markdown
    assert "No token-Zipf run found" in build.markdown
    assert build.corpus_content_hash == "unknown"


def test_build_report_lists_each_stored_candidate_with_what_it_was_stored_with(
    workspace_roots: WorkspaceRoots,
):
    promoted = _store(workspace_roots, model_text="model one")
    naive = _store(workspace_roots, model_text="model two", profile="naive")

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=promoted)

    assert build.tokenizers_included == ["wordpiece"]
    assert f"- Candidate: `{promoted.uri}`" in build.markdown
    assert "### Tokenizer: wordpiece (this report's candidate)" in build.markdown
    rows = {
        line.split("|")[1].strip(): line
        for line in build.markdown.splitlines()
        if line.startswith("| ") and "_v64_mf1_" in line
    }
    assert set(rows) == {promoted.fields["key"], naive.fields["key"]}
    assert (
        "| promoted | train | 64 | 1 | 1.2000 | 0.012300 |"
        in rows[str(promoted.fields["key"])]
    )
    assert "| naive | train |" in rows[str(naive.fields["key"])]
    assert build.corpus_content_hash != "unknown"


def test_the_reports_own_tokenizer_comes_first_and_is_marked(
    workspace_roots: WorkspaceRoots,
):
    """A reader scanning top-down meets the candidate's own section before any
    other tokenizer's, whatever order the ids are declared in."""
    _store(workspace_roots, identifier="wordpiece", profile=None)
    own = _store(workspace_roots, identifier="sentencepiece_unigram")

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=own)

    assert build.tokenizers_included == ["sentencepiece_unigram", "wordpiece"]
    assert build.markdown.index(
        "### Tokenizer: sentencepiece_unigram (this report's candidate)"
    ) < build.markdown.index("### Tokenizer: wordpiece")
    assert "### Tokenizer: wordpiece (this report's candidate)" not in build.markdown


def test_build_report_names_the_settings_the_optimize_run_chose(
    workspace_roots: WorkspaceRoots,
):
    chosen = _store(
        workspace_roots,
        optimize_summary={
            "winner_candidate": {
                "vocab_size_requested": 25000,
                "min_frequency": 8,
                "median_fertility_distance": 0.01,
                "pass_rate": 1.0,
                "accepted_runs": 3,
                "total_runs": 3,
            }
        },
    )

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=chosen)

    assert "picked vocab=25000, min_frequency=8" in build.markdown
    assert "(3/3 seeds)" in build.markdown


def test_build_report_embeds_live_elbow_curve_when_present(
    workspace_roots: WorkspaceRoots,
):
    candidate = _store(workspace_roots)
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "training_corpus.parquet"
    )
    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_scope_dir(workspace_roots, system="ie"),
        trainer="wordpiece",
        tokenizer_encoding=None,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"grid_hash": "testgrid"}), encoding="utf-8")
    run_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(corpus_path),
        grid_hash="testgrid",
    )
    run_paths.sweep_dir.mkdir(parents=True, exist_ok=True)
    elbow_png = run_paths.sweep_dir / "elbow_curve.wordpiece.png"
    elbow_png.write_bytes(b"not a real png, just bytes for the test")

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=candidate)

    assert build.image_files == {"elbow.wordpiece.png": elbow_png}
    assert "![wordpiece elbow curve](elbow.wordpiece.png)" in build.markdown


def test_build_report_notes_missing_elbow_curve_without_failing(
    workspace_roots: WorkspaceRoots,
):
    candidate = _store(workspace_roots)

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=candidate)

    assert build.image_files == {}
    assert "No elbow-curve image available for tokenizer=wordpiece" in build.markdown
    assert "stored metrics table above is real, complete evidence" in build.markdown


@pytest.mark.integration
@pytest.mark.graphics
def test_build_report_embeds_zipf_tier_plots_when_available(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    candidate = _store(workspace_roots)
    _write_zipf_cleansed_fixture(
        layer_fixture_dir("ie", layer=CLEANSED_LAYER_NAME) / "ie-001.parquet"
    )
    run_token_zipf_analysis(
        workspace_roots, systems=["ie"], run_date="2026-08-28", wordfreq_top_n=200
    )

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=candidate)

    assert build.zipf_run_date == "2026-08-28"
    assert "Corpus Zipf / OOV Divergence" in build.markdown
    assert (
        "raw" in build.markdown
        and "basic" in build.markdown
        and "cleansed" in build.markdown
    )
    zipf_image_names = [name for name in build.image_files if name.startswith("zipf.")]
    # The curve and its head-words pair, per tier.
    assert sorted(zipf_image_names) == sorted(
        f"zipf.{tier}.{suffix}"
        for tier in ("raw", "basic", "cleansed")
        for suffix in ("png", "head_words.png")
    )
    for name in zipf_image_names:
        assert build.image_files[name].exists()


def test_build_report_skips_zipf_section_for_global_scope(
    workspace_roots: WorkspaceRoots,
):
    candidate = _store(workspace_roots, system=None)

    build = build_tokenizer_corpus_report(roots=workspace_roots, candidate=candidate)

    assert "not applicable here" in build.markdown
    assert "No token-Zipf run found" not in build.markdown


def test_the_promoted_candidates_report_is_published_under_fixed_names(
    workspace_roots: WorkspaceRoots,
):
    promoted = _store(workspace_roots)

    report = generate_tokenizer_corpus_report(
        roots=workspace_roots, system="ie", trainer="wordpiece"
    )

    assert report.candidate == promoted
    assert report.published_dir == tokenizer_reports_dir(workspace_roots, scope="ie")
    published = (report.published_dir / PUBLISHED_REPORT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert published == report.build.markdown
    assert f"- Candidate: `{promoted.uri}`" in published


def test_a_report_is_not_published_with_the_off_switch(
    workspace_roots: WorkspaceRoots,
):
    _store(workspace_roots)

    report = generate_tokenizer_corpus_report(
        roots=workspace_roots, system="ie", trainer="wordpiece", publish_report=False
    )

    assert report.published_dir is None
    assert not tokenizer_reports_dir(workspace_roots).exists()


def test_a_candidate_that_is_not_promoted_has_no_report(
    workspace_roots: WorkspaceRoots,
):
    _store(workspace_roots, profile="naive")

    with pytest.raises(NothingPromotedError):
        generate_tokenizer_corpus_report(
            roots=workspace_roots, system="ie", trainer="wordpiece"
        )

    assert not tokenizer_reports_dir(workspace_roots).exists()


def test_publishing_replaces_the_scopes_files(workspace_roots: WorkspaceRoots):
    elbow = workspace_roots.checkout / "elbow.png"
    elbow.write_bytes(b"elbow")
    stale = tokenizer_reports_dir(workspace_roots, scope="ie") / "zipf.raw.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"from an earlier promotion")
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")
    candidate = reference(Kind.TOKENIZER, Side.DATA, **selection.fields, key="k")
    build = TokenizerCorpusReportBuild(
        markdown="# Report\n\n![curve](elbow.wordpiece.png)\n",
        image_files={"elbow.wordpiece.png": elbow},
        corpus_content_hash="abcd",
    )

    directory = publish_tokenizer_corpus_report(
        roots=workspace_roots, build=build, candidate=candidate
    )

    assert sorted(path.name for path in directory.iterdir()) == [
        "elbow.wordpiece.png",
        PUBLISHED_REPORT_FILENAME,
    ]
    assert (directory / PUBLISHED_REPORT_FILENAME).read_text(encoding="utf-8") == (
        "# Report\n\n![curve](elbow.wordpiece.png)\n"
    )
