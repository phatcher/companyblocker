from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import compare_tokenizer_scopes as compare_script
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.artifact_layout import tokenizer_artifact_root
from workspace.published_reports import tokenizer_reports_dir
from workspace.roots import WorkspaceRoots


def test_main_writes_markdown_and_json_covering_every_scope(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    promoted_tokenizer_files(
        workspace_roots, system="ie", key="chosen"
    ).metrics.write_text(
        json.dumps({"fertility_distance": 0.01, "rows": 10.0}), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_tokenizer_scopes.py",
            "--root",
            str(tmp_path),
            "--tokenizer",
            "wordpiece",
            "--systems",
            "ie",
        ],
    )
    exit_code = compare_script.main()
    assert exit_code == 0

    output_dir = tokenizer_artifact_root(workspace_roots)
    markdown_path = output_dir / "scope_comparison.wordpiece.md"
    json_path = output_dir / "scope_comparison.wordpiece.json"
    assert markdown_path.exists()
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert {row["scope"] for row in payload["quality_summary"]} == {"ie", "global"}
    ie_row = next(row for row in payload["quality_summary"] if row["scope"] == "ie")
    assert ie_row["status"] == "ok"
    assert ie_row["candidate"] == "chosen"

    global_row = next(
        row for row in payload["quality_summary"] if row["scope"] == "global"
    )
    assert global_row["status"] == "nothing promoted"

    captured = capsys.readouterr()
    assert "wrote" in captured.out
    published = tokenizer_reports_dir(workspace_roots) / markdown_path.name
    assert published.read_text(encoding="utf-8") == markdown_path.read_text(
        encoding="utf-8"
    )


def test_main_publishes_nothing_with_the_off_switch(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_tokenizer_scopes.py",
            "--root",
            str(tmp_path),
            "--systems",
            "ie",
            "--no-publish",
        ],
    )

    assert compare_script.main() == 0
    assert (
        tokenizer_artifact_root(workspace_roots) / "scope_comparison.wordpiece.md"
    ).exists()
    assert not tokenizer_reports_dir(workspace_roots).exists()
