"""Direct tests for `workspace.published_reports`."""

from __future__ import annotations

from workspace.published_reports import publish, tokenizer_reports_dir
from workspace.roots import WorkspaceRoots


def test_tokenizer_reports_sit_under_the_checkouts_docs(
    workspace_roots: WorkspaceRoots,
):
    base = workspace_roots.checkout / "docs" / "reports" / "tokenizers"
    assert tokenizer_reports_dir(workspace_roots) == base
    assert tokenizer_reports_dir(workspace_roots, scope="ie") == base / "ie"


def test_publish_writes_texts_and_copies_files(workspace_roots: WorkspaceRoots):
    source = workspace_roots.checkout / "plot.png"
    source.write_bytes(b"png")
    directory = tokenizer_reports_dir(workspace_roots)
    (directory / "ie").mkdir(parents=True)
    (directory / "kept.md").write_text("other trainer", encoding="utf-8")

    publish(directory, texts={"report.md": "# r\n"}, files={"fixed.png": source})

    assert (directory / "report.md").read_text(encoding="utf-8") == "# r\n"
    assert (directory / "fixed.png").read_bytes() == b"png"
    assert (directory / "kept.md").exists()


def test_publish_with_replace_removes_the_files_already_there_but_not_directories(
    workspace_roots: WorkspaceRoots,
):
    directory = tokenizer_reports_dir(workspace_roots, scope="ie")
    (directory / "nested").mkdir(parents=True)
    (directory / "zipf.raw.png").write_bytes(b"stale")

    publish(directory, texts={"report.md": "# r\n"}, replace=True)

    assert sorted(path.name for path in directory.iterdir()) == ["nested", "report.md"]
