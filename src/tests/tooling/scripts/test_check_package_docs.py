from __future__ import annotations

from pathlib import Path

from check_package_docs import check_no_workspace_layout


def _write_src_file(package_dir: Path, name: str, content: str) -> None:
    src_dir = package_dir / "src" / package_dir.name
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / name).write_text(content, encoding="utf-8")


def test_check_no_workspace_layout_allows_clean_file(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        "def resolve(source_files):\n    return sorted(source_files)\n",
    )

    assert check_no_workspace_layout(package_dir, tmp_path) == []


def test_check_no_workspace_layout_flags_bare_import(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(package_dir, "ops.py", "import workspace\n")

    errors = check_no_workspace_layout(package_dir, tmp_path)

    assert len(errors) == 1
    assert "imports 'workspace'" in errors[0]


def test_check_no_workspace_layout_flags_from_import_of_submodule(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        "from workspace.layer_layout import resolve_primary_files\n",
    )

    errors = check_no_workspace_layout(package_dir, tmp_path)

    assert len(errors) == 1
    assert "imports from 'workspace.layer_layout'" in errors[0]


def test_check_no_workspace_layout_flags_fstring_partition_path(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        'def path_for(root, value):\n    return root / f"jurisdiction_code={value}"\n',
    )

    errors = check_no_workspace_layout(package_dir, tmp_path)

    assert len(errors) == 1
    assert "hand-builds a 'jurisdiction_code='-style partition path" in errors[0]


def test_check_no_workspace_layout_flags_plain_string_partition_path(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        'PREFIX = "jurisdiction_code=gb"\n',
    )

    errors = check_no_workspace_layout(package_dir, tmp_path)

    assert len(errors) == 1
    assert "hand-builds a 'jurisdiction_code='-style partition path" in errors[0]


def test_check_no_workspace_layout_ignores_docstring_mention(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        '"""Module doc mentioning `jurisdiction_code=` partitions under data/."""\n\n'
        "def resolve(source_files):\n"
        "    return sorted(source_files)\n",
    )

    assert check_no_workspace_layout(package_dir, tmp_path) == []


def test_check_no_workspace_layout_ignores_class_and_function_docstrings(
    tmp_path: Path,
):
    package_dir = tmp_path / "company_widget"
    _write_src_file(
        package_dir,
        "ops.py",
        "class Entry:\n"
        '    """Matches `jurisdiction_code=` partitions under data/."""\n\n'
        "def resolve():\n"
        '    """Matches `jurisdiction_code=` partitions under data/."""\n'
        "    return None\n",
    )

    assert check_no_workspace_layout(package_dir, tmp_path) == []


def test_check_no_workspace_layout_missing_src_dir_returns_no_errors(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    package_dir.mkdir()

    assert check_no_workspace_layout(package_dir, tmp_path) == []


def test_check_no_workspace_layout_reports_multiple_files(tmp_path: Path):
    package_dir = tmp_path / "company_widget"
    _write_src_file(package_dir, "a.py", "import workspace\n")
    _write_src_file(package_dir, "b.py", "import workspace\n")

    errors = check_no_workspace_layout(package_dir, tmp_path)

    assert len(errors) == 2
