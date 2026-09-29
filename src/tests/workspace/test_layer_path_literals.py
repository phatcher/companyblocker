"""Direct tests for `scripts/check_layer_path_literals.py`, and the repo-wide backlogs
it enforces, which must never grow.

The checker is the enforcing half, run over a commit's own files at the pre-commit
stage. The two backlog tests are the other half, deliberately wider: they count every
offence across `src/` and `scripts/` (excluding `src/workspace/`, which owns the
shapes, and `src/tests/`, whose fixtures legitimately build literal paths), so neither
pattern creeps in somewhere a touched-files run happens to miss.

`src/tests/baselines/layer_path_literals.txt` lists the files still carrying a
`jurisdiction_code=` literal, and `workspace_paths.txt` each file's offences against
the workspace path contract by rule. Delete or lower a line in the same commit that
drains it, never add one.
"""

from __future__ import annotations

from pathlib import Path

import check_layer_path_literals
import pytest
from baseline import check_baseline
from check_layer_path_literals import (
    in_scope,
    layer_path_literals,
    main,
    workspace_path_offences,
)


def _scoped_files(repo_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for root in check_layer_path_literals.SCAN_ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(repo_root).as_posix()
            if in_scope(relative):
                files.append((path, relative))
    return files


def test_layer_path_literal_backlog_does_not_grow(repo_root: Path) -> None:
    """A new file, or a higher count in a known one, is a caller building a data/ layer
    path by hand: route it through workspace.layer_layout's resolve_partition_dir/
    resolve_primary_files/layer_partition_dir/partition_values instead."""
    found: dict[str, int] = {}
    for path, relative in _scoped_files(repo_root):
        count = len(layer_path_literals(path))
        if count:
            found[relative] = count
    check_baseline("layer_path_literals", found)


def test_workspace_path_backlog_does_not_grow(repo_root: Path) -> None:
    """A new line is a script taking or building a location the workspace should own:
    name it with a reference and a cli_common reference helper instead."""
    found: dict[str, int] = {}
    for path, relative in _scoped_files(repo_root):
        found.update(check_layer_path_literals.workspace_path_counts(path, relative))
    check_baseline("workspace_paths", found)


def _rules(
    tmp_path: Path, source: str, *, relative: str = "scripts/example.py"
) -> list[str]:
    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")
    return [
        offence.rule for offence in workspace_path_offences(path, relative=relative)
    ]


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ('parser.add_argument("--output-dir")\n', "path-flag"),
        ('parser.add_argument("--tokenizer-path")\n', "path-flag"),
        ('parser.add_argument("--json-out")\n', "path-flag"),
        ('parser.add_argument("--root", default=".")\n', "path-flag"),
        ('parser.add_argument("--source", type=Path)\n', "path-flag"),
        ("source = Path(args.source)\n", "path-from-flag"),
        ('corpus = roots.checkout / "artifacts"\n', "root-join"),
        ('scratch = base / "tmp"\n', "root-join"),
        ('default = "artifacts/tokenizers/gb/noise_words.json"\n', "root-literal"),
        ('pattern = f"data/{system}/matched"\n', "root-literal"),
        ("here = Path.cwd()\n", "cwd"),
        ("here = os.getcwd()\n", "cwd"),
        ("scratch = tempfile.mkdtemp()\n", "tempfile"),
        ("roots = default_workspace_roots(checkout)\n", "default-roots"),
        ('spec = REPO_ROOT / "config" / "p279.json"\n', "repo-join"),
        ('spec = repository_root(Path(__file__)) / "docs"\n', "repo-join"),
        ('spec = Path(__file__).resolve().parents[1] / "docs"\n', "repo-join"),
    ],
)
def test_each_rule_is_reported_on_a_fixture(tmp_path: Path, source: str, rule: str):
    assert rule in _rules(tmp_path, source)


@pytest.mark.parametrize(
    "source",
    [
        'parser.add_argument("--source", dest="source_system")\n',
        'parser.add_argument("--representation", default="tfidf")\n',
        'parser.add_argument("--rows-per-file", type=int)\n',
        "scratch = tempfile.mkdtemp(dir=roots.temp)\n",
        'def f():\n    """Writes under artifacts/blocking/data."""\n',
        "location = locate(roots, reference)\n",
        'name = f"{prefix}/data"\n',
    ],
)
def test_what_the_contract_allows_is_not_reported(tmp_path: Path, source: str):
    assert _rules(tmp_path, source) == []


def test_only_cli_commons_roots_helper_may_add_a_location_flag(tmp_path: Path):
    source = (
        "def add_workspace_roots_args(parser):\n"
        '    parser.add_argument("--data-dir")\n'
        "def add_tokenizer_path_arg(parser):\n"
        '    parser.add_argument("--tokenizer-path")\n'
    )

    assert _rules(tmp_path, source, relative="scripts/cli_common.py") == ["path-flag"]
    assert _rules(tmp_path, source, relative="scripts/other.py") == [
        "path-flag",
        "path-flag",
    ]


def test_the_workspace_and_the_tests_are_out_of_scope():
    assert in_scope("scripts/run_blocking.py")
    assert in_scope("src/blocking/run_layout.py")
    assert not in_scope("src/workspace/roots.py")
    assert not in_scope("src/tests/blocking/test_workflow.py")
    assert not in_scope("tooling/baseline.py")
    assert not in_scope("packages/company_tokenize/src/company_tokenize/io.py")
    assert not in_scope("scripts/_bootstrap.py")


def test_the_hook_fails_only_beyond_a_files_baseline(
    tmp_path: Path, monkeypatch, capsys
):
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text(
        'parser.add_argument("--json-out")\nparser.add_argument("--csv-out")\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    baselines = {
        "layer_path_literals": {},
        "workspace_paths": {"scripts/example.py path-flag": 2},
    }
    monkeypatch.setattr(
        check_layer_path_literals, "read_baseline", baselines.__getitem__
    )

    assert main(["scripts/example.py"]) == 0

    baselines["workspace_paths"]["scripts/example.py path-flag"] = 1
    assert main(["scripts/example.py"]) == 1
    assert "2 path-flag offence(s), 1 allowed" in capsys.readouterr().out
