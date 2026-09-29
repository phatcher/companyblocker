from __future__ import annotations

import subprocess
from pathlib import Path

import update_design_artifacts as uda


def _completed(returncode: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_update_pyscn_invokes_expected_command(mocker, tmp_path: Path) -> None:
    run_mock = mocker.patch(
        "update_design_artifacts.subprocess.run",
        return_value=_completed(0),
    )

    result = uda.update_pyscn(tmp_path)

    assert result is True
    run_mock.assert_called_once_with(
        ["uv", "run", "pyscn", "analyze", "--html", "."], cwd=tmp_path, check=False
    )


def test_update_pyscn_returns_false_on_failure(mocker, tmp_path: Path) -> None:
    mocker.patch(
        "update_design_artifacts.subprocess.run",
        return_value=_completed(1),
    )

    assert uda.update_pyscn(tmp_path) is False


def test_update_graphify_runs_update_then_cluster_only(mocker, tmp_path: Path) -> None:
    run_mock = mocker.patch(
        "update_design_artifacts.subprocess.run",
        return_value=_completed(0),
    )

    result = uda.update_graphify(tmp_path)

    assert result is True
    assert run_mock.call_count == 2
    first_call, second_call = run_mock.call_args_list
    assert first_call.args[0] == ["uv", "run", "graphify", "update", "."]
    assert second_call.args[0] == [
        "uv",
        "run",
        "graphify",
        "cluster-only",
        ".",
        "--no-label",
    ]


def test_update_graphify_skips_cluster_only_when_update_fails(
    mocker, tmp_path: Path
) -> None:
    run_mock = mocker.patch(
        "update_design_artifacts.subprocess.run",
        return_value=_completed(1),
    )

    result = uda.update_graphify(tmp_path)

    assert result is False
    run_mock.assert_called_once()


def test_main_only_runs_selected_tool(mocker, tmp_path: Path) -> None:
    pyscn_mock = mocker.Mock(return_value=True)
    graphify_mock = mocker.Mock(return_value=True)
    mocker.patch.dict(
        uda.TOOLS, {"pyscn": pyscn_mock, "graphify": graphify_mock}, clear=False
    )
    mocker.patch(
        "sys.argv",
        ["update_design_artifacts.py", "--scan", str(tmp_path), "--only", "pyscn"],
    )

    exit_code = uda.main()

    assert exit_code == 0
    pyscn_mock.assert_called_once_with(tmp_path)
    graphify_mock.assert_not_called()


def test_main_returns_nonzero_when_a_tool_fails(mocker, tmp_path: Path) -> None:
    mocker.patch.dict(
        uda.TOOLS,
        {
            "pyscn": mocker.Mock(return_value=False),
            "graphify": mocker.Mock(return_value=True),
        },
        clear=False,
    )
    mocker.patch("sys.argv", ["update_design_artifacts.py", "--scan", str(tmp_path)])

    exit_code = uda.main()

    assert exit_code == 1


def test_build_parser_only_choices_match_tools() -> None:
    parser = uda.build_parser()
    only_action = next(a for a in parser._actions if a.dest == "only")
    assert only_action.choices is not None
    assert set(only_action.choices) == set(uda.TOOLS)


def test_main_without_scan_analyzes_this_checkout_not_the_cwd(
    mocker, tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, Path] = {}

    def fake_tool(root: Path) -> bool:
        captured["root"] = root
        return True

    mocker.patch.dict(
        uda.TOOLS, {"pyscn": fake_tool, "graphify": fake_tool}, clear=False
    )
    monkeypatch.chdir(tmp_path)
    mocker.patch("sys.argv", ["update_design_artifacts.py"])

    exit_code = uda.main()

    assert exit_code == 0
    assert captured["root"] == uda.REPO_ROOT
    assert captured["root"] != tmp_path.resolve()
