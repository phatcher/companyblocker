from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import cli_common
from scripts.cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    OUTPUT_MIGRATE,
    PlannedOutput,
    add_dry_run_arg,
    coerce_scalar_value,
    describe_output_contents,
    parse_bool_literal,
    parse_optional_bool,
    parse_run_date,
    report_dry_run,
    report_output_plan,
    run_reporting_argument_errors,
)
from workspace.kind_layout import Kind
from workspace.reference import Side, locate, reference, reference_at


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026-06-15", "2026-06-15"),
        ("2026-6-15", "2026-06-15"),
        ("2026-6-5", "2026-06-05"),
        (" 2026-6-5 ", "2026-06-05"),
    ],
)
def test_parse_run_date_normalizes_iso_padding(raw_value: str, expected: str):
    assert parse_run_date(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value", ["2026/06/15", "15-06-2026", "2026-13-01", "2026-02-30"]
)
def test_parse_run_date_rejects_invalid_values(raw_value: str):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_run_date(raw_value)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (True, True),
        (False, False),
        (None, None),
        ("true", True),
        (" OFF ", False),
    ],
)
def test_parse_bool_literal_supports_common_inputs(
    raw_value: object, expected: bool | None
):
    assert parse_bool_literal(raw_value) is expected


def test_parse_bool_literal_rejects_invalid_values():
    with pytest.raises(ValueError, match="Invalid boolean value"):
        parse_bool_literal("maybe")


def test_parse_optional_bool_defaults_none_to_true():
    assert parse_optional_bool(None, argument_name="--enabled") is True


def test_parse_optional_bool_raises_argument_type_error_for_invalid_values():
    with pytest.raises(
        argparse.ArgumentTypeError, match="Expected boolean value for --enabled"
    ):
        parse_optional_bool("maybe", argument_name="--enabled")


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("No", False),
        ("42", 42),
        ("3.5", 3.5),
        ("wordpiece", "wordpiece"),
    ],
)
def test_coerce_scalar_value_coerces_bool_numeric_and_string_values(
    raw_value: str, expected: object
):
    assert coerce_scalar_value(raw_value) == expected


def test_resolve_existing_path_and_rows_returns_normalized_values(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    resolved_source, rows = cli_common.resolve_existing_path_and_rows(
        source=source, rows=25
    )

    assert resolved_source == source
    assert rows == 25


def test_resolve_existing_path_and_rows_rejects_non_positive_rows(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--rows must be >= 1"):
        cli_common.resolve_existing_path_and_rows(source=source, rows=0)


def test_resolve_existing_path_and_rows_requires_existing_source(tmp_path: Path):
    missing_source = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        cli_common.resolve_existing_path_and_rows(source=missing_source, rows=10)


def test_add_dry_run_arg_defaults_to_false_and_is_a_flag():
    parser = argparse.ArgumentParser()
    add_dry_run_arg(parser)

    assert parser.parse_args([]).dry_run is False
    assert parser.parse_args(["--dry-run"]).dry_run is True


def test_add_dry_run_arg_accepts_custom_help_text():
    parser = argparse.ArgumentParser()
    add_dry_run_arg(parser, help_text="Custom help.")

    [dry_run_action] = [a for a in parser._actions if a.dest == "dry_run"]
    assert dry_run_action.help == "Custom help."


def test_report_dry_run_prints_the_script_name_and_resolved_values(capsys):
    report_dry_run("some_script", source="gb", rows=10)

    out = capsys.readouterr().out
    assert "[dry-run] some_script: would proceed with:" in out
    assert "source='gb'" in out
    assert "rows=10" in out


def test_describe_output_contents_reports_an_absent_path(tmp_path: Path):
    assert describe_output_contents(tmp_path / "missing") == "does not exist"


def test_describe_output_contents_reports_an_empty_directory(tmp_path: Path):
    assert describe_output_contents(tmp_path) == "exists, empty"


def test_describe_output_contents_counts_and_sizes_the_matching_files(tmp_path: Path):
    (tmp_path / "part=a").mkdir()
    (tmp_path / "part=a" / "one.parquet").write_bytes(b"x" * 1024)
    (tmp_path / "two.parquet").write_bytes(b"x" * 1024)
    (tmp_path / "notes.txt").write_bytes(b"x")

    assert describe_output_contents(tmp_path) == "3 file(s), 2.0 KiB"
    assert (
        describe_output_contents(tmp_path, pattern="*.parquet") == "1 file(s), 1.0 KiB"
    )


def test_describe_output_contents_reports_a_single_file(tmp_path: Path):
    target = tmp_path / "model.json"
    target.write_bytes(b"x" * 10)

    assert describe_output_contents(target) == "1 file(s), 10 B"


@pytest.mark.parametrize("effect", [OUTPUT_CLEAR, OUTPUT_EXTEND])
def test_report_output_plan_names_the_effect_and_what_the_path_holds(
    tmp_path: Path, capsys, effect: str
):
    (tmp_path / "a.parquet").write_bytes(b"x")

    report_output_plan("[dry-run] [gb]", [PlannedOutput(tmp_path, effect, note="why")])

    assert capsys.readouterr().out == (
        f"[dry-run] [gb] would {effect} {tmp_path} [1 file(s), 1 B] -- why\n"
    )


def test_report_output_plan_names_where_a_migrated_output_goes(tmp_path: Path, capsys):
    legacy = tmp_path / "legacy"
    legacy.mkdir()

    report_output_plan(
        "p", [PlannedOutput(legacy, OUTPUT_MIGRATE, into=tmp_path / "current")]
    )

    assert capsys.readouterr().out == (
        f"p would migrate {legacy} into {tmp_path / 'current'} [exists, empty]\n"
    )


def test_report_output_plan_reports_an_absent_path_unless_it_is_optional(
    tmp_path: Path, capsys
):
    report_output_plan(
        "p",
        [
            PlannedOutput(tmp_path / "required", OUTPUT_CLEAR),
            PlannedOutput(tmp_path / "legacy", OUTPUT_CLEAR, optional=True),
        ],
    )

    out = capsys.readouterr().out
    assert out == f"p would clear {tmp_path / 'required'} [does not exist]\n"
    assert not (tmp_path / "required").exists()


def test_report_output_plan_uses_the_callers_description_of_an_existing_path(
    tmp_path: Path, capsys
):
    report_output_plan(
        "p",
        [
            PlannedOutput(tmp_path, OUTPUT_CLEAR, holds=lambda path: "3 partition(s)"),
            PlannedOutput(
                tmp_path / "missing", OUTPUT_CLEAR, holds=lambda path: "never called"
            ),
        ],
    )

    out = capsys.readouterr().out
    assert f"p would clear {tmp_path} [3 partition(s)]\n" in out
    assert f"p would clear {tmp_path / 'missing'} [does not exist]\n" in out


def test_planned_output_rejects_an_unknown_effect(tmp_path: Path):
    with pytest.raises(ValueError, match="not one of"):
        PlannedOutput(tmp_path, "overwrite")


@pytest.mark.parametrize(
    ("effect", "into"), [(OUTPUT_MIGRATE, None), (OUTPUT_CLEAR, Path("elsewhere"))]
)
def test_planned_output_requires_a_destination_exactly_when_migrating(
    tmp_path: Path, effect: str, into: Path | None
):
    with pytest.raises(ValueError, match="names where it goes"):
        PlannedOutput(tmp_path, effect, into=into)


def _reference_parser(*prefixes: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(exit_on_error=False)
    for prefix in prefixes:
        cli_common.add_reference_args(parser, Kind.BLOCKING, Side.DATA, prefix=prefix)
    cli_common.add_reference_args(parser, Kind.PERTURBATION, Side.PROFILE)
    return parser


@pytest.mark.parametrize(
    ("kind", "side", "argv", "uri"),
    [
        (
            Kind.PERTURBATION,
            Side.PROFILE,
            ["--name", "en-lite", "--version", "v2", "--draft"],
            "perturbation://en-lite/v2/draft",
        ),
        (
            Kind.PERTURBATION,
            Side.DATA,
            [
                "--source",
                "ie",
                "--profile",
                "en-lite",
                "--version",
                "v1",
                "--seed",
                "42",
            ],
            "perturbed://ie/en-lite/v1/42",
        ),
        (
            Kind.BLOCKING,
            Side.DATA,
            [
                "--source",
                "gleif",
                "--target",
                "gb",
                "--representation",
                "tfidf",
                "--key",
                "0123456789ab",
            ],
            "blocking://gleif/gb/tfidf/0123456789ab",
        ),
    ],
)
def test_a_reference_round_trips_between_flags_reference_uri_and_location(
    workspace_roots, kind: Kind, side: Side, argv: list[str], uri: str
):
    parser = argparse.ArgumentParser()
    cli_common.add_reference_args(parser, kind, side)

    ref = cli_common.reference_from_args(parser.parse_args(argv), kind, side)

    assert ref.uri == uri
    assert cli_common.reference_flags(ref) == argv
    assert reference_at(workspace_roots, locate(workspace_roots, ref)) == ref


def test_reference_flags_take_a_prefix_for_two_references_of_one_kind():
    parser = _reference_parser("left", "right")

    args = parser.parse_args(["--left-target", "gb", "--right-target", "ie"])

    assert (
        cli_common.reference_from_args(
            args, Kind.BLOCKING, Side.DATA, prefix="right"
        ).uri
        == "blocking://*/ie"
    )


def test_a_reference_flag_refuses_a_value_its_segment_does_not_admit():
    parser = argparse.ArgumentParser()
    cli_common.add_reference_args(parser, Kind.PERTURBATION, Side.PROFILE)

    with pytest.raises(SystemExit):
        parser.parse_args(["--name", "en-lite", "--version", "2.0.0"])


def test_requiring_a_reference_from_flags_names_what_exists(workspace_roots, capsys):
    locate(
        workspace_roots,
        reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v1"),
    ).mkdir(parents=True)
    parser = argparse.ArgumentParser()
    cli_common.add_reference_args(parser, Kind.PERTURBATION, Side.PROFILE)
    args = parser.parse_args(["--name", "en-lite", "--version", "v2"])

    with pytest.raises(SystemExit):
        cli_common.require_reference_from_args(
            parser, args, workspace_roots, Kind.PERTURBATION, Side.PROFILE
        )

    assert "perturbation://en-lite/v1" in capsys.readouterr().err


def test_requiring_a_reference_from_flags_returns_the_one_that_exists(workspace_roots):
    existing = reference(Kind.PERTURBATION, Side.PROFILE, name="en-lite", version="v1")
    locate(workspace_roots, existing).mkdir(parents=True)
    parser = argparse.ArgumentParser()
    cli_common.add_reference_args(parser, Kind.PERTURBATION, Side.PROFILE)

    assert (
        cli_common.require_reference_from_args(
            parser,
            parser.parse_args(["--name", "en-lite"]),
            workspace_roots,
            Kind.PERTURBATION,
            Side.PROFILE,
        )
        == existing
    )


def test_script_invocation_keeps_the_flags_and_drops_the_roots():
    assert cli_common.script_invocation(
        [
            "C:/checkout/scripts/run_blocking.py",
            "--data-dir",
            "D:/data",
            "--target",
            "gb",
            "--output-dir=E:/out",
            "--temp-dir",
            "F:/tmp",
            "--dry-run",
        ]
    ) == ("run_blocking.py", "--target", "gb", "--dry-run")


_DATES = (
    cli_common.RoleSetting("date", "source", "The source snapshot."),
    cli_common.RoleSetting("date", "target", "The target snapshot."),
)


def _role_args(settings, argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli_common.add_role_settings(parser, settings)
    return parser.parse_args(argv)


def test_a_bare_flag_with_one_role_is_that_role_s() -> None:
    args = _role_args(_DATES[:1], ["--date", "2026-06-01"])

    assert cli_common.resolve_role_settings(args, _DATES[:1]) == {
        "source_date": "2026-06-01"
    }


def test_a_bare_flag_is_given_to_every_role_that_can_take_it() -> None:
    args = _role_args(_DATES, ["--date", "2026-06-01"])

    assert cli_common.resolve_role_settings(args, _DATES, accepts=lambda *_: True) == {
        "source_date": "2026-06-01",
        "target_date": "2026-06-01",
    }


def test_a_bare_flag_one_role_cannot_take_is_refused_naming_the_qualified_forms() -> (
    None
):
    args = _role_args(_DATES, ["--date", "2026-06-01"])

    with pytest.raises(
        cli_common.RoleSettingError, match="--source-date, --target-date"
    ):
        cli_common.resolve_role_settings(
            args, _DATES, accepts=lambda setting, _: setting.role == "source"
        )


def test_a_qualified_flag_wins_for_its_role_beside_the_bare_one() -> None:
    args = _role_args(_DATES, ["--date", "2026-06-01", "--target-date", "2026-01-05"])

    assert cli_common.resolve_role_settings(args, _DATES) == {
        "source_date": "2026-06-01",
        "target_date": "2026-01-05",
    }


def test_a_seed_is_never_shared_between_two_roles() -> None:
    seeds = (
        cli_common.RoleSetting("seed", "perturbed", "", type=int, shared=False),
        cli_common.RoleSetting("seed", "sample", "", type=int, shared=False),
    )
    args = _role_args(seeds, ["--seed", "42"])

    with pytest.raises(
        cli_common.RoleSettingError, match="--perturbed-seed, --sample-seed"
    ):
        cli_common.resolve_role_settings(args, seeds)


def test_a_script_s_own_seed_keeps_the_bare_flag() -> None:
    seeds = (
        cli_common.RoleSetting("seed", None, "", type=int, shared=False),
        cli_common.RoleSetting("seed", "perturbed", "", type=int, shared=False),
    )
    args = _role_args(seeds, ["--seed", "7", "--perturbed-seed", "42"])

    assert cli_common.resolve_role_settings(args, seeds) == {
        "seed": 7,
        "perturbed_seed": 42,
    }


def test_a_perturbed_source_is_named_by_its_profile_and_pinned_to_what_is_on_disk(
    workspace_roots,
) -> None:
    dataset = reference(
        Kind.PERTURBATION,
        Side.DATA,
        source="ie",
        profile="en-lite",
        version="v1",
        seed="42",
    )
    locate(workspace_roots, dataset).mkdir(parents=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="source_system")
    cli_common.add_perturbed_source_args(parser)

    plain = parser.parse_args(["--source", "ie"])
    perturbed = parser.parse_args(["--source", "ie", "--perturbed", "en-lite"])
    typed = parser.parse_args(["--source", "perturbed://ie/en-lite/v1/42"])

    assert cli_common.source_system_from_args(plain, workspace_roots) == "ie"
    selector = cli_common.source_system_from_args(perturbed, workspace_roots)
    assert selector == "perturbed://ie/en-lite/v1/42"
    assert cli_common.source_flags(selector) == [
        "--source", "ie", "--perturbed", "en-lite",
        "--perturbed-version", "v1", "--perturbed-seed", "42",
    ]  # fmt: skip
    with pytest.raises(cli_common.RoleSettingError, match="--perturbed"):
        cli_common.source_system_from_args(typed, workspace_roots)


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (FileExistsError("a run for '2026-09-26' already exists; pass --force"), "a run for '2026-09-26' already exists; pass --force"),
        (KeyError("Unknown system 'xx'. Known values: ie, gb"), "Unknown system 'xx'. Known values: ie, gb"),
        (LookupError("nothing promoted for wordpiece/ie"), "nothing promoted for wordpiece/ie"),
    ],
)  # fmt: skip
def test_run_reporting_argument_errors_reports_one_line_and_exits_two(
    error: Exception, expected_message: str, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("sys.argv", ["scripts/analyze_token_zipf.py"])

    def main() -> int:
        raise error

    assert run_reporting_argument_errors(main) == 2
    captured = capsys.readouterr()
    assert captured.err == f"analyze_token_zipf.py: error: {expected_message}\n"
    assert captured.out == ""


def test_run_reporting_argument_errors_passes_a_return_code_through() -> None:
    assert run_reporting_argument_errors(lambda: 3) == 3


def test_run_reporting_argument_errors_lets_any_other_failure_propagate() -> None:
    def main() -> int:
        raise RuntimeError("the computation itself failed")

    with pytest.raises(RuntimeError, match="the computation itself failed"):
        run_reporting_argument_errors(main)
