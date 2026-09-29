from __future__ import annotations

import argparse
from collections.abc import Mapping

import pytest

from scripts.cli_common import (
    SOURCE_DEFAULT,
    SOURCE_GIVEN,
    SOURCE_SCRIPT_DEFAULT,
    SettingSurface,
    add_declared_arguments,
    applicable_settings_record,
    backend_options_from_settings,
    declared_keys_help,
    declared_settings,
    parse_stage_key_value_args,
    report_resolved_settings,
    resolve_declared_settings,
    resolved_setting_values,
    setting_applies,
    setting_was_given,
)

_TOP_K = {
    "name": "top_k",
    "type": "int",
    "default": 20,
    "help": "Neighbours per source row.",
}
_PREFIX_FILTER = {
    "name": "prefix_filter",
    "type": "bool",
    "default": False,
    "help": "Prune before scoring.",
    "applies": {"backend": ("sklearn",)},
}
_NGRAM_MIN = {
    "name": "tfidf_ngram_min",
    "type": "int",
    "default": 2,
    "help": "Minimum n-gram length.",
    "applies": {"representation": ("tfidf",)},
}


def _parse(
    mapping: Mapping[str, SettingSurface], argv: list[str]
) -> tuple[argparse.Namespace, dict, dict]:
    declarations = declared_settings((_TOP_K,), (_PREFIX_FILTER, _NGRAM_MIN))
    parser = argparse.ArgumentParser()
    add_declared_arguments(parser, declarations, mapping)
    parser.add_argument("--additional-args", nargs="+", action="append")
    args = parser.parse_args(argv)
    additional = parse_stage_key_value_args(
        args.additional_args,
        invalid_message="bad {token}",
        empty_message="empty {token}",
    )
    return args, declarations, additional


def _by_name(resolved: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(setting["name"]): setting for setting in resolved}


def test_declared_settings_refuses_a_name_two_owners_declare() -> None:
    with pytest.raises(ValueError, match="declared twice"):
        declared_settings((_TOP_K,), (dict(_TOP_K),))


@pytest.mark.parametrize(
    ("override", "message"),
    [({"type": "list"}, "has type"), ({"applies": {"country": ("ie",)}}, "unknown")],
)
def test_declared_settings_refuses_an_unknown_type_or_dimension(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        declared_settings(({**_TOP_K, **override},))


def test_declared_settings_refuses_two_settings_filling_one_backend_option() -> None:
    first = {**_PREFIX_FILTER, "option": "prefix_filter"}
    second = {**_PREFIX_FILTER, "name": "other_filter", "option": "prefix_filter"}
    with pytest.raises(ValueError, match="both fill backend option"):
        declared_settings((first, second))


def test_backend_options_from_settings_records_the_applicable_ones() -> None:
    """Each backend setting that applies, by its option key, as given or as
    its default; a setting with no option is never a backend option."""
    declarations = declared_settings(
        (
            _TOP_K,
            {**_PREFIX_FILTER, "option": "prefix_filter"},
            {
                "name": "kmeans_clusters",
                "type": "int",
                "default": 300,
                "help": "Partitions.",
                "applies": {"backend": ("kmeans",)},
                "option": "kmeans_clusters",
            },
        )
    )

    assert backend_options_from_settings(
        declarations, {"backend": "kmeans"}, {"top_k": 5}
    ) == {"kmeans_clusters": 300}
    assert backend_options_from_settings(
        declarations, {"backend": "sklearn"}, {"prefix_filter": True}
    ) == {"prefix_filter": True}
    assert backend_options_from_settings(declarations, {"backend": "lsh"}, {}) == {}


def test_add_declared_arguments_names_the_effective_default_in_help() -> None:
    declarations = declared_settings((_TOP_K, _PREFIX_FILTER))
    parser = argparse.ArgumentParser()
    add_declared_arguments(
        parser,
        declarations,
        {
            "top_k": {"flag": "--top-k"},
            "prefix_filter": {"flag": "--prefix-filter", "default": True},
        },
    )

    help_text = " ".join(parser.format_help().split())

    assert "Neighbours per source row. (default: 20)" in help_text
    assert "Prune before scoring. (default: True)" in help_text


def test_a_declared_bool_flag_takes_true_or_false_and_has_no_negated_spelling() -> None:
    mapping = {"prefix_filter": {"flag": "--prefix-filter"}}

    args, _, _ = _parse(mapping, ["--prefix-filter", "true"])
    assert args.prefix_filter is True

    args, _, _ = _parse(mapping, ["--prefix-filter", "false"])
    assert args.prefix_filter is False

    with pytest.raises(SystemExit):
        _parse(mapping, ["--no-prefix-filter"])


def test_a_declared_bool_flag_requires_a_value_and_refuses_a_bad_one() -> None:
    mapping = {"prefix_filter": {"flag": "--prefix-filter"}}

    with pytest.raises(SystemExit):
        _parse(mapping, ["--prefix-filter"])

    with pytest.raises(SystemExit):
        _parse(mapping, ["--prefix-filter", "maybe"])


def test_a_script_note_is_appended_to_the_owner_help() -> None:
    declarations = declared_settings((_TOP_K,))
    parser = argparse.ArgumentParser()
    add_declared_arguments(
        parser, declarations, {"top_k": {"flag": "--top-k", "note": "Per cell."}}
    )

    help_text = " ".join(parser.format_help().split())

    assert "Neighbours per source row. Per cell. (default: 20)" in help_text


def test_setting_was_given_is_true_only_for_a_flag_the_caller_passed() -> None:
    mapping = {
        "top_k": {"flag": "--top-k"},
        "prefix_filter": {"flag": "--prefix-filter"},
    }

    args, _, _ = _parse(mapping, ["--top-k", "20"])

    assert setting_was_given(args, "top_k")
    assert not setting_was_given(args, "prefix_filter")


def test_resolve_declared_settings_tells_given_from_owner_and_script_defaults() -> None:
    mapping: dict[str, SettingSurface] = {
        "top_k": {"flag": "--top-k"},
        "prefix_filter": {"flag": "--prefix-filter", "default": True},
    }

    args, declarations, additional = _parse(mapping, ["--top-k", "5"])
    defaulted = _by_name(
        resolve_declared_settings(args, declarations, mapping, additional=additional)
    )
    args, declarations, additional = _parse(mapping, ["--prefix-filter", "false"])
    given = _by_name(
        resolve_declared_settings(args, declarations, mapping, additional=additional)
    )

    assert (defaulted["top_k"]["value"], defaulted["top_k"]["source"]) == (
        5,
        SOURCE_GIVEN,
    )
    assert (
        defaulted["prefix_filter"]["value"],
        defaulted["prefix_filter"]["source"],
    ) == (
        True,
        SOURCE_SCRIPT_DEFAULT,
    )
    assert (given["top_k"]["value"], given["top_k"]["source"]) == (20, SOURCE_DEFAULT)
    assert (given["prefix_filter"]["value"], given["prefix_filter"]["source"]) == (
        False,
        SOURCE_GIVEN,
    )


def test_resolve_declared_settings_reads_a_prefixed_key_as_its_declared_type() -> None:
    mapping = {"tfidf_ngram_min": {"key": "tfidf.ngram_min"}}

    args, declarations, additional = _parse(
        mapping, ["--additional-args", "tfidf.ngram_min=4"]
    )
    given = _by_name(
        resolve_declared_settings(args, declarations, mapping, additional=additional)
    )
    args, declarations, additional = _parse(mapping, [])
    defaulted = _by_name(
        resolve_declared_settings(args, declarations, mapping, additional=additional)
    )

    assert given["tfidf_ngram_min"] == {
        "name": "tfidf_ngram_min",
        "value": 4,
        "surface": "tfidf.ngram_min",
        "source": SOURCE_GIVEN,
        "applies": True,
    }
    assert (
        defaulted["tfidf_ngram_min"]["value"],
        defaulted["tfidf_ngram_min"]["source"],
    ) == (
        2,
        SOURCE_DEFAULT,
    )


def test_resolve_declared_settings_refuses_a_key_value_of_the_wrong_type() -> None:
    mapping = {"tfidf_ngram_min": {"key": "tfidf.ngram_min"}}
    args, declarations, additional = _parse(
        mapping, ["--additional-args", "tfidf.ngram_min=wide"]
    )

    with pytest.raises(ValueError, match="expects int"):
        resolve_declared_settings(args, declarations, mapping, additional=additional)


def test_resolve_declared_settings_refuses_a_mapping_to_an_undeclared_setting() -> None:
    declarations = declared_settings((_TOP_K,))

    with pytest.raises(ValueError, match="no owner declares"):
        resolve_declared_settings(
            argparse.Namespace(), declarations, {"top_n": {"flag": "--top-n"}}
        )


def test_setting_applies_matches_named_dimensions_and_ignores_unnamed_ones() -> None:
    assert setting_applies(_NGRAM_MIN, {"representation": "tfidf"})
    assert not setting_applies(_NGRAM_MIN, {"representation": "wordpiece"})
    assert setting_applies(_NGRAM_MIN, {"backend": "sklearn"})
    assert setting_applies(_TOP_K, {"representation": "sbert", "backend": "kmeans"})


def test_report_and_record_keep_only_applicable_settings_but_values_keep_all(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mapping = {
        "top_k": {"flag": "--top-k"},
        "prefix_filter": {"flag": "--prefix-filter"},
        "tfidf_ngram_min": {"key": "tfidf.ngram_min"},
    }
    args, declarations, additional = _parse(mapping, [])
    resolved = resolve_declared_settings(
        args,
        declarations,
        mapping,
        additional=additional,
        context={"representation": "tfidf", "backend": "kmeans"},
    )

    report_resolved_settings("demo", resolved, output_dir=None)
    out = capsys.readouterr().out

    assert "  top_k=20 (default)" in out
    assert "  tfidf_ngram_min=2 (default)" in out
    assert "prefix_filter" not in out
    assert "  output_dir=None" in out
    assert set(applicable_settings_record(resolved)) == {"top_k", "tfidf_ngram_min"}
    assert resolved_setting_values(resolved) == {
        "top_k": 20,
        "prefix_filter": False,
        "tfidf_ngram_min": 2,
    }


def test_declared_keys_help_lists_each_prefixed_key_with_its_default() -> None:
    declarations = declared_settings((_TOP_K, _NGRAM_MIN))

    text = declared_keys_help(
        declarations,
        {"top_k": {"flag": "--top-k"}, "tfidf_ngram_min": {"key": "tfidf.ngram_min"}},
    )

    assert text == "tfidf.ngram_min (default: 2)"
