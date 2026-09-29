from __future__ import annotations

from collections.abc import Mapping

from company_tokenize.training import SUPPORTED_TRAINERS
from company_vectorize.clustering_contract import TFIDF_ANALYZERS

from validation._cli_helper import SETTINGS
from validation.config import resolve_prepared_rows_per_file

_TYPES = {"str", "int", "float", "bool"}
_DIMENSIONS = {"representation", "text_view", "backend"}


def _by_name() -> dict[str, dict[str, object]]:
    return {str(setting["name"]): setting for setting in SETTINGS}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES
        applies = setting.get("applies")
        applies_keys = set(applies) if isinstance(applies, Mapping) else set()
        assert applies_keys <= _DIMENSIONS


def test_declared_defaults_are_the_ones_the_area_itself_uses() -> None:
    settings = _by_name()

    assert settings["prepared_rows_per_file"][
        "default"
    ] == resolve_prepared_rows_per_file(None)
    assert settings["tfidf_analyzer"]["choices"] == tuple(sorted(TFIDF_ANALYZERS))
    assert settings["tokenizer"]["choices"] == SUPPORTED_TRAINERS
    assert settings["representation"]["default"] == "tfidf"
    assert settings["similarity_backend"]["default"] == "sklearn"
    assert settings["text_view"]["default"] == "auto"
    assert settings["tokenizer"]["default"] == "wordpiece"
    assert settings["tokenizer_scope"]["default"] == "country"
