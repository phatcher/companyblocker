from __future__ import annotations

import inspect

from blocking import workflow
from blocking._cli_helper import SETTINGS
from blocking.name_transform import (
    DEFAULT_CLEANSE_PROFILE,
    DEFAULT_NAME_TRANSFORM,
    NAME_TRANSFORMS,
)
from blocking.truth import MATCH_URI_COLUMN

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
        assert set(setting.get("applies") or {}) <= _DIMENSIONS


def test_declared_defaults_are_the_ones_the_run_itself_uses() -> None:
    settings = _by_name()

    assert settings["name_transform"]["default"] == DEFAULT_NAME_TRANSFORM
    assert settings["name_transform"]["choices"] == tuple(sorted(NAME_TRANSFORMS))
    assert settings["cleanse_profile"]["default"] == DEFAULT_CLEANSE_PROFILE
    assert settings["match_col"]["default"] == MATCH_URI_COLUMN
    assert (
        settings["source_chunk_size"]["default"]
        == inspect.signature(workflow.execute_blocking_run)
        .parameters["source_chunk_size"]
        .default
    )
