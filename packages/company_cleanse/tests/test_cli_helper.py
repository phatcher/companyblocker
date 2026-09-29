from __future__ import annotations

from company_cleanse._cli_helper import SETTINGS

_TYPES = {"str", "int", "float", "bool"}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES


def test_iso20275_countries_defaults_to_every_country() -> None:
    setting = next(s for s in SETTINGS if s["name"] == "iso20275_countries")

    assert setting["type"] == "str"
    assert setting["default"] is None


def test_short_name_layer_seed_defaults_to_zero() -> None:
    setting = next(s for s in SETTINGS if s["name"] == "short_name_layer_seed")

    assert setting["type"] == "int"
    assert setting["default"] == 0


def test_short_name_layer_german_decompounding_defaults_to_off() -> None:
    setting = next(
        s for s in SETTINGS if s["name"] == "short_name_layer_german_decompounding"
    )

    assert setting["type"] == "bool"
    assert setting["default"] is False
