from __future__ import annotations

from company_tokenize._cli_helper import SETTINGS
from company_tokenize.training import SUPPORTED_TRAINERS

_TYPES = {"str", "int", "float", "bool"}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES


def test_every_tokenizer_setting_applies_only_to_the_tokens_text_view() -> None:
    for setting in SETTINGS:
        assert setting["applies"] == {"text_view": ("tokens",)}


def test_the_trainer_choices_are_the_trainers_the_package_supports() -> None:
    trainer = next(s for s in SETTINGS if s["name"] == "tokenizer")

    assert trainer["choices"] == SUPPORTED_TRAINERS
    assert trainer["default"] in SUPPORTED_TRAINERS
