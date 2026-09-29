from __future__ import annotations

from company_classify._cli_helper import NEGATIVE_STRATEGIES, SETTINGS
from company_classify.entity_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_RATIO,
    EntitySplitter,
    SplitRatios,
)
from company_classify.negatives import (
    DEFAULT_HARD_POOL_SIZE,
    DEFAULT_NEGATIVE_SEED,
    NegativeSamplingConfig,
    NegativeStrategy,
)

_TYPES = {"str", "int", "float", "bool"}


def _by_name() -> dict[str, dict[str, object]]:
    return {str(setting["name"]): setting for setting in SETTINGS}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES


def test_declared_defaults_are_the_ones_the_package_dataclasses_use() -> None:
    settings = _by_name()

    assert settings["split_seed"]["default"] == DEFAULT_SPLIT_SEED
    assert settings["split_seed"]["default"] == EntitySplitter().seed
    assert settings["split_train_ratio"]["default"] == DEFAULT_TRAIN_RATIO
    assert settings["split_train_ratio"]["default"] == SplitRatios().train
    assert settings["split_validation_ratio"]["default"] == DEFAULT_VALIDATION_RATIO
    assert settings["split_validation_ratio"]["default"] == SplitRatios().validation

    negatives = NegativeSamplingConfig()
    assert settings["negative_strategy"]["default"] == negatives.strategy.value
    assert settings["negative_strategy"]["choices"] == NEGATIVE_STRATEGIES
    assert set(NEGATIVE_STRATEGIES) == {strategy.value for strategy in NegativeStrategy}
    assert (
        settings["negatives_per_positive"]["default"]
        == negatives.negatives_per_positive
    )
    assert settings["negative_seed"]["default"] == DEFAULT_NEGATIVE_SEED
    assert settings["negative_seed"]["default"] == negatives.seed
    assert settings["hard_pool_size"]["default"] == DEFAULT_HARD_POOL_SIZE
    assert settings["hard_pool_size"]["default"] == negatives.hard_pool_size
