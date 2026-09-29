from pathlib import Path
from typing import Any

import pytest

from blocking.contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingStrategyConfig,
    blocking_settings_key,
    validate_blocking_run_config,
)
from blocking.run_layout import resolve_run_location_for
from blocking.truth import ColumnTruth, MatchedLayerTruth, SourceTruthResolver
from workspace.identity import RunKeys
from workspace.roots import default_workspace_roots

_ROOTS = default_workspace_roots(Path("/root"))


def _descriptor(
    *,
    system: str,
    has_ground_truth: bool = False,
    matched_target_systems: tuple[str, ...] = (),
    layer: str = "cleansed",
) -> BlockingDatasetDescriptor:
    return BlockingDatasetDescriptor(
        system=system,
        system_dir=Path(f"/data/{system}/{layer}"),
        layer=layer,
        has_ground_truth=has_ground_truth,
        matched_target_systems=matched_target_systems,
        available_countries=("gb",),
    )


def _strategy(**overrides: Any) -> BlockingStrategyConfig:
    defaults: dict[str, Any] = {
        "representation": "tfidf",
        "top_k": 5,
        "min_similarity": 0.5,
        "max_candidates_per_source": None,
    }
    defaults.update(overrides)
    return BlockingStrategyConfig(**defaults)


def _config(
    *,
    source: BlockingDatasetDescriptor,
    target: BlockingDatasetDescriptor,
    strategy: BlockingStrategyConfig,
    countries: tuple[str, ...] | None = None,
    truth: SourceTruthResolver | None = None,
) -> BlockingRunConfig:
    return BlockingRunConfig(
        roots=_ROOTS,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=countries,
        strategy=strategy,
        truth=truth if truth is not None else MatchedLayerTruth(),
    )


_READ_KEYS = {
    "source_population": "aaaa",
    "target_population": "bbbb",
    "truth": "tttt",
    "index": "iiii",
}


def _keys(config: BlockingRunConfig, **read_keys: str) -> RunKeys:
    """A run's keys: the settings key from `config`, and the keys hashed from
    what it read, varying whichever the caller names."""
    return RunKeys(
        settings=blocking_settings_key(config), **{**_READ_KEYS, **read_keys}
    )


def _run_dir(**overrides: Any) -> Path:
    """One resolved run directory, varying whatever the caller names."""
    read_keys = overrides.pop("read_keys", {})
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(**overrides.pop("strategy", {})),
        **overrides,
    )
    return resolve_run_location_for(config, keys=_keys(config, **read_keys)).directory


def test_validate_blocking_run_config_accepts_valid_config() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_mismatched_tokenizer() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(representation="sentencepiece", tokenizer="wordpiece"),
    )

    with pytest.raises(ValueError, match="tokenizer"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_matching_tokenizer() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(representation="sentencepiece", tokenizer="sentencepiece"),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_candidate_similarity_ratio() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(candidate_similarity_ratio=1.5),
    )

    with pytest.raises(ValueError, match="candidate_similarity_ratio"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_valid_candidate_similarity_ratio() -> (
    None
):
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(candidate_similarity_ratio=0.8),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_max_candidates_per_target() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(max_candidates_per_target=0),
    )

    with pytest.raises(ValueError, match="max_candidates_per_target"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_one_sided_target_neighbor_settings() -> (
    None
):
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(target_neighbor_min_similarity=0.7),
    )

    with pytest.raises(ValueError, match="target_neighbor_min_similarity"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_target_neighbor_min_similarity() -> (
    None
):
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(
            target_neighbor_min_similarity=1.5, target_neighbor_max_per_target=5
        ),
    )

    with pytest.raises(ValueError, match="target_neighbor_min_similarity"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_target_neighbor_max_per_target() -> (
    None
):
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(
            target_neighbor_min_similarity=0.7, target_neighbor_max_per_target=0
        ),
    )

    with pytest.raises(ValueError, match="target_neighbor_max_per_target"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_valid_target_neighbor_settings() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(
            target_neighbor_min_similarity=0.7, target_neighbor_max_per_target=5
        ),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_unknown_representation() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(representation="unknown"),
    )

    with pytest.raises(ValueError, match="strategy.representation"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_top_k() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(top_k=0),
    )

    with pytest.raises(ValueError, match="top_k"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_bad_min_similarity() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(min_similarity=1.5),
    )

    with pytest.raises(ValueError, match="min_similarity"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_empty_countries() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(),
        countries=(),
    )

    with pytest.raises(ValueError, match="countries"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_target_not_in_matched_set() -> None:
    config = _config(
        source=_descriptor(
            system="gleif",
            has_ground_truth=True,
            matched_target_systems=("gleif",),
            layer="matched",
        ),
        target=_descriptor(system="offeneregister"),
        strategy=_strategy(),
    )

    with pytest.raises(ValueError, match="matched/ was joined against"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_target_in_matched_set() -> None:
    config = _config(
        source=_descriptor(
            system="ie",
            has_ground_truth=True,
            matched_target_systems=("gleif", "gb"),
            layer="matched",
        ),
        target=_descriptor(system="gb"),
        strategy=_strategy(),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_skips_ground_truth_check_without_truth() -> None:
    config = _config(
        source=_descriptor(system="gleif", has_ground_truth=False),
        target=_descriptor(system="offeneregister"),
        strategy=_strategy(),
    )

    validate_blocking_run_config(config)


def test_validate_blocking_run_config_column_truth_needs_no_match_metadata() -> None:
    """A perturbed set scored against the system it was made from carries its
    truth as a column; there is no matched/ metadata to check the target
    against, and the pairing is checked from the rows when they are read."""
    config = _config(
        source=_descriptor(
            system="perturbed://ie/p1/v1/1", has_ground_truth=True, layer="cleansed"
        ),
        target=_descriptor(system="ie"),
        strategy=_strategy(),
        truth=ColumnTruth("source_uri"),
    )

    validate_blocking_run_config(config)


def test_blocking_run_config_truth_defaults_to_the_matched_layer_rule() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(),
    )

    assert config.truth == MatchedLayerTruth()


def test_validate_blocking_run_config_rejects_unknown_name_source() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(name_source="acronym"),
    )

    with pytest.raises(ValueError, match="strategy.name_source"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_short_name_source() -> None:
    """The source-only column choice belongs to the raw-name transform: every
    other transform scores its own column on both sides."""
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(name_source="short_name", name_transform="identity"),
    )

    validate_blocking_run_config(config)


def test_blocking_strategy_config_name_source_defaults_to_name() -> None:
    assert _strategy().name_source == "name"


def test_blocking_strategy_config_name_transform_defaults_to_the_cleansed_name() -> (
    None
):
    """The scan compares the text the cleanse profile shapes, the same text
    the exact-name joins compare."""
    assert _strategy().name_transform == "cleanse"
    assert _strategy().cleanse_profile == "default"


def test_validate_blocking_run_config_rejects_unknown_name_transform() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(name_transform="uppercase"),
    )

    with pytest.raises(ValueError, match="strategy.name_transform"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_rejects_name_source_with_a_transform() -> None:
    """A transform scores its own column on both sides, so the source-only
    column choice has nothing to select and the two cannot combine."""
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(name_transform="cleanse", name_source="short_name"),
    )

    with pytest.raises(ValueError, match="name_source"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_each_registered_transform() -> None:
    for name in ("identity", "cleanse", "acronym", "short_name"):
        validate_blocking_run_config(
            _config(
                source=_descriptor(system="gleif"),
                target=_descriptor(system="gb"),
                strategy=_strategy(name_transform=name),
            )
        )


def test_run_location_separates_runs_differing_only_in_name_transform() -> None:
    """What both sides' names passed through changes what was compared, so
    the transform and its cleanse profile both enter the identity."""
    assert _run_dir() != _run_dir(strategy={"name_transform": "identity"})
    assert _run_dir(strategy={"name_transform": "cleanse"}) != _run_dir(
        strategy={
            "name_transform": "cleanse",
            "cleanse_profile": "default|geographic_terms",
        }
    )


def test_blocking_strategy_config_sbert_model_name_defaults_to_none() -> None:
    assert _strategy().sbert_model_name is None


def test_validate_blocking_run_config_rejects_blank_sbert_model_name() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(representation="sbert", sbert_model_name="   "),
    )

    with pytest.raises(ValueError, match="strategy.sbert_model_name"):
        validate_blocking_run_config(config)


def test_validate_blocking_run_config_accepts_explicit_sbert_model_name() -> None:
    config = _config(
        source=_descriptor(system="gleif"),
        target=_descriptor(system="gb"),
        strategy=_strategy(
            representation="sbert", sbert_model_name="fr-sentence-camembert-base"
        ),
    )

    validate_blocking_run_config(config)


def test_run_location_separates_runs_differing_only_in_countries() -> None:
    """The countries scored are part of the identity. The nickname
    `compare-<country>` existed to carry exactly this distinction because the
    old key omitted it."""
    assert _run_dir(countries=("gb",)) != _run_dir(countries=("ie",))


def test_run_location_separates_same_settings_over_a_different_population() -> None:
    """Two runs with the same settings over different rows are not the same
    run, whichever of the keys hashed from what was read differs; keying on
    settings alone let the second overwrite the first."""
    assert _run_dir() != _run_dir(read_keys={"source_population": "cccc"})
    assert _run_dir() != _run_dir(read_keys={"target_population": "cccc"})
    assert _run_dir() != _run_dir(read_keys={"truth": "cccc"})
    assert _run_dir() != _run_dir(read_keys={"index": "cccc"})


def test_run_location_separates_runs_differing_only_in_a_strategy_field() -> None:
    """The identity is derived from the whole strategy dataclass, so a field
    added to it extends the identity with no edit here."""
    assert _run_dir() != _run_dir(strategy={"top_k": 50})


def test_run_location_separates_runs_differing_only_in_truth_rule() -> None:
    """How ground truth is read is part of what a run measured, so the rule
    and the column it reads both enter the identity."""
    assert _run_dir() != _run_dir(truth=ColumnTruth("source_uri"))
    assert _run_dir(truth=ColumnTruth("source_uri")) != _run_dir(
        truth=ColumnTruth("match_uri")
    )
