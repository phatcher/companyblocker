from acquisition.constants_pipeline import (
    PIPELINE_INPUT_LAYER_CANONICAL,
    PIPELINE_INPUT_LAYER_CLEANSED,
    PIPELINE_STAGE_INPUT_LAYERS,
    PROCESS_STAGE_CLEANSE,
    PROCESS_STAGE_TOKENIZE,
    stage_input_layer,
    stage_input_layers,
)


def test_stage_input_layer_contract_for_cleanse_and_tokenize() -> None:
    assert stage_input_layer(PROCESS_STAGE_CLEANSE) == PIPELINE_INPUT_LAYER_CANONICAL
    assert stage_input_layer(PROCESS_STAGE_TOKENIZE) == PIPELINE_INPUT_LAYER_CLEANSED


def test_stage_input_layer_returns_none_for_stages_without_layer_contract() -> None:
    assert stage_input_layer("shard") is None


def test_stage_input_layer_mapping_contains_expected_entries() -> None:
    assert PIPELINE_STAGE_INPUT_LAYERS == {
        PROCESS_STAGE_CLEANSE: (PIPELINE_INPUT_LAYER_CANONICAL,),
        PROCESS_STAGE_TOKENIZE: (PIPELINE_INPUT_LAYER_CLEANSED,),
    }


def test_stage_input_layers_returns_ordered_tokenize_preferences() -> None:
    assert stage_input_layers(PROCESS_STAGE_TOKENIZE) == (
        PIPELINE_INPUT_LAYER_CLEANSED,
    )
