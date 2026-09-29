from __future__ import annotations

PROCESS_STAGE_ACQUIRE = "acquire"
PROCESS_STAGE_SHARD = "shard"
PROCESS_STAGE_CANONICAL = "canonical"
PROCESS_STAGE_CLEANSE = "cleanse"
PROCESS_STAGE_MATCH = "match"
PROCESS_STAGE_TOKENIZE = "tokenize"

PIPELINE_INPUT_LAYER_CANONICAL = "canonical"
PIPELINE_INPUT_LAYER_CLEANSED = "cleansed"

PIPELINE_STAGE_NAMES = (
    PROCESS_STAGE_ACQUIRE,
    PROCESS_STAGE_SHARD,
    PROCESS_STAGE_CANONICAL,
    PROCESS_STAGE_CLEANSE,
    PROCESS_STAGE_MATCH,
    PROCESS_STAGE_TOKENIZE,
)
DEFAULT_PROCESS_STAGES = PIPELINE_STAGE_NAMES[1:]
OPTIONAL_PROCESS_STAGES = PIPELINE_STAGE_NAMES[:1]
ALLOWED_PROCESS_STAGES = frozenset(PIPELINE_STAGE_NAMES)

PIPELINE_STAGE_INPUT_LAYERS: dict[str, tuple[str, ...]] = {
    PROCESS_STAGE_CLEANSE: (PIPELINE_INPUT_LAYER_CANONICAL,),
    PROCESS_STAGE_TOKENIZE: (PIPELINE_INPUT_LAYER_CLEANSED,),
}


def stage_input_layer(stage: str) -> str | None:
    """Return the expected input layer for a pipeline stage when stage consumes prior output.

    Returns None for stages that do not have a fixed layer-to-layer handoff contract.
    """
    return next(iter(stage_input_layers(stage)), None)


def stage_input_layers(stage: str) -> tuple[str, ...]:
    """Return ordered preferred input layers for a stage."""

    return PIPELINE_STAGE_INPUT_LAYERS.get(stage, ())
