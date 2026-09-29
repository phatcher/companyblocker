"""Default-fail gate on checkpoints with no declared pooling configuration.

`sentence_transformers.SentenceTransformer("Jean-Baptiste/camembert-ner")`
does not fail. When a checkpoint carries no sentence-transformers
configuration, the library wraps the bare transformer in mean pooling and
returns a working-looking encoder -- but the underlying weights were trained
for token classification or masked-language modelling, never for the pooled
cosine similarity `SbertClusteringStrategy` scores on. The run completes, the
similarities are plausible numbers, and nothing anywhere says the
representation is meaningless. That silent-success shape is why this is a
default-fail gate with an explicit override rather than a warning, the same
shape `dense_vocabulary_gate.py` uses for the dense-vocabulary/exhaustive-backend
mismatch.

What is actually checked is whether the checkpoint *declares* a
sentence-embedding configuration -- the marker files a published
sentence-transformers model ships (`_POOLING_MARKER_FILES`). That is a
provenance check, not a quality check: it separates "the author published
this as a sentence encoder" from "this is a NER/MLM checkpoint being
mean-pooled by accident". Mean pooling itself has no learned parameters, so
there is nothing else to inspect at load time; the trained-ness lives in the
encoder weights, and the marker files are the only durable evidence of what
those weights were trained to produce.

Verified against real checkpoints (2026-08-31): every checkpoint in
`resources/sbert_models.json` carries at least one marker, and
`Jean-Baptiste/camembert-ner`, `fhswf/bert_de_ner`, `bert-base-uncased`,
`camembert-base` and `dbmdz/bert-base-german-cased` carry none.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .sbert_model_registry import is_registered_checkpoint

# Files a published sentence-transformers checkpoint ships to declare its
# pooling configuration. `modules.json` plus `1_Pooling/config.json` is the
# current export layout; `sentence_bert_config.json` alone is the older
# layout, still used by checkpoints published years ago (for example
# `dangvantuan/sentence-camembert-large`), so requiring only the modern pair
# would reject genuine sentence encoders.
_POOLING_MARKER_FILES = (
    "modules.json",
    "1_Pooling/config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
)

PoolingStatus = Literal["declared", "absent", "unknown"]

PoolingStatusProbe = Callable[[str], PoolingStatus]


class UntrainedPoolingError(ValueError):
    """A checkpoint with no declared sentence-embedding configuration was requested.

    Raised when no explicit override was supplied.
    """


def _local_checkpoint_status(directory: Path) -> PoolingStatus:
    for marker in _POOLING_MARKER_FILES:
        if (directory / marker).exists():
            return "declared"
    return "absent"


def _cached_hub_status(model_name: str) -> PoolingStatus:
    """Answer from the local Hugging Face cache alone, without network use.

    Returns `"unknown"` when the cache has nothing to say, so an already
    downloaded checkpoint keeps working offline instead of being rejected for
    an unreachable hub.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return "unknown"

    known_absent = 0
    for marker in _POOLING_MARKER_FILES:
        try:
            cached = try_to_load_from_cache(model_name, marker)
        except Exception:  # noqa: BLE001 - malformed repo id, unreadable cache
            return "unknown"
        if isinstance(cached, str):
            return "declared"
        if cached is not None:
            # The `_CACHED_NO_EXIST` sentinel: the hub was asked before and
            # said this file does not exist.
            known_absent += 1
    if known_absent == len(_POOLING_MARKER_FILES):
        return "absent"
    return "unknown"


def _remote_hub_status(model_name: str) -> PoolingStatus:
    try:
        from huggingface_hub import file_exists
    except ImportError:
        return "unknown"

    seen_any_answer = False
    for marker in _POOLING_MARKER_FILES:
        try:
            if file_exists(model_name, marker):
                return "declared"
            seen_any_answer = True
        except Exception:  # noqa: BLE001 - offline, rate-limited, gated, bad id
            return "unknown"
    return "absent" if seen_any_answer else "unknown"


def checkpoint_pooling_status(model_name: str) -> PoolingStatus:
    """Whether `model_name` declares a sentence-embedding configuration.

    Args:
        model_name: A local checkpoint directory or a hub checkpoint
            identifier.

    Returns:
        `"declared"` when a marker file is present, `"absent"` when the
        checkpoint is readable and has none, and `"unknown"` when it could not
        be determined (no `huggingface_hub` installed, offline, an
        unresolvable identifier). `"unknown"` is treated as failing by
        `ensure_sentence_embedding_checkpoint()`: an unverifiable checkpoint
        is exactly the case the gate exists for.
    """
    candidate = str(model_name).strip()
    local = Path(candidate)
    if local.is_dir():
        return _local_checkpoint_status(local)

    cached = _cached_hub_status(candidate)
    if cached != "unknown":
        return cached
    return _remote_hub_status(candidate)


def ensure_sentence_embedding_checkpoint(
    model_name: str,
    *,
    force: bool = False,
    status_probe: PoolingStatusProbe = checkpoint_pooling_status,
) -> None:
    """Reject a checkpoint that is not a declared sentence-embedding model.

    Args:
        model_name: The checkpoint about to be loaded.
        force: Bypass the gate. For a deliberate experiment with a checkpoint
            whose provenance cannot be established here -- accepting that
            `sentence_transformers` may be mean-pooling weights never trained
            for pooled cosine similarity.
        status_probe: Seam for tests and for callers with their own way of
            answering the question; defaults to
            `checkpoint_pooling_status()`.

    Raises:
        UntrainedPoolingError: The checkpoint declares no sentence-embedding
            configuration, or none could be found, and `force` is not set.
    """
    if force:
        return

    candidate = str(model_name).strip()
    # Registry entries were checked when they were added; re-probing them
    # would put a network round trip in front of every default run.
    if is_registered_checkpoint(candidate):
        return

    status = status_probe(candidate)
    if status == "declared":
        return

    detail = (
        "declares no sentence-embedding configuration"
        if status == "absent"
        else "could not be verified as a sentence-embedding model"
    )
    known = ", ".join(_POOLING_MARKER_FILES)
    raise UntrainedPoolingError(
        f"sbert checkpoint '{candidate}' {detail} (looked for: {known}). "
        "sentence_transformers would load it by wrapping the bare transformer "
        "in untrained mean pooling, producing embeddings never trained for the "
        "cosine similarity this strategy scores on -- a silent failure that "
        "still returns plausible numbers. Select a registered model by slug "
        "(company_vectorize.sbert_model_registry.list_sbert_models()), or set "
        "force_untrained_pooling=True on SbertTargetIndexBuildSettings to load "
        "it anyway."
    )
