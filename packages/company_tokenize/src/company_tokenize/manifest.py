"""The run manifest a promoted tokenizer carries, and the checks a dual-tokenizer pair must pass.

The manifest, `metadata.json` in the tokenizer's folder, records what one training run
took and produced. It is distinct from an optimize run's summary, the evidence for
its choice, and from the archive index, the history across runs. `build_run_manifest()`
validates what it builds, and `validate_run_manifest()` rejects a missing mandatory
field, an unrecognised field or an unknown mode, so a field added to one mode's
manifest and not the other fails rather than drifting. `corpus_content_hash` records
the corpus's bytes, since `corpus_path` is routinely reused across a resample.

A dual tokenization feeds one name column through a country-scope tokenizer and a
global-scope fallback. `validate_dual_tokenizer_artifacts()` checks the pair when both
carry a manifest, and skips silently when either does not, since a fixture or an
older tokenizer carries none. `validate_dual_tokenizer_coherence()` requires the
scopes to be `country` and `global`, each manifest's trainer to match what it is
loaded as, and both to agree on `name_col`. The two trainers need not match each
other, and systems, vocabulary and corpus are expected to differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import tokenizer_directory_files

# The run manifest is the `metadata[.{trainer}].json` artifact written by
# both `--mode train` (`finalize_train_target`) and `--mode optimize
# --finalize` (`finalize_optimize_target`) in `src/training/optimize_execution.py`
# immediately after a tokenizer is promoted to operational. Both call sites
# used to build this dict independently and by hand; this module is the one
# place that now defines what shape that dict must have, so a future field
# added to one mode's manifest without the other doesn't silently drift.
#
# Fields present for every run, regardless of mode.
RUN_MANIFEST_MANDATORY_FIELDS: tuple[str, ...] = (
    "created_utc",
    "mode",
    "scope",
    "systems",
    "profile",
    "trainer",
    "all_rows",
    "vocab_size",
    "min_frequency",
    "name_col",
    "preprocess_profile",
    "corpus_path",
    "corpus_content_hash",
    "tokenizer_path",
    "line_count",
    "trainer_params",
    "token_score_artifact",
)

# Fields that only some modes populate. `run_log_path`/`summary_path` only
# exist for `--mode optimize`, which is the only mode with a candidate sweep
# and a resulting run log/summary to point back at -- plain `--mode train`
# has no sweep, so it never sets these. `optimize_preset`/
# `optimize_preset_overrides` are populated by both modes' normal
# CLI invocation path (`apply_optimize_preset` runs before either mode's
# finalize step and stashes its result onto `args`), but stay optional here
# too: a caller that builds a manifest directly, bypassing that CLI path
# (a hand-built test fixture, an older/partial `args` namespace), never
# resolved a preset at all.
RUN_MANIFEST_OPTIONAL_FIELDS: tuple[str, ...] = (
    "run_log_path",
    "summary_path",
    "optimize_preset",
    "optimize_preset_overrides",
)

RUN_MANIFEST_MODES: tuple[str, ...] = ("train", "optimize")

_RUN_MANIFEST_INT_FIELDS: tuple[str, ...] = (
    "vocab_size",
    "min_frequency",
    "line_count",
)
_RUN_MANIFEST_STR_FIELDS: tuple[str, ...] = (
    "created_utc",
    "scope",
    "trainer",
    "name_col",
    "preprocess_profile",
    "corpus_path",
    "corpus_content_hash",
    "tokenizer_path",
)
_RUN_MANIFEST_DICT_FIELDS: tuple[str, ...] = ("trainer_params", "token_score_artifact")
# Optional fields split by type: `run_log_path`/`summary_path`/
# `optimize_preset` are strings, `optimize_preset_overrides` is a dict, so
# they can't share one "every optional field is a str" loop.
_RUN_MANIFEST_OPTIONAL_STR_FIELDS: tuple[str, ...] = (
    "run_log_path",
    "summary_path",
    "optimize_preset",
)
_RUN_MANIFEST_OPTIONAL_DICT_FIELDS: tuple[str, ...] = ("optimize_preset_overrides",)


def build_run_manifest(
    *,
    created_utc: str,
    mode: str,
    scope: str,
    systems: list[str],
    profile: str | None,
    trainer: str,
    all_rows: bool,
    vocab_size: int,
    min_frequency: int,
    name_col: str,
    preprocess_profile: str,
    corpus_path: str,
    corpus_content_hash: str,
    tokenizer_path: str,
    line_count: int,
    trainer_params: dict[str, Any],
    token_score_artifact: dict[str, Any],
    run_log_path: str | None = None,
    summary_path: str | None = None,
    optimize_preset: str | None = None,
    optimize_preset_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one run manifest payload shared by train and optimize modes.

    Mandatory fields (`RUN_MANIFEST_MANDATORY_FIELDS`) are always present.
    Every field in `RUN_MANIFEST_OPTIONAL_FIELDS` (`run_log_path`/
    `summary_path`/`optimize_preset`/`optimize_preset_overrides`) is only
    included when given, omitted rather than written as `null` -- `--mode
    train` has no sweep run log or summary to point at, and any caller that
    never resolved an optimize preset has no name or override set to record.

    `name_col` and `preprocess_profile` say which column of the cleansed
    layer the corpus was built from and what it passed through before
    training, as information nothing refuses on.

    `corpus_content_hash` is the caller's job to compute (see
    `compute_corpus_content_hash`, already used by the optimize canary/sweep
    layout for the same reason): `corpus_path` alone identifies where a
    corpus lives, not what was actually in it at train time, and that path is
    routinely reused across a resample. The hash is what lets a manifest be
    checked against a corpus file later and answer "is this still the same
    corpus this tokenizer was trained on" without relying on mtime.

    Validates the built payload against `validate_run_manifest` before
    returning it, so a caller can never write a manifest that fails its own
    schema.
    """
    manifest: dict[str, Any] = {
        "created_utc": created_utc,
        "mode": mode,
        "scope": scope,
        "systems": systems,
        "profile": profile,
        "trainer": trainer,
        "all_rows": all_rows,
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "name_col": name_col,
        "preprocess_profile": preprocess_profile,
        "corpus_path": corpus_path,
        "corpus_content_hash": corpus_content_hash,
        "tokenizer_path": tokenizer_path,
        "line_count": line_count,
        "trainer_params": trainer_params,
        "token_score_artifact": token_score_artifact,
    }
    if run_log_path is not None:
        manifest["run_log_path"] = run_log_path
    if summary_path is not None:
        manifest["summary_path"] = summary_path
    if optimize_preset is not None:
        manifest["optimize_preset"] = optimize_preset
    if optimize_preset_overrides is not None:
        manifest["optimize_preset_overrides"] = optimize_preset_overrides

    validate_run_manifest(manifest)
    return manifest


def validate_run_manifest(payload: dict[str, Any]) -> None:
    """Validate a run-manifest payload against the shared train/optimize schema.

    Raises `ValueError` on the first problem found, naming the offending
    field, so a schema drift in either `finalize_train_target` or
    `finalize_optimize_target` fails loudly at write time instead of
    silently shipping a malformed `metadata.json`.

    Checks, in order: payload is a dict; every mandatory field is present;
    no field outside the mandatory/optional set is present (an unreviewed
    field added by only one mode is exactly the drift this schema exists to
    catch); `mode` is one of `RUN_MANIFEST_MODES`; and each field with a
    fixed expected type (`systems` a list of strings, the `int` fields, the
    `dict` fields, `all_rows` a bool) actually has it.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Run manifest must be a dict, got {type(payload).__name__}.")

    missing = [field for field in RUN_MANIFEST_MANDATORY_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            "Run manifest missing mandatory field(s): " + ", ".join(missing)
        )

    allowed = set(RUN_MANIFEST_MANDATORY_FIELDS) | set(RUN_MANIFEST_OPTIONAL_FIELDS)
    unknown = [key for key in payload if key not in allowed]
    if unknown:
        raise ValueError(
            "Run manifest has unrecognized field(s): " + ", ".join(sorted(unknown))
        )

    mode = payload["mode"]
    if mode not in RUN_MANIFEST_MODES:
        raise ValueError(
            f"Run manifest 'mode' must be one of {RUN_MANIFEST_MODES}, got {mode!r}."
        )

    systems = payload["systems"]
    if not isinstance(systems, list) or not all(
        isinstance(item, str) for item in systems
    ):
        raise TypeError("Run manifest 'systems' must be a list of strings.")

    for field in _RUN_MANIFEST_STR_FIELDS:
        if not isinstance(payload[field], str):
            raise TypeError(f"Run manifest '{field}' must be a str.")

    for field in _RUN_MANIFEST_INT_FIELDS:
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"Run manifest '{field}' must be an int.")

    for field in _RUN_MANIFEST_DICT_FIELDS:
        if not isinstance(payload[field], dict):
            raise TypeError(f"Run manifest '{field}' must be a dict.")

    if not isinstance(payload["all_rows"], bool):
        raise TypeError("Run manifest 'all_rows' must be a bool.")

    if payload["profile"] is not None and not isinstance(payload["profile"], str):
        raise TypeError("Run manifest 'profile' must be a str or None.")

    for field in _RUN_MANIFEST_OPTIONAL_STR_FIELDS:
        if field in payload and not isinstance(payload[field], str):
            raise TypeError(f"Run manifest '{field}' must be a str when present.")

    for field in _RUN_MANIFEST_OPTIONAL_DICT_FIELDS:
        if field in payload and not isinstance(payload[field], dict):
            raise TypeError(f"Run manifest '{field}' must be a dict when present.")


# The dual-tokenizer contract below governs `tokenize_name_dual()` /
# `tokenize_name_dataframe_dual()` (see `ops.py`/`tokenization.py`): one
# name_col fed through a "country" (per-system) tokenizer and a "global"
# (pooled) tokenizer in the same pass, written to two separate token
# columns. Each side's tokenizer has its own independently-promoted run
# manifest (`RUN_MANIFEST_MANDATORY_FIELDS`); this contract is about what
# must be true of the *pair* for the dual output to mean what a downstream
# consumer (a blocking-key builder comparing a local-semantic country
# signal against a global-recall fallback signal) expects it to mean, not
# about either manifest alone.
DUAL_TOKENIZER_ROLES: tuple[str, str] = ("country", "global")


def validate_dual_tokenizer_coherence(
    *,
    country_manifest: dict[str, Any],
    global_manifest: dict[str, Any],
    country_trainer: str,
    global_trainer: str,
) -> None:
    """Validate a country/global run-manifest pair for one dual-tokenizer run.

    Both manifests already pass `validate_run_manifest` on their own (this
    function re-checks that, since a caller may have loaded either one from
    disk without validating it first). What is specific to the *pair*:

    - `country_manifest["scope"]` must be ``"country"`` and
      `global_manifest["scope"]` must be ``"global"`` -- catches a path mix-up
      (for example `country_tokenizer_path` accidentally pointed at the
      pooled artifact).
    - Each manifest's own `trainer` field must match the trainer the caller
      is actually loading it as (`country_trainer`/`global_trainer`) -- the
      two trainers need not match *each other* (a wordpiece country
      tokenizer paired with a sentencepiece global one is a supported
      combination), only their own declared identity.
    - Both manifests' `name_col` must agree. `tokenize_name_dual()` /
      `tokenize_name_dataframe_dual()` read one `name_col` and feed it
      through both tokenizers in the same pass; if the two artifacts were
      fit on different name representations (for example cleansed vs. raw),
      the two token columns stop being comparable outputs of "the same
      name," which is the premise a downstream consumer combining them
      relies on.

    Deliberately *not* checked, because these are expected to differ and a
    mismatch is not a defect: `systems` (one system vs. many), `vocab_size`/
    `min_frequency`/`trainer_params` (independently tuned per scope),
    `corpus_path`/`corpus_content_hash` (different corpora entirely).

    Raises `ValueError` naming the first incoherence found.
    """
    validate_run_manifest(country_manifest)
    validate_run_manifest(global_manifest)

    if country_manifest["scope"] != "country":
        raise ValueError(
            "Dual-tokenizer country manifest has scope="
            f"{country_manifest['scope']!r}, expected 'country'. Check that "
            "country_tokenizer_path points at a per-system tokenizer, not "
            "the global one."
        )
    if global_manifest["scope"] != "global":
        raise ValueError(
            "Dual-tokenizer global manifest has scope="
            f"{global_manifest['scope']!r}, expected 'global'. Check that "
            "global_tokenizer_path points at the pooled tokenizer, not a "
            "per-system one."
        )

    if country_manifest["trainer"] != country_trainer:
        raise ValueError(
            "Dual-tokenizer country manifest trainer="
            f"{country_manifest['trainer']!r} does not match the requested "
            f"country_trainer={country_trainer!r}."
        )
    if global_manifest["trainer"] != global_trainer:
        raise ValueError(
            "Dual-tokenizer global manifest trainer="
            f"{global_manifest['trainer']!r} does not match the requested "
            f"global_trainer={global_trainer!r}."
        )

    if country_manifest["name_col"] != global_manifest["name_col"]:
        raise ValueError(
            "Dual-tokenizer country/global manifests disagree on name_col "
            f"({country_manifest['name_col']!r} vs "
            f"{global_manifest['name_col']!r}); both tokenizers must be "
            "trained on the same name representation, since a dual run "
            "feeds one name_col through both."
        )


def load_run_manifest_if_present(
    *, tokenizer_path: Path, trainer: str
) -> dict[str, Any] | None:
    """Load the trainer-scoped run manifest sibling of `tokenizer_path`.

    Resolves the tokenizer directory's `metadata.json` via
    `tokenizer_directory_files` and returns its parsed contents, or
    `None` when that sidecar does not exist -- not every tokenizer artifact
    has one (a hand-built test fixture, or a real artifact promoted before
    this manifest schema existed), so absence is a normal case a caller
    decides how to handle, not this function's decision to raise on.
    """
    metadata_path = tokenizer_directory_files(
        tokenizer_path.parent, trainer=trainer
    ).metadata
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def validate_dual_tokenizer_artifacts(
    *,
    country_tokenizer_path: Path,
    global_tokenizer_path: Path,
    country_trainer: str,
    global_trainer: str,
) -> None:
    """Best-effort coherence check for one dual-tokenizer artifact pair.

    Loads each side's run manifest via `load_run_manifest_if_present` and, if
    both are present, validates them with `validate_dual_tokenizer_coherence`.
    Silently skips validation when either sidecar is missing, so this is a
    defense against a manifest pair that looks coherent but isn't, not a hard
    requirement that a manifest exists in the first place -- `ops.py`'s
    `tokenize_name_dual` and `tokenization.py`'s `tokenize_name_dataframe_dual`
    both call this before doing any tokenization work.
    """
    country_manifest = load_run_manifest_if_present(
        tokenizer_path=country_tokenizer_path, trainer=country_trainer
    )
    global_manifest = load_run_manifest_if_present(
        tokenizer_path=global_tokenizer_path, trainer=global_trainer
    )
    if country_manifest is None or global_manifest is None:
        return

    validate_dual_tokenizer_coherence(
        country_manifest=country_manifest,
        global_manifest=global_manifest,
        country_trainer=country_trainer,
        global_trainer=global_trainer,
    )
