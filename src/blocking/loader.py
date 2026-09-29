"""Find the layer a run reads for each side, and load it a country at a time.

`load_dataset_descriptor()` resolves a system's layer and the countries it holds. The
source side reads the layer its truth rule names, `matched/` by default; the target
side reads the latest dated `canonical/` snapshot, since a target is never matched. A
perturbed dataset, named by its `perturbed://` reference, is read from the one layer
its materializer writes. A run derives its own name forms from the raw `name`, so no
carried name form is read. `load_name_variant_frame()` reads a system's recorded name
rows for the name-variant pairings. `load_run_config()` is reserved for reading a run
configuration from a file; no format is defined, so it raises `NotImplementedError`.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from validation.prepared_dataset import read_country_partition_frame
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    data_root,
    latest_canonical_snapshot_dir,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.layer_layout import (
    partition_values,
    resolve_name_files,
    resolve_partition_dir,
)
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import is_perturbed_source, perturbed_parts

from .contracts import BlockingDatasetDescriptor, BlockingRunConfig
from .truth import MatchedLayerTruth, SourceTruthResolver

_MATCHED_LAYER = MATCHED_LAYER_NAME
_CANONICAL_LAYER = CANONICAL_LAYER_NAME

_PERTURBED_DATASET_LAYER = CLEANSED_LAYER_NAME
"""The one layer a materialized perturbed dataset holds, the shape
`validation.perturbation_materializer` writes. It is that dataset's own layout,
not a system's cleansed layer, which no run reads."""


def _resolve_layer_dir(system_root: Path, layer: str) -> Path | None:
    """Where one layer's rows are for a system, or None when absent.

    Where a layer sits beneath the system, and which `canonical` snapshot is
    current, are `workspace.data_layout`'s call.
    """
    layer_dir = layer_directory(system_root, layer)
    if layer == _CANONICAL_LAYER:
        snapshot = latest_canonical_snapshot_dir(layer_dir)
        layer_dir = snapshot if snapshot is not None else layer_dir
    return layer_dir if _has_any_parquet_files(layer_dir) else None


def _system_root_of(descriptor: BlockingDatasetDescriptor) -> Path:
    """`<base_dir>/<system>` for a resolved descriptor: every layer's rows sit
    in a snapshot directory two levels beneath it, a date or `current`."""
    return descriptor.system_dir.parent.parent


def _has_any_parquet_files(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    if any(directory.glob("*.parquet")):
        return True
    return any(directory.glob("**/*.parquet"))


def _discover_available_countries(system_dir: Path) -> tuple[str, ...]:
    return tuple(
        country
        for country in partition_values(system_dir)
        if _has_any_parquet_files(
            resolve_partition_dir(system_dir, value=country) or system_dir
        )
    )


def _load_matched_target_systems(system_dir: Path, *, system: str) -> tuple[str, ...]:
    metadata_path = system_dir / "_match_metadata.json"
    if not metadata_path.exists():
        raise ValueError(
            f"matched/ for system '{system}' has parquet files but no "
            f"_match_metadata.json at {metadata_path} — cannot determine which "
            "target systems its match_uri values are valid against."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_systems = payload.get("target_systems") or []
    return tuple(str(target) for target in target_systems)


def load_dataset_descriptor(
    *,
    roots: WorkspaceRoots,
    system: str,
    prepared_base_dir: Path | None = None,
    require_ground_truth: bool = True,
    truth: SourceTruthResolver | None = None,
) -> BlockingDatasetDescriptor:
    """Resolve the layer a run reads `system` from and discover its available countries.

    `truth` is the rule the run will read ground truth by (`truth.py`), and
    it decides which layers may resolve and whether a resolved layer carries
    truth. Under the default `MatchedLayerTruth`, `require_ground_truth`
    (default `True`) accepts only the `matched` layer — callers that need
    `match_uri` truth should keep the default and let resolution fail fast
    when no match-stage output exists. `require_ground_truth=False`, which
    is how a target is loaded, resolves the system's latest dated
    `canonical` snapshot, since a target is never matched. No system's
    cleansed layer is read: the run derives every name form from the raw
    `name`, which canonical carries, so a carried form could only be one
    picked up by mistake.
    Under a `ColumnTruth`, either layer resolves and carries truth when its
    rows carry the named column, and `require_ground_truth` refuses a layer
    that does not. The returned descriptor's `has_ground_truth` field
    records what actually resolved either way, so callers never have to
    guess.

    `system` may be a perturbed dataset's URI, `perturbed://<system>/<profile>/<version>/<seed>`
    (`workspace.run_inputs.perturbed_source`), naming one materialized
    dataset. It lives under its own directory rather than `<base>/<system>/`,
    as one `cleansed/` layer whose rows carry their pre-perturbation
    identity as `source_uri`, so it is scored under `ColumnTruth("source_uri")`.
    """

    resolver: SourceTruthResolver = truth if truth is not None else MatchedLayerTruth()

    if is_perturbed_source(system):
        source_system, profile_id, version, seed = perturbed_parts(system)
        system_root = perturbed_dataset_dir(
            roots=roots,
            source_system=source_system,
            profile_id=profile_id,
            version=version,
            seed=seed,
        )
    else:
        base_dir = (
            Path(prepared_base_dir).resolve()
            if prepared_base_dir is not None
            else data_root(roots).resolve()
        )
        system_root = base_dir / system
    perturbed_hint = (
        " A materialized perturbed dataset's rows carry their pre-perturbation "
        "identity as source_uri: score it under ColumnTruth('source_uri') "
        "(--match-col source_uri on the scripts)."
        if is_perturbed_source(system)
        else ""
    )
    layers = (
        (_PERTURBED_DATASET_LAYER,)
        if is_perturbed_source(system)
        else resolver.layers(require_ground_truth=require_ground_truth)
    )

    for layer in layers:
        system_dir = _resolve_layer_dir(system_root, layer)
        if system_dir is not None:
            has_ground_truth = resolver.layer_has_ground_truth(
                layer=layer, layer_dir=system_dir
            )
            if require_ground_truth and not has_ground_truth:
                raise ValueError(
                    f"{system_dir} resolved for system '{system}' but carries no "
                    f"{resolver.truth_column!r} ground truth column under "
                    f"{resolver!r}; load it with require_ground_truth=False, or "
                    f"name the column its rows carry their target identity in.{perturbed_hint}"
                )
            matched_target_systems = (
                _load_matched_target_systems(system_dir, system=system)
                if layer == _MATCHED_LAYER and resolver.kind == MatchedLayerTruth().kind
                else ()
            )
            return BlockingDatasetDescriptor(
                system=system,
                system_dir=system_dir,
                layer=layer,
                has_ground_truth=has_ground_truth,
                matched_target_systems=matched_target_systems,
                available_countries=_discover_available_countries(system_dir),
            )

    tried = ", ".join(layers)
    raise FileNotFoundError(
        f"No prepared parquet files found for system '{system}' under {system_root}. "
        f"Tried layer(s): {tried}.{perturbed_hint}"
    )


def load_country_frame(
    descriptor: BlockingDatasetDescriptor, *, country: str
) -> pl.DataFrame:
    """Load all parquet rows for one country partition of a resolved dataset layer."""

    partition_dir = resolve_partition_dir(descriptor.system_dir, value=country)
    if partition_dir is None:
        raise FileNotFoundError(
            f"No '{country}' partition under {descriptor.system_dir} for system "
            f"'{descriptor.system}'."
        )
    return read_country_partition_frame(partition_dir)


def load_name_variant_frame(
    descriptor: BlockingDatasetDescriptor,
) -> pl.DataFrame | None:
    """Load the `<system>-names-*.parquet` sidecar for `descriptor.system`, if any.

    Looks under the current snapshot of `<base_dir>/<system>/canonical/`,
    which `workspace.data_layout.latest_canonical_snapshot_dir` picks; the
    `<base_dir>/<system>` root is recovered from `descriptor.system_dir`
    rather than taking a separate argument.

    Returns `None` when there is no `canonical/` directory, no dated
    subdirectory under it, or no `<system>-names-*.parquet` files in the
    latest one -- the expected case for a system this sidecar has not been
    derived for, which is only some of them (`acquisition.canonical`'s
    module docstring lists which). Callers should treat `None`
    as "no variant expansion available," not an error.
    """

    snapshot = latest_canonical_snapshot_dir(
        _system_root_of(descriptor) / _CANONICAL_LAYER
    )
    if snapshot is None:
        return None

    files = resolve_name_files(snapshot, system_code=descriptor.system)
    if not files:
        return None

    frames = [pl.read_parquet(path) for path in files]
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def load_run_config(config_path: Path) -> BlockingRunConfig:
    """Load a blocking run configuration from disk.

    Not implemented yet: no config file format is defined, so this always
    raises `NotImplementedError`. `scripts/run_blocking.py` and
    `scripts/compare_blocking_strategies.py` build their `BlockingRunConfig`
    from CLI flags directly instead, so nothing reads a config file today.
    """

    raise NotImplementedError("blocking config loading is not implemented yet")
