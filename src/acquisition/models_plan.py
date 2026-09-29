"""Typed plans for a system and its resources, built from the catalog and validated on construction.

A resource's metadata carries its runtime defaults: `read_defaults` per file extension (reader mode, chunk sizing, parallel decompression), `projection_defaults` (for Wikidata, which extraction engine runs), and `output_defaults.chunk_size`, where `null` writes one output file per stage. Sharding, canonical and the Wikidata projection all read them from here.
"""

from __future__ import annotations

# isort: skip_file
# fmt: off
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path
from typing import Literal, TypedDict

from .constants import (ALLOWED_ACCESS_MODES, ALLOWED_PROCESS_STAGES,
                        ALLOWED_RESEARCH_PRIORITIES, DEFAULT_RESEARCH_PRIORITY)
from .constants_status import (ALLOWED_SYSTEM_STATUSES, RUNNABLE_SYSTEM_STATUSES,
                               STATUS_BLOCKED, STATUS_LIVE,
                               STATUS_RESEARCH_REQUIRED, STATUS_STOPPED,
                               STATUS_SUPPORTED)

# fmt: on

_ALLOWED_RESOURCE_READER_MODES = {"line_iter", "chunked", "chunked_indexed"}


class ResourceReadDefaultsByExtensionEntry(TypedDict, total=False):
    reader_mode: str
    chunk_size_mb: int


class ResourceReadDefaultsRuntime(TypedDict, total=False):
    indexed_bzip2_parallelization: Literal["auto"] | int


class ResourceReadDefaults(TypedDict, total=False):
    by_extension: dict[str, ResourceReadDefaultsByExtensionEntry]
    runtime: ResourceReadDefaultsRuntime


class ResolvedReadOptions(TypedDict, total=False):
    reader_mode: str
    chunk_size_bytes: int
    indexed_bzip2_parallelization: Literal["auto"] | int


class ResourceProjectionDefaults(TypedDict, total=False):
    engine: Literal["python", "wikisieve"]
    binary_path: str
    output_mode: Literal["jsonl", "ids"]


class ResourceOutputDefaults(TypedDict, total=False):
    """Shared output-materialization defaults, consulted by every stage that
    splits its output into row-count-bounded files (shard, canonical, and
    cleanse's merged/deduped view -- which chunk_size governs via
    materialize_cleansed_merge's rows_per_file, and partition_by governs
    with a "jurisdiction_code" default there instead of canonical/shard's
    "don't partition" default, matching every real system's existing
    behavior). One section rather than a per-stage flag, since the
    underlying decision ("does this resource want its output split/partitioned
    at all") doesn't vary by stage.
    """

    chunk_size: int | None
    partition_by: str | None


def _validate_resource_read_defaults(
    resource_name: str, resource_read_defaults: ResourceReadDefaults
) -> None:
    by_extension = resource_read_defaults.get("by_extension")
    runtime = resource_read_defaults.get("runtime")

    if by_extension is not None and not isinstance(by_extension, dict):
        raise ValueError(
            f"{resource_name}: read_defaults.by_extension must be an object when provided"
        )
    if runtime is not None and not isinstance(runtime, dict):
        raise ValueError(
            f"{resource_name}: read_defaults.runtime must be an object when provided"
        )

    if isinstance(by_extension, dict):
        _validate_resource_read_defaults_by_extension(resource_name, by_extension)
    if isinstance(runtime, dict):
        _validate_resource_read_defaults_runtime(resource_name, runtime)


def _validate_resource_read_defaults_by_extension(
    resource_name: str,
    by_extension: Mapping[str, object],
) -> None:
    for extension, settings in by_extension.items():
        if not isinstance(extension, str) or not extension.startswith("."):
            raise ValueError(
                f"{resource_name}: read_defaults.by_extension keys must be dot-prefixed extensions"
            )
        if not isinstance(settings, dict):
            raise TypeError(
                f"{resource_name}: read_defaults.by_extension['{extension}'] must be an object"
            )

        reader_mode = settings.get("reader_mode")
        if reader_mode is not None and (
            not isinstance(reader_mode, str)
            or reader_mode not in _ALLOWED_RESOURCE_READER_MODES
        ):
            allowed = ", ".join(sorted(_ALLOWED_RESOURCE_READER_MODES))
            raise ValueError(
                f"{resource_name}: invalid reader_mode '{reader_mode}' for extension '{extension}'. "
                f"Allowed values: {allowed}"
            )

        chunk_size_mb = settings.get("chunk_size_mb")
        if chunk_size_mb is not None and (
            not isinstance(chunk_size_mb, int) or chunk_size_mb <= 0
        ):
            raise ValueError(
                f"{resource_name}: read_defaults.by_extension['{extension}'].chunk_size_mb must be a positive integer"
            )


def _validate_resource_read_defaults_runtime(
    resource_name: str, runtime: Mapping[str, object]
) -> None:
    indexed_parallelization = runtime.get("indexed_bzip2_parallelization")
    if indexed_parallelization is None:
        return
    if indexed_parallelization == "auto":
        return
    if isinstance(indexed_parallelization, int) and indexed_parallelization > 0:
        return
    raise ValueError(
        f"{resource_name}: read_defaults.runtime.indexed_bzip2_parallelization must be 'auto' or a positive integer"
    )


def _validate_resource_projection_defaults(
    resource_name: str,
    projection_defaults: ResourceProjectionDefaults,
) -> None:
    engine = projection_defaults.get("engine")
    if engine is not None and engine not in {"python", "wikisieve"}:
        raise ValueError(
            f"{resource_name}: projection_defaults.engine must be 'python' "
            "or 'wikisieve' when provided"
        )

    binary_path = projection_defaults.get("binary_path")
    if binary_path is not None and (
        not isinstance(binary_path, str) or not binary_path.strip()
    ):
        raise ValueError(
            f"{resource_name}: projection_defaults.binary_path must be a non-empty string when provided"
        )

    output_mode = projection_defaults.get("output_mode")
    if output_mode is not None and output_mode not in {"jsonl", "ids"}:
        raise ValueError(
            f"{resource_name}: projection_defaults.output_mode must be 'jsonl' or 'ids' when provided"
        )


def _validate_resource_output_defaults(
    resource_name: str,
    output_defaults: ResourceOutputDefaults,
) -> None:
    if "chunk_size" in output_defaults:
        chunk_size = output_defaults["chunk_size"]
        if chunk_size is not None and (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError(
                f"{resource_name}: output_defaults.chunk_size must be a positive integer or null"
            )

    if "partition_by" in output_defaults:
        partition_by = output_defaults["partition_by"]
        if partition_by is not None and not (
            isinstance(partition_by, str) and partition_by.strip()
        ):
            raise ValueError(
                f"{resource_name}: output_defaults.partition_by must be a non-empty string or null"
            )


def _validate_optional_nonempty_string_tuple(
    code: str,
    field_name: str,
    value: tuple[str, ...] | None,
) -> None:
    if isinstance(value, tuple):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(
                f"{code}: {field_name} list items must be non-empty strings"
            )
    elif value is not None:
        raise ValueError(f"{code}: {field_name} must be null or a string list")


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonempty_string_tuple(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(_is_nonempty_string(item) for item in value)
    )


def _is_field_candidates_entry(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_nonempty_string(value[0])
        and _is_nonempty_string_tuple(value[1])
    )


def _is_source_alias_entry(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and _is_nonempty_string(value[0])
        and _is_nonempty_string(value[1])
    )


def _is_canonical_derived_entry(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and _is_nonempty_string(value[0])
        and _is_nonempty_string_tuple(value[1])
        and isinstance(value[2], str)
    )


def _validate_optional_tuple_entries(
    code: str,
    field_name: str,
    value: tuple[object, ...] | None,
    *,
    entry_is_valid,
    invalid_entry_message: str,
    invalid_container_message: str,
) -> None:
    if isinstance(value, tuple):
        for entry in value:
            if not entry_is_valid(entry):
                raise ValueError(f"{code}: {invalid_entry_message}")
        return
    if value is not None:
        raise ValueError(f"{code}: {invalid_container_message}")


@dataclass(frozen=True)
class SourceResource:
    name: str
    url: str
    file_format: str
    file_name: str | None = None
    extract: bool = False
    file_name_template: str | None = None
    download_url_template: str | None = None
    notes: str | None = None
    publication_frequency: str = "unknown"
    snapshot_date_mode: str = "run_date"
    refresh_if_older_than_days: int | None = None
    adapter: str = "direct_download"
    request_headers: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    required_env_vars: tuple[str, ...] = field(default_factory=tuple)
    auth_header_name: str | None = None
    auth_header_env: str | None = None
    auth_header_prefix: str | None = None
    read_defaults: ResourceReadDefaults | None = None
    projection_defaults: ResourceProjectionDefaults | None = None
    output_defaults: ResourceOutputDefaults | None = None

    def __post_init__(self) -> None:
        if self.read_defaults is not None:
            if not isinstance(self.read_defaults, dict):
                raise ValueError(
                    f"{self.name}: read_defaults must be an object when provided"
                )
            _validate_resource_read_defaults(self.name, self.read_defaults)

        if self.projection_defaults is not None:
            if not isinstance(self.projection_defaults, dict):
                raise ValueError(
                    f"{self.name}: projection_defaults must be an object when provided"
                )
            _validate_resource_projection_defaults(self.name, self.projection_defaults)

        if self.output_defaults is not None:
            if not isinstance(self.output_defaults, dict):
                raise ValueError(
                    f"{self.name}: output_defaults must be an object when provided"
                )
            _validate_resource_output_defaults(self.name, self.output_defaults)

    def resolve_output_chunk_size(self, default_chunk_size: int) -> int | None:
        """Resource-declared chunk_size override for output materialization,
        shared across every stage that splits output into row-count-bounded
        files (shard, canonical), falling back to the caller's default when
        the resource declares none. A declared `null` means "don't split --
        write a single output file", distinct from the field being absent
        entirely.
        """
        if self.output_defaults is None or "chunk_size" not in self.output_defaults:
            return default_chunk_size
        return self.output_defaults["chunk_size"]

    def resolve_output_partition_by(
        self, default_partition_by: str | None = None
    ) -> str | None:
        """Resource-declared Hive-style partition column for output
        materialization (e.g. "jurisdiction_code"), consulted by the
        canonical stage. `None` (the default for every resource today) means
        "don't partition -- write flat row-count-bounded files", the same as
        not declaring the field at all.
        """
        if self.output_defaults is None or "partition_by" not in self.output_defaults:
            return default_partition_by
        return self.output_defaults["partition_by"]

    def resolve_read_options_for_path(self, source_path: Path) -> ResolvedReadOptions:
        if self.read_defaults is None:
            return {}

        by_extension = self.read_defaults.get("by_extension")
        runtime = self.read_defaults.get("runtime")
        if not isinstance(by_extension, dict):
            by_extension = {}
        if not isinstance(runtime, dict):
            runtime = {}

        extension = source_path.suffix.lower()
        extension_settings = by_extension.get(extension)
        if not isinstance(extension_settings, dict):
            extension_settings = {}

        resolved: ResolvedReadOptions = {}
        reader_mode = extension_settings.get("reader_mode")
        if isinstance(reader_mode, str):
            resolved["reader_mode"] = reader_mode

        chunk_size_mb = extension_settings.get("chunk_size_mb")
        if isinstance(chunk_size_mb, int) and chunk_size_mb > 0:
            resolved["chunk_size_bytes"] = chunk_size_mb * 1024 * 1024

        indexed_parallelization = runtime.get("indexed_bzip2_parallelization")
        if indexed_parallelization == "auto" or (
            isinstance(indexed_parallelization, int) and indexed_parallelization > 0
        ):
            resolved["indexed_bzip2_parallelization"] = indexed_parallelization

        return resolved

    def _resolve_date_template_kwargs(self, run_date: str) -> dict[str, object]:
        snapshot_date = self.resolve_snapshot_date(run_date)
        effective_date = date_type.fromisoformat(snapshot_date)
        return {
            "run_date": run_date,
            "snapshot_date": snapshot_date,
            "year": effective_date.year,
            "month": f"{effective_date.month:02d}",
            "day": f"{effective_date.day:02d}",
        }

    def resolve_file_name(self, run_date: str) -> str:
        if self.file_name_template is not None:
            return self.file_name_template.format(
                **self._resolve_date_template_kwargs(run_date)
            )
        if self.file_name is None:
            raise ValueError(
                f"SourceResource '{self.name}' must define file_name or file_name_template"
            )
        return self.file_name

    def resolve_download_url(self, run_date: str) -> str:
        if self.download_url_template is None:
            return self.url

        template_kwargs = self._resolve_date_template_kwargs(run_date)
        file_name = self.resolve_file_name(run_date)
        return self.download_url_template.format(
            **template_kwargs,
            file_name=file_name,
        )

    def resolve_snapshot_date(self, run_date: str) -> str:
        effective_date = date_type.fromisoformat(run_date)
        if self.snapshot_date_mode == "month_start":
            return effective_date.replace(day=1).isoformat()
        return effective_date.isoformat()


@dataclass(frozen=True)
class ResearchPlan:
    pass_date: str | None = None
    basis: str | None = None
    priority: str | None = DEFAULT_RESEARCH_PRIORITY
    approach: str | None = None
    next_action: str | None = None
    allow_research_runtime: bool = False
    allowed_stages: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.priority is None:
            pass
        elif self.priority not in ALLOWED_RESEARCH_PRIORITIES:
            allowed = ", ".join(sorted(ALLOWED_RESEARCH_PRIORITIES))
            raise ValueError(
                f"Invalid research priority '{self.priority}'. Allowed values: {allowed}"
            )

        invalid_stages = [
            stage
            for stage in self.allowed_stages
            if stage not in ALLOWED_PROCESS_STAGES
        ]
        if invalid_stages:
            allowed = ", ".join(sorted(ALLOWED_PROCESS_STAGES))
            raise ValueError(
                "Invalid research allowed_stages value(s): "
                + ", ".join(sorted(set(invalid_stages)))
                + f". Allowed values: {allowed}"
            )


@dataclass(frozen=True)
class SystemPlan:
    code: str
    display_name: str
    region: str
    status: str
    access_mode: str
    info_url: str
    notes: str
    all_systems_target: bool
    """Whether `--systems all` targets this system.

    Declared per system in the catalog rather than derived from `status`:
    status records how far onboarding got and why it stopped, which is a
    different question from who a regeneration should sweep. Required, with
    no default, so a system cannot be onboarded without someone deciding --
    an absent flag meaning "excluded" would put the membership rule back in
    whoever remembers it.
    """
    and_tokens: tuple[str, ...] | None = None
    personal_owner_markers: tuple[str, ...] | None = None
    research: ResearchPlan | None = None
    resources: tuple[SourceResource, ...] = field(default_factory=tuple)
    company_type_column: str | None = None
    supported_countries: tuple[str, ...] | None = None
    system_uri_identifier_candidates: tuple[str, ...] | None = None
    system_field_candidates: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    canonical_input_columns: tuple[str, ...] | None = None
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None = None
    canonical_derived_fields: tuple[tuple[str, tuple[str, ...], str], ...] | None = None
    name_variant_type_map: tuple[tuple[str, str], ...] | None = None
    primary_name_override_type_priority: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_SYSTEM_STATUSES:
            allowed = ", ".join(sorted(ALLOWED_SYSTEM_STATUSES))
            raise ValueError(
                f"{self.code}: invalid status '{self.status}'. Allowed values: {allowed}"
            )
        if self.access_mode not in ALLOWED_ACCESS_MODES:
            allowed = ", ".join(sorted(ALLOWED_ACCESS_MODES))
            raise ValueError(
                f"{self.code}: invalid access_mode '{self.access_mode}'. Allowed values: {allowed}"
            )
        if not isinstance(self.all_systems_target, bool):
            actual = type(self.all_systems_target).__name__
            raise TypeError(
                f"{self.code}: all_systems_target must be a boolean, got {actual}"
            )
        if self.all_systems_target and self.status not in RUNNABLE_SYSTEM_STATUSES:
            allowed = ", ".join(sorted(RUNNABLE_SYSTEM_STATUSES))
            raise ValueError(
                f"{self.code}: all_systems_target is set but status '{self.status}' "
                f"is not runnable. A system '--systems all' targets must carry one of: "
                f"{allowed}."
            )
        if self.status in {STATUS_LIVE, STATUS_SUPPORTED} and self.research is not None:
            raise ValueError(
                f"{self.code}: live/supported systems must not define research metadata"
            )
        if (
            self.status in {STATUS_RESEARCH_REQUIRED, STATUS_BLOCKED, STATUS_STOPPED}
            and self.research is None
        ):
            raise ValueError(
                f"{self.code}: status '{self.status}' requires research metadata"
            )
        _validate_optional_nonempty_string_tuple(
            self.code, "supported_countries", self.supported_countries
        )
        _validate_optional_nonempty_string_tuple(
            self.code, "and_tokens", self.and_tokens
        )
        _validate_optional_nonempty_string_tuple(
            self.code, "personal_owner_markers", self.personal_owner_markers
        )
        _validate_optional_nonempty_string_tuple(
            self.code,
            "system_uri_identifier_candidates",
            self.system_uri_identifier_candidates,
        )
        _validate_optional_tuple_entries(
            self.code,
            "system_field_candidates",
            self.system_field_candidates,
            entry_is_valid=_is_field_candidates_entry,
            invalid_entry_message="system_field_candidates items must be (field, non-empty string tuple) pairs",
            invalid_container_message="system_field_candidates must be null or an object map",
        )
        _validate_optional_nonempty_string_tuple(
            self.code, "canonical_input_columns", self.canonical_input_columns
        )
        _validate_optional_tuple_entries(
            self.code,
            "canonical_source_column_aliases",
            self.canonical_source_column_aliases,
            entry_is_valid=_is_source_alias_entry,
            invalid_entry_message="canonical_source_column_aliases items must be (source, target) non-empty string pairs",
            invalid_container_message="canonical_source_column_aliases must be null or an object map",
        )
        _validate_optional_tuple_entries(
            self.code,
            "canonical_derived_fields",
            self.canonical_derived_fields,
            entry_is_valid=_is_canonical_derived_entry,
            invalid_entry_message="canonical_derived_fields items must be (field, non-empty parts tuple, separator)",
            invalid_container_message="canonical_derived_fields must be null or an object map",
        )
        _validate_optional_tuple_entries(
            self.code,
            "name_variant_type_map",
            self.name_variant_type_map,
            entry_is_valid=_is_source_alias_entry,
            invalid_entry_message="name_variant_type_map items must be (source_type, name_type) non-empty string pairs",
            invalid_container_message="name_variant_type_map must be null or an object map",
        )
        _validate_optional_nonempty_string_tuple(
            self.code,
            "primary_name_override_type_priority",
            self.primary_name_override_type_priority,
        )
