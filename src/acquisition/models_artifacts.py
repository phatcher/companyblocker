from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workspace.roots import WorkspaceRoots


@dataclass(frozen=True)
class ArtifactRecord:
    country: str
    run_date: str
    artifact_type: str
    source_name: str
    source_url: str
    format: str
    file_path: str
    content_hash: str | None = None
    row_count: int | None = None
    chunk_index: int | None = None
    note: str | None = None
    source_snapshot_date: str | None = None
    publication_frequency: str | None = None
    refresh_if_older_than_days: int | None = None
    acquired_at_utc: str | None = None
    last_acquired_at_utc: str | None = None
    days_since_last_acquisition: int | None = None
    freshness_gate_applied: bool | None = None
    freshness_gate_passed: bool | None = None
    freshness_gate_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "country": self.country,
            "run_date": self.run_date,
            "artifact_type": self.artifact_type,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "format": self.format,
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "row_count": self.row_count,
            "chunk_index": self.chunk_index,
            "note": self.note,
            "source_snapshot_date": self.source_snapshot_date,
            "publication_frequency": self.publication_frequency,
            "refresh_if_older_than_days": self.refresh_if_older_than_days,
            "acquired_at_utc": self.acquired_at_utc,
            "last_acquired_at_utc": self.last_acquired_at_utc,
            "days_since_last_acquisition": self.days_since_last_acquisition,
            "freshness_gate_applied": self.freshness_gate_applied,
            "freshness_gate_passed": self.freshness_gate_passed,
            "freshness_gate_reason": self.freshness_gate_reason,
        }


@dataclass(frozen=True)
class AcquisitionPaths:
    roots: WorkspaceRoots
    data_root: Path
    acquire_dir: Path
    prepare_dir: Path
    source_dir: Path
    canonical_dir: Path


@dataclass(frozen=True)
class AcquisitionResult:
    country: str
    status: str
    message: str
    artifact_count: int = 0
