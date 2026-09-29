from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip_archive(archive_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            archive.extract(member, path=extract_dir)
            if not member.is_dir():
                extracted_paths.append(extract_dir / member.filename)

    return extracted_paths
