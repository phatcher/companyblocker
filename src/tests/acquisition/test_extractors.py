import zipfile
from pathlib import Path

from acquisition.extractors import extract_zip_archive


def test_extract_zip_archive_preserves_all_files(tmp_path: Path):
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a.txt", "alpha")
        archive.writestr("nested/b.txt", "beta")

    extracted = extract_zip_archive(archive_path, tmp_path / "extracted")

    assert sorted(
        path.relative_to(tmp_path / "extracted").as_posix() for path in extracted
    ) == [
        "a.txt",
        "nested/b.txt",
    ]
