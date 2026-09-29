import subprocess
import sys
from pathlib import Path

import pytest


def main() -> int:
    # Add src to path so tests can import acquisition
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    tach_validation = subprocess.run(
        [sys.executable, "-m", "tach", "check"],
        check=False,
    )
    if tach_validation.returncode != 0:
        return tach_validation.returncode

    catalog_validation = subprocess.run(
        [sys.executable, "scripts/validate_catalog.py"],
        check=False,
    )
    if catalog_validation.returncode != 0:
        return catalog_validation.returncode

    return pytest.main(
        [
            "-q",
            "--tb=short",
            "-m",
            "not performance",
            "--cov=src/acquisition",
            "--cov=src/analysis",
            "--cov=src/training",
            "--cov=src/validation",
            "--cov=packages/company_cleanse/src",
            "--cov=packages/company_tokenize/src",
            "--cov=packages/company_classify/src",
            "--cov-report=term-missing",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
