"""Pytest wiring for the cross-stage contract tests.

The corpus itself lives in `corpus.py`, which holds what it is and why it is
shaped that way; this only makes it available as a fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.pipeline.corpus import PipelineCorpus, write_pipeline_corpus


@pytest.fixture
def pipeline_corpus(tmp_path: Path) -> PipelineCorpus:
    """The corpus, materialised under `tmp_path` and no further.

    Deliberately stops at the Acquire layer rather than running any stage, so
    a test drives exactly the stage subset it is about and none of them
    inherits output another test's run left behind.
    """
    return write_pipeline_corpus(tmp_path)
