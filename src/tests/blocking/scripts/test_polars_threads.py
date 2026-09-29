from __future__ import annotations

import importlib

import pytest

from scripts import _polars_threads


def test_the_cap_is_set_when_the_environment_leaves_it_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLARS_MAX_THREADS", raising=False)

    importlib.reload(_polars_threads)

    assert _polars_threads.os.environ["POLARS_MAX_THREADS"] == (
        _polars_threads.DEFAULT_POLARS_MAX_THREADS
    )


def test_a_cap_the_environment_sets_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_MAX_THREADS", "12")

    importlib.reload(_polars_threads)

    assert _polars_threads.os.environ["POLARS_MAX_THREADS"] == "12"
