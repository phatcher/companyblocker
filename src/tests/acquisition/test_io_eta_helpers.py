from __future__ import annotations

from pathlib import Path

from acquisition.io_eta_helpers import (
    _estimate_stream_eta,
    _format_eta_band_and_stability,
)


class _FakeSourceHandle:
    def __init__(self, compressed_bytes: int) -> None:
        self._compressed_bytes = compressed_bytes

    def tell_compressed(self) -> int:
        return self._compressed_bytes


def test_estimate_stream_eta_applies_warmup_scaling_at_low_progress(tmp_path: Path):
    source_path = tmp_path / "sample.json.bz2"
    source_path.write_bytes(b"x" * 1000)

    # 2% compressed progress
    source_handle = _FakeSourceHandle(compressed_bytes=20)

    details = _estimate_stream_eta(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=100,
        elapsed_seconds=10,
    )

    assert details is not None
    estimated_total_lines, compressed_bytes_read, eta_seconds = details
    assert estimated_total_lines == 5000
    assert compressed_bytes_read == 20

    # Baseline ETA without warm-up multiplier would be 490s.
    # With a 0.20 / 0.02 = 10x warm-up scale this should be 4900s.
    assert eta_seconds == 4900


def test_estimate_stream_eta_no_scaling_after_warmup_progress(tmp_path: Path):
    source_path = tmp_path / "sample.json.bz2"
    source_path.write_bytes(b"x" * 1000)

    # 25% compressed progress, above warm-up threshold.
    source_handle = _FakeSourceHandle(compressed_bytes=250)

    details = _estimate_stream_eta(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=100,
        elapsed_seconds=10,
    )

    assert details is not None
    estimated_total_lines, compressed_bytes_read, eta_seconds = details
    assert estimated_total_lines == 400
    assert compressed_bytes_read == 250
    # Remaining lines 300 at 10 lines/s => 30s, unchanged.
    assert eta_seconds == 30


def test_format_eta_band_and_stability_uses_recent_rate_bounds() -> None:
    eta_band, stability = _format_eta_band_and_stability(
        global_eta_seconds=3600,
        estimated_total_lines=200_000,
        lines_processed=100_000,
        recent_line_rate=40.0,
    )

    assert eta_band.startswith("eta=")
    assert "-" in eta_band
    assert stability in {"stable", "moderate", "volatile"}


def test_format_eta_band_and_stability_falls_back_when_recent_rate_missing() -> None:
    eta_band, stability = _format_eta_band_and_stability(
        global_eta_seconds=1800,
        estimated_total_lines=None,
        lines_processed=0,
        recent_line_rate=None,
    )

    assert eta_band.startswith("eta=")
    assert "-" in eta_band
    assert stability in {"stable", "moderate", "volatile"}
