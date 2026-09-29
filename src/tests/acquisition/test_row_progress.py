from __future__ import annotations

from acquisition.row_progress import RowProgress


def test_advance_is_a_no_op_without_a_progress_callable():
    tracker = RowProgress(label="stage")

    tracker.advance(150_000)

    assert tracker.rows == 150_000


def test_advance_emits_once_per_crossed_interval_boundary():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(250)

    assert len(emitted) == 1
    assert "250 rows" in emitted[0]
    assert tracker.rows == 250


def test_advance_does_not_emit_again_below_the_next_boundary():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(150)
    tracker.advance(20)

    assert len(emitted) == 1


def test_advance_emits_again_once_a_further_boundary_is_crossed():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(150)
    tracker.advance(100)

    assert len(emitted) == 2


def test_advance_ignores_non_positive_row_counts():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=10)

    tracker.advance(0)
    tracker.advance(-5)

    assert tracker.rows == 0
    assert emitted == []


def test_finish_emits_unreported_tail_marked_complete():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(120)  # crosses the 100-row boundary, emits at 120
    tracker.advance(30)  # rows=150, below the next boundary: not yet emitted
    tracker.finish()

    assert len(emitted) == 2
    assert "(complete)" in emitted[-1]
    assert "150 rows" in emitted[-1]


def test_finish_does_not_repeat_a_line_already_emitted_at_the_final_boundary():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(100)
    tracker.finish()

    assert len(emitted) == 1


def test_finish_is_a_no_op_when_nothing_was_ever_advanced():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.finish()

    assert emitted == []


def test_progress_line_includes_percentage_when_total_rows_known():
    emitted: list[str] = []
    tracker = RowProgress(
        label="stage", progress=emitted.append, total_rows=1000, interval=100
    )

    tracker.advance(250)

    assert "25%" in emitted[0]


def test_progress_line_omits_percentage_when_total_rows_unknown():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=100)

    tracker.advance(250)

    assert "%" not in emitted[0]


def test_interval_of_zero_or_less_is_clamped_to_one():
    emitted: list[str] = []
    tracker = RowProgress(label="stage", progress=emitted.append, interval=0)

    tracker.advance(1)

    assert len(emitted) == 1
