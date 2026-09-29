import polars as pl
import pytest
from company_vectorize import compose_blocking_text
from company_vectorize.blocking_text_prep import DEFAULT_BLOCKING_TEXT_SEPARATOR


def test_composes_multiple_columns_in_caller_order():
    frame = pl.DataFrame(
        {
            "name": ["Acme Ltd"],
            "city": ["London"],
        }
    )
    composed = compose_blocking_text(frame, columns=["name", "city"])
    assert composed.get_column("blocking_text").to_list() == ["acme ltd london"]


def test_column_order_is_the_columns_argument_not_frame_order():
    frame = pl.DataFrame(
        {
            "city": ["London"],
            "name": ["Acme Ltd"],
        }
    )
    reversed_first = compose_blocking_text(frame, columns=["name", "city"])
    frame_order_first = compose_blocking_text(frame, columns=["city", "name"])
    assert reversed_first.get_column("blocking_text").to_list() == ["acme ltd london"]
    assert frame_order_first.get_column("blocking_text").to_list() == [
        "london acme ltd"
    ]


def test_null_and_blank_columns_collapse_without_stray_separators():
    frame = pl.DataFrame(
        {
            "name": ["Acme"],
            "middle": [None],
            "city": [""],
            "country": ["UK"],
        },
        schema={
            "name": pl.Utf8,
            "middle": pl.Utf8,
            "city": pl.Utf8,
            "country": pl.Utf8,
        },
    )
    composed = compose_blocking_text(
        frame, columns=["name", "middle", "city", "country"]
    )
    assert composed.get_column("blocking_text").to_list() == ["acme uk"]


def test_all_blank_columns_compose_to_empty_string():
    frame = pl.DataFrame({"a": [None], "b": [""]}, schema={"a": pl.Utf8, "b": pl.Utf8})
    composed = compose_blocking_text(frame, columns=["a", "b"])
    assert composed.get_column("blocking_text").to_list() == [""]


def test_lowercase_false_preserves_source_casing():
    frame = pl.DataFrame({"name": ["Acme LTD"]})
    composed = compose_blocking_text(frame, columns=["name"], lowercase=False)
    assert composed.get_column("blocking_text").to_list() == ["Acme LTD"]


def test_custom_separator_and_output_col():
    frame = pl.DataFrame({"a": ["x"], "b": ["y"]})
    composed = compose_blocking_text(
        frame, columns=["a", "b"], output_col="text_out", separator="|"
    )
    assert composed.get_column("text_out").to_list() == ["x|y"]


def test_default_separator_constant_matches_default_behaviour():
    frame = pl.DataFrame({"a": ["x"], "b": ["y"]})
    explicit = compose_blocking_text(
        frame, columns=["a", "b"], separator=DEFAULT_BLOCKING_TEXT_SEPARATOR
    )
    default = compose_blocking_text(frame, columns=["a", "b"])
    assert explicit.equals(default)


def test_empty_columns_raises():
    frame = pl.DataFrame({"a": ["x"]})
    with pytest.raises(ValueError):
        compose_blocking_text(frame, columns=[])


def test_missing_column_raises_before_touching_frame():
    frame = pl.DataFrame({"a": ["x"]})
    with pytest.raises(ValueError, match="missing"):
        compose_blocking_text(frame, columns=["a", "not_a_column"])


def test_deterministic_across_repeated_calls_and_row_order():
    frame = pl.DataFrame(
        {
            "name": ["Beta Co", "Acme Ltd"],
            "city": ["Paris", "London"],
        }
    )
    first = compose_blocking_text(frame, columns=["name", "city"])
    second = compose_blocking_text(frame, columns=["name", "city"])
    assert first.equals(second)

    shuffled = frame.reverse()
    shuffled_composed = compose_blocking_text(shuffled, columns=["name", "city"])
    assert shuffled_composed.get_column("blocking_text").to_list() == list(
        reversed(first.get_column("blocking_text").to_list())
    )


def test_non_string_column_casts_without_raising():
    frame = pl.DataFrame({"name": ["Acme"], "reg_number": [12345]})
    composed = compose_blocking_text(frame, columns=["name", "reg_number"])
    assert composed.get_column("blocking_text").to_list() == ["acme 12345"]


def test_single_column_composition_is_a_no_op_join():
    frame = pl.DataFrame({"name": ["Acme Ltd"]})
    composed = compose_blocking_text(frame, columns=["name"])
    assert composed.get_column("blocking_text").to_list() == ["acme ltd"]
