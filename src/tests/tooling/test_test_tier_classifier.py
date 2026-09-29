"""Direct tests of `scripts/test_tier_classifier`'s behavioural rule.

One example per category the tier rule names: a subprocess call, a call to
a `materialize_*`/`swap_layer_into_place` function, and a reference to the
repository's real `data/` directory as opposed to a synthetic `tmp_path`
fixture that merely mirrors its layout. Also covers the one level of
same-module helper indirection the classifier follows, since that is the
shape almost every real case in this repo takes: a fixture or a private
helper does the heavy work, not the test body itself.
"""

from __future__ import annotations

from test_tier_classifier import classify_callable, classify_source


def test_classify_source_finds_no_signal_in_an_ordinary_unit_test() -> None:
    source = """
def test_adds_two_numbers():
    assert 1 + 1 == 2
"""
    signal = classify_source(source)
    assert not signal
    assert signal.reasons == frozenset()


def test_classify_source_flags_a_subprocess_call() -> None:
    source = """
def test_runs_the_script():
    result = subprocess.run(["python", "script.py"], check=False)
    assert result.returncode == 0
"""
    signal = classify_source(source)
    assert signal.reasons == frozenset({"subprocess"})


def test_classify_source_flags_a_layer_materialization_call() -> None:
    source = """
def test_publishes_the_cleansed_layer():
    materialize_cleansed_merge(config)
"""
    signal = classify_source(source)
    assert signal.reasons == frozenset({"layer"})


def test_classify_source_flags_swap_layer_into_place_too() -> None:
    source = """
def test_swaps_the_staged_layer_live():
    swap_layer_into_place(staging=staging_dir, live=live_dir)
"""
    signal = classify_source(source)
    assert signal.reasons == frozenset({"layer"})


def test_classify_source_ignores_a_private_helper_of_a_similar_name() -> None:
    """`_materialize_cleansed_merge` is a local fixture-building helper, not
    the repo's `materialize_<something>` publishing convention -- the
    leading underscore breaks the word boundary the pattern requires."""
    source = """
def test_builds_a_small_fixture():
    _materialize_cleansed_merge(root, system)
"""
    signal = classify_source(source)
    assert not signal


def test_classify_source_flags_a_real_data_directory_reference() -> None:
    source = """
def test_reads_the_real_corpus():
    repo_root = Path(__file__).resolve().parents[3]
    files = list((repo_root / "data" / "gb" / "canonical").glob("*.parquet"))
"""
    signal = classify_source(source)
    assert "corpus" in signal.reasons


def test_classify_source_does_not_flag_a_synthetic_tmp_path_data_fixture() -> None:
    """`tmp_path / "data" / ...` mirrors the production layout for a
    throwaway fixture; it never touches the real `data/` tree and must not
    be flagged as reading a corpus."""
    source = """
def test_builds_a_synthetic_layer(tmp_path):
    data_dir = tmp_path / "data" / "gb" / "source" / "2026-06-01"
    data_dir.mkdir(parents=True)
"""
    signal = classify_source(source)
    assert not signal


def test_classify_callable_follows_one_level_of_same_module_helper() -> None:
    """The common real shape: the test's own body calls a helper defined
    beside it in the same module, and the helper is where the subprocess
    call actually lives (e.g. a `git()` wrapper used by many tests)."""
    function_source = """
def test_dispatches_a_workitem():
    git(tmp_path, "init", "-q", "-b", "main")
"""
    module_source = (
        function_source
        + """

def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True)
"""
    )
    signal = classify_callable(function_source, module_source)
    assert signal.reasons == frozenset({"subprocess"})


def test_classify_callable_returns_no_signal_when_no_helper_matches() -> None:
    function_source = """
def test_computes_a_total(fixture_value):
    assert total(fixture_value) == 3
"""
    module_source = (
        function_source
        + """

def total(values):
    return sum(values)
"""
    )
    signal = classify_callable(function_source, module_source)
    assert not signal
