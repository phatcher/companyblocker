from __future__ import annotations

from scripts import compare_wikidata_rust_wikisieve
from workspace.reference_inputs import wikidata_legal_form_closure
from workspace.roots import default_workspace_roots


def test_default_p279_resolves_through_the_reference_input_function() -> None:
    """The script's default `p279.json` is the tracked reference input under the
    config root, resolved by its `workspace` function rather than a hand-composed
    path, and it exists in this checkout."""
    expected = wikidata_legal_form_closure(
        default_workspace_roots(compare_wikidata_rust_wikisieve.REPO_ROOT)
    )

    assert compare_wikidata_rust_wikisieve.DEFAULT_P279 == expected
    assert compare_wikidata_rust_wikisieve.DEFAULT_P279 == (
        compare_wikidata_rust_wikisieve.REPO_ROOT
        / "config"
        / "reference"
        / "wikidata"
        / "p279.json"
    )
    assert compare_wikidata_rust_wikisieve.DEFAULT_P279.is_file()


def test_default_reference_is_the_tracked_expected_output_and_no_extractor_is_run() -> (
    None
):
    """The other side of the diff is a file this repository keeps, the reviewed
    output over the tracked sample, so the script needs no Rust extractor built."""
    reference = compare_wikidata_rust_wikisieve.DEFAULT_REFERENCE

    assert reference == (
        compare_wikidata_rust_wikisieve.REPO_ROOT
        / "src"
        / "tests"
        / "acquisition"
        / "fixtures"
        / "wikisieve-expected-output.jsonl"
    )
    assert reference.is_file()
    assert not hasattr(compare_wikidata_rust_wikisieve, "DEFAULT_RUST_BINARY")
    flags = {
        action.option_strings[0]
        for action in compare_wikidata_rust_wikisieve._build_parser()._actions
        if action.option_strings
    }
    assert "--reference" in flags
    assert "--write-reference" in flags
    assert "--rust-binary" not in flags
