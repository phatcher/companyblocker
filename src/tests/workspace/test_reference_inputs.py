"""`reference_inputs.py`'s own contract: each tracked input resolves under the config root."""

from __future__ import annotations

from pathlib import Path

from workspace.reference_inputs import (
    reference_input_root,
    wikidata_legal_form_closure,
)
from workspace.roots import default_workspace_roots


def test_reference_inputs_resolve_under_the_config_root(tmp_path: Path) -> None:
    """Under `config/reference`, never `artifacts/`, whatever the artifact root is."""
    roots = default_workspace_roots(tmp_path)

    assert reference_input_root(roots) == tmp_path / "config" / "reference"
    assert wikidata_legal_form_closure(roots) == (
        tmp_path / "config" / "reference" / "wikidata" / "p279.json"
    )
    assert not wikidata_legal_form_closure(roots).is_relative_to(roots.artifacts)
