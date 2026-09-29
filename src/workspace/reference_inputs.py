"""Where a tracked reference input lives: `config/reference/<concern>/<file>`.

A reference input is authored or downloaded once, read by a run and tracked in
git: the Wikidata legal-form closure that the extractor comparison reads. It is
input, so it sits under the config root beside the checkout's other configuration,
and never under `artifacts/`, which holds only what runs generate and which a
worktree links whole to shared storage. Each input has a function here naming it,
so a caller names the thing and this module says where it is.
"""

from __future__ import annotations

from pathlib import Path

from .roots import WorkspaceRoots

REFERENCE_DIR_NAME = "reference"

WIKIDATA_DIR_NAME = "wikidata"
LEGAL_FORM_CLOSURE_FILENAME = "p279.json"


def reference_input_root(roots: WorkspaceRoots) -> Path:
    """The directory every reference input sits beneath: `config/reference`."""
    return roots.config / REFERENCE_DIR_NAME


def wikidata_legal_form_closure(roots: WorkspaceRoots) -> Path:
    """The `P279` subclass closure of Wikidata's legal-form classes, the file the
    extractors read as `p279.json`: `config/reference/wikidata/p279.json`."""
    return reference_input_root(roots) / WIKIDATA_DIR_NAME / LEGAL_FORM_CLOSURE_FILENAME
