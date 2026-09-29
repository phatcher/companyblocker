"""The repo-wide set of layers holding two layout generations of a partition must
never grow.

`scripts/check_stale_layout_generations.py` is the detecting half: it walks every
layer directory under `data/` and reports one where a partition value exists both
under `primary/` and at the layer's own top level, the shape
`workspace.layer_layout.resolve_partition_dir` silently disambiguates by preferring
`primary/`.

This is the ratchet half. `src/tests/baselines/stale_layout_generations.txt` lists
today's known instances, one line per layer with the count of doubled partition
values; delete or lower a line only once a fresh run against the real `data/`
(shared physical storage, the same across every worktree) no longer finds it --
never to silence a violation this repo's own change introduced, the same baseline
convention `test_layer_path_literals.py` and `test_docstring_item_ids.py` use.
`data/gleif/matched/` is today's only known instance, left behind when the family
split moved the write target and nothing removed the old shape; deleting it is a
destructive change to shared storage and is an operator's own run, not something
this test or any commit does.
"""

from __future__ import annotations

from pathlib import Path

from baseline import check_baseline
from check_stale_layout_generations import measure


def test_stale_layout_generation_backlog_does_not_grow(repo_root: Path) -> None:
    """A new layer, or a higher doubled-partition count for a known one, means a
    write path left a second generation behind: fix the write path rather than
    adding a line here."""
    check_baseline("stale_layout_generations", measure(repo_root / "data"))
