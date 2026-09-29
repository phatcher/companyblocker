"""Cap Polars' thread pool for a script, before anything imports Polars.

Import this first in a script whose memory is bounded by Polars, bare, as
`_bootstrap` is imported::

    import _polars_threads  # noqa: F401

Polars sizes its thread pool once, at its first import, from
`POLARS_MAX_THREADS`, so the cap cannot be a command-line flag: by the time
arguments are parsed, Polars is imported. This sets the variable to
`DEFAULT_POLARS_MAX_THREADS` unless the environment already sets it, which is
how a run overrides it.

**Why a cap at all.** Each Polars thread keeps its own allocator heap, and
memory freed in one thread's heap is not reused by another, so a process
holds far more than its live data once many threads have each run a large
join. On `gleif -> fr` (12.9M target rows) the exact-name joins took the
process from 11 GiB to 28 GiB at 32 threads while its live frames stayed at
2.3 GiB; at 4 threads the same steps ended at 18 GiB.
`docs/findings/polars-memory.md` carries the figures.

This module imports nothing, so it is safe before `_bootstrap`.
"""

from __future__ import annotations

import os

DEFAULT_POLARS_MAX_THREADS = "4"

os.environ.setdefault("POLARS_MAX_THREADS", DEFAULT_POLARS_MAX_THREADS)
