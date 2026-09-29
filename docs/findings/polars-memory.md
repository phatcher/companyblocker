# Polars memory in a blocking run

A run's memory is set by what its process keeps, not by its data. On `gleif -> fr` (165,749 source rows, 12.9M target rows) the frames a run holds before its target index are 2.3 GiB at most, yet the process held 28 GiB at that point.

Polars is why. Each of its threads keeps its own allocator heap, and memory freed in one thread's heap is not reused by another, so every large join leaves memory behind in whichever threads ran it. Returning freed pages at once (`MIMALLOC_PURGE_DELAY=0`) barely changes this; fewer threads does:

| Step on `gleif -> fr` | 32 threads | 4 threads | Live frames |
| --- | --- | --- | --- |
| Name forms derived | 10.9 GiB | 11.3 GiB | 2.2 GiB |
| Exact join on raw names | 16.4 GiB | 15.3 GiB | 2.3 GiB |
| Exact join on basic names | 19.0 GiB | 15.8 GiB | 2.3 GiB |
| Exact join on cleansed names | 23.8 GiB | 16.6 GiB | 2.3 GiB |
| Target reduced to its ids and names | 28.0 GiB | 17.9 GiB | 0.4 GiB |

`run_blocking.py` therefore caps Polars at 4 threads (`scripts/_polars_threads.py`). The cap is the environment variable `POLARS_MAX_THREADS`, which Polars reads once when it is first imported, so it is not a command-line flag; set it before a run to override the default. The run's start line and its `--dry-run` report name the thread count in effect. Loading and deriving the name forms take about 11 GiB for 2.2 GiB of frames whatever the thread count, and the cap does not touch that.
