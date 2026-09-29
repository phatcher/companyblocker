# wikisieve benchmarks

Measurements behind the README's performance summary. Each section names its dump, spec, build and machine; figures are single runs unless a table says otherwise.

## The engine on an 8,000,000-line window

### An external decompressor

This measurement predates the parallel reader described in the next section, and is what motivated it. The built-in reader decompressed the gzip dump on the calling thread at the time, and `--input -` let an external parallel decompressor feed the scan instead, so what parallel decompression was worth could be measured before building one in. `pigz` has no Windows build, so only `rapidgzip` 0.16.0 was measured, as `rapidgzip -d -c <dump.json.gz> | wikisieve --input - ...`. `--input -` is still supported, but it is no longer how a run takes decompression off the main thread.

On the first 8,000,000 lines of the `2026-07-16` dump (`data/wikidata/acquire/2026-07-16/wikidata-all.json.gz`), with a release build and the production company spec, wall time from `System.Diagnostics.Stopwatch` and CPU from `Process.TotalProcessorTime` (bash's `time` under Git Bash reports near-zero user and system time for native Windows processes):

| reader | wall | `wikisieve` CPU | records emitted |
| --- | --- | --- | --- |
| built-in (`--input <path>`) | 439.2 s | 1,399.7 s | 29,244 |
| `rapidgzip -d -c` piped to `--input -` | 358.3 s | 1,212.5 s | 29,244, byte-identical |

Piping through `rapidgzip` cuts wall time by 18.4% on this window. Decompression is a real but minority share of wall time, not the ceiling: most of `wikisieve`'s CPU in both configurations goes to the parallel candidate parse. `rapidgzip`'s own process reported near-zero CPU under both measurements, which its wall-time win contradicts, so its CPU cost is not recorded here.

### Decompression off the main thread

`crate::dump_source::open_parallel_gzip_source` decompresses the dump through [`rapidgzip-core`](https://crates.io/crates/rapidgzip-core) 0.3.1 (pure Rust, `zlib-rs` inflate backend) instead of on the calling thread, and replaces the sequential `flate2::read::MultiGzDecoder` reader the section above measured. It was evaluated ahead of the C++-backed `rapidgzip` crate and adopted without needing that crate's build risk. The dump is one trivial stub member followed by one continuous member carrying essentially all the content, so there is no free multi-member parallelism to exploit. `rapidgzip-core` can index DEFLATE block boundaries within one member and decode the blocks in parallel, but on this dump its admission screen decodes the 22-byte stub, reaches its end and selects its sequential decoder, so the whole dump inflates on one of its worker threads whatever thread budget it is given ([the floors](#the-floors-raw-read-then-decompression)). What the reader changes is where decoding runs: off the caller's thread, which receives only the ordered decoded bytes, so decoding overlaps the scan instead of running in series with it.

On the same first 8,000,000 lines of the `2026-07-16` dump, production company spec and release build as the table above, wall time and process CPU measured the same way, and the main thread's own CPU sampled from `Process.Threads` once a second (its lowest native thread id, created before any worker thread and never retired, and that thread's last-sampled `TotalProcessorTime` before the process exited):

| reader | wall | `wikisieve` CPU | main thread CPU | records emitted |
| --- | --- | --- | --- | --- |
| built-in, sequential reader | 439.2 s | 1,399.7 s | not sampled | 29,244 |
| `rapidgzip -d -c` piped to `--input -` | 358.3 s | 1,212.5 s | not sampled | 29,244, byte-identical |
| built-in, `rapidgzip-core` reader | 267.3 s | 1,386.9 s | 227.2 s | 29,244, byte-identical |

The `rapidgzip-core` reader cuts wall time 39.1% against the sequential built-in reader and 25.4% against the piped external decompressor, while process CPU stays close to the sequential figure (0.9% less): the same total decompression-plus-parse work now overlaps, the decode on its own thread while the main thread scans, instead of running in series behind one. The main thread's own CPU, 227.2 s of 267.3 s wall or about 85% of a core, is no longer decompression: `fill_wikidata_batch`'s line-splitting and byte-level prefilter, and the main thread's own share of the parallel candidate parse it joins by calling `rayon`'s `par_iter().collect()`, both still run there. Per-thread sampling was added for this measurement, so the two earlier rows have no comparable figure; an earlier profile on Linux, by a different method, put the main thread at about 90% of a core before this change. The full-dump section below measures this reader over the whole dump, without a sequential-reader counterpart to compare it against and without main-thread sampling.

### The value prefilter

`CompiledSpec::load` also compiles the literal quoted form of every QID any marker's `qid_set` or `qid_closure_file` can match (`"Q6881511"`, say) into an Aho-Corasick automaton, searched alongside the existing marker-property-name search. `Prefilter::build` holds both stages and `Prefilter::admits` requires both of them to admit a line before it reaches the candidate parse. A true match always carries its marker's own property literal and its matched QID literal somewhere in the line, so each stage is independently a necessary condition and requiring both stays conservative: a line either search rejects can never be a real match, and a QID found under an unrelated property only admits an extra line, which the full marker check then rejects. Requiring both is far more selective than the property search alone, which tests only the spec's marker property literals, two of them (`"P31"` and `"P1454"`) for the production company spec, and 97%+ of the dump's lines carry one of those whether or not the entity is a company. Projected-field properties are not searched by either stage: only markers decide whether a line can match.

The value stage uses Aho-Corasick rather than one `memchr::memmem::Finder` per QID, which is what `build_prefilter_finders` does for the handful of marker properties: the production company spec's `P1454` closure alone holds 49,176 entries, and testing every line against that many separate finders would cost more than the JSON parse the prefilter exists to avoid. One automaton tests all of them in a single pass over the line.

Measured over the same first 8,000,000 lines of the `2026-07-16` dump, production company spec and release build as the tables above, one run per configuration, interleaved (old prefilter then new, immediately in succession, rather than as two separate blocks), wall time and process CPU measured the same way as the parallel-reader table:

| prefilter | pass rate | wall | `wikisieve` CPU | records emitted |
| --- | --- | --- | --- | --- |
| property-name only (old) | 97.43% (7,794,023 / 8,000,000) | 262.4 s | 1,331.3 s | 29,244 |
| property-name + value (new) | 8.75% (700,278 / 8,000,000) | 159.7 s | 343.8 s | 29,244, byte-identical |

The value search cuts the candidate parse's input by 91%, wall time by 39.1% and process CPU by 74.2%. Both runs emitted the same 29,244 records and wrote byte-identical output (`md5sum` `6796c675286ec041f62f7ee7d209e240` on both). The remaining wall time is no longer dominated by the candidate parse the way the parallel-reader section's CPU breakdown described: with 91% fewer candidates reaching `project_candidate`, decompression and the two-stage prefilter's own sequential scan, which still runs once per scanned line rather than only per candidate, make up a larger share of what is left. The section below measures the same prefilter over the whole dump.

## The full dump

The production company spec over the whole `2026-07-16` dump, 120,905,360 lines and 144 GiB gzipped, with a release build, the `rapidgzip-core` reader and flat output rather than `--resume`, wall time and process CPU measured as the tables above:

| measure | value |
| --- | --- |
| wall | 2,238.4 s |
| `wikisieve` CPU | 4,923.1 s |
| peak working set | 1,408 MB |
| pass rate | 8.13% (9,828,510 / 120,905,360) |
| records emitted | 884,070 |
| parse errors | 0 |

The prefilter's pass rate holds at full scale, 8.13% against the head window's 8.75%, so the window did not flatter it; the head is the less favourable end, since the company candidate rate there is 0.37% against 0.73% across the dump. Peak working set is 1,408 MB against the window's 1,396 MB, so memory does not grow with the input.

The output is byte-identical to `data/wikidata/prepare/2026-07-16/wikidata-companies.jsonl`, the extract in use, which was produced under the property-name-only prefilter: SHA-256 `72f008f2e352119cd9512926cce7115062c09b6fcf51d3789878b44d7efba3a2` on both files, 884,070 records each. That is the first confirmation at full scale that the value stage changes no selection, every earlier parity check having covered the 8,000,000-line window alone. That earlier run's completed resume state records 118,633,260 candidates scanned of the same 120,905,360 lines, a 98.12% pass rate against 8.13% here, but no wall time: the two runs are therefore comparable on pass rate and on output, not on speed. No summary JSON yet carries a run's elapsed wall and CPU seconds, which is why no earlier full run's speed can be recovered.

These figures are an upper bound rather than a clean measurement: the run shared the machine with a full `pytest` suite and interactive work throughout.

### What the engine gained, on one fixed window

Every row is the same first 8,000,000 lines of the `2026-07-16` dump, the same production company spec and the same measurement method, so the series is comparable end to end; each is one run, not a repeated mean.

| engine | wall | `wikisieve` CPU | pass rate |
| --- | --- | --- | --- |
| sequential reader, property-name prefilter | 439.2 s | 1,399.7 s | 97.43% |
| `rapidgzip-core` reader, property-name prefilter | 262.4 s | 1,331.3 s | 97.43% |
| `rapidgzip-core` reader, property-name + value prefilter | 159.7 s | 343.8 s | 8.75% |

Wall time on the window fell 2.75x across the series and process CPU 4.07x, and every row emitted the same 29,244 records with byte-identical output. The first row is the engine the full extract above was made with, whose own wall time was never recorded.

Re-measured on the current build, that last configuration ran the window in 163.2 s wall and 355.3 s CPU, 2.2% and 3.3% above the figures in the table, on a machine running a `pytest` suite at the time.

The reader that preceded the `rapidgzip-core` reader measured what read size costs on the same window: read from WSL over `/mnt/c`, 32 KiB reads cost 17 to 24% more wall time than 4 to 64 MiB reads, while on native local disk read size made no difference, the run being CPU-bound.

## Syncing chunks before rename

Measured over the same first 8,000,000 lines of the `2026-07-16` dump and the same production company spec as the tables above, both release builds, one run per configuration, in `--resume` mode with `--max-rows 8000000`:

| configuration    |    wall | `wikisieve` CPU | records emitted |
| ---------------- | ------: | --------------: | --------------: |
| without syncing  | 425.8 s |       1,382.4 s |          29,244 |
| with syncing     | 436.2 s |       1,387.0 s |          29,244 |

Syncing costs 2.4% more wall time on this window and does not move CPU time: the run stays CPU-bound on the parallel candidate parse either way, and fsync's cost is wall time spent waiting on the disk rather than processor work. Both configurations wrote the same 80 chunks of the same total size, so the sync and rename change alone accounts for the difference. This is the resumable-chunked-output path's cost; a flat, non-`--resume` run renames nothing and pays none of it.

## Against published subsetting tools

### Selection parity with the Wikidata subsetting evaluation

Hosseini Beghaeiraveri et al., "Wikidata subsetting: approaches, tools, and evaluation", Semantic Web 15(6), counted the instances of four classes in the 3 January 2022 JSON dump (95,900,304 items; Academic Torrents `229cfeb2331ad43d4706efd435f6d78f40a3c438`) and ran four subsetting tools over it. Its scripts are at https://github.com/kg-subsetting/paper-wikidata-subsetting-2023, commit `2f0a552`; the repository holds no results, so the expected counts are the paper's Table 5.

The rule, from `performance-experiments/count_instances_json_iter.py`: an item is in a class when any `P31` statement, at any rank, names that class; a statement with no `datavalue` is skipped; an item in two classes counts in both. A `qid_set` marker reads no rank and skips a snak with no value, which `qid_set_marker_matches_any_rank_and_skips_a_novalue_snak` in `src/lib.rs` checks.

Spec, with no subclass closure:

```json
{
  "markers": [
    { "property": "P31", "match": { "type": "qid_set", "qids": ["Q7187", "Q8054", "Q11173", "Q12136"] } }
  ],
  "match_logic": "any",
  "projected_fields": [
    { "property": "P31", "field": "matched_classes", "shape": "entity_id_list", "match_only": true }
  ]
}
```

Run with a release build reporting `wikisieve 0.1.0 (commit 8d0735817ac1102083b6587d37ebe90f87c73020)`, not dirty, on Windows:

```
wikisieve --input wikidata-20220103-all.json.gz --spec gene_protein_disease_chemicals.json --output wikisieve.jsonl --summary-json wikisieve-summary.json
```

95,900,305 lines scanned, 3,434,538 records emitted, no parse errors, 3,982.7 s wall. Counting each output record once per distinct class in `matched_classes`:

| class | wikisieve | evaluation, input dump |
| --- | --- | --- |
| gene (`Q7187`) | 1,196,532 | 1,196,532 |
| protein (`Q8054`) | 987,636 | 987,636 |
| chemical compound (`Q11173`) | 1,244,881 | 1,244,881 |
| disease (`Q12136`) | 5,513 | 5,513 |

Operon (`Q139677`) and acid (`Q11158`), which a subclass closure would pull in, select nothing: neither is ever a matched class, and no output item is an operon. Seven output items do carry `P31` acid beside chemical compound, `Q409602` (boron trifluoride) and `Q189298` (picric acid) among them; they are selected as chemical compounds, the evaluation's chemical compound count includes them, and its Table 5 reports the same 7 acids in wikibase-dump-filter's output.

The same dump through wikibase-dump-filter 6.1.1 with the evaluation's `tool_runner.py` command, under WSL with Node 22.23.2:

```
cat wikidata-20220103-all.json.gz | gzip -d | wikibase-dump-filter --claim P31:Q11173,Q12136,Q7187,Q8054 > wdf.ndjson
```

kept 3,434,538 entities in 24,819 s, reading the dump over `/mnt/c`. `find_unique_ids.py` over the two matched id lists finds no id in one that is not in the other.

#### Platform and storage: the same run on Linux and Windows

The projection run above was made twice from the same build, on the same dump and spec, one run each and differing only in platform and where the dump sat:

| platform | dump on | wall | compressed input consumed |
| --- | --- | --- | --- |
| WSL, native `ext4` in a vhdx | spinning disk | 2,839 s | 38.4 MB/s |
| Windows, native | SSD | 3,982.7 s | 27.4 MB/s |

Linux is 1.40x faster on storage that reads at an eighth the rate, so this is a platform difference and not a storage one. Sequential reads of 4 GiB taken 64 GiB into the file, large enough to defeat the page cache, put the three access paths far apart: the SSD read natively by Windows at 1,259 MB/s, the vhdx's `ext4` volume on the spinning disk at 166 MB/s, and the same SSD reached through WSL's `/mnt/c` mount at 86 MB/s. The mount, not the disk, is what costs: it gives back 7% of what the drive beneath it can do.

None of that reaches either tool. The faster of the two runs above consumed 23% of the volume it read from and 3% of what the SSD can deliver, and wikibase-dump-filter's 24,819 s works out at 4.4 MB/s, a twentieth of even the `/mnt/c` figure, so its time is spent in its single Node thread rather than waiting on the mount it read through. A comparison between the two tools is therefore not biased by which of these volumes it runs on, provided both sides run on the same one.

The other tools' shortfalls in the evaluation's Table 5 are a rank rule, not a parse failure. Counting only truthy `P31` statements, as SPARQL's `wdt:` does (preferred if any exist, otherwise normal, never deprecated), over the same selected entities gives gene 1,196,503, protein 987,636, chemical compound 1,244,874 and disease 5,512, exactly WDumper's counts. All 29 genes the evaluation lists as missed by WDumper and KGTK hold gene only at deprecated rank. `wikisieve`, wikibase-dump-filter and the evaluation's counting script match any rank.

The engines' rules differ in two places, and neither changed a count or an id here: the counting script compares `numeric-id` without checking `entity-type`, where `wikisieve` accepts only a `Q` id, so a `P31` value naming a property whose number equals a counted class would count there and not here; and the script parses `line[:-2]`, which fails on the dump's last entity line, which has no trailing comma, and skips it, where `wikisieve` reads that line.

### Speed against wikibase-dump-filter, and what bounds each tool

The parity experiment above establishes that both tools select the same 3,434,538 entities from the 2022-01-03 dump. This times them on one machine, and places each against the floors beneath it: the raw sequential read of the volume it ran on, and the decompressor feeding it.

Machine: 2 Intel Xeon E5-2640 v3 (16 cores, 32 threads, 2.6 GHz), 64 GB, Windows 11 26200 with WSL 2.7.14, Ubuntu 22.04.5, kernel 6.18.33.2. Two volumes: C: is a Samsung NVMe SSD 960; the other is an `ext4` volume in a vhdx on a Storage Spaces pool over two WD Red WD40EFRX spinning drives. Tools: `wikisieve` 0.1.0 (commit `5a64b617` on Linux, `e1fef52d` on Windows; the two commits differ only in documentation, so the code is identical), rustc 1.97.0, wikibase-dump-filter 6.1.1 under Node v22.23.2, `gzip` 1.10, `pigz` 2.6, `rapidgzip-rust` 0.3.1.

Every Linux leg ran inside WSL on the `ext4` volume, reading and writing there, so no leg crossed `/mnt/c`. Three runs of each `wikisieve` configuration; wikibase-dump-filter ran once here and is paired with the parity experiment's run on the same dump.

Two things beyond `pigz` having no Windows build put the Linux legs in WSL, and both bear on what the figures support. Linux reports per-thread CPU directly, where the Windows legs above had to poll `Process.Threads` for the lowest native thread id, so a core count attributed to the decoder is measured rather than inferred. And the dump is held on an `ext4` vhdx rather than reached through `/mnt/c`, so the mount's cost, 17 times at a 1 MiB read, sits inside no Linux figure here.

What that leaves unseparated is platform from storage. Every Linux leg here ran on the pool's `ext4` vhdx, so the Windows-against-Linux comparison below sets a Windows run on the NVMe against a Linux run on the pool and the two move together. A second `ext4` vhdx now holds the same dump on the NVMe, beside the pool's, so the separation is available to a later run although no figure here uses it: against the pool vhdx it isolates storage with the platform fixed, against the Windows run on the same drive it isolates the platform with the storage fixed, and against `/mnt/c` on that drive it isolates what the mount costs with both fixed. The `/mnt/c` figures recorded here are taken against native Windows instead, so they carry the platform difference inside them.

| leg | runs | mean wall | lines/s | peak RSS | output |
| --- | --- | --- | --- | --- | --- |
| `wikisieve` projection | 1,376.8 / 1,251.1 / 1,270.6 s | 1,299.5 s | 73,800 | 1.07 GB | 927 MB |
| `wikisieve` `--raw-candidate-output` | 1,474.8 / 1,404.8 / 1,391.0 s | 1,423.5 s | 67,400 | 1.07 GB | 927 MB + 4.14 GB gzipped |
| wikibase-dump-filter | 22,731.5 s here, 24,819 s in the parity experiment | - | 4,220 | 292 MB | 37.6 GB |

`wikisieve` is 17.5 times faster than the filter on the projection and 16.0 times on whole entities, against the run measured here; against the mean of the two filter runs, 18.3 and 16.7. The two filter runs are 8.4% apart, inside the projection's own 10.1% spread, and the parity experiment's read over `/mnt/c` where this one read `ext4`, which is evidence that the mount did not bind it.

Only the `--raw-candidate-output` leg is a subsetting comparison: it writes every matched entity's whole line as the filter does, though gzipped, so compression cost falls on `wikisieve`'s side. The projection leg is an extraction comparison, writing 927 MB of chosen fields against 37.6 GB.

Commands, run from the volume:

```
wikisieve --input wikidata-20220103-all.json.gz --spec gene_protein_disease_chemicals.json \
  --output ws-projection.jsonl --summary-json ws-projection-summary.json
wikisieve --input wikidata-20220103-all.json.gz --spec gene_protein_disease_chemicals.json \
  --output ws-raw.jsonl --raw-candidate-output ws-raw-raw.jsonl.gz --summary-json ws-raw-summary.json
cat wikidata-20220103-all.json.gz | gzip -d | wikibase-dump-filter --claim P31:Q11173,Q12136,Q7187,Q8054 > wdf.ndjson
```

#### The floors: raw read, then decompression

Raw sequential read, 8 GiB of real bytes, cache bypassed on every path (`O_DIRECT` under WSL, `FILE_FLAG_NO_BUFFERING` on Windows), one program on all three, three reads each, MB/s:

| path | 64 KiB | 1 MiB | 8 MiB |
| --- | --- | --- | --- |
| Windows on C: (NVMe) | 643 | 1,942 | 2,094 |
| WSL through `/mnt/c` (same NVMe) | 68 | 113 | 118 |
| WSL on `ext4` (vhdx on the spinning pool) | 92 | 97 | 97 |

The `/mnt/c` mount costs 17 times at 1 MiB against Windows reading the same file on the same device. The `ext4` volume is flat across block sizes at 97 MB/s, so no read size helps there. These are unbuffered reads of a separate file; the parity experiment's 166 MB/s for this volume came from one buffered 4 GiB read inside the dump, and the two are not measuring the same thing.

Decompression alone over the full dump into `/dev/null`, back to back on the `ext4` volume:

| decompressor | wall | compressed MB/s | cores |
| --- | --- | --- | --- |
| `gzip -dc` | 6,080.7 s | 17.9 | 0.98 |
| `pigz -dc` | 6,937.5 s | 15.7 | 1.29 |
| `rapidgzip-rust -c -P 1` | 1,128.5 s | 96.6 | 0.99 |
| `rapidgzip-rust -c -P 2` | 1,131.7 s | 96.4 | 0.99 |
| `rapidgzip-rust -c -P 4` | 1,127.9 s | 96.7 | 0.99 |
| `rapidgzip-rust -c -P 8` | 1,131.9 s | 96.3 | 0.99 |
| `rapidgzip-rust -c -P 16` | 1,139.2 s | 95.7 | 0.99 |
| `rapidgzip-rust -c -P 0` (32) | 1,121.7 s | 97.2 | 0.99 |

`pigz` is slower than `gzip` on this file and spends 4,366.9 s in the kernel against `gzip`'s 141.7 s: the dump is one continuous member, so its threads have nothing to split and only add handoff.

`rapidgzip-rust -l -P 16` counts 95,900,307 newlines and reports 1,428,353,996,731 decompressed bytes, which is the 95,900,305 entity lines `wikisieve` scans plus the dump's opening `[` and closing `]`.

#### The ladder, and what actually limits each tool

Same dump, same build, each rung measured:

| path | raw read | `rapidgzip-rust -P 0` | `wikisieve` projection | wikibase-dump-filter |
| --- | --- | --- | --- | --- |
| WSL on `ext4` | 1,124 s (97 MB/s) | 1,121.7 s (0.93 cores) | 1,299.5 s (1.90 cores) | 22,731.5 s (1.02 cores) |
| Windows on C: | 52 s (2,090 MB/s) | 1,074.3 s (0.99 cores) | 1,646.9 s (1.71 cores) | not run |

As multiples of its own decompression floor, `wikisieve` is 1.16 times the `rapidgzip-rust` floor for the projection and 1.27 times with raw output; wikibase-dump-filter is 3.74 times its `gzip -d` floor, which includes writing 37.6 GB where the projection writes 927 MB. As multiples of the raw read floor of the volume each ran on, `wikisieve` is 1.16 times on `ext4` and 31.6 times on C:, and the filter 20.2 times on `ext4`.

The decode is single-threaded on this dump. `rapidgzip-rust` reports a 32-worker budget and 1,041 framing units, then takes 0.99 cores and about 100 MB/s of compressed input on both volumes, whose raw read rates differ by 21 times. That it matched the `ext4` volume's 97 MB/s floor is a coincidence, and the Windows run is what separates them. Per-thread sampling of the `wikisieve` runs shows the same from inside: `rapidgzip-coord` at 1,322 CPU-s over a 1,377 s wall, 96% of one core, with the 32 parse threads sharing 1,117 s. So `wikisieve`'s 1,299.5 s is a 1,074 s single-threaded decode plus about 225 s of everything else, which is why no threaded rearrangement of the scan could win when five were tried. The cause is the crate, not the framing: `rapidgzip-core`'s admission screen decodes the dump's 22-byte first gzip member, reaches its end and selects the sequential decoder before measuring anything, so every `-P` row above runs the same path. Overriding that choice decodes a cached 652 MB slice 1.61 times faster with identical output.

Both tools on one thread over a 200,000-line sample, each behind the same `gzip -d`, separates per-entity work from parallelism: wikibase-dump-filter 70.0 s wall and 73.3 s of CPU, `wikisieve` 26.9 s wall and 3.8 s of CPU under `RAYON_NUM_THREADS=1`, both keeping the same 886 entities. The filter does about 19 times the work per entity; the wall ratio understates it because `wikisieve`'s side of that pipeline is waiting on `gzip`.

#### Against the evaluation's timing table

The evaluation ran on 2 AMD EPYC 7302 CPUs with a spinning disk. Its times as ratios to its own WDF time, with `wikisieve` placed through the ratio measured here:

| tool | evaluation's time | ratio to WDF |
| --- | --- | --- |
| WDF | 13,876 s | 1.00 |
| KGTK | 17,148 s | 1.24 |
| WDumper | 23,427 s | 1.69 |
| WDSub | 43,060 s | 3.10 |
| `wikisieve`, whole entities | - | 0.063 |
| `wikisieve`, projection | - | 0.057 |

WDF took 22,731.5 s here against its published 13,876 s. It is one Node thread at 4.8 MB/s of compressed input, 5% of the slowest floor measured, so neither storage nor the mount explains the gap; the likeliest cause is per-core speed, the E5-2640 v3 being a 2014 Haswell against the EPYC's 2019 Zen 2, and that is measured rather than assumed only by running the same sample elsewhere.
