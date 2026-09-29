//! Batched, parallel scan -- ported from `src/rust/wikidata/main.rs`'s `fill_wikidata_batch`/
//! `process_dump`. Decompression + the byte-level prefilter stay sequential
//! (`fill_wikidata_batch`); the CPU-bound parse/match/project work (`project_candidate`) runs in
//! parallel per batch via `rayon`. Batch size is keyed to `DEFAULT_PROGRESS_EVERY` *scanned*
//! lines, which is also the resume-checkpoint cadence -- chunk finalization and the resume
//! checkpoint both happen right after the same batch completes, so a kill (or a power loss)
//! can only ever cost the single in-flight batch, never leave a resume-state.json
//! pointing past data that isn't durably on disk yet: every chunk, raw companion and resume-state
//! write is fsynced before the rename that makes it visible under its final name, and (on Unix)
//! the containing directory is fsynced after, so a killed process and an OS crash or power loss
//! give the same guarantee -- the difference between the two is only how much of the in-flight
//! batch is lost, never whether the checkpoint can claim data that a restart won't find.
//!
//! Shared by `extract` (which writes a projected record per row) and `profile` (which
//! accumulates a schema report instead): "which lines are candidates" has to agree between the
//! two, so the decompress + prefilter + batched-parallel-parse machinery lives here once.

use serde::Serialize;
use std::io::{self, BufRead};

use crate::{trim_wikidata_line, Prefilter};

/// Batch size for both the resume-checkpoint cadence and the `rayon` parallel-parse unit --
/// ported from `src/rust/wikidata/main.rs`'s `DEFAULT_PROGRESS_EVERY`, same value.
pub(crate) const DEFAULT_PROGRESS_EVERY: usize = 100_000;

#[derive(Default, Serialize)]
pub(crate) struct Stats {
    pub(crate) lines_scanned: usize,
    pub(crate) candidates_scanned: usize,
    pub(crate) records_emitted: usize,
    pub(crate) skipped_no_properties: usize,
    pub(crate) skipped_parse_errors: usize,
}

/// Reads and prefilters lines sequentially, collecting up to `DEFAULT_PROGRESS_EVERY` scanned
/// lines' worth of candidate lines into `batch` for parallel parsing by the caller. Returns
/// `true` once the source is exhausted or `max_rows` is reached (caller should stop after
/// processing the returned batch).
///
/// Generic over `R: BufRead` rather than the one concrete gzip-file reader type: the
/// same prefilter/batch machinery reads either the gzip-compressed dump file or, for `--input
/// -`, uncompressed newline-delimited entities from standard input (`crate::extract::InputSource`).
///
/// A line is a candidate only if `prefilter` admits it -- both of its stages have to
/// agree: does it carry a marker property literal at all, and does it carry a marker QID literal
/// anywhere. Each is independently a necessary condition for a real match, so requiring both is
/// still conservative -- never rejects a line `project_candidate`/`profile_candidate` would have
/// matched -- while being far more selective than either alone.
pub(crate) fn fill_wikidata_batch<R: BufRead>(
    reader: &mut R,
    raw_line: &mut Vec<u8>,
    batch: &mut Vec<Vec<u8>>,
    lines_to_skip: &mut usize,
    prefilter: &Prefilter<'_>,
    stats: &mut Stats,
    max_rows: Option<usize>,
) -> io::Result<bool> {
    batch.clear();
    let mut batch_lines_scanned = 0usize;

    loop {
        raw_line.clear();
        let bytes_read = reader.read_until(b'\n', raw_line)?;
        if bytes_read == 0 {
            return Ok(true);
        }

        let trimmed = trim_wikidata_line(raw_line);
        if trimmed.is_empty() {
            continue;
        }

        if *lines_to_skip > 0 {
            *lines_to_skip -= 1;
            continue;
        }

        stats.lines_scanned += 1;
        batch_lines_scanned += 1;

        if prefilter.admits(trimmed) {
            stats.candidates_scanned += 1;
            batch.push(trimmed.to_vec());
        } else {
            stats.skipped_no_properties += 1;
        }

        if reached_limit(stats.lines_scanned, max_rows) {
            eprintln!(
                "[wikisieve] stopping early after max_rows={} content rows",
                stats.lines_scanned
            );
            return Ok(true);
        }

        if batch_lines_scanned >= DEFAULT_PROGRESS_EVERY {
            return Ok(false);
        }
    }
}

pub(crate) fn reached_limit(count: usize, limit: Option<usize>) -> bool {
    match limit {
        Some(limit) => count >= limit,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reached_limit_reports_false_when_unbounded() {
        assert!(!reached_limit(1_000_000, None));
    }

    #[test]
    fn reached_limit_reports_false_below_bound() {
        assert!(!reached_limit(5, Some(10)));
    }

    #[test]
    fn reached_limit_reports_true_at_and_above_bound() {
        assert!(reached_limit(10, Some(10)));
        assert!(reached_limit(11, Some(10)));
    }
}
