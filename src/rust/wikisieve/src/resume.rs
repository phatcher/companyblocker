//! On-disk resume checkpoint machinery -- ported from `src/rust/wikidata/main.rs`'s already-fixed
//! design: tying both events to the same batch boundary is what makes a
//! `resume-state.json` this run persists always correspond to output that's already durable on
//! disk, never to rows still sitting in an in-progress chunk's write buffer.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::extract::Config;
use crate::output_sink::sync_parent_dir;
use crate::scan::Stats;

/// On-disk resume checkpoint, written after each completed batch alongside that same batch's
/// finalized chunk (see `process_dump`).
#[derive(Default, Deserialize, Serialize)]
pub(crate) struct ResumeState {
    pub(crate) lines_scanned: usize,
    pub(crate) candidates_scanned: usize,
    pub(crate) records_emitted: usize,
    pub(crate) skipped_no_properties: usize,
    pub(crate) skipped_parse_errors: usize,
    pub(crate) next_chunk_index: usize,
}

pub(crate) fn load_resume_state(config: &Config) -> io::Result<ResumeState> {
    if !config.resume_enabled {
        return Ok(ResumeState::default());
    }

    let Some(state_path) = config.state_path.as_ref() else {
        return Ok(ResumeState::default());
    };

    if !state_path.exists() {
        return Ok(ResumeState::default());
    }

    let bytes = fs::read(state_path)?;
    let state: ResumeState = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to parse resume state {}: {error}",
                state_path.display()
            ),
        )
    })?;
    eprintln!(
        "[wikisieve] resume state loaded lines_scanned={} records_emitted={} next_chunk_index={}",
        state.lines_scanned, state.records_emitted, state.next_chunk_index
    );
    Ok(state)
}

pub(crate) fn maybe_write_resume_state(
    config: &Config,
    stats: &Stats,
    next_chunk_index: usize,
) -> io::Result<()> {
    if !config.resume_enabled {
        return Ok(());
    }
    let Some(state_path) = config.state_path.as_ref() else {
        return Ok(());
    };

    if let Some(parent) = state_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let payload = ResumeState {
        lines_scanned: stats.lines_scanned,
        candidates_scanned: stats.candidates_scanned,
        records_emitted: stats.records_emitted,
        skipped_no_properties: stats.skipped_no_properties,
        skipped_parse_errors: stats.skipped_parse_errors,
        next_chunk_index,
    };
    let encoded = serde_json::to_vec_pretty(&payload).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to serialize resume state {}: {error}",
                state_path.display()
            ),
        )
    })?;

    // Written to a sibling temp file, synced, then renamed over the real path -- the same
    // write-sync-rename shape as a chunk or a merge output, and for the same reason: this is the
    // record that a resume trusts to say how much output already exists, so its own on-disk
    // write must never claim more than a power loss can leave behind, nor land as a half-written
    // file if the process dies mid-write.
    let temp_path = resume_state_temp_path(state_path);
    write_and_sync(&temp_path, &encoded)?;
    fs::rename(&temp_path, state_path)?;
    sync_parent_dir(state_path)
}

/// `<state-path>.tmp`, a sibling of the resume state file rather than a system temp file, so the
/// rename that commits a checkpoint is same-filesystem and therefore atomic.
fn resume_state_temp_path(state_path: &Path) -> PathBuf {
    let mut temp_path = state_path.as_os_str().to_owned();
    temp_path.push(".tmp");
    PathBuf::from(temp_path)
}

fn write_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Renames `resume-state.json` to `resume-state.completed.json` rather than deleting it: it is
/// tiny metadata, not a multi-GB artifact, and no reuse decision anywhere reads it (a caller
/// checks the destination file's own existence), so keeping the record that this run reached a
/// verified merge costs nothing and is worth having when debugging the next incident in this
/// class of code.
pub(crate) fn mark_resume_state_completed(state_path: &Path) {
    if !state_path.exists() {
        return;
    }
    let Some(marker_path) = resume_state_completed_marker_path(state_path) else {
        return;
    };
    // The state file's content is already synced -- `maybe_write_resume_state` guarantees that --
    // so only the rename itself needs the follow-up directory sync, on the same basis as every
    // other rename in the chunk/merge/resume paths.
    let result = fs::rename(state_path, &marker_path).and_then(|()| sync_parent_dir(&marker_path));
    report_cleanup_failure(result, state_path);
}

pub(crate) fn resume_state_completed_marker_path(state_path: &Path) -> Option<PathBuf> {
    let mut marker_name = state_path.file_stem()?.to_owned();
    marker_name.push(".completed.json");
    Some(state_path.with_file_name(marker_name))
}

pub(crate) fn report_cleanup_failure(result: io::Result<()>, path: &Path) {
    if let Err(error) = result {
        eprintln!(
            "[wikisieve] merge-chunks could not clean up {}: {error}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::Stats;
    use crate::test_support::{config_for, path_arg, ScopedTempDir};

    #[test]
    fn resume_state_completed_marker_path_appends_beside_the_state_file() {
        let marker_path =
            resume_state_completed_marker_path(Path::new("runs/wikidata/resume-state.json"))
                .unwrap();
        assert_eq!(
            marker_path,
            PathBuf::from("runs/wikidata/resume-state.completed.json")
        );
    }

    fn resume_config(state_path: &Path) -> Config {
        config_for(&[
            "--input",
            "unused.jsonl.gz",
            "--spec",
            "unused-spec.json",
            "--output",
            "unused.jsonl",
            "--resume",
            "--chunk-dir",
            "unused-chunk-dir",
            "--state-path",
            &path_arg(state_path),
        ])
    }

    // -----------------------------------------------------------------------------------------
    // `maybe_write_resume_state`: write-to-temp, sync, then rename onto the real path.
    // Every assertion reads the checkpoint back rather than just checking the call returned Ok,
    // for the same reason `output_sink`'s chunk tests do: a durable-looking write that silently
    // lands wrong content is exactly the failure mode this item exists to close off.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn maybe_write_resume_state_writes_the_checkpoint_and_leaves_no_temp_file_behind() {
        let dir = ScopedTempDir::new("resume-state-write");
        let state_path = dir.path().join("resume-state.json");
        let config = resume_config(&state_path);
        let stats = Stats {
            lines_scanned: 10,
            candidates_scanned: 8,
            records_emitted: 3,
            skipped_no_properties: 2,
            skipped_parse_errors: 1,
        };

        maybe_write_resume_state(&config, &stats, 4).unwrap();

        assert!(state_path.exists());
        assert!(!resume_state_temp_path(&state_path).exists());
        let loaded = load_resume_state(&config).unwrap();
        assert_eq!(loaded.lines_scanned, 10);
        assert_eq!(loaded.candidates_scanned, 8);
        assert_eq!(loaded.records_emitted, 3);
        assert_eq!(loaded.skipped_no_properties, 2);
        assert_eq!(loaded.skipped_parse_errors, 1);
        assert_eq!(loaded.next_chunk_index, 4);
    }

    #[test]
    fn maybe_write_resume_state_overwrites_the_previous_checkpoint_and_no_stale_temp_survives() {
        let dir = ScopedTempDir::new("resume-state-overwrite");
        let state_path = dir.path().join("resume-state.json");
        let config = resume_config(&state_path);

        maybe_write_resume_state(&config, &Stats::default(), 1).unwrap();
        let second_stats = Stats {
            lines_scanned: 100_000,
            records_emitted: 42,
            ..Stats::default()
        };
        maybe_write_resume_state(&config, &second_stats, 2).unwrap();

        // The rename is over the same destination name every time, so a second checkpoint
        // replaces the first rather than appending or leaving the first version's bytes behind
        // under the temp name.
        let loaded = load_resume_state(&config).unwrap();
        assert_eq!(loaded.lines_scanned, 100_000);
        assert_eq!(loaded.records_emitted, 42);
        assert_eq!(loaded.next_chunk_index, 2);
        assert!(!resume_state_temp_path(&state_path).exists());
    }

    #[test]
    fn mark_resume_state_completed_leaves_the_marker_content_unchanged_by_the_rename() {
        let dir = ScopedTempDir::new("resume-state-completed");
        let state_path = dir.path().join("resume-state.json");
        let config = resume_config(&state_path);
        maybe_write_resume_state(&config, &Stats::default(), 1).unwrap();
        let before = fs::read(&state_path).unwrap();

        mark_resume_state_completed(&state_path);

        assert!(!state_path.exists());
        let marker_path = resume_state_completed_marker_path(&state_path).unwrap();
        assert_eq!(fs::read(&marker_path).unwrap(), before);
    }
}
