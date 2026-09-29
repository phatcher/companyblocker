//! `merge-chunks` -- assembling a completed run's chunk directory into the flat destination file.
//!
//! Chunks are a resumability mechanism, not the final artifact, so something has to concatenate
//! them once a run finishes. Until now that step existed only as this repo's Python pipeline glue
//! (`_merge_wikidata_chunk_dir_into_destination` in `downloader_wikidata.py`), which left a caller
//! holding just the binary with no way to finish a resumable run. The sequence below is that
//! function's, natively: concatenate in chunk-index order, reverify the merged row count against
//! the run's own reported total, rename onto the destination only after that check passes, then
//! clean up.
//!
//! The row-count reverification is what makes the rename safe to do at all -- a partial or
//! mis-ordered merge must never replace a destination file that is already good, so everything is
//! written to a sibling temp file and the destination is untouched until the count agrees.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::cli::{next_arg_value, parse_usize_arg, print_usage, require_arg};
use crate::output_sink::{
    count_non_empty_gz_lines, create_parent_dir, iter_existing_chunk_files,
    iter_existing_raw_chunk_files, iter_existing_temp_files, raw_chunk_path, sync_parent_dir,
    DEFAULT_CHUNK_PREFIX,
};
use crate::resume::{mark_resume_state_completed, report_cleanup_failure};

/// Settings for the `merge-chunks` subcommand -- assembling a completed run's chunk directory
/// into the flat destination file (see `run_merge_chunks`).
#[derive(Debug)]
pub struct MergeConfig {
    pub(crate) chunk_dir: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) chunk_prefix: String,
    pub(crate) summary_json_path: Option<PathBuf>,
    pub(crate) state_path: Option<PathBuf>,
    pub(crate) expected_rows: Option<usize>,
    /// `--raw-candidate-output`: where the run's raw-candidate companions are assembled to. Unset
    /// means the companions are neither merged nor removed -- see `cleanup_after_merge`.
    pub(crate) raw_candidate_output: Option<PathBuf>,
}

impl MergeConfig {
    /// Parses the `merge-chunks` subcommand's flags.
    ///
    /// # Errors
    /// Returns an error if a required flag (`--chunk-dir`, `--output`) is missing, if a flag's
    /// value fails to parse, if an unknown argument is given, if no expected-row-count source
    /// (`--summary-json`, `--state-path`, or `--expected-rows`) is given, or if `--output` or
    /// `--raw-candidate-output` names stdout (`-`).
    pub fn from_args(args: impl Iterator<Item = String>) -> io::Result<Self> {
        let mut chunk_dir: Option<PathBuf> = None;
        let mut output_path: Option<PathBuf> = None;
        let mut chunk_prefix = DEFAULT_CHUNK_PREFIX.to_string();
        let mut summary_json_path: Option<PathBuf> = None;
        let mut state_path: Option<PathBuf> = None;
        let mut expected_rows: Option<usize> = None;
        let mut raw_candidate_output: Option<PathBuf> = None;

        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--chunk-dir" => {
                    chunk_dir = Some(PathBuf::from(next_arg_value(&mut args, "--chunk-dir")?));
                }
                "--output" => {
                    output_path = Some(PathBuf::from(next_arg_value(&mut args, "--output")?));
                }
                "--chunk-prefix" => {
                    chunk_prefix = next_arg_value(&mut args, "--chunk-prefix")?;
                }
                "--summary-json" => {
                    summary_json_path =
                        Some(PathBuf::from(next_arg_value(&mut args, "--summary-json")?));
                }
                "--state-path" => {
                    state_path = Some(PathBuf::from(next_arg_value(&mut args, "--state-path")?));
                }
                "--expected-rows" => {
                    expected_rows = Some(parse_usize_arg(
                        &next_arg_value(&mut args, "--expected-rows")?,
                        "--expected-rows",
                    )?);
                }
                "--raw-candidate-output" => {
                    raw_candidate_output = Some(PathBuf::from(next_arg_value(
                        &mut args,
                        "--raw-candidate-output",
                    )?));
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("unknown argument: {other}"),
                    ));
                }
            }
        }

        let resolved = Self {
            chunk_dir: require_arg(chunk_dir, "--chunk-dir")?,
            output_path: require_arg(output_path, "--output")?,
            chunk_prefix,
            summary_json_path,
            state_path,
            expected_rows,
            raw_candidate_output,
        };

        // The row-count reverification is the whole safety property of this mode, so a caller
        // cannot opt out of it by simply naming no expected total. One of the three sources has
        // to be present; a resumable run always leaves at least `--state-path` behind, since
        // `--resume` requires it.
        if resolved.expected_rows.is_none()
            && resolved.summary_json_path.is_none()
            && resolved.state_path.is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "merge-chunks needs an expected row count: pass --summary-json, --state-path \
                 or --expected-rows",
            ));
        }

        // `-` means stdout for the extract mode's `--output`, but this mode's guarantee is
        // write-to-temp-then-rename onto a real path -- there is nothing to rename onto for a
        // stream, so accepting `-` here would quietly create a file literally named `-`.
        if resolved.output_path.as_os_str() == "-" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "merge-chunks --output must be a file path, not '-' (stdout)",
            ));
        }
        if resolved
            .raw_candidate_output
            .as_ref()
            .is_some_and(|path| path.as_os_str() == "-")
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "merge-chunks --raw-candidate-output must be a file path, not '-' (stdout)",
            ));
        }

        Ok(resolved)
    }
}

/// Assembles a completed `--resume` run's chunk directory into the flat destination named by
/// `config.output_path`, and the raw-candidate companions into `config.raw_candidate_output` if
/// requested, verifying the merged row count against the run's own reported total first.
///
/// # Errors
/// Returns an error if the expected row count can't be resolved, if a chunk or companion file
/// can't be read, or if the merged row count disagrees with the expected total.
pub fn run_merge_chunks(config: &MergeConfig) -> io::Result<()> {
    let expected_rows = resolve_expected_rows(config)?;
    eprintln!(
        "[wikisieve] merge-chunks chunk_dir={} output={} prefix={} expected_rows={}",
        config.chunk_dir.display(),
        config.output_path.display(),
        config.chunk_prefix,
        expected_rows,
    );

    let chunks = iter_existing_chunk_files(&config.chunk_dir, &config.chunk_prefix)?;
    let merged_rows = merge_chunks_into_destination(config, &chunks, expected_rows)?;
    let merged_raw_rows = merge_raw_companions_into_destination(config, &chunks, expected_rows)?;
    cleanup_after_merge(config, &chunks, merged_raw_rows.is_some());

    eprintln!(
        "[wikisieve] merge-chunks complete merged_rows={} output={} raw_candidate_rows={}",
        merged_rows,
        config.output_path.display(),
        merged_raw_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string()),
    );
    Ok(())
}

/// Assembles the run's raw-candidate companions into one gzip artifact at
/// `--raw-candidate-output`, returning the row count it verified -- or `None` when no
/// raw-candidate output was asked for.
///
/// Runs after the projected destination is already committed. That ordering costs nothing: the
/// projected merge verifies itself, so a failure here leaves a correct destination file plus every
/// chunk and companion still in place (`cleanup_after_merge` has not run yet), and re-running
/// `merge-chunks` repeats both halves from the same inputs.
fn merge_raw_companions_into_destination(
    config: &MergeConfig,
    chunks: &[(usize, PathBuf)],
    expected_rows: usize,
) -> io::Result<Option<usize>> {
    let Some(output_path) = config.raw_candidate_output.as_ref() else {
        return Ok(None);
    };
    create_parent_dir(output_path)?;

    let temp_path = merge_temp_path(output_path);
    let merged_rows = match write_merged_raw_companions(config, chunks, &temp_path) {
        Ok(merged_rows) => merged_rows,
        Err(error) => {
            fs::remove_file(&temp_path).ok();
            return Err(error);
        }
    };

    // The same equality check the projected merge makes, and for the same reason -- but it is
    // also the only thing standing between a caller and a cache that reprojects short. Every row
    // in the cache is one emitted row's dump line, so the two counts are equal by construction and
    // any inequality means companions are missing, duplicated or from another run.
    if merged_rows != expected_rows {
        fs::remove_file(&temp_path).ok();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "raw-candidate merge row count mismatch: \
                 merged={merged_rows} expected={expected_rows}"
            ),
        ));
    }

    fs::rename(&temp_path, output_path)?;
    sync_parent_dir(output_path)?;
    Ok(Some(merged_rows))
}

/// Concatenates each chunk's companion into `temp_path` in chunk-index order, returning the number
/// of rows the result decompresses to.
///
/// The copy is byte for byte, with no decode in between: gzip members concatenate, so the
/// resulting file is one valid multi-member stream that `MultiGzDecoder` -- and therefore this
/// binary's own `--input` -- reads as a single sequence of lines. The row count is then taken by
/// decoding the assembled file once, since a gzip stream's length says nothing about how many
/// rows are inside it.
fn write_merged_raw_companions(
    config: &MergeConfig,
    chunks: &[(usize, PathBuf)],
    temp_path: &Path,
) -> io::Result<usize> {
    let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(temp_path)?);
    for (index, chunk_path) in chunks {
        let companion_path = raw_chunk_path(&config.chunk_dir, &config.chunk_prefix, *index);
        let mut reader = BufReader::with_capacity(
            1024 * 1024,
            File::open(&companion_path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "chunk {} has no raw-candidate companion at {}: {error} (the run that \
                         wrote {} did not capture raw candidates)",
                        index,
                        companion_path.display(),
                        chunk_path.display(),
                    ),
                )
            })?,
        );
        io::copy(&mut reader, &mut writer)?;
    }
    writer.flush()?;
    // Synced while still a plain `.tmp` file, before the caller's row-count check can clear it
    // for the rename onto the destination -- the same requirement as every other rename in this
    // crate's chunk/merge/resume paths.
    writer.get_ref().sync_all()?;
    drop(writer);

    count_non_empty_gz_lines(temp_path)
}

/// Resolves how many rows the merged file must contain, from whichever of the three sources the
/// caller supplied. `--expected-rows` is the explicit override and wins outright; otherwise the
/// count comes from a JSON file's `records_emitted`, which both `--summary-json` (`Stats`) and
/// `--state-path` (`ResumeState`) carry under that same name.
///
/// Supplying both is not redundant: they are written at different moments -- the summary once the
/// run ends, the resume state after each completed batch -- so a disagreement between them means
/// the two files are not from the same run, and merging against either would be guesswork.
fn resolve_expected_rows(config: &MergeConfig) -> io::Result<usize> {
    if let Some(expected_rows) = config.expected_rows {
        return Ok(expected_rows);
    }

    let from_summary = match config.summary_json_path.as_ref() {
        Some(path) => Some(read_records_emitted(path)?),
        None => None,
    };
    let from_state = match config.state_path.as_ref() {
        Some(path) if path.exists() => Some(read_records_emitted(path)?),
        _ => None,
    };

    match (from_summary, from_state) {
        (Some(summary_rows), Some(state_rows)) if summary_rows != state_rows => {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expected row count disagrees between sources: \
                     summary={summary_rows} state={state_rows}"
                ),
            ))
        }
        (Some(rows), _) | (None, Some(rows)) => Ok(rows),
        (None, None) => Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no expected row count available: --state-path does not exist and neither \
             --summary-json nor --expected-rows was given",
        )),
    }
}

fn read_records_emitted(path: &Path) -> io::Result<usize> {
    let bytes = fs::read(path)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse {}: {error}", path.display()),
        )
    })?;
    value
        .get("records_emitted")
        .and_then(serde_json::Value::as_u64)
        .and_then(|records_emitted| usize::try_from(records_emitted).ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} has no integer records_emitted field", path.display()),
            )
        })
}

/// `<output>.tmp`, a sibling of the destination rather than a system temp file, so the rename that
/// commits the merge is same-filesystem and therefore atomic.
fn merge_temp_path(output_path: &Path) -> PathBuf {
    let mut temp_path = output_path.as_os_str().to_owned();
    temp_path.push(".tmp");
    PathBuf::from(temp_path)
}

fn merge_chunks_into_destination(
    config: &MergeConfig,
    chunks: &[(usize, PathBuf)],
    expected_rows: usize,
) -> io::Result<usize> {
    if let Some(parent) = config.output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let temp_path = merge_temp_path(&config.output_path);
    let merged_rows = match write_merged_chunks(chunks, &temp_path) {
        Ok(merged_rows) => merged_rows,
        Err(error) => {
            fs::remove_file(&temp_path).ok();
            return Err(error);
        }
    };

    if merged_rows != expected_rows {
        fs::remove_file(&temp_path).ok();
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "chunk merge row count mismatch: merged={merged_rows} expected={expected_rows}"
            ),
        ));
    }

    fs::rename(&temp_path, &config.output_path)?;
    sync_parent_dir(&config.output_path)?;
    Ok(merged_rows)
}

/// Concatenates `chunks` into `temp_path` byte for byte, returning the number of rows written.
///
/// Blank lines are dropped and a chunk whose last row has no trailing newline still gets one, so
/// the merged file's row count is exactly what `count_non_empty_lines` reports across the chunks
/// -- the same normalization the Python implementation applies, and what makes the reverification
/// above an equality check rather than an approximation.
fn write_merged_chunks(chunks: &[(usize, PathBuf)], temp_path: &Path) -> io::Result<usize> {
    let mut writer = BufWriter::with_capacity(1024 * 1024, File::create(temp_path)?);
    let mut merged_rows = 0usize;
    let mut raw_line: Vec<u8> = Vec::with_capacity(64 * 1024);

    for (_, chunk_path) in chunks {
        let mut reader = BufReader::with_capacity(1024 * 1024, File::open(chunk_path)?);
        loop {
            raw_line.clear();
            if reader.read_until(b'\n', &mut raw_line)? == 0 {
                break;
            }
            if raw_line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            writer.write_all(&raw_line)?;
            if raw_line.last() != Some(&b'\n') {
                writer.write_all(b"\n")?;
            }
            merged_rows += 1;
        }
    }

    writer.flush()?;
    // Synced before the row-count check that clears this temp file for its rename onto the
    // destination -- see the identical requirement on the raw-companion merge above.
    writer.get_ref().sync_all()?;
    Ok(merged_rows)
}

/// Drops what the merge made redundant: the chunks whose content now lives, verified, inside the
/// destination file, then the chunk directory itself, then the run's live resume state.
///
/// Nothing here can fail the command. By the time this runs the destination is already committed
/// and its row count already checked, so a non-zero exit would tell the caller the merge did not
/// happen when it did -- and the natural response, re-running `merge-chunks`, would then fail
/// against a chunk directory that is half cleaned up. Failures are reported and left; every one
/// of them leaves behind redundant files, never missing ones. This is the same posture as the
/// Python implementation's `shutil.rmtree(..., ignore_errors=True)`, extended to cover its
/// unguarded resume-state rename too.
///
/// Two deliberate narrowings against that implementation, which removes the chunk directory
/// wholesale: only the chunk files this merge actually consumed are removed, plus this run's own
/// abandoned `.tmp` files, and the directory goes only if that leaves it empty. A chunk directory
/// is not necessarily the run's exclusive property -- the fresh-start path already assumes it may
/// not be -- and a recursive delete of somebody else's files, immediately after a merge that
/// never read them, is the shape of loss already produced once in this crate's sibling.
///
/// `merged_raw` extends that same narrowing to the raw-candidate companions: they go only when
/// this merge actually assembled them into an artifact. Without `--raw-candidate-output` they are
/// left where they are, which keeps the chunk directory behind too -- a visible leftover rather
/// than a silent deletion of the only copy of data that cost a full-dump scan to produce.
fn cleanup_after_merge(config: &MergeConfig, chunks: &[(usize, PathBuf)], merged_raw: bool) {
    for (index, chunk_path) in chunks {
        report_cleanup_failure(fs::remove_file(chunk_path), chunk_path);
        if merged_raw {
            let companion_path = raw_chunk_path(&config.chunk_dir, &config.chunk_prefix, *index);
            report_cleanup_failure(fs::remove_file(&companion_path), &companion_path);
        }
    }

    if !merged_raw {
        report_unmerged_raw_companions(config);
    }

    match iter_existing_temp_files(&config.chunk_dir, &config.chunk_prefix) {
        Ok(temp_paths) => {
            for temp_path in temp_paths {
                report_cleanup_failure(fs::remove_file(&temp_path), &temp_path);
            }
        }
        Err(error) => {
            eprintln!(
                "[wikisieve] merge-chunks could not list temp files in {}: {error}",
                config.chunk_dir.display()
            );
        }
    }

    // `remove_dir`, not `remove_dir_all`: it succeeds only on an empty directory, which is
    // exactly the condition under which removing it takes nothing with it.
    if let Err(error) = fs::remove_dir(&config.chunk_dir) {
        eprintln!(
            "[wikisieve] merge-chunks left {} in place: {error}",
            config.chunk_dir.display()
        );
    }

    if let Some(state_path) = config.state_path.as_ref() {
        mark_resume_state_completed(state_path);
    }
}

/// Says why the chunk directory is about to survive a successful merge, when it survives because
/// the run captured raw candidates and this merge was not asked to assemble them. Silence there
/// reads as an unexplained failure to clean up, when it is the deliberate refusal to delete an
/// artifact nobody claimed.
fn report_unmerged_raw_companions(config: &MergeConfig) {
    let Ok(companions) = iter_existing_raw_chunk_files(&config.chunk_dir, &config.chunk_prefix)
    else {
        return;
    };
    if companions.is_empty() {
        return;
    }
    eprintln!(
        "[wikisieve] merge-chunks kept {} raw-candidate companion(s) in {}: pass \
         --raw-candidate-output to assemble them",
        companions.len(),
        config.chunk_dir.display(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{run_extract, write_summary};
    use crate::resume::maybe_write_resume_state;
    use crate::scan::Stats;
    use crate::test_support::{
        args, as_dump_array, candidate_line, chunk_path, config_for, entry_names, ids_in,
        non_candidate_line, path_arg, read_file, read_gz_file, raw_chunk_ids, seed_chunk,
        seed_raw_companion, summary_field, temp_chunk_path, unmatched_candidate_line,
        write_extended_spec, write_gzipped, write_spec, ScopedTempDir, TEST_PREFIX,
    };
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn merge_config_for(values: &[&str]) -> MergeConfig {
        MergeConfig::from_args(args(values)).unwrap()
    }

    fn merge_config_error(values: &[&str]) -> io::Error {
        MergeConfig::from_args(args(values)).unwrap_err()
    }

    #[test]
    fn merge_config_parses_required_flags_with_defaults() {
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--state-path",
            "state.json",
        ]);
        assert_eq!(config.chunk_dir, PathBuf::from("chunks"));
        assert_eq!(config.output_path, PathBuf::from("out.jsonl"));
        assert_eq!(config.state_path, Some(PathBuf::from("state.json")));
        assert_eq!(config.summary_json_path, None);
        assert_eq!(config.expected_rows, None);
        // The default has to be the same prefix the extract mode writes under, or the common case
        // -- one run, then its merge -- would find no chunks at all.
        assert_eq!(config.chunk_prefix, DEFAULT_CHUNK_PREFIX);
    }

    #[test]
    fn merge_config_parses_all_optional_flags() {
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--chunk-prefix",
            "part-",
            "--summary-json",
            "summary.json",
            "--state-path",
            "state.json",
            "--expected-rows",
            "9",
        ]);
        assert_eq!(config.chunk_prefix, "part-");
        assert_eq!(
            config.summary_json_path,
            Some(PathBuf::from("summary.json"))
        );
        assert_eq!(config.state_path, Some(PathBuf::from("state.json")));
        assert_eq!(config.expected_rows, Some(9));
    }

    #[test]
    fn merge_config_rejects_missing_required_flags() {
        let error = merge_config_error(&["--output", "out.jsonl", "--expected-rows", "1"]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--chunk-dir"));

        let error = merge_config_error(&["--chunk-dir", "chunks", "--expected-rows", "1"]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--output"));
    }

    #[test]
    fn merge_config_requires_some_expected_row_source() {
        // Without one there is nothing to reverify the merge against, which is the only reason
        // the rename onto a destination is safe -- so this is refused rather than defaulted.
        let error = merge_config_error(&["--chunk-dir", "chunks", "--output", "out.jsonl"]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--summary-json"));
        assert!(error.to_string().contains("--state-path"));
        assert!(error.to_string().contains("--expected-rows"));
    }

    #[test]
    fn merge_config_rejects_stdout_as_the_destination() {
        let error = merge_config_error(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "-",
            "--expected-rows",
            "1",
        ]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("stdout"));
    }

    #[test]
    fn merge_config_rejects_unknown_and_valueless_arguments() {
        let error = merge_config_error(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--expected-rows",
            "1",
            "--resume",
        ]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--resume"));

        let error = merge_config_error(&["--chunk-dir"]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--chunk-dir"));
    }

    // -----------------------------------------------------------------------------------------
    // merge-chunks: where the expected row count comes from.
    // -----------------------------------------------------------------------------------------

    fn write_json(path: &Path, records_emitted: usize) {
        fs::write(
            path,
            serde_json::json!({ "records_emitted": records_emitted }).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn resolve_expected_rows_prefers_the_explicit_flag_without_reading_any_file() {
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--expected-rows",
            "7",
            // A path that does not exist: reaching it at all would be an error, so this asserts
            // the explicit count short-circuits rather than merely wins a comparison.
            "--summary-json",
            "no-such-summary.json",
        ]);

        assert_eq!(resolve_expected_rows(&config).unwrap(), 7);
    }

    #[test]
    fn resolve_expected_rows_reads_records_emitted_from_either_file() {
        let dir = ScopedTempDir::new("merge-expected-rows");
        let summary_path = dir.path().join("summary.json");
        let state_path = dir.path().join("resume-state.json");
        write_json(&summary_path, 41);
        write_json(&state_path, 12);

        let summary_arg = path_arg(&summary_path);
        let state_arg = path_arg(&state_path);
        let from_summary = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--summary-json",
            summary_arg.as_str(),
        ]);
        let from_state = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--state-path",
            state_arg.as_str(),
        ]);

        assert_eq!(resolve_expected_rows(&from_summary).unwrap(), 41);
        assert_eq!(resolve_expected_rows(&from_state).unwrap(), 12);
    }

    #[test]
    fn resolve_expected_rows_rejects_sources_that_disagree() {
        // A summary and a resume state from different runs: merging against either would be a
        // guess, and the wrong guess silently publishes a truncated destination file.
        let dir = ScopedTempDir::new("merge-expected-rows-conflict");
        let summary_path = dir.path().join("summary.json");
        let state_path = dir.path().join("resume-state.json");
        write_json(&summary_path, 41);
        write_json(&state_path, 12);

        let summary_arg = path_arg(&summary_path);
        let state_arg = path_arg(&state_path);
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--summary-json",
            summary_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
        ]);

        let error = resolve_expected_rows(&config).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("41"));
        assert!(error.to_string().contains("12"));
    }

    #[test]
    fn resolve_expected_rows_accepts_agreeing_sources() {
        let dir = ScopedTempDir::new("merge-expected-rows-agree");
        let summary_path = dir.path().join("summary.json");
        let state_path = dir.path().join("resume-state.json");
        write_json(&summary_path, 33);
        write_json(&state_path, 33);

        let summary_arg = path_arg(&summary_path);
        let state_arg = path_arg(&state_path);
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--summary-json",
            summary_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
        ]);

        assert_eq!(resolve_expected_rows(&config).unwrap(), 33);
    }

    #[test]
    fn resolve_expected_rows_errors_when_the_only_source_is_an_absent_state_file() {
        let dir = ScopedTempDir::new("merge-expected-rows-absent");
        let state_arg = path_arg(&dir.path().join("resume-state.json"));
        let config = merge_config_for(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--state-path",
            state_arg.as_str(),
        ]);

        let error = resolve_expected_rows(&config).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn read_records_emitted_rejects_a_file_without_an_integer_field() {
        let dir = ScopedTempDir::new("merge-records-emitted");
        let missing_field = dir.path().join("no-field.json");
        let wrong_type = dir.path().join("wrong-type.json");
        fs::write(&missing_field, r#"{"lines_scanned": 10}"#).unwrap();
        fs::write(&wrong_type, r#"{"records_emitted": "12"}"#).unwrap();

        for path in [&missing_field, &wrong_type] {
            let error = read_records_emitted(path).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains("records_emitted"));
        }
    }

    #[test]
    fn read_records_emitted_reads_a_real_summary_and_resume_state_file() {
        // Both file kinds this mode accepts are produced by code in this file, so read them back
        // through their real writers rather than through a hand-written JSON literal that could
        // drift from the field name they actually serialize.
        let dir = ScopedTempDir::new("merge-records-emitted-real");
        let summary_path = dir.path().join("summary.json");
        write_summary(
            &summary_path,
            &Stats {
                records_emitted: 5,
                ..Stats::default()
            },
        )
        .unwrap();

        let state_path = dir.path().join("resume-state.json");
        let chunk_arg = path_arg(&dir.path().join("chunks"));
        let state_arg = path_arg(&state_path);
        let run_config = config_for(&[
            "--input",
            "unused.jsonl.gz",
            "--spec",
            "unused-spec.json",
            "--output",
            "unused.jsonl",
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
        ]);
        maybe_write_resume_state(
            &run_config,
            &Stats {
                records_emitted: 5,
                ..Stats::default()
            },
            2,
        )
        .unwrap();

        assert_eq!(read_records_emitted(&summary_path).unwrap(), 5);
        assert_eq!(read_records_emitted(&state_path).unwrap(), 5);
    }

    // -----------------------------------------------------------------------------------------
    // merge-chunks: the concatenate-verify-rename core.
    //
    // The destination is a file the rest of a pipeline consumes, so every failure assertion below
    // checks that a pre-existing destination still holds its original bytes -- "the call returned
    // Err" says nothing about whether good data survived.
    // -----------------------------------------------------------------------------------------

    /// A `MergeConfig` over `chunk_dir` writing to `output`, with the count supplied explicitly so
    /// a test can state the expected total inline rather than seeding a summary file for it.
    fn merge_config_over(chunk_dir: &Path, output: &Path, expected_rows: usize) -> MergeConfig {
        let chunk_arg = path_arg(chunk_dir);
        let output_arg = path_arg(output);
        let expected_arg = expected_rows.to_string();
        merge_config_for(&[
            "--chunk-dir",
            chunk_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
            "--expected-rows",
            expected_arg.as_str(),
        ])
    }

    fn merge_all(config: &MergeConfig) -> io::Result<usize> {
        let chunks = iter_existing_chunk_files(&config.chunk_dir, &config.chunk_prefix)?;
        merge_chunks_into_destination(config, &chunks, config.expected_rows.unwrap())
    }

    #[test]
    fn merge_concatenates_chunks_in_index_order_into_the_destination() {
        let dir = ScopedTempDir::new("merge-order");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\nb\n");
        seed_chunk(&chunk_dir, 2, "c\n");
        seed_chunk(&chunk_dir, 3, "d\ne\n");
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 5);
        assert_eq!(merge_all(&config).unwrap(), 5);

        assert_eq!(read_file(&output), "a\nb\nc\nd\ne\n");
        // The temp file is gone: it was renamed onto the destination, not copied.
        assert_eq!(entry_names(dir.path()), vec!["chunks", "merged.jsonl"]);
    }

    #[test]
    fn merge_orders_by_numeric_index_not_file_name() {
        // Unpadded names sort as "10" < "7" by name and 7 < 10 by index. The Python
        // implementation this mirrors sorts by file name, which is only equivalent while every
        // index is the same width -- ordering by the parsed index is right at any width.
        let dir = ScopedTempDir::new("merge-numeric-order");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        fs::write(chunk_dir.join("part-7.jsonl"), "seven\n").unwrap();
        fs::write(chunk_dir.join("part-10.jsonl"), "ten\n").unwrap();
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 2);
        merge_all(&config).unwrap();

        assert_eq!(read_file(&output), "seven\nten\n");
    }

    #[test]
    fn merge_drops_blank_lines_and_terminates_a_chunk_missing_its_final_newline() {
        // A chunk sealed mid-write by a crash can end without a newline; concatenating it raw
        // would glue its last row onto the next chunk's first, producing one corrupt line that
        // still parses as neither. The row count alone would not catch it -- it would come out
        // one short, but so does plain truncation.
        let dir = ScopedTempDir::new("merge-normalize");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n\n   \nb");
        seed_chunk(&chunk_dir, 2, "c\n");
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 3);
        assert_eq!(merge_all(&config).unwrap(), 3);

        assert_eq!(read_file(&output), "a\nb\nc\n");
    }

    #[test]
    fn merge_leaves_a_good_destination_untouched_when_the_row_count_disagrees() {
        let dir = ScopedTempDir::new("merge-mismatch");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\nb\n");
        let output = dir.path().join("merged.jsonl");
        fs::write(&output, "previous good content\n").unwrap();

        let config = merge_config_over(&chunk_dir, &output, 5);
        let error = merge_all(&config).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("merged=2"));
        assert!(error.to_string().contains("expected=5"));
        assert_eq!(read_file(&output), "previous good content\n");
        // The abandoned temp file is cleaned up too, so a later successful merge cannot inherit
        // a half-written one and so a retry sees a clean directory.
        assert_eq!(entry_names(dir.path()), vec!["chunks", "merged.jsonl"]);
    }

    #[test]
    fn merge_ignores_files_that_are_not_this_runs_chunks() {
        // A chunk directory is not necessarily the run's exclusive property -- the same
        // assumption the fresh-start cleanup path already makes.
        let dir = ScopedTempDir::new("merge-foreign-files");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "ours\n");
        fs::write(chunk_dir.join("other-000001.jsonl"), "not ours\n").unwrap();
        fs::write(temp_chunk_path(&chunk_dir, 2), "half-written").unwrap();
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 1);
        assert_eq!(merge_all(&config).unwrap(), 1);

        assert_eq!(read_file(&output), "ours\n");
    }

    #[test]
    fn merge_creates_missing_destination_parent_directories() {
        let dir = ScopedTempDir::new("merge-nested-destination");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        let output = dir.path().join("nested").join("deep").join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 1);
        merge_all(&config).unwrap();

        assert_eq!(read_file(&output), "a\n");
    }

    // -----------------------------------------------------------------------------------------
    // merge-chunks: what the merge removes once the destination is committed.
    //
    // Cleanup deletes things, so every test here also asserts what survived. The narrowing
    // against the Python implementation's whole-directory delete is only real if a foreign file
    // demonstrably outlives the merge.
    // -----------------------------------------------------------------------------------------

    fn merge_config_with_state(
        chunk_dir: &Path,
        output: &Path,
        state_path: &Path,
        expected_rows: usize,
    ) -> MergeConfig {
        let chunk_arg = path_arg(chunk_dir);
        let output_arg = path_arg(output);
        let state_arg = path_arg(state_path);
        let expected_arg = expected_rows.to_string();
        merge_config_for(&[
            "--chunk-dir",
            chunk_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
            "--state-path",
            state_arg.as_str(),
            "--expected-rows",
            expected_arg.as_str(),
        ])
    }

    #[test]
    fn merge_removes_the_consumed_chunks_and_the_emptied_chunk_directory() {
        let dir = ScopedTempDir::new("merge-cleanup");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        seed_chunk(&chunk_dir, 2, "b\n");
        // An abandoned temp file from the killed run whose chunks these are: redundant now that
        // the merge is verified, and it would otherwise block the directory removal forever.
        fs::write(temp_chunk_path(&chunk_dir, 3), "half-written").unwrap();
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 2);
        run_merge_chunks(&config).unwrap();

        assert_eq!(read_file(&output), "a\nb\n");
        assert_eq!(entry_names(dir.path()), vec!["merged.jsonl"]);
        assert!(!chunk_dir.exists());
    }

    #[test]
    fn merge_keeps_a_chunk_directory_that_still_holds_files_it_did_not_merge() {
        let dir = ScopedTempDir::new("merge-cleanup-foreign");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "ours\n");
        fs::write(chunk_dir.join("notes.txt"), "not ours\n").unwrap();
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 1);
        run_merge_chunks(&config).unwrap();

        // Our chunk is gone; the file we never read is still there, and so is the directory.
        assert_eq!(entry_names(&chunk_dir), vec!["notes.txt"]);
        assert_eq!(read_file(&chunk_dir.join("notes.txt")), "not ours\n");
    }

    #[test]
    fn merge_renames_the_resume_state_to_a_completed_marker() {
        let dir = ScopedTempDir::new("merge-cleanup-state");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        let output = dir.path().join("merged.jsonl");
        let state_path = dir.path().join("resume-state.json");
        write_json(&state_path, 1);

        let config = merge_config_with_state(&chunk_dir, &output, &state_path, 1);
        run_merge_chunks(&config).unwrap();

        // Renamed, not deleted: the record that this run reached a verified merge survives, and
        // its content is unchanged by the rename.
        assert!(!state_path.exists());
        let marker_path = dir.path().join("resume-state.completed.json");
        assert_eq!(read_records_emitted(&marker_path).unwrap(), 1);
        assert_eq!(
            entry_names(dir.path()),
            vec!["merged.jsonl", "resume-state.completed.json"]
        );
    }

    #[test]
    fn merge_tolerates_a_state_path_whose_file_is_already_gone() {
        let dir = ScopedTempDir::new("merge-cleanup-state-absent");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        let output = dir.path().join("merged.jsonl");
        let state_path = dir.path().join("resume-state.json");

        let config = merge_config_with_state(&chunk_dir, &output, &state_path, 1);
        run_merge_chunks(&config).unwrap();

        assert_eq!(read_file(&output), "a\n");
        assert_eq!(entry_names(dir.path()), vec!["merged.jsonl"]);
    }

    #[test]
    fn merge_cleans_up_nothing_when_the_row_count_disagrees() {
        // The counterpart to the destination-untouched assertion: a rejected merge must also
        // leave the chunks and the resume state exactly as it found them, or the caller loses
        // the ability to investigate the mismatch and to resume the run.
        let dir = ScopedTempDir::new("merge-cleanup-mismatch");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        let output = dir.path().join("merged.jsonl");
        let state_path = dir.path().join("resume-state.json");
        write_json(&state_path, 1);

        let config = merge_config_with_state(&chunk_dir, &output, &state_path, 9);
        run_merge_chunks(&config).unwrap_err();

        assert_eq!(entry_names(&chunk_dir), vec!["part-000001.jsonl"]);
        assert_eq!(read_file(&chunk_path(&chunk_dir, 1)), "a\n");
        assert!(state_path.exists());
        assert!(!output.exists());
    }

    // -----------------------------------------------------------------------------------------
    // merge-chunks end to end: a real resumed run, then its merge.
    //
    // Everything below goes through `run_extract`/`run_merge_chunks` -- the same functions `main`
    // calls once it has parsed arguments -- over a fixture dump the test builds, never a real
    // one. The claim being tested is the item's own: a chunked run assembled by this binary
    // produces the file a single unchunked run would have written, with no Python step involved.
    // -----------------------------------------------------------------------------------------

    #[test]
    fn merge_chunks_reassembles_a_resumed_run_into_the_file_a_flat_run_would_have_written() {
        let dir = ScopedTempDir::new("merge-end-to-end");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        write_gzipped(
            &input_path,
            &as_dump_array(
                (1..=6)
                    .flat_map(|index| [candidate_line(index), non_candidate_line(index)])
                    .collect(),
            ),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);

        // The reference: one unchunked run straight to a flat file.
        let flat_path = dir.path().join("flat.jsonl");
        let flat_arg = path_arg(&flat_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            flat_arg.as_str(),
        ]))
        .unwrap();

        // The subject: the same input run in chunked resume mode, interrupted after three
        // scanned rows (`--max-rows` standing in for a kill) and resumed to completion. No
        // `--chunk-prefix` on either side, so this also exercises the default the two modes
        // share.
        let chunk_dir = dir.path().join("chunks");
        let state_path = dir.path().join("resume-state.json");
        let summary_path = dir.path().join("summary.json");
        let merged_path = dir.path().join("merged.jsonl");
        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&state_path);
        let summary_arg = path_arg(&summary_path);
        let merged_arg = path_arg(&merged_path);
        let mut resume_flags = vec![
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            merged_arg.as_str(),
            "--summary-json",
            summary_arg.as_str(),
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
        ];

        resume_flags.extend_from_slice(&["--max-rows", "3"]);
        run_extract(&config_for(&resume_flags)).unwrap();
        resume_flags.truncate(resume_flags.len() - 2);
        run_extract(&config_for(&resume_flags)).unwrap();

        // More than one chunk, or the merge would have nothing to assemble and the test would
        // pass on a single rename.
        assert!(entry_names(&chunk_dir).len() > 1);
        // The run wrote no destination file: producing it is entirely the merge's job.
        assert!(!merged_path.exists());

        // Both count sources come from the run itself -- nothing about the expected total is
        // supplied by hand here, which is what a caller holding only the binary would have.
        run_merge_chunks(
            &MergeConfig::from_args(args(&[
                "--chunk-dir",
                chunk_arg.as_str(),
                "--output",
                merged_arg.as_str(),
                "--summary-json",
                summary_arg.as_str(),
                "--state-path",
                state_arg.as_str(),
            ]))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(read_file(&merged_path), read_file(&flat_path));
        assert_eq!(
            ids_in(&merged_path),
            ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
                .map(ToString::to_string)
                .to_vec()
        );
        assert!(!chunk_dir.exists());
        assert!(dir.path().join("resume-state.completed.json").exists());
    }

    // -----------------------------------------------------------------------------------------
    // Re-deriving a new field from the raw-candidate cache.
    //
    // The claim is that a `projected_fields` addition can be served by replaying the cache rather
    // than by rescanning the dump. Both tests below check it the same way: project the new spec
    // twice, once from the cache and once from the dump, and require the two files to be
    // identical -- an approximation would not be a substitute for a rescan, it would be a
    // different answer.
    // -----------------------------------------------------------------------------------------

    /// Runs the extract mode over `input`, returning `(output_path, summary_path)`.
    fn extract_with(dir: &Path, label: &str, input: &Path, spec: &Path) -> (PathBuf, PathBuf) {
        let output_path = dir.join(format!("{label}.jsonl"));
        let summary_path = dir.join(format!("{label}-summary.json"));
        let input_arg = path_arg(input);
        let spec_arg = path_arg(spec);
        let output_arg = path_arg(&output_path);
        let summary_arg = path_arg(&summary_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--summary-json",
            summary_arg.as_str(),
        ]))
        .unwrap();
        (output_path, summary_path)
    }

    #[test]
    fn a_new_projected_field_is_re_derivable_from_the_raw_candidate_cache() {
        let dir = ScopedTempDir::new("raw-replay-flat");
        let spec_path = write_spec(dir.path());
        let extended_spec_path = write_extended_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let cache_path = dir.path().join("cache.jsonl.gz");

        // Mostly chaff, as the real dump is: 4 companies among 12 lines, so replaying the cache
        // and rescanning the dump are visibly different amounts of work.
        let mut entries = Vec::new();
        for index in 1..=4 {
            entries.push(candidate_line(index));
            entries.push(non_candidate_line(index));
            entries.push(unmatched_candidate_line(index));
        }
        write_gzipped(&input_path, &as_dump_array(entries));

        // The original run: today's spec, plus the cache.
        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let original_path = dir.path().join("original.jsonl");
        let original_arg = path_arg(&original_path);
        let cache_arg = path_arg(&cache_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            original_arg.as_str(),
            "--raw-candidate-output",
            cache_arg.as_str(),
        ]))
        .unwrap();

        // The field being added is absent from what the original run produced -- otherwise there
        // would be nothing to re-derive and the comparison below would prove nothing.
        let original: serde_json::Value =
            serde_json::from_str(read_file(&original_path).lines().next().unwrap()).unwrap();
        assert!(original.get("inception").is_none());

        let (from_cache_path, from_cache_summary) =
            extract_with(dir.path(), "from-cache", &cache_path, &extended_spec_path);
        let (from_dump_path, from_dump_summary) =
            extract_with(dir.path(), "from-dump", &input_path, &extended_spec_path);

        // Identical output, from an input that is the cache rather than the dump.
        assert_eq!(read_file(&from_cache_path), read_file(&from_dump_path));
        let replayed: serde_json::Value =
            serde_json::from_str(read_file(&from_cache_path).lines().next().unwrap()).unwrap();
        assert_eq!(replayed["inception"], "+2001-01-01T00:00:00Z");
        assert_eq!(replayed["lei"], serde_json::json!(["LEI-1"]));

        // And it cost 4 scanned lines against the dump's 12: the cache holds the matched
        // candidates and nothing else, which is the entire reason it is cheaper than a rescan.
        assert_eq!(summary_field(&from_cache_summary, "lines_scanned"), 4);
        assert_eq!(summary_field(&from_dump_summary, "lines_scanned"), 12);
        assert_eq!(summary_field(&from_cache_summary, "records_emitted"), 4);
        assert_eq!(summary_field(&from_dump_summary, "records_emitted"), 4);
    }

    #[test]
    fn a_resumed_run_s_merged_cache_re_derives_the_same_new_field() {
        // The same claim for the shape a real Prepare run actually takes: chunked, interrupted,
        // resumed, then merged. The cache has to survive all of that intact, because the run it
        // is produced by is the one that takes hours.
        let dir = ScopedTempDir::new("raw-replay-resumed");
        let spec_path = write_spec(dir.path());
        let extended_spec_path = write_extended_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        write_gzipped(
            &input_path,
            &as_dump_array(
                (1..=6)
                    .flat_map(|index| [candidate_line(index), non_candidate_line(index)])
                    .collect(),
            ),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let chunk_dir = dir.path().join("chunks");
        let state_path = dir.path().join("resume-state.json");
        let summary_path = dir.path().join("summary.json");
        let merged_path = dir.path().join("merged.jsonl");
        let cache_path = dir.path().join("cache.jsonl.gz");
        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&state_path);
        let summary_arg = path_arg(&summary_path);
        let merged_arg = path_arg(&merged_path);
        let cache_arg = path_arg(&cache_path);
        let mut resume_flags = vec![
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            merged_arg.as_str(),
            "--summary-json",
            summary_arg.as_str(),
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
            "--raw-candidate-output",
            cache_arg.as_str(),
        ];

        resume_flags.extend_from_slice(&["--max-rows", "3"]);
        run_extract(&config_for(&resume_flags)).unwrap();
        resume_flags.truncate(resume_flags.len() - 2);
        run_extract(&config_for(&resume_flags)).unwrap();

        // More than one chunk, each with its companion, and no cache written yet: assembling it
        // is the merge's job, exactly as it is for the projected destination.
        assert!(
            iter_existing_raw_chunk_files(&chunk_dir, DEFAULT_CHUNK_PREFIX)
                .unwrap()
                .len()
                > 1
        );
        assert!(!cache_path.exists());

        run_merge_chunks(
            &MergeConfig::from_args(args(&[
                "--chunk-dir",
                chunk_arg.as_str(),
                "--output",
                merged_arg.as_str(),
                "--summary-json",
                summary_arg.as_str(),
                "--state-path",
                state_arg.as_str(),
                "--raw-candidate-output",
                cache_arg.as_str(),
            ]))
            .unwrap(),
        )
        .unwrap();

        assert!(!chunk_dir.exists());
        assert_eq!(
            raw_chunk_ids(&cache_path),
            ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
        );

        let (from_cache_path, _) =
            extract_with(dir.path(), "from-cache", &cache_path, &extended_spec_path);
        let (from_dump_path, _) =
            extract_with(dir.path(), "from-dump", &input_path, &extended_spec_path);

        assert_eq!(read_file(&from_cache_path), read_file(&from_dump_path));
        let replayed: serde_json::Value =
            serde_json::from_str(read_file(&from_cache_path).lines().next().unwrap()).unwrap();
        assert_eq!(replayed["inception"], "+2001-01-01T00:00:00Z");
    }

    // -----------------------------------------------------------------------------------------
    // merge-chunks: assembling the raw-candidate companions.
    // -----------------------------------------------------------------------------------------

    fn merge_config_with_raw(
        chunk_dir: &Path,
        output: &Path,
        raw_output: &Path,
        expected_rows: usize,
    ) -> MergeConfig {
        let chunk_arg = path_arg(chunk_dir);
        let output_arg = path_arg(output);
        let raw_arg = path_arg(raw_output);
        let expected_arg = expected_rows.to_string();
        merge_config_for(&[
            "--chunk-dir",
            chunk_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
            "--expected-rows",
            expected_arg.as_str(),
            "--raw-candidate-output",
            raw_arg.as_str(),
        ])
    }

    #[test]
    fn merge_assembles_companions_into_one_readable_gzip_stream() {
        // The load-bearing property: gzip members concatenate, so the merged file is one valid
        // multi-member stream rather than three files glued into something only the first member
        // of which decodes. Everything downstream -- `--input`, the row-count reverification --
        // depends on that holding.
        let dir = ScopedTempDir::new("merge-raw-order");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\nb\n");
        seed_raw_companion(&chunk_dir, 1, "raw-a\nraw-b\n");
        seed_chunk(&chunk_dir, 2, "c\n");
        seed_raw_companion(&chunk_dir, 2, "raw-c\n");
        seed_chunk(&chunk_dir, 3, "d\n");
        seed_raw_companion(&chunk_dir, 3, "raw-d\n");
        let output = dir.path().join("merged.jsonl");
        let raw_output = dir.path().join("raw").join("cache.jsonl.gz");

        let config = merge_config_with_raw(&chunk_dir, &output, &raw_output, 4);
        run_merge_chunks(&config).unwrap();

        assert_eq!(read_file(&output), "a\nb\nc\nd\n");
        assert_eq!(read_gz_file(&raw_output), "raw-a\nraw-b\nraw-c\nraw-d\n");
    }

    #[test]
    fn merge_removes_the_companions_it_assembled_along_with_their_chunks() {
        let dir = ScopedTempDir::new("merge-raw-cleanup");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        seed_raw_companion(&chunk_dir, 1, "raw-a\n");
        let output = dir.path().join("merged.jsonl");
        let raw_output = dir.path().join("cache.jsonl.gz");

        let config = merge_config_with_raw(&chunk_dir, &output, &raw_output, 1);
        run_merge_chunks(&config).unwrap();

        assert!(!chunk_dir.exists());
        assert_eq!(
            entry_names(dir.path()),
            vec!["cache.jsonl.gz", "merged.jsonl"]
        );
    }

    #[test]
    fn merge_keeps_companions_nobody_asked_it_to_assemble() {
        // Without `--raw-candidate-output` the companions are the only copy of data that cost a
        // full-dump scan, and no destination has been named for them. They survive, and so does
        // the directory holding them.
        let dir = ScopedTempDir::new("merge-raw-unclaimed");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        seed_raw_companion(&chunk_dir, 1, "raw-a\n");
        let output = dir.path().join("merged.jsonl");

        let config = merge_config_over(&chunk_dir, &output, 1);
        run_merge_chunks(&config).unwrap();

        assert_eq!(read_file(&output), "a\n");
        assert_eq!(entry_names(&chunk_dir), vec!["part-000001.raw.jsonl.gz"]);
        assert_eq!(
            read_gz_file(&raw_chunk_path(&chunk_dir, TEST_PREFIX, 1)),
            "raw-a\n"
        );
    }

    #[test]
    fn merge_refuses_when_a_chunk_has_no_companion_and_keeps_everything() {
        // Capture switched on partway through a resumed run: the chunks are all there, so the
        // projected merge succeeds, but a cache assembled from what exists would silently be
        // missing the earlier chunk's rows.
        let dir = ScopedTempDir::new("merge-raw-missing-companion");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\n");
        seed_chunk(&chunk_dir, 2, "b\n");
        seed_raw_companion(&chunk_dir, 2, "raw-b\n");
        let output = dir.path().join("merged.jsonl");
        let raw_output = dir.path().join("cache.jsonl.gz");

        let config = merge_config_with_raw(&chunk_dir, &output, &raw_output, 2);
        let error = run_merge_chunks(&config).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error.to_string().contains("raw-candidate companion"));
        // No cache was written, and nothing was cleaned up -- the run stays re-mergeable.
        assert!(!raw_output.exists());
        assert_eq!(
            entry_names(&chunk_dir),
            vec![
                "part-000001.jsonl",
                "part-000002.jsonl",
                "part-000002.raw.jsonl.gz"
            ]
        );
    }

    #[test]
    fn merge_leaves_a_good_raw_cache_untouched_when_the_companion_row_count_disagrees() {
        let dir = ScopedTempDir::new("merge-raw-mismatch");
        let chunk_dir = dir.path().join("chunks");
        fs::create_dir_all(&chunk_dir).unwrap();
        seed_chunk(&chunk_dir, 1, "a\nb\n");
        // One row where the chunk has two: a companion from a different run, or one truncated
        // before its trailer. Either way the cache would reproject short.
        seed_raw_companion(&chunk_dir, 1, "raw-a\n");
        let output = dir.path().join("merged.jsonl");
        let raw_output = dir.path().join("cache.jsonl.gz");
        let mut previous = GzEncoder::new(File::create(&raw_output).unwrap(), Compression::fast());
        previous.write_all(b"previous good cache\n").unwrap();
        previous.finish().unwrap();

        let config = merge_config_with_raw(&chunk_dir, &output, &raw_output, 2);
        let error = run_merge_chunks(&config).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("merged=1"));
        assert!(error.to_string().contains("expected=2"));
        assert_eq!(read_gz_file(&raw_output), "previous good cache\n");
        // The projected destination committed before this half ran, and its own verification
        // passed -- so it is correct and stays. The chunks stay too, for the retry.
        assert_eq!(read_file(&output), "a\nb\n");
        assert_eq!(
            entry_names(&chunk_dir),
            vec!["part-000001.jsonl", "part-000001.raw.jsonl.gz"]
        );
    }

    #[test]
    fn merge_config_rejects_stdout_as_the_raw_candidate_destination() {
        let error = merge_config_error(&[
            "--chunk-dir",
            "chunks",
            "--output",
            "out.jsonl",
            "--expected-rows",
            "1",
            "--raw-candidate-output",
            "-",
        ]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--raw-candidate-output"));
    }

    #[test]
    fn merge_reports_a_missing_chunk_directory_rather_than_writing_an_empty_destination() {
        let dir = ScopedTempDir::new("merge-missing-chunk-dir");
        let output = dir.path().join("merged.jsonl");
        fs::write(&output, "previous good content\n").unwrap();

        let config = merge_config_over(&dir.path().join("absent-chunks"), &output, 1);
        let error = merge_all(&config).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert_eq!(read_file(&output), "previous good content\n");
    }

}
