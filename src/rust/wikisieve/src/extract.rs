//! The default `wikisieve` mode: spec-driven extraction from a Wikidata dump into projected JSONL
//! (or bare ids), optionally resumable via chunked output (see `crate::output_sink`) and always
//! checkpointed via `crate::resume`.

use rapidgzip_core::DecoderReader;
use rayon::prelude::*;
use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::cli::{next_arg_value, parse_usize_arg, print_usage, require_arg};
use crate::output_sink::{OutputSink, DEFAULT_CHUNK_PREFIX};
use crate::resume::{load_resume_state, maybe_write_resume_state, ResumeState};
use crate::scan::{fill_wikidata_batch, Stats, DEFAULT_PROGRESS_EVERY};
use crate::{project_candidate, CompiledSpec};

#[derive(Debug)]
pub struct Config {
    pub(crate) input_path: PathBuf,
    pub(crate) spec_path: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) summary_json_path: Option<PathBuf>,
    pub(crate) max_rows: Option<usize>,
    pub(crate) max_records: Option<usize>,
    pub(crate) output_mode: OutputMode,
    pub(crate) resume_enabled: bool,
    pub(crate) state_path: Option<PathBuf>,
    pub(crate) chunk_dir: Option<PathBuf>,
    pub(crate) chunk_prefix: String,
    /// `--raw-candidate-output`: where the raw-candidate cache goes, and the switch that turns
    /// capture on at all. Set in flat mode and unset in `--resume` mode, exactly as `output_path`
    /// is -- a resumable run writes chunks, and `merge-chunks` assembles both artifacts.
    pub(crate) raw_candidate_output: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum OutputMode {
    Ids,
    Jsonl,
}

/// Where a run's lines come from: the existing gzip-compressed dump file, decompressed in
/// parallel by `crate::dump_source`, or (`--input -`) uncompressed
/// newline-delimited entities read from standard input -- the shape
/// `rapidgzip -d -c <dump.gz> | wikisieve --input -` produces, letting an external parallel
/// decompressor be measured against the built-in one without this binary doing its own piping.
/// Implements `Read`/`BufRead` by delegating to whichever variant is open, so
/// `crate::scan::fill_wikidata_batch` and `process_dump` (both generic over `R: BufRead`) don't
/// need to know which -- the same pattern as `crate::output_sink::ProjectedWriter` on the output
/// side.
pub(crate) enum InputSource {
    // Boxed: `DecoderReader` carries `rapidgzip-core`'s own runtime/channel state, which makes
    // this variant far larger than `Stdin`'s (clippy::large_enum_variant); boxing keeps the
    // common `Stdin` case from paying for headroom it never uses.
    GzipFile(Box<BufReader<DecoderReader>>),
    Stdin(BufReader<io::Stdin>),
}

impl InputSource {
    /// Opens `input_path` as a gzip-compressed dump file, or, when `input_path` is exactly `-`,
    /// as uncompressed standard input -- the same `-` convention `OutputSink::open` already uses
    /// for `--output`. The gzip-file variant reads through `crate::dump_source`'s parallel reader,
    /// so decompression runs on the decoder's own worker threads rather than this one.
    pub(crate) fn open(input_path: &Path) -> io::Result<Self> {
        if input_path.as_os_str() == "-" {
            return Ok(Self::Stdin(BufReader::with_capacity(
                1024 * 1024,
                io::stdin(),
            )));
        }
        let decoder = crate::dump_source::open_parallel_gzip_source(input_path)?;
        Ok(Self::GzipFile(Box::new(BufReader::with_capacity(
            1024 * 1024,
            decoder,
        ))))
    }
}

impl Read for InputSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::GzipFile(reader) => reader.read(buf),
            Self::Stdin(reader) => reader.read(buf),
        }
    }
}

impl BufRead for InputSource {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        match self {
            Self::GzipFile(reader) => reader.fill_buf(),
            Self::Stdin(reader) => reader.fill_buf(),
        }
    }

    fn consume(&mut self, amt: usize) {
        match self {
            Self::GzipFile(reader) => reader.consume(amt),
            Self::Stdin(reader) => reader.consume(amt),
        }
    }
}

/// Runs the default extraction mode: decompresses `config.input_path`, projects each matched
/// candidate through `config.spec_path`, and writes the result to `config.output_path` (or a
/// chunk directory, in `--resume` mode).
///
/// # Errors
/// Returns an error if the spec or input can't be read, if the output sink (flat file, stdout,
/// or chunk directory) can't be opened or written to, or if the resume state or summary JSON
/// can't be read or written.
pub fn run_extract(config: &Config) -> io::Result<()> {
    let spec = CompiledSpec::load(&config.spec_path)?;
    eprintln!(
        "[wikisieve] input={} spec={} output={} mode={} markers={} projected_fields={} \
         resume_enabled={} max_rows={} raw_candidate_output={}",
        config.input_path.display(),
        config.spec_path.display(),
        config.output_path.display(),
        config.output_mode.as_str(),
        spec.markers.len(),
        spec.projected_fields.len(),
        config.resume_enabled,
        config
            .max_rows
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        config
            .raw_candidate_output
            .as_ref()
            .map_or_else(|| "none".to_string(), |path| path.display().to_string()),
    );

    let mut reader = InputSource::open(&config.input_path)?;

    let mut resume_state = load_resume_state(config)?;
    let mut chunk_rows_restored = 0usize;
    let mut output_sink = if config.resume_enabled {
        let (sink, restored_rows) = OutputSink::stream_mode(config, &resume_state)?;
        chunk_rows_restored = restored_rows;
        sink
    } else {
        OutputSink::open(&config.output_path, config.raw_candidate_output.as_deref())?
    };

    if config.resume_enabled
        && chunk_rows_restored > 0
        && resume_state.records_emitted != chunk_rows_restored
    {
        eprintln!(
            "[wikisieve] resume sync: records_emitted {} -> {} from existing chunks",
            resume_state.records_emitted, chunk_rows_restored
        );
        resume_state.records_emitted = chunk_rows_restored;
    }

    let stats = process_dump(
        &mut reader,
        &mut output_sink,
        &spec,
        config,
        &mut resume_state,
    )?;
    output_sink.finish()?;
    maybe_write_resume_state(config, &stats, output_sink.next_chunk_index())?;

    if let Some(summary_json_path) = config.summary_json_path.as_ref() {
        write_summary(summary_json_path, &stats)?;
    }

    eprintln!(
        "[wikisieve] complete lines_scanned={} candidates={} emitted={} skipped_no_properties={} skipped_parse_errors={}",
        stats.lines_scanned,
        stats.candidates_scanned,
        stats.records_emitted,
        stats.skipped_no_properties,
        stats.skipped_parse_errors,
    );
    Ok(())
}

impl Config {
    /// Parses the default (extract) subcommand's flags.
    ///
    /// # Errors
    /// Returns an error if a required flag (`--input`, `--spec`, `--output`) is missing, if a
    /// flag's value fails to parse, if an unknown argument is given, or if `--resume` is passed
    /// without both `--chunk-dir` and `--state-path`.
    pub fn from_args(args: impl Iterator<Item = String>) -> io::Result<Self> {
        let mut input_path: Option<PathBuf> = None;
        let mut spec_path: Option<PathBuf> = None;
        let mut output_path: Option<PathBuf> = None;
        let mut summary_json_path: Option<PathBuf> = None;
        let mut max_rows: Option<usize> = None;
        let mut max_records: Option<usize> = None;
        let mut output_mode = OutputMode::Jsonl;
        let mut resume_enabled = false;
        let mut state_path: Option<PathBuf> = None;
        let mut chunk_dir: Option<PathBuf> = None;
        let mut chunk_prefix = DEFAULT_CHUNK_PREFIX.to_string();
        let mut raw_candidate_output: Option<PathBuf> = None;

        let mut args = args;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => {
                    input_path = Some(PathBuf::from(next_arg_value(&mut args, "--input")?));
                }
                "--spec" => spec_path = Some(PathBuf::from(next_arg_value(&mut args, "--spec")?)),
                "--output" => {
                    output_path = Some(PathBuf::from(next_arg_value(&mut args, "--output")?));
                }
                "--summary-json" => {
                    summary_json_path =
                        Some(PathBuf::from(next_arg_value(&mut args, "--summary-json")?));
                }
                "--max-rows" => {
                    max_rows = Some(parse_usize_arg(
                        &next_arg_value(&mut args, "--max-rows")?,
                        "--max-rows",
                    )?);
                }
                "--max-records" => {
                    max_records = Some(parse_usize_arg(
                        &next_arg_value(&mut args, "--max-records")?,
                        "--max-records",
                    )?);
                }
                "--output-mode" => {
                    output_mode = OutputMode::parse(&next_arg_value(&mut args, "--output-mode")?)?;
                }
                "--resume" => {
                    resume_enabled = true;
                }
                "--state-path" => {
                    state_path = Some(PathBuf::from(next_arg_value(&mut args, "--state-path")?));
                }
                "--chunk-dir" => {
                    chunk_dir = Some(PathBuf::from(next_arg_value(&mut args, "--chunk-dir")?));
                }
                "--chunk-prefix" => {
                    chunk_prefix = next_arg_value(&mut args, "--chunk-prefix")?;
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
            input_path: require_arg(input_path, "--input")?,
            spec_path: require_arg(spec_path, "--spec")?,
            output_path: require_arg(output_path, "--output")?,
            summary_json_path,
            max_rows,
            max_records,
            output_mode,
            resume_enabled,
            state_path,
            chunk_dir,
            chunk_prefix,
            raw_candidate_output,
        };

        if resolved.resume_enabled && resolved.chunk_dir.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--chunk-dir is required when --resume is used",
            ));
        }
        if resolved.resume_enabled && resolved.state_path.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "--state-path is required when --resume is used",
            ));
        }

        Ok(resolved)
    }
}

impl OutputMode {
    fn parse(value: &str) -> io::Result<Self> {
        match value {
            "ids" => Ok(Self::Ids),
            "jsonl" => Ok(Self::Jsonl),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported --output-mode '{other}' (expected 'ids' or 'jsonl')"),
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ids => "ids",
            Self::Jsonl => "jsonl",
        }
    }
}

/// Writes `stats` to `path` as pretty JSON, with the running binary's build provenance
/// (`crate_version`, `git_commit`, `git_dirty`) folded into the same object: this is what lets a downstream manifest (the Wikidata downloader's) tie a
/// prepare run's output back to the exact binary that produced it, without a separate
/// `--version` invocation.
pub(crate) fn write_summary(path: &Path, stats: &Stats) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut payload = serde_json::to_value(stats).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to serialize summary JSON for {}: {error}",
                path.display()
            ),
        )
    })?;
    let build_info = crate::version::build_info();
    if let Value::Object(map) = &mut payload {
        map.insert(
            "crate_version".to_string(),
            Value::String(build_info.version.to_string()),
        );
        map.insert(
            "git_commit".to_string(),
            Value::String(build_info.commit.to_string()),
        );
        map.insert("git_dirty".to_string(), Value::Bool(build_info.dirty));
    }
    let bytes = serde_json::to_vec_pretty(&payload).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to serialize summary JSON for {}: {error}",
                path.display()
            ),
        )
    })?;
    fs::write(path, bytes)
}

pub(crate) fn process_dump<R: BufRead>(
    reader: &mut R,
    output_sink: &mut OutputSink,
    spec: &CompiledSpec,
    config: &Config,
    resume_state: &mut ResumeState,
) -> io::Result<Stats> {
    let mut stats = Stats {
        lines_scanned: resume_state.lines_scanned,
        candidates_scanned: resume_state.candidates_scanned,
        records_emitted: resume_state.records_emitted,
        skipped_no_properties: resume_state.skipped_no_properties,
        skipped_parse_errors: resume_state.skipped_parse_errors,
    };
    let mut lines_to_skip = if config.resume_enabled {
        resume_state.lines_scanned
    } else {
        0
    };
    let mut raw_line = Vec::with_capacity(64 * 1024);
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(DEFAULT_PROGRESS_EVERY);
    // Built once per run, not once per line: `Finder::new`/`AhoCorasick::new` do the search-table
    // setup, which would dominate a naive per-line rebuild -- see `Prefilter::build`.
    let prefilter = crate::Prefilter::build(spec);

    loop {
        let source_exhausted_or_capped = fill_wikidata_batch(
            reader,
            &mut raw_line,
            &mut batch,
            &mut lines_to_skip,
            &prefilter,
            &mut stats,
            config.max_rows,
        )?;

        let results: Vec<Result<Option<String>, serde_json::Error>> = batch
            .par_iter()
            .map(|line| project_candidate(line, spec))
            .collect();

        let mut hit_max_records = false;
        // Zipped against `batch`, which holds the trimmed dump line each result was projected
        // from -- that pairing is the whole raw-candidate cache, and it exists here for free
        // because the batch is still in memory. Recovering it later would mean the full-dump
        // rescan this item exists to avoid.
        for (raw_line, result) in batch.iter().zip(results) {
            match result {
                Ok(Some(projected_line)) => {
                    let output_line = match config.output_mode {
                        OutputMode::Jsonl => projected_line,
                        OutputMode::Ids => {
                            extract_id_field(&projected_line).unwrap_or(projected_line)
                        }
                    };
                    output_sink.write_row(&output_line, raw_line)?;
                    stats.records_emitted += 1;
                    if crate::scan::reached_limit(stats.records_emitted, config.max_records) {
                        eprintln!(
                            "[wikisieve] stopping early after max_records={} emitted rows",
                            stats.records_emitted
                        );
                        hit_max_records = true;
                        break;
                    }
                }
                Ok(None) => {}
                Err(_) => {
                    stats.skipped_parse_errors += 1;
                }
            }
        }

        // Finalizing the chunk here, before the resume checkpoint below, keeps the two events
        // atomic with each other -- see the module-level comment above and the resume-checkpoint postmortem
        // for why a row-count-based chunk rollover, independent of the checkpoint cadence, is
        // unsafe.
        output_sink.finalize_pending_chunk()?;

        if !batch.is_empty() {
            eprintln!(
                "[wikisieve] progress lines_scanned={} candidates={} emitted={}",
                stats.lines_scanned, stats.candidates_scanned, stats.records_emitted,
            );
            maybe_write_resume_state(config, &stats, output_sink.next_chunk_index())?;
        }

        if hit_max_records || source_exhausted_or_capped {
            break;
        }
    }

    maybe_write_resume_state(config, &stats, output_sink.next_chunk_index())?;

    Ok(stats)
}

fn extract_id_field(projected_line: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(projected_line).ok()?;
    value.get("id")?.as_str().map(ToString::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        args, as_dump_array, candidate_line, chunk_path, config_for, entry_names, ids_in,
        malformed_candidate_line, non_candidate_line, open_gzipped, path_arg, read_file,
        read_gz_file, raw_chunk_ids, unmatched_candidate_line, write_gzipped, write_spec,
        ScopedTempDir, TEST_PREFIX,
    };

    #[test]
    fn output_mode_parses_known_values() {
        assert!(matches!(OutputMode::parse("ids").unwrap(), OutputMode::Ids));
        assert!(matches!(
            OutputMode::parse("jsonl").unwrap(),
            OutputMode::Jsonl
        ));
    }

    #[test]
    fn output_mode_rejects_unknown_value() {
        let error = OutputMode::parse("csv").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("csv"));
    }

    #[test]
    fn output_mode_as_str_round_trips_parse() {
        assert_eq!(OutputMode::Ids.as_str(), "ids");
        assert_eq!(OutputMode::Jsonl.as_str(), "jsonl");
        assert!(matches!(
            OutputMode::parse(OutputMode::Ids.as_str()).unwrap(),
            OutputMode::Ids
        ));
    }

    #[test]
    fn input_source_open_reads_a_gzip_file_through_the_shared_bufread_impl() {
        let dir = ScopedTempDir::new("input-source-gzip-file");
        let input_path = dir.path().join("dump.jsonl.gz");
        write_gzipped(&input_path, &as_dump_array(vec![candidate_line(1)]));

        let mut reader = InputSource::open(&input_path).unwrap();
        assert!(matches!(reader, InputSource::GzipFile(_)));

        let mut first_line = Vec::new();
        reader.read_until(b'\n', &mut first_line).unwrap();
        assert_eq!(String::from_utf8(first_line).unwrap().trim_end(), "[");
    }

    #[test]
    fn input_source_open_dash_selects_the_stdin_variant() {
        let reader = InputSource::open(Path::new("-")).unwrap();
        assert!(matches!(reader, InputSource::Stdin(_)));
    }

    #[test]
    fn extract_id_field_returns_id_string() {
        let line = r#"{"id": "Q42", "official_name": "Example"}"#;
        assert_eq!(extract_id_field(line).as_deref(), Some("Q42"));
    }

    #[test]
    fn extract_id_field_returns_none_when_field_missing() {
        let line = r#"{"official_name": "Example"}"#;
        assert_eq!(extract_id_field(line), None);
    }

    #[test]
    fn extract_id_field_returns_none_for_invalid_json() {
        assert_eq!(extract_id_field("not json"), None);
    }

    #[test]
    fn extract_id_field_returns_none_when_id_is_not_a_string() {
        let line = r#"{"id": 42}"#;
        assert_eq!(extract_id_field(line), None);
    }

    #[test]
    fn from_args_parses_required_arguments_with_defaults() {
        let config = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "out.jsonl",
        ]))
        .unwrap();
        assert_eq!(config.input_path, PathBuf::from("in.jsonl.gz"));
        assert_eq!(config.spec_path, PathBuf::from("spec.json"));
        assert_eq!(config.output_path, PathBuf::from("out.jsonl"));
        assert_eq!(config.summary_json_path, None);
        assert_eq!(config.max_rows, None);
        assert_eq!(config.max_records, None);
        assert!(matches!(config.output_mode, OutputMode::Jsonl));
        assert!(!config.resume_enabled);
        assert_eq!(config.chunk_prefix, "wikisieve-part-");
    }

    #[test]
    fn from_args_parses_all_optional_flags() {
        let config = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "-",
            "--summary-json",
            "summary.json",
            "--max-rows",
            "100",
            "--max-records",
            "7",
            "--output-mode",
            "ids",
            "--resume",
            "--state-path",
            "state.json",
            "--chunk-dir",
            "chunks",
            "--chunk-prefix",
            "part-",
        ]))
        .unwrap();
        assert_eq!(
            config.summary_json_path,
            Some(PathBuf::from("summary.json"))
        );
        assert_eq!(config.max_rows, Some(100));
        assert_eq!(config.max_records, Some(7));
        assert!(matches!(config.output_mode, OutputMode::Ids));
        assert!(config.resume_enabled);
        assert_eq!(config.state_path, Some(PathBuf::from("state.json")));
        assert_eq!(config.chunk_dir, Some(PathBuf::from("chunks")));
        assert_eq!(config.chunk_prefix, "part-");
    }

    #[test]
    fn from_args_rejects_missing_required_argument() {
        let error = Config::from_args(args(&["--input", "in.jsonl.gz", "--spec", "spec.json"]))
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--output"));
    }

    #[test]
    fn from_args_rejects_unknown_argument() {
        let error = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "out.jsonl",
            "--bogus",
        ]))
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--bogus"));
    }

    #[test]
    fn from_args_rejects_flag_missing_its_value() {
        let error = Config::from_args(args(&["--input"])).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--input"));
    }

    #[test]
    fn from_args_resume_requires_chunk_dir() {
        let error = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "out.jsonl",
            "--resume",
            "--state-path",
            "state.json",
        ]))
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--chunk-dir"));
    }

    #[test]
    fn from_args_resume_requires_state_path() {
        let error = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "out.jsonl",
            "--resume",
            "--chunk-dir",
            "chunks",
        ]))
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--state-path"));
    }

    #[test]
    fn from_args_resume_succeeds_with_both_required_flags() {
        let config = Config::from_args(args(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "out.jsonl",
            "--resume",
            "--chunk-dir",
            "chunks",
            "--state-path",
            "state.json",
        ]))
        .unwrap();
        assert!(config.resume_enabled);
    }

    #[test]
    fn write_summary_serializes_stats_fields() {
        let dir = ScopedTempDir::new("write-summary");
        let path = dir.path().join("summary.json");
        let stats = Stats {
            lines_scanned: 100,
            candidates_scanned: 40,
            records_emitted: 12,
            skipped_no_properties: 60,
            skipped_parse_errors: 3,
        };

        write_summary(&path, &stats).unwrap();

        let contents = fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert_eq!(value["lines_scanned"], 100);
        assert_eq!(value["candidates_scanned"], 40);
        assert_eq!(value["records_emitted"], 12);
        assert_eq!(value["skipped_no_properties"], 60);
        assert_eq!(value["skipped_parse_errors"], 3);
        // Every summary ties itself back to the binary that wrote it.
        assert_eq!(value["crate_version"], env!("CARGO_PKG_VERSION"));
        assert!(!value["git_commit"].as_str().unwrap().is_empty());
        assert!(value["git_dirty"].is_boolean());
    }

    #[test]
    fn write_summary_creates_missing_parent_directories() {
        let dir = ScopedTempDir::new("write-summary-nested");
        let path = dir.path().join("nested").join("deep").join("summary.json");
        let stats = Stats::default();

        write_summary(&path, &stats).unwrap();

        assert!(path.exists());
        let value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["lines_scanned"], 0);
    }

    // End-to-end `process_dump` over a fixture dump this test builds itself. Never a real dump:
    // `data/` is shared physical storage here, and these need to run on any machine.
    // -----------------------------------------------------------------------------------------

    fn run_process_dump(
        config: &Config,
        sink: &mut OutputSink,
        resume_state: &mut ResumeState,
    ) -> Stats {
        let spec = CompiledSpec::load(&config.spec_path).unwrap();
        let mut reader = open_gzipped(&config.input_path);
        process_dump(&mut reader, sink, &spec, config, resume_state).unwrap()
    }

    #[test]
    fn process_dump_writes_projected_records_in_input_order() {
        let dir = ScopedTempDir::new("process-order");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.jsonl");

        // Rejected lines are interleaved between the candidates, so "output is in input order"
        // is a real claim about `par_iter().map(...).collect()` keeping its batch's order rather
        // than the trivial one that every scanned line was emitted.
        let mut entries = Vec::new();
        for index in 1..=250 {
            entries.push(candidate_line(index));
            if index % 5 == 0 {
                entries.push(non_candidate_line(index));
            }
            if index % 25 == 0 {
                entries.push(unmatched_candidate_line(index));
            }
        }
        entries.push(malformed_candidate_line());
        write_gzipped(&input_path, &as_dump_array(entries));

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let config = config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
        ]);

        let mut sink =
            OutputSink::open(&config.output_path, config.raw_candidate_output.as_deref()).unwrap();
        let mut resume_state = ResumeState::default();
        let stats = run_process_dump(&config, &mut sink, &mut resume_state);
        sink.finish().unwrap();

        let expected_ids: Vec<String> = (1..=250).map(|index| format!("Q{index}")).collect();
        assert_eq!(ids_in(&output_path), expected_ids);

        // Each rejection route is counted separately, so a regression that starts silently
        // dropping matched rows can't hide inside a single "not emitted" bucket.
        assert_eq!(stats.lines_scanned, 311);
        assert_eq!(stats.candidates_scanned, 261);
        assert_eq!(stats.records_emitted, 250);
        assert_eq!(stats.skipped_no_properties, 50);
        assert_eq!(stats.skipped_parse_errors, 1);

        let first: serde_json::Value =
            serde_json::from_str(read_file(&output_path).lines().next().unwrap()).unwrap();
        assert_eq!(first["id"], "Q1");
        assert_eq!(first["label_en"], "Company 1");
        assert_eq!(first["instance_of"], serde_json::json!(["Q783794"]));
        assert_eq!(first["lei"], serde_json::json!(["LEI-1"]));
    }

    #[test]
    fn process_dump_ids_mode_writes_bare_ids_rather_than_json_records() {
        let dir = ScopedTempDir::new("process-ids-mode");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.txt");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=3).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let config = config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--output-mode",
            "ids",
        ]);

        let mut sink =
            OutputSink::open(&config.output_path, config.raw_candidate_output.as_deref()).unwrap();
        let mut resume_state = ResumeState::default();
        run_process_dump(&config, &mut sink, &mut resume_state);
        sink.finish().unwrap();

        // The whole file, byte for byte: bare ids, no surrounding JSON object or quoting.
        assert_eq!(read_file(&output_path), "Q1\nQ2\nQ3\n");
    }

    #[test]
    fn process_dump_stops_at_max_records_without_writing_further_rows() {
        let dir = ScopedTempDir::new("process-max-records");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.txt");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=6).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let config = config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--output-mode",
            "ids",
            "--max-records",
            "2",
        ]);

        let mut sink =
            OutputSink::open(&config.output_path, config.raw_candidate_output.as_deref()).unwrap();
        let mut resume_state = ResumeState::default();
        let stats = run_process_dump(&config, &mut sink, &mut resume_state);
        sink.finish().unwrap();

        assert_eq!(read_file(&output_path), "Q1\nQ2\n");
        assert_eq!(stats.records_emitted, 2);
    }

    #[test]
    fn process_dump_resume_continues_after_a_capped_pass_without_duplicating_rows() {
        // The round trip verified once by hand with a forced kill on a 500K-row run, in
        // the form that runs again on every change to this file: `--max-rows` stands in for the
        // kill, and the second pass has to start exactly where the checkpoint left off.
        let dir = ScopedTempDir::new("process-resume");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let chunk_dir = dir.path().join("chunks");
        let state_path = dir.path().join("resume-state.json");
        let output_path = dir.path().join("unused.jsonl");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=6).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&state_path);
        let mut flags = vec![
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
        ];

        flags.extend_from_slice(&["--max-rows", "3"]);
        let first_pass = config_for(&flags);
        let mut first_resume_state = load_resume_state(&first_pass).unwrap();
        let (mut first_sink, restored_rows) =
            OutputSink::stream_mode(&first_pass, &first_resume_state).unwrap();
        assert_eq!(restored_rows, 0);
        let first_stats = run_process_dump(&first_pass, &mut first_sink, &mut first_resume_state);
        first_sink.finish().unwrap();

        assert_eq!(first_stats.records_emitted, 3);
        assert_eq!(entry_names(&chunk_dir), vec!["part-000001.jsonl"]);
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 1)), ["Q1", "Q2", "Q3"]);

        // The checkpoint is written alongside the chunk that is already durable on disk, so it
        // names rows that exist rather than rows still in a write buffer.
        let checkpoint: serde_json::Value = serde_json::from_str(&read_file(&state_path)).unwrap();
        assert_eq!(checkpoint["lines_scanned"], 3);
        assert_eq!(checkpoint["records_emitted"], 3);
        assert_eq!(checkpoint["next_chunk_index"], 2);

        flags.truncate(flags.len() - 2);
        let second_pass = config_for(&flags);
        let mut second_resume_state = load_resume_state(&second_pass).unwrap();
        assert_eq!(second_resume_state.lines_scanned, 3);
        let (mut second_sink, restored_rows) =
            OutputSink::stream_mode(&second_pass, &second_resume_state).unwrap();
        // What the chunks on disk actually hold agrees with what the checkpoint claims -- the
        // reconciliation `main` would otherwise have to correct.
        assert_eq!(restored_rows, 3);
        assert_eq!(second_resume_state.records_emitted, restored_rows);

        let second_stats =
            run_process_dump(&second_pass, &mut second_sink, &mut second_resume_state);
        second_sink.finish().unwrap();

        assert_eq!(
            entry_names(&chunk_dir),
            vec!["part-000001.jsonl", "part-000002.jsonl"]
        );
        // No overlap and no gap across the resume boundary: the first chunk is byte-identical to
        // what the capped pass sealed, and the second picks up at the next unscanned row.
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 1)), ["Q1", "Q2", "Q3"]);
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 2)), ["Q4", "Q5", "Q6"]);
        assert_eq!(second_stats.records_emitted, 6);
        // `--output` is inert in stream mode; the chunk directory is the only output.
        assert!(!output_path.exists());
    }

    #[test]
    fn run_extract_resume_rescans_a_chunk_sealed_after_the_last_checkpoint() {
        // A kill between sealing a batch's chunk and saving the checkpoint that covers it. The
        // resume rescans from the checkpoint, so the uncovered chunk must be dropped rather than
        // kept beside the rescanned rows.
        let dir = ScopedTempDir::new("resume-uncovered-chunk");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let chunk_dir = dir.path().join("chunks");
        let state_path = dir.path().join("resume-state.json");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=6).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&dir.path().join("unused.jsonl"));
        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&state_path);
        let mut flags = vec![
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
        ];
        flags.extend_from_slice(&["--max-rows", "3"]);
        run_extract(&config_for(&flags)).unwrap();
        // The next batch's chunk, sealed, with no checkpoint saved after it.
        fs::write(
            chunk_path(&chunk_dir, 2),
            "{\"id\":\"Q4\"}\n{\"id\":\"Q5\"}\n",
        )
        .unwrap();

        flags.truncate(flags.len() - 2);
        run_extract(&config_for(&flags)).unwrap();

        assert_eq!(
            entry_names(&chunk_dir),
            vec!["part-000001.jsonl", "part-000002.jsonl"]
        );
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 1)), ["Q1", "Q2", "Q3"]);
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 2)), ["Q4", "Q5", "Q6"]);
        let checkpoint: serde_json::Value = serde_json::from_str(&read_file(&state_path)).unwrap();
        assert_eq!(checkpoint["records_emitted"], 6);
    }

    #[test]
    fn process_dump_flat_mode_caches_the_dump_line_behind_every_emitted_row() {
        let dir = ScopedTempDir::new("raw-flat-capture");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.jsonl");
        let raw_path = dir.path().join("raw").join("cache.jsonl.gz");

        // Interleaved rejections again: the cache must hold the emitted rows' lines and only
        // those, not every line the prefilter let through.
        let mut entries = Vec::new();
        for index in 1..=4 {
            entries.push(candidate_line(index));
            entries.push(non_candidate_line(index));
            entries.push(unmatched_candidate_line(index));
        }
        write_gzipped(&input_path, &as_dump_array(entries));

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let raw_arg = path_arg(&raw_path);
        let config = config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--raw-candidate-output",
            raw_arg.as_str(),
        ]);

        run_extract(&config).unwrap();

        // Byte-identical to the dump's own lines, in the same order -- the array delimiters and
        // trailing commas `trim_wikidata_line` stripped stay stripped, which is what makes the
        // artifact a valid `--input` in its own right.
        let expected: String = (1..=4)
            .map(|index| format!("{}\n", candidate_line(index)))
            .collect();
        assert_eq!(read_gz_file(&raw_path), expected);
        assert_eq!(
            ids_in(&output_path),
            ["Q1", "Q2", "Q3", "Q4"].map(ToString::to_string).to_vec()
        );
    }

    #[test]
    fn process_dump_writes_no_raw_cache_unless_one_is_asked_for() {
        let dir = ScopedTempDir::new("raw-flat-absent");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.jsonl");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=3).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
        ]))
        .unwrap();

        assert_eq!(
            entry_names(dir.path()),
            vec!["dump.jsonl.gz", "out.jsonl", "spec.json"]
        );
    }

    #[test]
    fn process_dump_caches_raw_lines_in_ids_mode_too() {
        // `--output-mode ids` discards everything but the id from the projection. The cache is
        // the raw line regardless, which is the point: the projection's shape is what a later
        // spec change alters, and the cache has to outlive it.
        let dir = ScopedTempDir::new("raw-ids-mode");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.txt");
        let raw_path = dir.path().join("cache.jsonl.gz");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=2).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let raw_arg = path_arg(&raw_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--output-mode",
            "ids",
            "--raw-candidate-output",
            raw_arg.as_str(),
        ]))
        .unwrap();

        assert_eq!(read_file(&output_path), "Q1\nQ2\n");
        assert_eq!(raw_chunk_ids(&raw_path), ["Q1", "Q2"]);
    }

    #[test]
    fn process_dump_raw_cache_stops_with_the_projection_at_max_records() {
        // The cap breaks out mid-batch. Both artifacts have to stop on the same row, or the cache
        // reprojects to a row count the projected output never had.
        let dir = ScopedTempDir::new("raw-max-records");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("out.txt");
        let raw_path = dir.path().join("cache.jsonl.gz");
        write_gzipped(
            &input_path,
            &as_dump_array((1..=6).map(candidate_line).collect()),
        );

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&output_path);
        let raw_arg = path_arg(&raw_path);
        run_extract(&config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--output-mode",
            "ids",
            "--max-records",
            "2",
            "--raw-candidate-output",
            raw_arg.as_str(),
        ]))
        .unwrap();

        assert_eq!(read_file(&output_path), "Q1\nQ2\n");
        assert_eq!(raw_chunk_ids(&raw_path), ["Q1", "Q2"]);
    }

    #[test]
    fn process_dump_seals_one_chunk_per_batch() {
        // `DEFAULT_PROGRESS_EVERY` scanned lines is the rayon batch unit, the resume-checkpoint
        // cadence, and the chunk boundary all at once -- that coupling is the resume fix, and it
        // is only observable across a real batch boundary, which every other test here sits
        // inside. The filler lines carry no `"P31"`, so the prefilter drops them without a parse
        // and the cost of spanning the boundary stays in the milliseconds.
        let dir = ScopedTempDir::new("process-batch-boundary");
        let spec_path = write_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let chunk_dir = dir.path().join("chunks");
        let state_path = dir.path().join("resume-state.json");

        let mut entries = Vec::with_capacity(DEFAULT_PROGRESS_EVERY + 1);
        entries.push(candidate_line(1));
        while entries.len() < DEFAULT_PROGRESS_EVERY {
            entries.push(non_candidate_line(entries.len()));
        }
        // The first line of the second batch.
        entries.push(candidate_line(2));
        write_gzipped(&input_path, &as_dump_array(entries));

        let input_arg = path_arg(&input_path);
        let spec_arg = path_arg(&spec_path);
        let output_arg = path_arg(&dir.path().join("unused.jsonl"));
        let chunk_arg = path_arg(&chunk_dir);
        let state_arg = path_arg(&state_path);
        let config = config_for(&[
            "--input",
            input_arg.as_str(),
            "--spec",
            spec_arg.as_str(),
            "--output",
            output_arg.as_str(),
            "--resume",
            "--chunk-dir",
            chunk_arg.as_str(),
            "--state-path",
            state_arg.as_str(),
            "--chunk-prefix",
            TEST_PREFIX,
        ]);

        let mut resume_state = ResumeState::default();
        let (mut sink, _) = OutputSink::stream_mode(&config, &resume_state).unwrap();
        let stats = run_process_dump(&config, &mut sink, &mut resume_state);
        sink.finish().unwrap();

        assert_eq!(stats.lines_scanned, DEFAULT_PROGRESS_EVERY + 1);
        assert_eq!(stats.records_emitted, 2);
        // Two batches, two sealed chunks -- not one chunk holding both rows.
        assert_eq!(
            entry_names(&chunk_dir),
            vec!["part-000001.jsonl", "part-000002.jsonl"]
        );
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 1)), ["Q1"]);
        assert_eq!(ids_in(&chunk_path(&chunk_dir, 2)), ["Q2"]);
    }
}
