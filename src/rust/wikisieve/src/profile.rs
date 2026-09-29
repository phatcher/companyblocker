//! `profile` -- schema discovery/diagnostic mode (README, "Schema discovery: `profile`"). Shares the extract path's decompress +
//! prefilter + batched-parallel-parse machinery (`crate::scan::fill_wikidata_batch`) so "which
//! lines are candidates" agrees with `extract`, but calls `profile_candidate` in place of
//! `project_candidate` and accumulates a frequency+shape report in memory instead of writing a
//! projected record per line. No resume/chunking: a diagnostic pass over a bounded (`--max-rows`)
//! sample is the intended use, not a multi-hour full-dump run that needs to survive a restart.

use rayon::prelude::*;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

use crate::cli::{next_arg_value, parse_usize_arg, print_usage, require_arg};
use crate::output_sink::create_parent_dir;
use crate::scan::{fill_wikidata_batch, Stats, DEFAULT_PROGRESS_EVERY};
use crate::{profile_candidate, CandidateProfile, CompiledSpec, Prefilter, PropertyShape};

/// Settings for the `profile` subcommand -- the schema-discovery/diagnostic pass over
/// matched candidates, see `run_profile`.
#[derive(Debug)]
pub struct ProfileConfig {
    pub(crate) input_path: PathBuf,
    pub(crate) spec_path: PathBuf,
    pub(crate) output_path: PathBuf,
    pub(crate) max_rows: Option<usize>,
    /// The gap report's frequency floor: a simple-shaped, not-yet-projected property is included
    /// only once at least this many matched candidates carry it. Defaults to 1 (report
    /// everything), since "frequent enough to matter" is a judgment call for whoever reads the
    /// report, not one this tool should make silently by filtering rows out.
    pub(crate) gap_min_count: usize,
}

impl ProfileConfig {
    /// Parses the `profile` subcommand's flags.
    ///
    /// # Errors
    /// Returns an error if a required flag (`--input`, `--spec`, `--output`) is missing, if a
    /// flag's value fails to parse, or if an unknown argument is given.
    pub fn from_args(args: impl Iterator<Item = String>) -> io::Result<Self> {
        let mut input_path: Option<PathBuf> = None;
        let mut spec_path: Option<PathBuf> = None;
        let mut output_path: Option<PathBuf> = None;
        let mut max_rows: Option<usize> = None;
        let mut gap_min_count = 1usize;

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
                "--max-rows" => {
                    max_rows = Some(parse_usize_arg(
                        &next_arg_value(&mut args, "--max-rows")?,
                        "--max-rows",
                    )?);
                }
                "--gap-min-count" => {
                    gap_min_count = parse_usize_arg(
                        &next_arg_value(&mut args, "--gap-min-count")?,
                        "--gap-min-count",
                    )?;
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

        Ok(Self {
            input_path: require_arg(input_path, "--input")?,
            spec_path: require_arg(spec_path, "--spec")?,
            output_path: require_arg(output_path, "--output")?,
            max_rows,
            gap_min_count,
        })
    }
}

/// Aggregates `profile_candidate` results across a run: how many matched candidates carry each
/// property, broken down by that property's classified shape (usually one shape per property,
/// but real data can disagree with itself -- see `classify_property_shape`).
struct ProfileAccumulator {
    candidates_matched: usize,
    property_shape_counts: HashMap<String, HashMap<PropertyShape, usize>>,
}

impl ProfileAccumulator {
    fn new() -> Self {
        Self {
            candidates_matched: 0,
            property_shape_counts: HashMap::new(),
        }
    }

    fn record_candidate(&mut self, profile: &CandidateProfile<'_>) {
        self.candidates_matched += 1;
        for (&property, &shape) in profile {
            *self
                .property_shape_counts
                .entry(property.to_string())
                .or_default()
                .entry(shape)
                .or_insert(0) += 1;
        }
    }

    /// Builds the final report: per-property frequency+shape counts (`properties`, sorted by
    /// frequency descending) and the coverage-gap report derived from them (`gap_report`) --
    /// simple-shaped, not-yet-`projected_fields`-named properties seen on at least
    /// `gap_min_count` candidates.
    fn into_report(self, spec: &CompiledSpec, gap_min_count: usize) -> ProfileReport {
        let projected_properties: HashSet<&str> = spec
            .projected_fields
            .iter()
            .map(|field| field.property.as_str())
            .collect();
        let candidates_matched = self.candidates_matched;

        let mut properties: Vec<PropertyProfileEntry> = self
            .property_shape_counts
            .into_iter()
            .map(|(property, shape_counts)| {
                build_property_profile_entry(property, shape_counts, &projected_properties)
            })
            .collect();
        properties.sort_by(|left, right| {
            right
                .candidates_with_property
                .cmp(&left.candidates_with_property)
                .then_with(|| left.property.cmp(&right.property))
        });

        let gap_report = build_gap_report(&properties, candidates_matched, gap_min_count);

        ProfileReport {
            candidates_matched,
            properties,
            gap_report,
        }
    }
}

fn build_property_profile_entry(
    property: String,
    shape_counts: HashMap<PropertyShape, usize>,
    projected_properties: &HashSet<&str>,
) -> PropertyProfileEntry {
    let candidates_with_property: usize = shape_counts.values().sum();
    // Ties (equally-frequent shapes on the same property) break toward whichever `PropertyShape`
    // variant sorts first below -- rare in practice (real Wikidata properties overwhelmingly
    // agree on one datatype) and, when it happens, any deterministic choice is as defensible as
    // any other for a diagnostic reader to see and investigate further.
    let dominant_shape = shape_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then(right.0.cmp(left.0)))
        .map_or(PropertyShape::Complex, |(&shape, _)| shape);
    let shapes: BTreeMap<&'static str, usize> = shape_counts
        .into_iter()
        .map(|(shape, count)| (shape.as_str(), count))
        .collect();
    let already_projected = projected_properties.contains(property.as_str());

    PropertyProfileEntry {
        property,
        candidates_with_property,
        dominant_shape: dominant_shape.as_str(),
        shapes,
        already_projected,
    }
}

/// The properties `profile` would recommend adding to `projected_fields`: present on at least
/// `gap_min_count` matched candidates, classified with a simple (non-`Complex`) dominant shape,
/// and not already named by a `projected_fields` entry. Already sorted by frequency descending,
/// inherited from `properties`'s own order.
// A run's `candidates_matched`/`candidates_with_property` counts are bounded by lines scanned --
// nowhere near f64's 2^53 exact-integer range even against the full ~repository-sized dump -- so
// the cast below loses no actual precision; it only trips the lint that can't see that bound.
#[allow(clippy::cast_precision_loss)]
fn build_gap_report(
    properties: &[PropertyProfileEntry],
    candidates_matched: usize,
    gap_min_count: usize,
) -> Vec<GapReportEntry> {
    properties
        .iter()
        .filter(|entry| {
            !entry.already_projected
                && entry.dominant_shape != PropertyShape::Complex.as_str()
                && entry.candidates_with_property >= gap_min_count
        })
        .map(|entry| GapReportEntry {
            property: entry.property.clone(),
            shape: entry.dominant_shape,
            candidates_with_property: entry.candidates_with_property,
            candidate_fraction: if candidates_matched == 0 {
                0.0
            } else {
                entry.candidates_with_property as f64 / candidates_matched as f64
            },
        })
        .collect()
}

#[derive(Serialize)]
struct PropertyProfileEntry {
    property: String,
    candidates_with_property: usize,
    dominant_shape: &'static str,
    shapes: BTreeMap<&'static str, usize>,
    /// Whether this property is already named by a `projected_fields` entry in the spec that
    /// produced this report -- what `gap_report` filters on.
    already_projected: bool,
}

#[derive(Serialize)]
struct GapReportEntry {
    property: String,
    shape: &'static str,
    candidates_with_property: usize,
    candidate_fraction: f64,
}

#[derive(Serialize)]
struct ProfileReport {
    candidates_matched: usize,
    properties: Vec<PropertyProfileEntry>,
    gap_report: Vec<GapReportEntry>,
}

/// Runs the schema-discovery/diagnostic pass: classifies every present claim property on each
/// matched candidate into a shape, and writes a frequency+shape report plus a coverage-gap
/// report to `config.output_path`.
///
/// # Errors
/// Returns an error if the spec or input can't be read, or if the report can't be written.
pub fn run_profile(config: &ProfileConfig) -> io::Result<()> {
    let spec = CompiledSpec::load(&config.spec_path)?;
    eprintln!(
        "[wikisieve] profile input={} spec={} output={} markers={} projected_fields={} \
         max_rows={} gap_min_count={}",
        config.input_path.display(),
        config.spec_path.display(),
        config.output_path.display(),
        spec.markers.len(),
        spec.projected_fields.len(),
        config
            .max_rows
            .map_or_else(|| "none".to_string(), |value| value.to_string()),
        config.gap_min_count,
    );

    // Shares `crate::dump_source`'s parallel reader with `extract::InputSource::open`,
    // so a profile run and an extract run read the dump the same way.
    let decoder = crate::dump_source::open_parallel_gzip_source(&config.input_path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, decoder);
    let prefilter = Prefilter::build(&spec);

    let mut stats = Stats::default();
    let mut lines_to_skip = 0usize;
    let mut raw_line = Vec::with_capacity(64 * 1024);
    let mut batch: Vec<Vec<u8>> = Vec::with_capacity(DEFAULT_PROGRESS_EVERY);
    let mut accumulator = ProfileAccumulator::new();

    loop {
        let source_exhausted_or_capped = fill_wikidata_batch(
            &mut reader,
            &mut raw_line,
            &mut batch,
            &mut lines_to_skip,
            &prefilter,
            &mut stats,
            config.max_rows,
        )?;

        let results: Vec<Result<Option<CandidateProfile<'_>>, serde_json::Error>> = batch
            .par_iter()
            .map(|line| profile_candidate(line, &spec))
            .collect();

        for result in results {
            match result {
                Ok(Some(profile)) => accumulator.record_candidate(&profile),
                Ok(None) => {}
                Err(_) => stats.skipped_parse_errors += 1,
            }
        }

        if !batch.is_empty() {
            eprintln!(
                "[wikisieve] profile progress lines_scanned={} candidates={} matched={}",
                stats.lines_scanned, stats.candidates_scanned, accumulator.candidates_matched,
            );
        }

        if source_exhausted_or_capped {
            break;
        }
    }

    let properties_observed = accumulator.property_shape_counts.len();
    let report = accumulator.into_report(&spec, config.gap_min_count);
    write_profile_report(&config.output_path, &report)?;

    eprintln!(
        "[wikisieve] profile complete lines_scanned={} candidates_scanned={} \
         candidates_matched={} properties_observed={} gap_report_entries={}",
        stats.lines_scanned,
        stats.candidates_scanned,
        report.candidates_matched,
        properties_observed,
        report.gap_report.len(),
    );
    Ok(())
}

fn write_profile_report(path: &Path, report: &ProfileReport) -> io::Result<()> {
    create_parent_dir(path)?;
    let payload = serde_json::to_vec_pretty(report).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "failed to serialize profile report for {}: {error}",
                path.display()
            ),
        )
    })?;
    fs::write(path, payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        args, as_dump_array, path_arg, read_file, unmatched_candidate_line, write_gzipped,
        ScopedTempDir,
    };

    // -----------------------------------------------------------------------------------------
    // `profile` subcommand
    // -----------------------------------------------------------------------------------------

    fn profile_config_for(values: &[&str]) -> ProfileConfig {
        ProfileConfig::from_args(args(values)).unwrap()
    }

    fn profile_config_error(values: &[&str]) -> io::Error {
        ProfileConfig::from_args(args(values)).unwrap_err()
    }

    #[test]
    fn profile_config_from_args_parses_required_arguments_with_defaults() {
        let config = profile_config_for(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "report.json",
        ]);
        assert_eq!(config.input_path, PathBuf::from("in.jsonl.gz"));
        assert_eq!(config.spec_path, PathBuf::from("spec.json"));
        assert_eq!(config.output_path, PathBuf::from("report.json"));
        assert_eq!(config.max_rows, None);
        assert_eq!(config.gap_min_count, 1);
    }

    #[test]
    fn profile_config_from_args_parses_optional_flags() {
        let config = profile_config_for(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "report.json",
            "--max-rows",
            "500",
            "--gap-min-count",
            "10",
        ]);
        assert_eq!(config.max_rows, Some(500));
        assert_eq!(config.gap_min_count, 10);
    }

    #[test]
    fn profile_config_from_args_rejects_missing_required_argument() {
        let error = profile_config_error(&["--input", "in.jsonl.gz", "--spec", "spec.json"]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--output"));
    }

    #[test]
    fn profile_config_from_args_rejects_unknown_argument() {
        let error = profile_config_error(&[
            "--input",
            "in.jsonl.gz",
            "--spec",
            "spec.json",
            "--output",
            "report.json",
            "--bogus",
        ]);
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--bogus"));
    }

    /// A spec with one field already projected (`P1278`/`lei`), leaving `P31` (the marker) and
    /// any other property a candidate carries as gap-report candidates.
    fn write_profile_spec(dir: &Path) -> PathBuf {
        let path = dir.join("profile-spec.json");
        fs::write(
            &path,
            r#"{
                "markers": [
                    { "property": "P31", "match": { "type": "qid_set", "qids": ["Q783794"] } }
                ],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1278", "field": "lei" }
                ]
            }"#,
        )
        .unwrap();
        path
    }

    /// A matching candidate carrying: `P31` (`entity_id_list`, the marker), `P1278`
    /// (`text_list`, already projected), `P2853` (`entity_id_list`, not projected -- a gap
    /// candidate) present only when `with_duns` is set, and `P2049` (quantity, complex -- never
    /// a gap candidate).
    fn profile_candidate_line(index: usize, with_duns: bool) -> String {
        let mut claims = serde_json::json!({
            "P31": [{"mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q783794"}}}}],
            "P1278": [{"mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": format!("LEI-{index}")}}}],
            "P2049": [{"mainsnak": {"snaktype": "value", "datavalue": {"type": "quantity", "value": {"amount": "+1"}}}}]
        });
        if with_duns {
            claims["P2853"] = serde_json::json!(
                [{"mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q1"}}}}]
            );
        }
        serde_json::json!({"id": format!("Q{index}"), "claims": claims}).to_string()
    }

    fn read_profile_report(path: &Path) -> serde_json::Value {
        serde_json::from_str(&read_file(path)).unwrap()
    }

    fn property_entry<'a>(report: &'a serde_json::Value, property: &str) -> &'a serde_json::Value {
        report["properties"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["property"] == property)
            .unwrap_or_else(|| panic!("no profile entry for {property}"))
    }

    fn gap_properties(report: &serde_json::Value) -> Vec<String> {
        report["gap_report"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["property"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn run_profile_reports_frequency_shape_and_gap_entries() {
        let dir = ScopedTempDir::new("profile-basic");
        let spec_path = write_profile_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("report.json");

        // Three matching candidates, only the first two carrying P2853 -- and one line the P31
        // marker rejects outright, so `candidates_matched` (3) has to differ from lines scanned.
        write_gzipped(
            &input_path,
            &as_dump_array(vec![
                profile_candidate_line(1, true),
                profile_candidate_line(2, true),
                profile_candidate_line(3, false),
                unmatched_candidate_line(4),
            ]),
        );

        let config = profile_config_for(&[
            "--input",
            path_arg(&input_path).as_str(),
            "--spec",
            path_arg(&spec_path).as_str(),
            "--output",
            path_arg(&output_path).as_str(),
            "--gap-min-count",
            "2",
        ]);
        run_profile(&config).unwrap();

        let report = read_profile_report(&output_path);
        assert_eq!(report["candidates_matched"], 3);

        let lei_entry = property_entry(&report, "P1278");
        assert_eq!(lei_entry["candidates_with_property"], 3);
        assert_eq!(lei_entry["dominant_shape"], "text_list");
        assert_eq!(lei_entry["already_projected"], true);

        let duns_entry = property_entry(&report, "P2853");
        assert_eq!(duns_entry["candidates_with_property"], 2);
        assert_eq!(duns_entry["dominant_shape"], "entity_id_list");
        assert_eq!(duns_entry["already_projected"], false);

        let quantity_entry = property_entry(&report, "P2049");
        assert_eq!(quantity_entry["dominant_shape"], "complex");
        assert_eq!(quantity_entry["already_projected"], false);

        // The gap report answers "what could I add that I'm not already projecting": P2853
        // qualifies (simple shape, not projected, meets --gap-min-count 2); P1278 is excluded
        // because it's already projected, and P2049 because it's complex. P31 (the marker
        // property) is not named by any `projected_fields` entry either, so it qualifies too --
        // the report is literal about `projected_fields` coverage, not about the envelope's own
        // hardcoded `instance_of` field.
        let gap = gap_properties(&report);
        assert!(gap.contains(&"P2853".to_string()));
        assert!(gap.contains(&"P31".to_string()));
        assert!(!gap.contains(&"P1278".to_string()));
        assert!(!gap.contains(&"P2049".to_string()));
    }

    #[test]
    fn run_profile_gap_report_respects_gap_min_count() {
        let dir = ScopedTempDir::new("profile-gap-threshold");
        let spec_path = write_profile_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("report.json");

        write_gzipped(
            &input_path,
            &as_dump_array(vec![
                profile_candidate_line(1, true),
                profile_candidate_line(2, false),
            ]),
        );

        let config = profile_config_for(&[
            "--input",
            path_arg(&input_path).as_str(),
            "--spec",
            path_arg(&spec_path).as_str(),
            "--output",
            path_arg(&output_path).as_str(),
            "--gap-min-count",
            "2",
        ]);
        run_profile(&config).unwrap();

        let report = read_profile_report(&output_path);
        // P2853 is present on only one of the two matched candidates -- below the threshold, so
        // it's excluded from the gap report even though it still shows up in `properties`.
        let duns_entry = property_entry(&report, "P2853");
        assert_eq!(duns_entry["candidates_with_property"], 1);
        assert!(!gap_properties(&report).contains(&"P2853".to_string()));
    }

    #[test]
    fn run_profile_respects_max_rows() {
        let dir = ScopedTempDir::new("profile-max-rows");
        let spec_path = write_profile_spec(dir.path());
        let input_path = dir.path().join("dump.jsonl.gz");
        let output_path = dir.path().join("report.json");

        write_gzipped(
            &input_path,
            &as_dump_array(
                (1..=5)
                    .map(|index| profile_candidate_line(index, true))
                    .collect(),
            ),
        );

        let config = profile_config_for(&[
            "--input",
            path_arg(&input_path).as_str(),
            "--spec",
            path_arg(&spec_path).as_str(),
            "--output",
            path_arg(&output_path).as_str(),
            "--max-rows",
            "2",
        ]);
        run_profile(&config).unwrap();

        let report = read_profile_report(&output_path);
        assert_eq!(report["candidates_matched"], 2);
    }
}
