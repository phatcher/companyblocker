//! Test-only fixtures shared across this crate's module test suites: a temp-directory guard, a
//! CLI-args builder, and the small file-reading helpers most chunk/output tests need. Kept in one
//! place because every module that writes chunk or raw-candidate output tests it the same way --
//! duplicating these per module would let the copies drift.

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::extract::Config;
use crate::output_sink::raw_chunk_path;

pub(crate) fn args(values: &[&str]) -> impl Iterator<Item = String> {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .into_iter()
}

/// A temp directory scoped to one test, removed when the guard drops.
///
/// Deliberately hand-rolled rather than pulling in `tempfile` as a dev-dependency: the only
/// property these tests actually need beyond the crate's earlier
/// `temp_dir().join(format!("...-{pid}"))` pattern is cleanup that survives a failing
/// assertion (that pattern's trailing `remove_dir_all` never runs on a panic unwind, so every
/// red test leaked its fixture directory). `Drop` gives exactly that in a dozen lines, where
/// `tempfile` would add a transitive dependency tree to a crate whose five direct dependencies
/// are its whole surface -- and its randomised naming buys nothing here that a per-process
/// counter doesn't. Uniqueness within a run comes from that counter, so two tests sharing a
/// label can't collide the way the pid-only names could.
pub(crate) struct ScopedTempDir {
    path: PathBuf,
}

impl ScopedTempDir {
    pub(crate) fn new(label: &str) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "wikisieve-test-{label}-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        fs::remove_dir_all(&path).ok();
        fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScopedTempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).ok();
    }
}

pub(crate) const TEST_PREFIX: &str = "part-";

pub(crate) fn chunk_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("{TEST_PREFIX}{index:06}.jsonl"))
}

/// Every file name currently in `dir`, sorted -- lets a test assert what the chunk directory
/// holds as a whole, so a leftover `.tmp` or an un-removed stale chunk fails the assertion
/// rather than going unnoticed beside a narrower "the file I wanted exists" check.
pub(crate) fn entry_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

pub(crate) fn read_file(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

/// The decompressed content of a raw-candidate artifact. Every assertion about capture goes
/// through a real gzip decode rather than through the bytes on disk, so a stream finished
/// without its trailer -- which still leaves a plausible-looking file -- fails the test.
pub(crate) fn read_gz_file(path: &Path) -> String {
    let mut contents = String::new();
    MultiGzDecoder::new(File::open(path).unwrap())
        .read_to_string(&mut contents)
        .unwrap();
    contents
}

pub(crate) fn raw_chunk_ids(path: &Path) -> Vec<String> {
    read_gz_file(path)
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

pub(crate) fn path_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub(crate) fn config_for(values: &[&str]) -> Config {
    Config::from_args(args(values)).unwrap()
}

pub(crate) fn temp_chunk_path(dir: &Path, index: usize) -> PathBuf {
    dir.join(format!("{TEST_PREFIX}{index:06}.tmp"))
}

pub(crate) fn seed_chunk(dir: &Path, index: usize, contents: &str) {
    fs::write(chunk_path(dir, index), contents).unwrap();
}

pub(crate) fn seed_raw_companion(dir: &Path, index: usize, contents: &str) {
    let path = raw_chunk_path(dir, TEST_PREFIX, index);
    let mut encoder = GzEncoder::new(File::create(&path).unwrap(), Compression::fast());
    encoder.write_all(contents.as_bytes()).unwrap();
    encoder.finish().unwrap();
}

pub(crate) fn write_gzipped(path: &Path, lines: &[String]) {
    let mut encoder = GzEncoder::new(File::create(path).unwrap(), Compression::fast());
    for line in lines {
        encoder.write_all(line.as_bytes()).unwrap();
        encoder.write_all(b"\n").unwrap();
    }
    encoder.finish().unwrap();
}

pub(crate) fn open_gzipped(path: &Path) -> BufReader<MultiGzDecoder<File>> {
    BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(File::open(path).unwrap()))
}

/// Wraps `entries` as a Wikidata dump array: a leading `[`, a trailing `]`, and a comma after
/// every element -- all of which `trim_wikidata_line` has to strip on the way through.
pub(crate) fn as_dump_array(entries: Vec<String>) -> Vec<String> {
    let mut lines = vec!["[".to_string()];
    lines.extend(entries.into_iter().map(|entry| format!("{entry},")));
    lines.push("]".to_string());
    lines
}

pub(crate) fn ids_in(path: &Path) -> Vec<String> {
    read_file(path)
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap()["id"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

pub(crate) fn summary_field(path: &Path, field: &str) -> u64 {
    serde_json::from_str::<serde_json::Value>(&read_file(path)).unwrap()[field]
        .as_u64()
        .unwrap()
}

/// One P31 marker and one `text_list` projected field, matching `candidate_line`'s claims.
pub(crate) fn write_spec(dir: &Path) -> PathBuf {
    let path = dir.join("spec.json");
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

/// `write_spec`'s spec with one more `projected_fields` entry: the later field addition
/// (labels, registry numbers, or anything else) that the cache exists so
/// nobody has to rescan the dump for.
pub(crate) fn write_extended_spec(dir: &Path) -> PathBuf {
    let path = dir.join("extended-spec.json");
    fs::write(
        &path,
        r#"{
            "markers": [
                { "property": "P31", "match": { "type": "qid_set", "qids": ["Q783794"] } }
            ],
            "match_logic": "any",
            "projected_fields": [
                { "property": "P1278", "field": "lei" },
                { "property": "P571", "field": "inception", "shape": "time" }
            ]
        }"#,
    )
    .unwrap();
    path
}

/// A line the prefilter accepts and the P31 marker matches: it gets emitted.
///
/// Carries a P571 claim `write_spec` does not project. That is not decoration: it stands for
/// every claim the dump holds and today's spec ignores, which is exactly what a raw-candidate
/// cache preserves and a projected record throws away -- see
/// `write_extended_spec` and the replay tests that use it.
pub(crate) fn candidate_line(index: usize) -> String {
    serde_json::json!({
        "id": format!("Q{index}"),
        "type": "item",
        "labels": {"en": {"language": "en", "value": format!("Company {index}")}},
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
            "P1278": [{"mainsnak": {"datavalue": {"value": format!("LEI-{index}")}}}],
            "P571": [{"mainsnak": {"datavalue": {"value": {
                "time": format!("+20{index:02}-01-01T00:00:00Z")
            }}}}]
        }
    })
    .to_string()
}

/// A line the byte-level prefilter rejects outright -- it carries no `"P31"` substring.
pub(crate) fn non_candidate_line(index: usize) -> String {
    serde_json::json!({"id": format!("N{index}"), "type": "item", "claims": {}}).to_string()
}

/// A line the prefilter accepts but no marker matches, so `project_candidate` returns `None`.
///
/// Carries `write_spec`'s marker qid (`Q783794`) under an unrelated property (`P17`), not under
/// `P31` where the marker actually looks: the value search can't tell that apart from a
/// real match and admits the line anyway, so this also exercises its conservativeness -- the full
/// marker check inside `project_candidate`/`profile_candidate` is what correctly rejects it.
pub(crate) fn unmatched_candidate_line(index: usize) -> String {
    serde_json::json!({
        "id": format!("U{index}"),
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
            "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}]
        }
    })
    .to_string()
}

/// A line that passes the prefilter and then fails to parse: the `skipped_parse_errors` arm.
/// Carries both `write_spec`'s marker property (`P31`) and its qid (`Q783794`), quoted and intact,
/// so the value search still admits it despite the truncation -- otherwise this would be
/// rejected by the prefilter instead of reaching the JSON parser this fixture means to exercise.
pub(crate) fn malformed_candidate_line() -> String {
    r#"{"id":"BROKEN","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}"#
        .to_string()
}
