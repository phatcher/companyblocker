//! End-to-end test of the built `wikisieve` binary against the checked-in company spec and a
//! small sample of real entities drawn from the comparison script's own benchmark fixture --
//! The crate holds only what ships, proven by an integration test that reads nothing
//! outside `src/rust/wikisieve`.

use flate2::read::MultiGzDecoder;
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Parses a JSONL file into one `serde_json::Value` per non-empty line, so the comparison below
/// doesn't depend on exact byte formatting -- only on the same fields and values being present.
fn read_jsonl_records(path: &std::path::Path) -> Vec<serde_json::Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("invalid JSON line in {}: {error}", path.display()))
        })
        .collect()
}

#[test]
fn extracts_company_records_from_entity_sample_via_built_binary() {
    let fixtures = fixtures_dir();
    let spec_path = fixtures.join("company.json");
    let input_path = fixtures.join("entities_sample.jsonl.gz");
    let expected_output_path = fixtures.join("expected_output.jsonl");

    let output_dir = std::env::temp_dir().join(format!(
        "wikisieve-extract-company-test-{}",
        std::process::id()
    ));
    fs::create_dir_all(&output_dir).expect("create temp output dir");
    let output_path = output_dir.join("actual_output.jsonl");

    let binary_path = PathBuf::from(env!("CARGO_BIN_EXE_wikisieve"));
    let status = Command::new(&binary_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--output")
        .arg(&output_path)
        .status()
        .unwrap_or_else(|error| {
            panic!("failed to run {}: {error}", binary_path.display())
        });
    assert!(status.success(), "wikisieve exited with {status}");

    let actual_records = read_jsonl_records(&output_path);
    let expected_records = read_jsonl_records(&expected_output_path);

    assert_eq!(
        actual_records, expected_records,
        "extracted records did not match {}",
        expected_output_path.display()
    );

    fs::remove_dir_all(&output_dir).ok();
}

/// `--input -` (uncompressed lines on standard input, the shape `rapidgzip -d -c
/// <dump.gz> | wikisieve --input -` produces) has to emit exactly what `--input <path>` emits
/// from the same gzip-compressed dump, over the tracked benchmark sample -- the only thing this
/// item requires be true of the new input path, proven here rather than assumed from the
/// `InputSource` unit tests, which stop at variant selection and a single read.
#[test]
fn stdin_input_matches_gzip_file_input_on_entity_sample() {
    let fixtures = fixtures_dir();
    let spec_path = fixtures.join("company.json");
    let input_path = fixtures.join("entities_sample.jsonl.gz");

    let output_dir = std::env::temp_dir().join(format!(
        "wikisieve-extract-company-stdin-test-{}",
        std::process::id()
    ));
    fs::create_dir_all(&output_dir).expect("create temp output dir");
    let file_output_path = output_dir.join("file_output.jsonl");
    let stdin_output_path = output_dir.join("stdin_output.jsonl");

    let binary_path = PathBuf::from(env!("CARGO_BIN_EXE_wikisieve"));

    let file_status = Command::new(&binary_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--output")
        .arg(&file_output_path)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", binary_path.display()));
    assert!(file_status.success(), "file-input run exited with {file_status}");

    let mut decompressed = String::new();
    MultiGzDecoder::new(fs::File::open(&input_path).unwrap())
        .read_to_string(&mut decompressed)
        .unwrap_or_else(|error| panic!("failed to decompress {}: {error}", input_path.display()));

    let mut child = Command::new(&binary_path)
        .arg("--input")
        .arg("-")
        .arg("--spec")
        .arg(&spec_path)
        .arg("--output")
        .arg(&stdin_output_path)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("failed to spawn {}: {error}", binary_path.display()));
    child
        .stdin
        .take()
        .expect("child stdin")
        .write_all(decompressed.as_bytes())
        .expect("write decompressed dump to child stdin");
    let stdin_status = child.wait().expect("wait for stdin-input run");
    assert!(stdin_status.success(), "stdin-input run exited with {stdin_status}");

    let file_records = read_jsonl_records(&file_output_path);
    let stdin_records = read_jsonl_records(&stdin_output_path);
    assert_eq!(
        stdin_records, file_records,
        "--input - output did not match --input <path> output for {}",
        input_path.display()
    );
    assert!(!stdin_records.is_empty(), "expected at least one matched record");

    fs::remove_dir_all(&output_dir).ok();
}
