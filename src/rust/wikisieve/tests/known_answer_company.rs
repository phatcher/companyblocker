//! Known-answer test for the production company spec: runs the built `wikisieve`
//! binary with today's real production spec
//! (`src/acquisition/catalog/projections/wikidata-company.json`) over the tracked
//! `src/tests/acquisition/fixtures/wikidata-benchmark-sample.jsonl.gz` benchmark sample, and
//! diffs the result against a checked-in, field-by-field-reviewed expected output.
//!
//! Unlike `extract_company.rs`'s small in-crate fixture, proving parity with the frozen
//! `main.rs` extractor over five hand-picked entities, this test deliberately reads outside the
//! crate on both ends: the real spec (so a deliberate spec change is caught automatically
//! rather than checked against a copy that can drift) and the real tracked sample (so
//! post-freeze fields -- `multi_lang_text` variants, the jurisdiction-scoped
//! `company_number_gb`/`_fr`/`_de`, and anything added later -- are checked at 10,000-entity
//! scale instead of five). Only the spec's `qid_closure_file` marker path is patched, to a small
//! checked-in subset of the real P279 closure (`tests/fixtures/known_answer/p279.json`)
//! covering just the QIDs this sample's entities actually carry on `P1454`: a 61-entry subset
//! of the real, roughly 49K-entry closure, holding exactly the QIDs that appear on some
//! entity's `P1454` claim in the sample and are true members of the real closure. Every
//! candidate QID absent from the real closure is also absent from the subset, so matching and
//! `matched_company_type_qids` filtering behave as they would against the full closure for this
//! sample; entities whose only `P1454` value is one of the four candidate QIDs outside the real
//! closure were checked to fail to match.
//!
//! `tests/fixtures/known_answer/expected_output.jsonl` is the 117 records the production spec
//! emits over the sample, reviewed field by field against the sample's raw claims for these
//! entities, which between them exercise every post-freeze field:
//!
//! - `Q32141` (Cathay Pacific), `Q60188` (Mobico Group), `Q76473` (University of Southampton),
//!   `Q81230` (Siemens): `company_number_gb` matches each entity's own `P2622` value, distinct
//!   from `lei`'s `P1278` value.
//! - `Q75706` (Ministry of the Armed Forces): `company_number_fr` matches `P1616`, with
//!   `company_number_gb`/`_de` both empty.
//! - `Q8093` (Nintendo): `official_name`/`official_name_en`/`official_name_variants` match
//!   `P1448`'s four values (three Japanese, one English), each variant keeping its language and
//!   `official_name_en` holding only the English one.
//! - `Q81230` (Siemens), `Q8093` (Nintendo): `legal_form` and `matched_company_type_qids` are
//!   equal, since every `P1454` value either entity carries is a closure member.
//! - `Q1201` (Saarland): matches only through the `P1454` closure marker, proving the
//!   closure-only match path fires.
//!
//! A deliberate spec or projection change regenerates the expected output in the same change:
//!
//! 1. Build a release binary: `cargo build --release`.
//! 2. Run it over the tracked sample with the production spec, its `qid_closure_file` patched to
//!    `tests/fixtures/known_answer/p279.json` exactly as `write_resolved_spec` below does, with
//!    `--output tests/fixtures/known_answer/expected_output.jsonl`.
//! 3. Review the diff field by field for every entity it touches, and extend the list above if
//!    the change reaches new fields.
//! 4. If the change adds a new `P1454` value to any sample entity, re-derive the closure subset
//!    from the real `p279.json` so it still holds exactly the true closure members among the
//!    sample's `P1454` values, and check it in with the regenerated output.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("resolve repo root from CARGO_MANIFEST_DIR")
}

fn crate_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/known_answer")
}

/// Parses a JSONL file into a map keyed by each record's own `id` field, so the comparison
/// below can name a specific entity rather than only an ordinal line number.
fn read_jsonl_records_by_id(path: &Path) -> BTreeMap<String, serde_json::Value> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|error| {
                panic!("invalid JSON line in {}: {error}", path.display())
            });
            let id = record
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("record with no `id` field in {}", path.display()))
                .to_string();
            (id, record)
        })
        .collect()
}

/// Writes a copy of the real production spec at `destination_path`, with every
/// `qid_closure_file` marker's `path` repointed at the checked-in closure fixture, so the run
/// below exercises today's actual spec content without needing the real, ~49K-entry P279
/// closure file that a real acquisition run downloads fresh alongside its dump.
fn write_resolved_spec(destination_path: &Path) -> PathBuf {
    let real_spec_path =
        repo_root().join("src/acquisition/catalog/projections/wikidata-company.json");
    let spec_text = fs::read_to_string(&real_spec_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", real_spec_path.display()));
    let mut spec: serde_json::Value = serde_json::from_str(&spec_text)
        .unwrap_or_else(|error| panic!("invalid JSON in {}: {error}", real_spec_path.display()));

    let closure_path = crate_fixtures_dir().join("p279.json");
    let closure_path_text = closure_path
        .to_str()
        .expect("closure fixture path is valid UTF-8")
        .to_string();

    let markers = spec
        .get_mut("markers")
        .and_then(serde_json::Value::as_array_mut)
        .unwrap_or_else(|| panic!("{} has no `markers` array", real_spec_path.display()));
    let mut patched_markers = 0;
    for marker in markers.iter_mut() {
        let Some(match_value) = marker.get_mut("match") else {
            continue;
        };
        if match_value.get("type").and_then(serde_json::Value::as_str) == Some("qid_closure_file")
        {
            match_value["path"] = serde_json::Value::String(closure_path_text.clone());
            patched_markers += 1;
        }
    }
    assert!(
        patched_markers > 0,
        "expected at least one qid_closure_file marker in {}, found none to repoint at the \
         checked-in closure fixture",
        real_spec_path.display()
    );

    fs::write(
        destination_path,
        serde_json::to_string(&spec).expect("serialize resolved spec"),
    )
    .unwrap_or_else(|error| {
        panic!("failed to write {}: {error}", destination_path.display())
    });
    destination_path.to_path_buf()
}

/// Every field name present on either side where the two records disagree, for a failure
/// message naming exactly which fields drifted rather than dumping both full records.
fn differing_fields(expected: &serde_json::Value, actual: &serde_json::Value) -> Vec<String> {
    let mut fields = std::collections::BTreeSet::new();
    if let Some(object) = expected.as_object() {
        fields.extend(object.keys().cloned());
    }
    if let Some(object) = actual.as_object() {
        fields.extend(object.keys().cloned());
    }
    fields
        .into_iter()
        .filter(|field| expected.get(field) != actual.get(field))
        .collect()
}

#[test]
fn production_spec_matches_reviewed_output_over_benchmark_sample() {
    let expected_output_path = crate_fixtures_dir().join("expected_output.jsonl");
    let input_path =
        repo_root().join("src/tests/acquisition/fixtures/wikidata-benchmark-sample.jsonl.gz");
    assert!(
        input_path.exists(),
        "tracked benchmark sample not found at {}",
        input_path.display()
    );

    let output_dir = std::env::temp_dir().join(format!(
        "wikisieve-known-answer-test-{}",
        std::process::id()
    ));
    fs::create_dir_all(&output_dir).expect("create temp output dir");
    let resolved_spec_path = write_resolved_spec(&output_dir.join("resolved-spec.json"));
    let output_path = output_dir.join("actual_output.jsonl");

    let binary_path = PathBuf::from(env!("CARGO_BIN_EXE_wikisieve"));
    let status = Command::new(&binary_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--spec")
        .arg(&resolved_spec_path)
        .arg("--output")
        .arg(&output_path)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", binary_path.display()));
    assert!(status.success(), "wikisieve exited with {status}");

    let actual = read_jsonl_records_by_id(&output_path);
    let expected = read_jsonl_records_by_id(&expected_output_path);

    let mut failures = Vec::new();

    let missing_ids: Vec<&String> = expected.keys().filter(|id| !actual.contains_key(*id)).collect();
    if !missing_ids.is_empty() {
        failures.push(format!(
            "missing ids (expected but not produced): {missing_ids:?}"
        ));
    }
    let extra_ids: Vec<&String> = actual.keys().filter(|id| !expected.contains_key(*id)).collect();
    if !extra_ids.is_empty() {
        failures.push(format!(
            "extra ids (produced but not expected): {extra_ids:?}"
        ));
    }

    for (id, expected_record) in &expected {
        let Some(actual_record) = actual.get(id) else {
            continue; // already reported above as a missing id
        };
        let differing = differing_fields(expected_record, actual_record);
        if !differing.is_empty() {
            failures.push(format!("{id}: differing fields {differing:?}"));
        }
    }

    fs::remove_dir_all(&output_dir).ok();

    assert!(
        failures.is_empty(),
        "production spec output over the benchmark sample no longer matches {}:\n{}\n\
         if this is a deliberate spec or projection change, regenerate the fixture and review \
         the diff (see this test file's doc comment)",
        expected_output_path.display(),
        failures.join("\n")
    );
}
