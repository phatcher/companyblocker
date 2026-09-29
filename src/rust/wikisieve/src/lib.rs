//! Generic, spec-driven Wikidata candidate extraction engine.
//!
//! A `Spec` names one or more `markers` (which claim properties, matched against either an
//! inline QID set or a QID closure file, make a line a "candidate") and `projected_fields`
//! (which claim properties get projected into the output JSONL, and under what field name).
//! The engine never deep-parses a claim property the spec doesn't reference, and parses each
//! referenced property's claim array at most once per candidate line -- see `ClaimCache`.

use aho_corasick::AhoCorasick;
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

pub mod cli;
pub mod dump_source;
pub mod extract;
pub mod merge_chunks;
pub mod output_sink;
pub mod profile;
pub mod resume;
pub mod scan;
#[cfg(test)]
pub(crate) mod test_support;
pub mod version;

/// Zero-copy top-level view of an entity line: claim/label/description/alias/sitelink bodies
/// are captured as raw JSON spans instead of being fully parsed, since only the properties a
/// spec actually references are ever deep-parsed.
#[derive(Deserialize)]
pub struct ShallowEntity<'a> {
    id: Option<&'a str>,
    #[serde(rename = "type")]
    entity_type: Option<&'a str>,
    modified: Option<&'a str>,
    #[serde(default, borrow)]
    labels: HashMap<&'a str, &'a RawValue>,
    #[serde(default, borrow)]
    descriptions: HashMap<&'a str, &'a RawValue>,
    #[serde(default, borrow)]
    aliases: HashMap<&'a str, &'a RawValue>,
    #[serde(default, borrow)]
    sitelinks: HashMap<&'a str, &'a RawValue>,
    #[serde(default, borrow)]
    claims: HashMap<&'a str, &'a RawValue>,
}

#[derive(Deserialize)]
struct LangStringEntry {
    value: String,
}

#[derive(Deserialize)]
struct SubclassEntry {
    subclass: String,
}

// ---------------------------------------------------------------------------------------------
// Spec: the raw, on-disk JSON shape.
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawSpec {
    markers: Vec<RawMarker>,
    match_logic: String,
    projected_fields: Vec<RawProjectedField>,
}

#[derive(Deserialize)]
struct RawMarker {
    property: String,
    #[serde(rename = "match")]
    match_rule: RawMarkerMatch,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawMarkerMatch {
    QidSet { qids: Vec<String> },
    QidClosureFile { path: String },
}

#[derive(Deserialize)]
struct RawProjectedField {
    property: String,
    field: String,
    #[serde(default)]
    shape: Option<String>,
    /// Only meaningful for the `multi_lang_text` shape: the output field name for the
    /// English-only subset (mirrors `_MULTI_LANG_TEXT_PROPERTIES`'s `en_key` in
    /// `wikidata_projection_helpers.py`).
    #[serde(default)]
    field_en: Option<String>,
    /// Only meaningful for the `multi_lang_text` shape: the output field name for the
    /// full `{value, language}`-tagged list (mirrors `_MULTI_LANG_TEXT_PROPERTIES`'s
    /// `variants_key`).
    #[serde(default)]
    field_variants: Option<String>,
    /// Cardinality modifier, valid for the `text_list`/`entity_id_list` shapes only: `"list"`
    /// (default, today's behavior) emits the full array; `"first"` emits just the first value
    /// (or `null` if none) -- e.g. `company_number` as the first value of the `lei` property.
    #[serde(default)]
    take: Option<String>,
    /// Modifier valid for the `entity_id_list` shape only: when `true`, only claim ids that
    /// are also members of a marker's qid set on the *same* property are kept -- e.g.
    /// `matched_company_type_qids`, the subset of `legal_form`'s (P1454) claim ids that
    /// matched the P1454 marker's own `qid_closure_file`. Requires a marker on the same
    /// property to exist in the spec.
    #[serde(default)]
    match_only: Option<bool>,
}

// ---------------------------------------------------------------------------------------------
// CompiledSpec: the resolved, ready-to-run form -- QID sets loaded, closure files read, prefilter
// byte patterns built.
// ---------------------------------------------------------------------------------------------

pub struct CompiledMarker {
    pub property: String,
    pub qids: std::collections::HashSet<u32>,
}

/// Which extraction algorithm a `projected_fields` entry uses. `TextList` is the default
/// (unnamed) shape from Phase 1; the other three mirror
/// `_ENTITY_ID_PROPERTIES`/`_TIME_PROPERTIES`/`_MULTI_LANG_TEXT_PROPERTIES` in
/// `wikidata_projection_helpers.py` and their hand-written equivalents in the old `main.rs`.
pub enum ProjectedFieldShape {
    TextList,
    /// `match_only_qids`, when set (the `match_only: true` modifier), restricts the extracted
    /// claim ids to those also present in a marker's qid set on the same property -- computed
    /// once at spec-load time, not recomputed from `is_candidate_match`'s per-line matching.
    EntityIdList {
        match_only_qids: Option<std::collections::HashSet<u32>>,
    },
    Time,
    /// A single claim property fans out into three output fields: the full text-value list,
    /// the English-only subset, and the `{value, language}`-tagged full list.
    MultiLangText {
        field_en: String,
        field_variants: String,
    },
}

pub struct CompiledProjectedField {
    pub property: String,
    pub field: String,
    pub shape: ProjectedFieldShape,
    /// The `take: "first"` cardinality modifier -- see `RawProjectedField::take`. Only ever
    /// `true` for `TextList`/`EntityIdList` shapes; validated at load time.
    pub take_first: bool,
}

pub struct CompiledSpec {
    pub markers: Vec<CompiledMarker>,
    pub projected_fields: Vec<CompiledProjectedField>,
    pub prefilter_patterns: Vec<Vec<u8>>,
    /// The literal quoted form (`"Q6881511"`) of every QID any marker's `qid_set` or
    /// `qid_closure_file` can match, deduplicated across markers. A true match
    /// always carries its marker's matched QID as a literal somewhere in the line (the value of
    /// the `id` field inside that property's claim), so a line containing none of these can't
    /// possibly match, regardless of which property it's actually under.
    pub prefilter_value_patterns: Vec<Vec<u8>>,
}

impl CompiledSpec {
    /// Loads a spec JSON file and resolves it into a ready-to-run form. `qid_closure_file`
    /// paths are resolved relative to the spec file's own directory unless already absolute.
    ///
    /// # Errors
    /// Returns an error if the spec file can't be read or parsed, if `match_logic` isn't
    /// `"any"`, if a `qid_closure_file` can't be read/parsed, or if a `projected_fields` entry
    /// names an unsupported `shape`.
    pub fn load(spec_path: &Path) -> io::Result<Self> {
        let bytes = std::fs::read(spec_path)?;
        let raw: RawSpec = serde_json::from_slice(&bytes).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse spec {}: {error}", spec_path.display()),
            )
        })?;

        if raw.match_logic != "any" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported match_logic '{}' (only 'any' is supported)",
                    raw.match_logic
                ),
            ));
        }

        let spec_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));
        let mut markers = Vec::with_capacity(raw.markers.len());
        for marker in raw.markers {
            let qids = match marker.match_rule {
                RawMarkerMatch::QidSet { qids } => parse_qid_strings(&qids)?,
                RawMarkerMatch::QidClosureFile { path } => {
                    let resolved = resolve_spec_relative_path(spec_dir, &path);
                    load_closure_qids(&resolved)?
                }
            };
            markers.push(CompiledMarker {
                property: marker.property,
                qids,
            });
        }

        let mut projected_fields = Vec::with_capacity(raw.projected_fields.len());
        for field in raw.projected_fields {
            projected_fields.push(compile_projected_field(field, &markers)?);
        }

        let prefilter_patterns = markers
            .iter()
            .map(|marker| format!("\"{}\"", marker.property).into_bytes())
            .collect();

        // Union across every marker, not per-marker: the value search only has to know "could
        // this line possibly match *some* marker", the same question `has_candidate_properties`
        // already answers with an `any()` over its own patterns -- see `has_candidate_value`.
        let mut value_qids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for marker in &markers {
            value_qids.extend(marker.qids.iter().copied());
        }
        let mut prefilter_value_patterns: Vec<Vec<u8>> = value_qids
            .into_iter()
            .map(|qid| format!("\"Q{qid}\"").into_bytes())
            .collect();
        prefilter_value_patterns.sort_unstable();

        Ok(Self {
            markers,
            projected_fields,
            prefilter_patterns,
            prefilter_value_patterns,
        })
    }
}

/// Resolves one raw `projected_fields` entry into its compiled form: validates `take`/`shape`/
/// `match_only`/`field_en`/`field_variants` against each other, and (for `match_only`) looks up
/// the already-compiled marker on the same property to borrow its qid set from.
fn compile_projected_field(
    field: RawProjectedField,
    markers: &[CompiledMarker],
) -> io::Result<CompiledProjectedField> {
    let take_first = match field.take.as_deref() {
        None | Some("list") => false,
        Some("first") => true,
        Some(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported projected_fields take '{other}' for property {} \
                     (supported: list, first)",
                    field.property
                ),
            ));
        }
    };
    let match_only = field.match_only.unwrap_or(false);

    let shape = match field.shape.as_deref() {
        None | Some("text_list") => {
            reject_match_only(&field, match_only, "text_list")?;
            ProjectedFieldShape::TextList
        }
        Some("entity_id_list") => {
            let match_only_qids = if match_only {
                let marker = markers
                    .iter()
                    .find(|marker| marker.property == field.property)
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "projected_fields entry for property {} sets 'match_only' but \
                                 no marker exists on that property",
                                field.property
                            ),
                        )
                    })?;
                Some(marker.qids.clone())
            } else {
                None
            };
            ProjectedFieldShape::EntityIdList { match_only_qids }
        }
        Some("time") => {
            reject_take_first(&field, take_first, "time")?;
            reject_match_only(&field, match_only, "time")?;
            ProjectedFieldShape::Time
        }
        Some("multi_lang_text") => {
            reject_take_first(&field, take_first, "multi_lang_text")?;
            reject_match_only(&field, match_only, "multi_lang_text")?;
            let field_en = field.field_en.clone().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "projected_fields entry for property {} has shape 'multi_lang_text' \
                         but is missing 'field_en'",
                        field.property
                    ),
                )
            })?;
            let field_variants = field.field_variants.clone().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "projected_fields entry for property {} has shape 'multi_lang_text' \
                         but is missing 'field_variants'",
                        field.property
                    ),
                )
            })?;
            ProjectedFieldShape::MultiLangText {
                field_en,
                field_variants,
            }
        }
        Some(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "unsupported projected_fields shape '{other}' for property {} (supported \
                     shapes: text_list, entity_id_list, time, multi_lang_text)",
                    field.property
                ),
            ));
        }
    };

    Ok(CompiledProjectedField {
        property: field.property,
        field: field.field,
        shape,
        take_first,
    })
}

fn reject_take_first(
    field: &RawProjectedField,
    take_first: bool,
    shape_name: &str,
) -> io::Result<()> {
    if take_first {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "projected_fields entry for property {} sets 'take: first' but shape \
                 '{shape_name}' doesn't support it",
                field.property
            ),
        ));
    }
    Ok(())
}

fn reject_match_only(
    field: &RawProjectedField,
    match_only: bool,
    shape_name: &str,
) -> io::Result<()> {
    if match_only {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "projected_fields entry for property {} sets 'match_only' but shape \
                 '{shape_name}' doesn't support it (only entity_id_list does)",
                field.property
            ),
        ));
    }
    Ok(())
}

fn resolve_spec_relative_path(spec_dir: &Path, path: &str) -> PathBuf {
    let candidate = Path::new(path);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        spec_dir.join(candidate)
    }
}

fn parse_qid_strings(qids: &[String]) -> io::Result<std::collections::HashSet<u32>> {
    let mut parsed = std::collections::HashSet::with_capacity(qids.len());
    for qid in qids {
        let Some(value) = parse_qid_number(qid) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid qid '{qid}' in spec qid_set"),
            ));
        };
        parsed.insert(value);
    }
    Ok(parsed)
}

fn load_closure_qids(path: &Path) -> io::Result<std::collections::HashSet<u32>> {
    let handle = File::open(path).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to open closure file {}: {error}", path.display()),
        )
    })?;
    // Buffered deliberately: `serde_json::from_reader` on a bare `File` pulls
    // through its `IoRead` adapter a byte at a time, which cost ~10s to parse
    // p279.json's 49,176 entries (2.62 MB) against ~0.1s buffered. That is a
    // fixed startup charge on every run regardless of input size, so it
    // dominated short benchmark runs rather than merely adding to long ones.
    let entries: Vec<SubclassEntry> =
        serde_json::from_reader(io::BufReader::new(handle)).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse closure file {}: {error}", path.display()),
            )
        })?;

    let mut qids = std::collections::HashSet::with_capacity(entries.len());
    for entry in entries {
        if let Some(qid) = parse_qid_number(&entry.subclass) {
            qids.insert(qid);
        }
    }
    Ok(qids)
}

// ---------------------------------------------------------------------------------------------
// Prefilter
// ---------------------------------------------------------------------------------------------

/// Builds a reusable, precompiled substring searcher for each of `patterns` -- call once per
/// spec (not once per line) and pass the result to `has_candidate_properties`. `memchr`'s
/// `Finder` amortizes the SIMD-accelerated Two-Way search setup across every line scanned,
/// unlike a naive `windows(pattern.len())` scan whose window length isn't known until runtime
/// and so can't be specialized by the compiler the way the old hardcoded extractor's
/// compile-time-constant `windows(5)`/`windows(7)` calls could -- this was measured to be the
/// dominant cost in wikisieve's own throughput-parity gap against that hardcoded extractor (the
/// per-line prefilter scan runs sequentially over every scanned line, not just candidates, and
/// isn't parallelized the way `project_candidate` is).
#[must_use]
pub fn build_prefilter_finders(patterns: &[Vec<u8>]) -> Vec<memchr::memmem::Finder<'_>> {
    patterns.iter().map(memchr::memmem::Finder::new).collect()
}

/// Cheap byte-level check for whether a raw line is worth deep-parsing at all: does it contain
/// the literal `"<property>"` tag for at least one of the spec's marker properties? Generalizes
/// today's hardcoded Rust extractor's fixed `"P31"`/`"P1454"` window scan to whatever properties
/// the spec names. `finders` must come from `build_prefilter_finders` over the same patterns.
#[must_use]
pub fn has_candidate_properties(line: &[u8], finders: &[memchr::memmem::Finder<'_>]) -> bool {
    finders.iter().any(|finder| finder.find(line).is_some())
}

/// Builds a reusable Aho-Corasick multi-pattern byte matcher for `patterns` -- call once per spec
/// (not once per line) and pass the result to `has_candidate_value`. A spec's marker QIDs can
/// number in the tens of thousands (the production company spec's `P1454` closure alone holds
/// 49,176 entries), so testing each line against that many separate `memchr::memmem::Finder`s the
/// way `build_prefilter_finders` does for the handful of marker properties would cost more than
/// the JSON parse this prefilter exists to avoid; Aho-Corasick's single automaton tests every
/// pattern in one pass over the line instead.
///
/// # Panics
/// Panics if `patterns` fails to compile into an automaton. `CompiledSpec::load` builds
/// `patterns` from parsed QIDs as plain ASCII byte strings, which always compiles; this can only
/// fire on a caller-constructed `CompiledSpec` whose `prefilter_value_patterns` breaks that
/// invariant.
#[must_use]
pub fn build_prefilter_value_matcher(patterns: &[Vec<u8>]) -> AhoCorasick {
    AhoCorasick::new(patterns).expect("prefilter value patterns must compile into an automaton")
}

/// Cheap byte-level check for whether a raw line could possibly be a candidate: does it contain
/// the literal quoted form of at least one QID that some marker's `qid_set`/`qid_closure_file`
/// can match (e.g. `"Q6881511"`), anywhere in the line -- not necessarily as the value of the
/// marker's own property. That makes this check conservative rather than exact: a marker's QID
/// appearing under an unrelated property admits the line here, and is rejected later by the full
/// marker check (`is_candidate_match`, inside `project_candidate`/`profile_candidate`), never the
/// reverse -- this can only ever admit extra lines, not lose a real match. `value_matcher` must
/// come from `build_prefilter_value_matcher` over the same patterns.
#[must_use]
pub fn has_candidate_value(line: &[u8], value_matcher: &AhoCorasick) -> bool {
    value_matcher.is_match(line)
}

/// Bundles both prefilter stages behind the single question `fill_wikidata_batch` actually asks
/// per line: could this possibly be a candidate. Built once per run from a `CompiledSpec` (never
/// once per line -- see `build_prefilter_finders`/`build_prefilter_value_matcher`), and keeping
/// both stages behind one type is also what keeps that function's own argument list from growing
/// every time the prefilter gains another stage.
pub struct Prefilter<'a> {
    property_finders: Vec<memchr::memmem::Finder<'a>>,
    value_matcher: AhoCorasick,
}

impl<'a> Prefilter<'a> {
    /// Builds both prefilter stages from `spec`'s compiled patterns.
    #[must_use]
    pub fn build(spec: &'a CompiledSpec) -> Self {
        Self {
            property_finders: build_prefilter_finders(&spec.prefilter_patterns),
            value_matcher: build_prefilter_value_matcher(&spec.prefilter_value_patterns),
        }
    }

    /// Whether `line` could possibly be a candidate: both `has_candidate_properties` and
    /// `has_candidate_value` have to admit it. Each is independently a necessary condition for a
    /// real match (see their own docs), so requiring both stays conservative while being far more
    /// selective than either alone, the value search's whole point.
    #[must_use]
    pub fn admits(&self, line: &[u8]) -> bool {
        has_candidate_properties(line, &self.property_finders)
            && has_candidate_value(line, &self.value_matcher)
    }
}

// ---------------------------------------------------------------------------------------------
// Per-candidate claim cache -- the performance-equivalence requirement: a property referenced
// by both a marker and a projected field (e.g. legal_form's P1454) is parsed at most once.
// ---------------------------------------------------------------------------------------------

/// `'p` is the lifetime of the property-name strings callers pass to `get` (borrowed from
/// `CompiledSpec`'s own `String` fields, or a `&'static str` literal like `"P31"`) -- keying
/// `parsed` by `&'p str` instead of an owned `String` means a cache *hit* costs one hash lookup
/// and zero allocations, not a fresh heap `String` on every single call regardless of hit/miss
/// (the property-lookup cost dominates real-dump throughput at ~15 distinct properties checked
/// per candidate line, measured at throughput parity with the frozen extractor).
struct ClaimCache<'a, 'p> {
    claims: &'a HashMap<&'a str, &'a RawValue>,
    parsed: HashMap<&'p str, Option<Value>>,
    /// Memoizes `extract_claim_ids` per property, on top of `parsed`'s raw-JSON-parse caching --
    /// a property referenced by both a marker check and one or more `entity_id_list` projected
    /// fields on the *same* property (e.g. `P1454`'s marker match plus its `legal_form`/
    /// `matched_company_type_qids` fields, or `P31`'s marker match plus the `instance_of`
    /// envelope field) would otherwise re-walk the same already-parsed claim array and
    /// re-derive the same `Vec<u32>` once per reference instead of once per candidate line.
    entity_ids: HashMap<&'p str, Vec<u32>>,
    /// Memoizes `extract_claim_text_values` per property -- same rationale as `entity_ids`, for
    /// `text_list` fields that share a property (e.g. `lei`/`company_number`, both `P1278`).
    text_values: HashMap<&'p str, Vec<Value>>,
}

impl<'a, 'p> ClaimCache<'a, 'p> {
    fn new(claims: &'a HashMap<&'a str, &'a RawValue>) -> Self {
        Self {
            claims,
            parsed: HashMap::new(),
            entity_ids: HashMap::new(),
            text_values: HashMap::new(),
        }
    }

    fn get(&mut self, property: &'p str) -> Option<&Value> {
        self.parsed
            .entry(property)
            .or_insert_with(|| claim_array(self.claims, property))
            .as_ref()
    }

    /// Returns `property`'s claim ids (mirrors `extract_claim_ids`), computed at most once per
    /// candidate line regardless of how many markers/projected fields reference `property`.
    fn entity_ids(&mut self, property: &'p str) -> &[u32] {
        if !self.entity_ids.contains_key(property) {
            let ids = extract_claim_ids(self.get(property));
            self.entity_ids.insert(property, ids);
        }
        self.entity_ids.get(property).map_or(&[], Vec::as_slice)
    }

    /// Returns `property`'s claim text values (mirrors `extract_claim_text_values`), computed at
    /// most once per candidate line regardless of how many projected fields reference `property`.
    fn text_values(&mut self, property: &'p str) -> &[Value] {
        if !self.text_values.contains_key(property) {
            let values = extract_claim_text_values(self.get(property));
            self.text_values.insert(property, values);
        }
        self.text_values.get(property).map_or(&[], Vec::as_slice)
    }
}

fn claim_array(claims: &HashMap<&str, &RawValue>, property_id: &str) -> Option<Value> {
    claims
        .get(property_id)
        .and_then(|raw| serde_json::from_str(raw.get()).ok())
}

// ---------------------------------------------------------------------------------------------
// Candidate matching + projection
// ---------------------------------------------------------------------------------------------

fn is_candidate_match<'p>(cache: &mut ClaimCache<'_, 'p>, spec: &'p CompiledSpec) -> bool {
    spec.markers.iter().any(|marker| {
        cache
            .entity_ids(&marker.property)
            .iter()
            .any(|qid| marker.qids.contains(qid))
    })
}

/// Parses a candidate line against `spec`, returning the projected JSONL record if it matches
/// at least one marker, or `None` if it doesn't (or the line has no `id`).
///
/// # Errors
/// Returns an error if `line` isn't valid JSON.
pub fn project_candidate(
    line: &[u8],
    spec: &CompiledSpec,
) -> Result<Option<String>, serde_json::Error> {
    let entity: ShallowEntity = serde_json::from_slice(line)?;
    let Some(entity_id) = entity.id else {
        return Ok(None);
    };

    let mut cache = ClaimCache::new(&entity.claims);
    if !is_candidate_match(&mut cache, spec) {
        return Ok(None);
    }

    let mut payload = Map::new();
    payload.insert("id".to_string(), Value::String(entity_id.to_string()));
    payload.insert(
        "entity_type".to_string(),
        entity
            .entity_type
            .map_or(Value::Null, |value| Value::String(value.to_string())),
    );
    payload.insert(
        "modified".to_string(),
        entity
            .modified
            .map_or(Value::Null, |value| Value::String(value.to_string())),
    );
    payload.insert(
        "label_en".to_string(),
        extract_langstring_value(&entity.labels, "en").map_or(Value::Null, Value::String),
    );
    payload.insert(
        "description_en".to_string(),
        extract_langstring_value(&entity.descriptions, "en").map_or(Value::Null, Value::String),
    );
    payload.insert(
        "aliases_en".to_string(),
        extract_aliases(&entity.aliases, "en"),
    );
    payload.insert(
        "instance_of".to_string(),
        Value::Array(
            cache
                .entity_ids("P31")
                .iter()
                .copied()
                .map(format_qid)
                .map(Value::String)
                .collect(),
        ),
    );
    payload.insert(
        "sitelinks_count".to_string(),
        Value::Number(entity.sitelinks.len().into()),
    );

    for field in &spec.projected_fields {
        match &field.shape {
            ProjectedFieldShape::TextList => {
                let values =
                    list_or_first_ref(cache.text_values(&field.property), field.take_first);
                payload.insert(field.field.clone(), values);
            }
            ProjectedFieldShape::EntityIdList { match_only_qids } => {
                let values: Vec<Value> = cache
                    .entity_ids(&field.property)
                    .iter()
                    .copied()
                    .filter(|qid| {
                        match_only_qids
                            .as_ref()
                            .is_none_or(|qids| qids.contains(qid))
                    })
                    .map(format_qid)
                    .map(Value::String)
                    .collect();
                payload.insert(field.field.clone(), list_or_first(values, field.take_first));
            }
            ProjectedFieldShape::Time => {
                let value = extract_time_claim(cache.get(&field.property));
                payload.insert(field.field.clone(), value);
            }
            ProjectedFieldShape::MultiLangText {
                field_en,
                field_variants,
            } => {
                let (all_values, english_values, tagged_values) =
                    extract_claim_text_values_with_english(cache.get(&field.property));
                payload.insert(field.field.clone(), Value::Array(all_values));
                payload.insert(field_en.clone(), Value::Array(english_values));
                payload.insert(field_variants.clone(), Value::Array(tagged_values));
            }
        }
    }

    Ok(Some(Value::Object(payload).to_string()))
}

fn extract_claim_ids(claims_array: Option<&Value>) -> Vec<u32> {
    let Some(claims) = claims_array.and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut ids = Vec::with_capacity(claims.len());
    for claim_value in claims {
        let Some(value_id) = claim_mainsnak_value(claim_value).and_then(|value| {
            value
                .as_object()
                .and_then(|object| object.get("id"))
                .and_then(Value::as_str)
        }) else {
            continue;
        };
        if let Some(qid) = parse_qid_number(value_id) {
            ids.push(qid);
        }
    }
    ids
}

fn extract_claim_text_values(claims_array: Option<&Value>) -> Vec<Value> {
    let Some(claims) = claims_array.and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    for claim_value in claims {
        let Some(value) = claim_mainsnak_value(claim_value) else {
            continue;
        };

        if let Some(text_value) = value.as_str() {
            values.push(Value::String(text_value.to_string()));
            continue;
        }

        let Some(text_object) = value.as_object() else {
            continue;
        };
        let Some(text_value) = text_object.get("text").and_then(Value::as_str) else {
            continue;
        };
        values.push(Value::String(text_value.to_string()));
    }

    values
}

/// Applies the `take: "first"` cardinality modifier: `false` (the `"list"` default) wraps
/// `values` as a JSON array unchanged; `true` collapses it to just the first value, or `null`
/// if empty -- e.g. `company_number` as the first value of the `lei` property.
fn list_or_first(values: Vec<Value>, take_first: bool) -> Value {
    if take_first {
        values.into_iter().next().unwrap_or(Value::Null)
    } else {
        Value::Array(values)
    }
}

/// Same cardinality modifier as `list_or_first`, but reads from a cached `&[Value]` instead of
/// consuming an owned `Vec` -- lets `take: "first"` fields (e.g. `company_number`) avoid cloning
/// the full list when only the first element is ever needed.
fn list_or_first_ref(values: &[Value], take_first: bool) -> Value {
    if take_first {
        values.first().cloned().unwrap_or(Value::Null)
    } else {
        Value::Array(values.to_vec())
    }
}

/// The `time` shape: the first claim's `mainsnak.datavalue.value.time` string, or `null` if
/// the property has no claims or none carry a time value -- mirrors `_TIME_PROPERTIES` in
/// `wikidata_projection_helpers.py` (e.g. `inception`/`dissolved`).
fn extract_time_claim(claims_array: Option<&Value>) -> Value {
    let Some(claims) = claims_array.and_then(Value::as_array) else {
        return Value::Null;
    };

    for claim_value in claims {
        let Some(time_value) = claim_mainsnak_value(claim_value)
            .and_then(|value| value.as_object())
            .and_then(|object| object.get("time"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        return Value::String(time_value.to_string());
    }

    Value::Null
}

/// The `multi_lang_text` shape: returns `(all_values, english_values, tagged_values)` in one
/// scan over the claim array -- mirrors `_extract_claim_text_values_with_english` in
/// `wikidata_projection_helpers.py` (e.g. `official_name`/`short_name`). `tagged_values` keeps
/// each value's own language ("en", another ISO code, or `null` for an untyped plain-string
/// datavalue) instead of collapsing it to the all/english split.
fn extract_claim_text_values_with_english(
    claims_array: Option<&Value>,
) -> (Vec<Value>, Vec<Value>, Vec<Value>) {
    let tagged = extract_claim_text_values_tagged(claims_array);

    let mut all_values = Vec::with_capacity(tagged.len());
    let mut english_values = Vec::new();
    for entry in &tagged {
        let Some(text_value) = entry.get("value").and_then(Value::as_str) else {
            continue;
        };
        all_values.push(Value::String(text_value.to_string()));
        if entry.get("language").and_then(Value::as_str) == Some("en") {
            english_values.push(Value::String(text_value.to_string()));
        }
    }

    (all_values, english_values, tagged)
}

fn extract_claim_text_values_tagged(claims_array: Option<&Value>) -> Vec<Value> {
    let Some(claims) = claims_array.and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    for claim_value in claims {
        let Some(value) = claim_mainsnak_value(claim_value) else {
            continue;
        };

        if let Some(text_value) = value.as_str() {
            values.push(serde_json::json!({"value": text_value, "language": Value::Null}));
            continue;
        }

        let Some(text_object) = value.as_object() else {
            continue;
        };
        let Some(text_value) = text_object.get("text").and_then(Value::as_str) else {
            continue;
        };
        let value_language = text_object.get("language").and_then(Value::as_str);
        values.push(serde_json::json!({"value": text_value, "language": value_language}));
    }

    values
}

fn claim_mainsnak_value(claim_value: &Value) -> Option<&Value> {
    claim_value
        .as_object()?
        .get("mainsnak")?
        .as_object()?
        .get("datavalue")?
        .as_object()?
        .get("value")
}

fn extract_langstring_value(field: &HashMap<&str, &RawValue>, language: &str) -> Option<String> {
    let raw = field.get(language)?;
    serde_json::from_str::<LangStringEntry>(raw.get())
        .ok()
        .map(|entry| entry.value)
}

fn extract_aliases(field: &HashMap<&str, &RawValue>, language: &str) -> Value {
    let Some(raw) = field.get(language) else {
        return Value::Array(Vec::new());
    };
    let entries: Vec<LangStringEntry> = serde_json::from_str(raw.get()).unwrap_or_default();
    Value::Array(
        entries
            .into_iter()
            .map(|entry| Value::String(entry.value))
            .collect(),
    )
}

fn parse_qid_number(value: &str) -> Option<u32> {
    let qid = value.rsplit('/').next().unwrap_or(value);
    let digits = qid.strip_prefix('Q')?;
    digits.parse::<u32>().ok()
}

fn format_qid(value: u32) -> String {
    format!("Q{value}")
}

// ---------------------------------------------------------------------------------------------
// Schema discovery / profiling (README, "Schema discovery: `profile`") -- a deliberately separate, slower diagnostic pass over
// matched candidates: for every claim property a candidate carries (not just the properties a
// spec's markers/projected_fields reference), classify its shape. The performance-
// equivalence principle (only claim properties the spec actually references get deep-parsed)
// binds `project_candidate`'s per-line hot path; it does not bind this mode, which deep-parses
// every present property on every matched candidate by design, in order to answer "what could I
// be extracting that I'm not".
// ---------------------------------------------------------------------------------------------

/// One of the four extraction shapes the spec format already supports (mirrors
/// `ProjectedFieldShape`, minus the `match_only_qids`/`field_en`/`field_variants` extras that
/// only make sense once a shape has been chosen for a specific spec entry), or `Complex` for
/// everything else: `quantity`, `globe-coordinate`, a `novalue`/`somevalue` snak, or a property
/// whose claims don't all agree on one of the four.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum PropertyShape {
    EntityIdList,
    Time,
    TextList,
    MultiLangText,
    Complex,
}

impl PropertyShape {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EntityIdList => "entity_id_list",
            Self::Time => "time",
            Self::TextList => "text_list",
            Self::MultiLangText => "multi_lang_text",
            Self::Complex => "complex",
        }
    }
}

/// Classifies a Wikidata `mainsnak.datavalue.type` string into the shape it would compile to as
/// a `projected_fields` entry. Anything outside the four simple shapes -- `quantity`,
/// `globecoordinate`, `commonsMedia`, `url`, and so on -- is `Complex`.
fn classify_datavalue_type(datavalue_type: &str) -> PropertyShape {
    match datavalue_type {
        "wikibase-entityid" => PropertyShape::EntityIdList,
        "time" => PropertyShape::Time,
        "string" => PropertyShape::TextList,
        "monolingualtext" => PropertyShape::MultiLangText,
        _ => PropertyShape::Complex,
    }
}

/// The shape one claim's `mainsnak` would classify as, or `None` for anything that collapses the
/// whole property to `Complex`: a `novalue`/`somevalue` snak (no `datavalue` at all), or a
/// `datavalue.type` outside the four simple shapes.
fn classify_claim_shape(claim: &Value) -> Option<PropertyShape> {
    let mainsnak = claim.as_object()?.get("mainsnak")?.as_object()?;
    let snaktype = mainsnak
        .get("snaktype")
        .and_then(Value::as_str)
        .unwrap_or("value");
    if snaktype != "value" {
        return None;
    }
    let datavalue_type = mainsnak
        .get("datavalue")?
        .as_object()?
        .get("type")?
        .as_str()?;
    match classify_datavalue_type(datavalue_type) {
        PropertyShape::Complex => None,
        shape => Some(shape),
    }
}

/// Classifies one property's already-parsed claim array into a single shape: the shape shared by
/// every claim's `mainsnak`, or `Complex` if any claim is a `novalue`/`somevalue` snak, carries a
/// datatype outside the four simple shapes, or the claims disagree with each other (mixed-type
/// data is real on Wikidata but rare enough that reporting it as unclassified, rather than
/// picking a shape by majority vote, is the safer default for a diagnostic reader deciding
/// whether a property is worth adding to a spec).
#[must_use]
pub fn classify_property_shape(claims_array: &Value) -> PropertyShape {
    let Some(claims) = claims_array.as_array() else {
        return PropertyShape::Complex;
    };

    let mut shape: Option<PropertyShape> = None;
    for claim in claims {
        let Some(claim_shape) = classify_claim_shape(claim) else {
            return PropertyShape::Complex;
        };
        match shape {
            None => shape = Some(claim_shape),
            Some(existing) if existing == claim_shape => {}
            Some(_) => return PropertyShape::Complex,
        }
    }
    shape.unwrap_or(PropertyShape::Complex)
}

/// One profiled candidate's observations: every claim property it carries (not just
/// spec-referenced ones), mapped to that property's classified shape. Borrows property-name keys
/// from the input line, same as `ShallowEntity`.
pub type CandidateProfile<'a> = HashMap<&'a str, PropertyShape>;

/// Profiles one candidate line against `spec`: the same match test `project_candidate` uses
/// (`is_candidate_match`), so a profile run and an extract run over the same input agree on
/// which lines are candidates, but the payload is every present claim property's shape rather
/// than a spec-projected record.
///
/// # Errors
/// Returns an error if `line` isn't valid JSON.
pub fn profile_candidate<'a>(
    line: &'a [u8],
    spec: &CompiledSpec,
) -> Result<Option<CandidateProfile<'a>>, serde_json::Error> {
    let entity: ShallowEntity<'a> = serde_json::from_slice(line)?;

    let mut cache = ClaimCache::new(&entity.claims);
    if !is_candidate_match(&mut cache, spec) {
        return Ok(None);
    }

    let mut profile = HashMap::with_capacity(entity.claims.len());
    for (&property, raw_claims) in &entity.claims {
        let Ok(claims_array) = serde_json::from_str::<Value>(raw_claims.get()) else {
            continue;
        };
        profile.insert(property, classify_property_shape(&claims_array));
    }
    Ok(Some(profile))
}

/// Trims a raw Wikidata dump line: strips whitespace, drops the enclosing `[`/`]` array
/// delimiter lines, and drops a trailing comma from a mid-array element.
#[must_use]
pub fn trim_wikidata_line(line: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = line.len();

    while start < end && matches!(line[start], b' ' | b'\t' | b'\r' | b'\n') {
        start += 1;
    }
    while end > start && matches!(line[end - 1], b' ' | b'\t' | b'\r' | b'\n') {
        end -= 1;
    }
    if start == end {
        return &[];
    }
    if line[start] == b'[' || line[start] == b']' {
        return &[];
    }
    if end > start && line[end - 1] == b',' {
        end -= 1;
    }
    &line[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_json(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        path
    }

    /// Derives `prefilter_patterns`/`prefilter_value_patterns` for a hand-built `CompiledSpec`
    /// the same way `CompiledSpec::load` does, so the tests below that construct a spec directly
    /// (rather than through a spec file) don't each hand-duplicate that derivation.
    fn compiled_spec(
        markers: Vec<CompiledMarker>,
        projected_fields: Vec<CompiledProjectedField>,
    ) -> CompiledSpec {
        let prefilter_patterns = markers
            .iter()
            .map(|marker| format!("\"{}\"", marker.property).into_bytes())
            .collect();
        let mut value_qids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for marker in &markers {
            value_qids.extend(marker.qids.iter().copied());
        }
        let mut prefilter_value_patterns: Vec<Vec<u8>> = value_qids
            .into_iter()
            .map(|qid| format!("\"Q{qid}\"").into_bytes())
            .collect();
        prefilter_value_patterns.sort_unstable();
        CompiledSpec {
            markers,
            projected_fields,
            prefilter_patterns,
            prefilter_value_patterns,
        }
    }

    #[test]
    fn compiles_qid_set_marker_and_text_list_field() {
        let dir = std::env::temp_dir().join(format!("wikisieve-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [
                    { "property": "P31", "match": { "type": "qid_set", "qids": ["Q783794"] } }
                ],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1278", "field": "lei" }
                ]
            }"#,
        );

        let spec = CompiledSpec::load(&spec_path).unwrap();
        assert_eq!(spec.markers.len(), 1);
        assert!(spec.markers[0].qids.contains(&783_794));
        assert_eq!(spec.projected_fields.len(), 1);
        assert_eq!(spec.projected_fields[0].field, "lei");
        assert_eq!(spec.prefilter_patterns, vec![b"\"P31\"".to_vec()]);
        assert_eq!(spec.prefilter_value_patterns, vec![b"\"Q783794\"".to_vec()]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compiles_qid_closure_file_marker_relative_to_spec_dir() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-closure-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_temp_json(
            &dir,
            "closure.json",
            r#"[{"subclass": "Q5"}, {"subclass": "Q6"}]"#,
        );
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [
                    { "property": "P1454", "match": { "type": "qid_closure_file", "path": "closure.json" } }
                ],
                "match_logic": "any",
                "projected_fields": []
            }"#,
        );

        let spec = CompiledSpec::load(&spec_path).unwrap();
        assert!(spec.markers[0].qids.contains(&5));
        assert!(spec.markers[0].qids.contains(&6));
        assert_eq!(
            spec.prefilter_value_patterns,
            vec![b"\"Q5\"".to_vec(), b"\"Q6\"".to_vec()]
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_unsupported_shape() {
        let dir = std::env::temp_dir().join(format!("wikisieve-test-shape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P625", "field": "coordinates", "shape": "geo_point" }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail for an unsupported shape");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_multi_lang_text_missing_field_names() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-multilang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1448", "field": "official_name", "shape": "multi_lang_text" }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail for a multi_lang_text entry missing field_en");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compiles_entity_id_list_time_and_multi_lang_text_shapes() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-shapes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P17", "field": "country", "shape": "entity_id_list" },
                    { "property": "P571", "field": "inception", "shape": "time" },
                    {
                        "property": "P1448",
                        "field": "official_name",
                        "shape": "multi_lang_text",
                        "field_en": "official_name_en",
                        "field_variants": "official_name_variants"
                    }
                ]
            }"#,
        );

        let spec = CompiledSpec::load(&spec_path).unwrap();
        assert_eq!(spec.projected_fields.len(), 3);
        assert!(matches!(
            spec.projected_fields[0].shape,
            ProjectedFieldShape::EntityIdList {
                match_only_qids: None
            }
        ));
        assert!(matches!(
            spec.projected_fields[1].shape,
            ProjectedFieldShape::Time
        ));
        assert!(matches!(
            &spec.projected_fields[2].shape,
            ProjectedFieldShape::MultiLangText { field_en, field_variants }
                if field_en == "official_name_en" && field_variants == "official_name_variants"
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn projects_matching_candidate_and_parses_shared_property_once() {
        let line = br#"{"id":"Q1","type":"item","modified":"2026-01-01T00:00:00Z",
            "labels":{"en":{"language":"en","value":"Acme"}},
            "descriptions":{},"aliases":{},"sitelinks":{"enwiki":{}},
            "claims":{
                "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}],
                "P1278":[{"mainsnak":{"datavalue":{"value":"5493001KJTIIGC8Y1R12"}}}]
            }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let projected_fields = vec![CompiledProjectedField {
            property: "P1278".to_string(),
            field: "lei".to_string(),
            shape: ProjectedFieldShape::TextList,
            take_first: false,
        }];
        let spec = compiled_spec(markers, projected_fields);

        let finders = build_prefilter_finders(&spec.prefilter_patterns);
        assert!(has_candidate_properties(line, &finders));
        let value_matcher = build_prefilter_value_matcher(&spec.prefilter_value_patterns);
        assert!(has_candidate_value(line, &value_matcher));
        assert!(Prefilter::build(&spec).admits(line));

        let projected = project_candidate(line, &spec).unwrap().unwrap();
        let record: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(record["id"], "Q1");
        assert_eq!(record["label_en"], "Acme");
        assert_eq!(record["instance_of"], serde_json::json!(["Q783794"]));
        assert_eq!(record["lei"], serde_json::json!(["5493001KJTIIGC8Y1R12"]));
        assert_eq!(record["sitelinks_count"], 1);
    }

    /// The value search's conservativeness requirement, tested directly: a marker's target QID appearing
    /// only under an unrelated property still passes the value search (it can only admit extra
    /// lines, never reject a true one), and is then correctly rejected by the full marker check,
    /// which requires the QID under the marker's *own* property.
    #[test]
    fn value_search_admits_a_qid_present_only_under_an_unrelated_property() {
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}],
            "P17":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]
        }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());

        let value_matcher = build_prefilter_value_matcher(&spec.prefilter_value_patterns);
        assert!(has_candidate_value(line, &value_matcher));
        assert!(project_candidate(line, &spec).unwrap().is_none());
    }

    /// The converse of the case above, so the two together pin down what the value search
    /// actually rejects: a line carrying none of the spec's marker QIDs anywhere, under any
    /// property, is rejected outright.
    #[test]
    fn value_search_rejects_a_line_with_no_marker_qid_anywhere() {
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]
        }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());

        let value_matcher = build_prefilter_value_matcher(&spec.prefilter_value_patterns);
        assert!(!has_candidate_value(line, &value_matcher));
    }

    /// `Prefilter::admits` requires both stages to agree, tested directly rather than only
    /// through each half's own test above: a line can carry the marker's property literal
    /// without ever carrying its qid (rejected here, by the value stage), and the converse
    /// (a qid with no marker property present at all, rejected by the property stage).
    #[test]
    fn prefilter_admits_requires_both_the_property_and_value_stage() {
        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());
        let prefilter = Prefilter::build(&spec);

        // "P31" present, but its value is Q5, not the marker's Q783794: the value stage rejects.
        let property_only =
            br#"{"id":"Q1","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}"#;
        assert!(!prefilter.admits(property_only));

        // Q783794 present, but under P17, and no "P31" literal anywhere: the property stage
        // rejects, even though the value stage alone would have admitted it.
        let value_only = br#"{"id":"Q1","claims":{"P17":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}"#;
        assert!(!prefilter.admits(value_only));

        // Both present: admitted.
        let both = br#"{"id":"Q1","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}"#;
        assert!(prefilter.admits(both));
    }

    #[test]
    fn projects_entity_id_list_time_and_multi_lang_text_shapes() {
        let line = br#"{"id":"Q1","type":"item","modified":"2026-01-01T00:00:00Z",
            "labels":{},"descriptions":{},"aliases":{},"sitelinks":{},
            "claims":{
                "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}],
                "P17":[{"mainsnak":{"datavalue":{"value":{"id":"Q30"}}}}],
                "P571":[{"mainsnak":{"datavalue":{"value":{"time":"+2001-01-01T00:00:00Z"}}}}],
                "P1448":[
                    {"mainsnak":{"datavalue":{"value":{"text":"Acme Corp","language":"en"}}}},
                    {"mainsnak":{"datavalue":{"value":{"text":"Acme SA","language":"fr"}}}}
                ]
            }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let projected_fields = vec![
            CompiledProjectedField {
                property: "P17".to_string(),
                field: "country".to_string(),
                shape: ProjectedFieldShape::EntityIdList {
                    match_only_qids: None,
                },
                take_first: false,
            },
            CompiledProjectedField {
                property: "P571".to_string(),
                field: "inception".to_string(),
                shape: ProjectedFieldShape::Time,
                take_first: false,
            },
            CompiledProjectedField {
                property: "P1448".to_string(),
                field: "official_name".to_string(),
                shape: ProjectedFieldShape::MultiLangText {
                    field_en: "official_name_en".to_string(),
                    field_variants: "official_name_variants".to_string(),
                },
                take_first: false,
            },
        ];
        let spec = compiled_spec(markers, projected_fields);

        let projected = project_candidate(line, &spec).unwrap().unwrap();
        let record: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(record["country"], serde_json::json!(["Q30"]));
        assert_eq!(record["inception"], "+2001-01-01T00:00:00Z");
        assert_eq!(
            record["official_name"],
            serde_json::json!(["Acme Corp", "Acme SA"])
        );
        assert_eq!(record["official_name_en"], serde_json::json!(["Acme Corp"]));
        assert_eq!(
            record["official_name_variants"],
            serde_json::json!([
                {"value": "Acme Corp", "language": "en"},
                {"value": "Acme SA", "language": "fr"}
            ])
        );
    }

    #[test]
    fn time_shape_is_null_when_property_absent() {
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]
        }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let projected_fields = vec![CompiledProjectedField {
            property: "P576".to_string(),
            field: "dissolved".to_string(),
            shape: ProjectedFieldShape::Time,
            take_first: false,
        }];
        let spec = compiled_spec(markers, projected_fields);

        let projected = project_candidate(line, &spec).unwrap().unwrap();
        let record: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(record["dissolved"], Value::Null);
    }

    #[test]
    fn compiles_take_first_and_match_only_modifiers() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-modifiers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [
                    { "property": "P1454", "match": { "type": "qid_set", "qids": ["Q6881511"] } }
                ],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1278", "field": "lei" },
                    { "property": "P1278", "field": "company_number", "take": "first" },
                    { "property": "P1454", "field": "legal_form", "shape": "entity_id_list" },
                    {
                        "property": "P1454",
                        "field": "matched_company_type_qids",
                        "shape": "entity_id_list",
                        "match_only": true
                    }
                ]
            }"#,
        );

        let spec = CompiledSpec::load(&spec_path).unwrap();
        assert_eq!(spec.projected_fields.len(), 4);
        assert!(!spec.projected_fields[0].take_first);
        assert!(spec.projected_fields[1].take_first);
        assert!(matches!(
            spec.projected_fields[2].shape,
            ProjectedFieldShape::EntityIdList {
                match_only_qids: None
            }
        ));
        assert!(matches!(
            &spec.projected_fields[3].shape,
            ProjectedFieldShape::EntityIdList {
                match_only_qids: Some(qids)
            } if qids.contains(&6_881_511)
        ));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_match_only_without_marker_on_property() {
        let dir = std::env::temp_dir().join(format!(
            "wikisieve-test-match-only-no-marker-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    {
                        "property": "P1454",
                        "field": "matched_company_type_qids",
                        "shape": "entity_id_list",
                        "match_only": true
                    }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail: match_only with no marker on the property");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_take_first_for_unsupported_shape() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-take-shape-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P571", "field": "inception", "shape": "time", "take": "first" }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail: take:first isn't supported by shape 'time'");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_match_only_for_unsupported_shape() {
        let dir = std::env::temp_dir().join(format!(
            "wikisieve-test-match-only-shape-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [
                    { "property": "P1278", "match": { "type": "qid_set", "qids": ["Q1"] } }
                ],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1278", "field": "lei", "match_only": true }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail: match_only isn't supported by shape 'text_list'");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rejects_unsupported_take_value() {
        let dir =
            std::env::temp_dir().join(format!("wikisieve-test-take-value-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec_path = write_temp_json(
            &dir,
            "spec.json",
            r#"{
                "markers": [],
                "match_logic": "any",
                "projected_fields": [
                    { "property": "P1278", "field": "lei", "take": "last" }
                ]
            }"#,
        );

        let result = CompiledSpec::load(&spec_path);
        let Err(error) = result else {
            panic!("expected spec load to fail for an unsupported take value");
        };
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn projects_company_number_and_matched_company_type_qids() {
        // Mirrors the old hardcoded extractor's derived fields: company_number (first LEI
        // value) and matched_company_type_qids (legal_form claim ids that matched the P1454
        // marker's own closure set).
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}],
            "P1278":[
                {"mainsnak":{"datavalue":{"value":"5493001KJTIIGC8Y1R12"}}},
                {"mainsnak":{"datavalue":{"value":"SECONDLEIVALUE0000AA"}}}
            ],
            "P1454":[
                {"mainsnak":{"datavalue":{"value":{"id":"Q6881511"}}}},
                {"mainsnak":{"datavalue":{"value":{"id":"Q99999999"}}}}
            ]
        }}"#;

        let markers = vec![
            CompiledMarker {
                property: "P31".to_string(),
                qids: std::collections::HashSet::from([783_794]),
            },
            CompiledMarker {
                property: "P1454".to_string(),
                qids: std::collections::HashSet::from([6_881_511]),
            },
        ];
        let projected_fields = vec![
            CompiledProjectedField {
                property: "P1278".to_string(),
                field: "lei".to_string(),
                shape: ProjectedFieldShape::TextList,
                take_first: false,
            },
            CompiledProjectedField {
                property: "P1278".to_string(),
                field: "company_number".to_string(),
                shape: ProjectedFieldShape::TextList,
                take_first: true,
            },
            CompiledProjectedField {
                property: "P1454".to_string(),
                field: "legal_form".to_string(),
                shape: ProjectedFieldShape::EntityIdList {
                    match_only_qids: None,
                },
                take_first: false,
            },
            CompiledProjectedField {
                property: "P1454".to_string(),
                field: "matched_company_type_qids".to_string(),
                shape: ProjectedFieldShape::EntityIdList {
                    match_only_qids: Some(std::collections::HashSet::from([6_881_511])),
                },
                take_first: false,
            },
        ];
        let spec = compiled_spec(markers, projected_fields);

        let projected = project_candidate(line, &spec).unwrap().unwrap();
        let record: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(
            record["lei"],
            serde_json::json!(["5493001KJTIIGC8Y1R12", "SECONDLEIVALUE0000AA"])
        );
        assert_eq!(record["company_number"], "5493001KJTIIGC8Y1R12");
        assert_eq!(
            record["legal_form"],
            serde_json::json!(["Q6881511", "Q99999999"])
        );
        assert_eq!(
            record["matched_company_type_qids"],
            serde_json::json!(["Q6881511"])
        );
    }

    #[test]
    fn take_first_is_null_when_property_absent() {
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]
        }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let projected_fields = vec![CompiledProjectedField {
            property: "P1278".to_string(),
            field: "company_number".to_string(),
            shape: ProjectedFieldShape::TextList,
            take_first: true,
        }];
        let spec = compiled_spec(markers, projected_fields);

        let projected = project_candidate(line, &spec).unwrap().unwrap();
        let record: Value = serde_json::from_str(&projected).unwrap();
        assert_eq!(record["company_number"], Value::Null);
    }

    #[test]
    fn non_matching_candidate_returns_none() {
        let line =
            br#"{"id":"Q2","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}"#;
        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());

        assert!(project_candidate(line, &spec).unwrap().is_none());
    }

    /// The rule the Wikidata subsetting evaluation's `count_instances_json_iter.py` counts by: a
    /// `P31` statement matches at any rank, and one with no `datavalue` is skipped rather than
    /// failing the line.
    #[test]
    fn qid_set_marker_matches_any_rank_and_skips_a_novalue_snak() {
        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([7_187, 8_054]),
        }];
        let projected_fields = vec![CompiledProjectedField {
            property: "P31".to_string(),
            field: "matched_classes".to_string(),
            shape: ProjectedFieldShape::EntityIdList {
                match_only_qids: Some(markers[0].qids.clone()),
            },
            take_first: false,
        }];
        let spec = compiled_spec(markers, projected_fields);
        let item_statement = |qid: &str, rank: &str| {
            serde_json::json!({
                "mainsnak": {"snaktype": "value", "property": "P31", "datavalue": {
                    "type": "wikibase-entityid",
                    "value": {"entity-type": "item", "numeric-id": qid[1..].parse::<u32>().unwrap(), "id": qid}
                }},
                "type": "statement", "rank": rank
            })
        };
        let novalue = serde_json::json!({
            "mainsnak": {"snaktype": "novalue", "property": "P31"},
            "type": "statement", "rank": "normal"
        });
        let matched_classes = |statements: Vec<Value>| {
            let line =
                serde_json::json!({"id": "Q1", "type": "item", "claims": {"P31": statements}});
            project_candidate(line.to_string().as_bytes(), &spec)
                .unwrap()
                .map(|record| {
                    serde_json::from_str::<Value>(&record).unwrap()["matched_classes"].clone()
                })
        };

        assert_eq!(
            matched_classes(vec![item_statement("Q7187", "deprecated")]),
            Some(serde_json::json!(["Q7187"]))
        );
        assert_eq!(
            matched_classes(vec![item_statement("Q8054", "preferred")]),
            Some(serde_json::json!(["Q8054"]))
        );
        assert_eq!(
            matched_classes(vec![
                novalue.clone(),
                item_statement("Q5", "normal"),
                item_statement("Q7187", "deprecated"),
                item_statement("Q8054", "preferred"),
            ]),
            Some(serde_json::json!(["Q7187", "Q8054"]))
        );
        assert_eq!(matched_classes(vec![novalue]), None);
    }

    #[test]
    fn classify_property_shape_recognizes_each_simple_shape() {
        let entity_id = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q30"}}}}
        ]);
        assert_eq!(
            classify_property_shape(&entity_id),
            PropertyShape::EntityIdList
        );

        let time = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "time", "value": {"time": "+2001-01-01T00:00:00Z"}}}}
        ]);
        assert_eq!(classify_property_shape(&time), PropertyShape::Time);

        let text = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": "5493001KJTIIGC8Y1R12"}}}
        ]);
        assert_eq!(classify_property_shape(&text), PropertyShape::TextList);

        let multi_lang = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "monolingualtext", "value": {"text": "Acme", "language": "en"}}}}
        ]);
        assert_eq!(
            classify_property_shape(&multi_lang),
            PropertyShape::MultiLangText
        );
    }

    #[test]
    fn classify_property_shape_treats_quantity_and_globecoordinate_as_complex() {
        let quantity = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "quantity", "value": {"amount": "+1"}}}}
        ]);
        assert_eq!(classify_property_shape(&quantity), PropertyShape::Complex);

        let coordinate = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "globecoordinate", "value": {}}}}
        ]);
        assert_eq!(classify_property_shape(&coordinate), PropertyShape::Complex);
    }

    #[test]
    fn classify_property_shape_treats_novalue_and_somevalue_snaks_as_complex() {
        let novalue = serde_json::json!([{"mainsnak": {"snaktype": "novalue"}}]);
        assert_eq!(classify_property_shape(&novalue), PropertyShape::Complex);

        let somevalue = serde_json::json!([{"mainsnak": {"snaktype": "somevalue"}}]);
        assert_eq!(classify_property_shape(&somevalue), PropertyShape::Complex);
    }

    #[test]
    fn classify_property_shape_treats_mixed_shape_claims_as_complex() {
        let mixed = serde_json::json!([
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "string", "value": "text"}}},
            {"mainsnak": {"snaktype": "value", "datavalue": {"type": "time", "value": {"time": "+2001-01-01T00:00:00Z"}}}}
        ]);
        assert_eq!(classify_property_shape(&mixed), PropertyShape::Complex);
    }

    #[test]
    fn classify_property_shape_is_complex_for_a_property_with_no_claims() {
        assert_eq!(
            classify_property_shape(&serde_json::json!([])),
            PropertyShape::Complex
        );
        assert_eq!(
            classify_property_shape(&Value::Null),
            PropertyShape::Complex
        );
    }

    #[test]
    fn profile_candidate_classifies_every_present_property_on_a_match() {
        let line = br#"{"id":"Q1","claims":{
            "P31":[{"mainsnak":{"snaktype":"value","datavalue":{"type":"wikibase-entityid","value":{"id":"Q783794"}}}}],
            "P1278":[{"mainsnak":{"snaktype":"value","datavalue":{"type":"string","value":"5493001KJTIIGC8Y1R12"}}}],
            "P571":[{"mainsnak":{"snaktype":"value","datavalue":{"type":"time","value":{"time":"+2001-01-01T00:00:00Z"}}}}],
            "P2124":[{"mainsnak":{"snaktype":"value","datavalue":{"type":"quantity","value":{"amount":"+1"}}}}]
        }}"#;

        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());

        let profile = profile_candidate(line, &spec).unwrap().unwrap();
        assert_eq!(profile.len(), 4);
        assert_eq!(profile["P31"], PropertyShape::EntityIdList);
        assert_eq!(profile["P1278"], PropertyShape::TextList);
        assert_eq!(profile["P571"], PropertyShape::Time);
        assert_eq!(profile["P2124"], PropertyShape::Complex);
    }

    #[test]
    fn profile_candidate_returns_none_for_a_non_matching_line() {
        let line =
            br#"{"id":"Q2","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}"#;
        let markers = vec![CompiledMarker {
            property: "P31".to_string(),
            qids: std::collections::HashSet::from([783_794]),
        }];
        let spec = compiled_spec(markers, Vec::new());

        assert!(profile_candidate(line, &spec).unwrap().is_none());
    }

    #[test]
    fn trims_array_brackets_and_trailing_comma() {
        assert_eq!(trim_wikidata_line(b"["), b"");
        assert_eq!(trim_wikidata_line(b"]"), b"");
        assert_eq!(
            trim_wikidata_line(b"  {\"id\":\"Q1\"},  "),
            b"{\"id\":\"Q1\"}"
        );
        assert_eq!(trim_wikidata_line(b"{\"id\":\"Q1\"}"), b"{\"id\":\"Q1\"}");
    }
}
