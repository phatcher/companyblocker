# Source Data Adjustments

This document tracks source-specific adjustments made during acquisition, canonicalization and cleanse: corrections for known bad, missing or misleading upstream data, as distinct from the generic field-mapping logic in `docs/canonical_schema.md`.

Two reasons to keep it current:

1. **Guidance going forward.** A future change to one of these systems' handling starts from why the adjustment exists rather than by rediscovering the underlying data issue.
2. **Explaining row-count and value differences from the raw source.** Where a canonical or match output has fewer rows, different `company_number` values or different counts than a naive read of the raw source suggests, the cause is traceable to an entry below.

Only adjustments that exist in code are listed.

## GLEIF

Implemented in `canonical_system_config.py` and `canonical_frame_transform.py`.

- **Registration-authority whitelist per jurisdiction** (`GLEIF_VALID_REGISTRATION_AUTHORITIES`). GLEIF reports a registration number for many non-target-registry authorities per jurisdiction: GB carries charity and credit-union-style registries distinct from Companies House, and DE carries BaFin, Bundesanzeiger's Unternehmensregister, and foundations, lawyers, tax advisors and political parties all filed under jurisdiction DE. Numbers from authorities outside the whitelist are nulled rather than passed through, so a per-system match target is not asked to resolve an identifier it was never going to contain. Covers GB, FR, IE and DE.
- **GB registration-number zero-padding** (`GLEIF_COMPANY_NUMBER_PAD_WIDTH`). GLEIF sometimes reports GB numbers without leading zeros, while Companies House's own numbers are fixed 8 characters. Short all-numeric values are zero-padded to 8 before matching.
- **DE authority-scoped company_number** (`GLEIF_COMPANY_NUMBER_AUTHORITY_SCOPED_JURISDICTIONS`). Germany runs ~150 independent Amtsgerichte, each issuing its own HRB/HRA sequence from 1, so a bare register number is not nationally unique: "HRB 1005" alone collided across 208 different courts in offeneregister's data. For DE, GLEIF's own registration-authority code is folded into `company_number` as a scoping prefix (`RA000234|HRB150148`) rather than only being used to gate or pad it. GB, FR and IE are single national registries and are excluded from this scoping.
- **Successor-LEI filter**, general rather than DE-specific. A row with a populated `Entity_SuccessorEntity_SuccessorLEI` is a superseded or merged LEI record, and the successor's own row always exists independently in the same GLEIF extract, verified at 200 of 200 sampled successor LEIs found as their own row. The obsolete predecessor row is dropped rather than left to collide with the current one. This cut genuine DE registration-authority-scoped duplicate-key collisions from 304 groups and 613 rows to 36 groups and 72 rows, 0.036% of resolved DE keys.
- **FUND entity-type exclusion** (`company_type_mappings/gleif-exclude.json`). Rows classified `FUND` are excluded before canonicalization.
- **Previous names come from two sources.** Canonical `previous_names` carries GLEIF's native `Entity_OtherEntityNames` rows typed `PREVIOUS_LEGAL_NAME`, plus predecessor names folded in by walking `Entity_SuccessorEntity_SuccessorLEI` chains: on the real 2026-06-21 snapshot, 34,001 records carry a `SuccessorLEI` and 23,777 chain-derived rows were folded onto 18,391 surviving entities. A chain whose ultimate terminal has no canonical row, because it was `FUND`-excluded or the hop-capped walk declined a fork or cycle, has nothing to fold into.

## Offeneregister (Germany)

Implemented in `canonical_frame_transform.py`, `authority_scoped_ids.py` and `resources/authority_crosswalks/xjustiz.json`.

- **`company_number` is derived from register fields**, not passed through raw. Offeneregister's raw `company_number` is an OpenCorporates-style id with an embedded court prefix, for example `K1101R_HRB150148`.
- **Court (XJustizId) scoping**, mirroring the GLEIF DE adjustment above. The court-code segment is extracted from the raw id and resolved to a canonical GLEIF registration-authority code via a crosswalk (`resources/authority_crosswalks/xjustiz.json`), so both sides of a DE match use the same scoping vocabulary. **Crosswalk coverage is a first pass**: 89 of 427 distinct court codes are mapped, covering 77.1% of DE rows (4,088,328 of 5,305,727). It is built from the official GLEIF registration authorities list cross-referenced against offeneregister's own `registered_address` city field; an earlier attempt to fetch the GLEIF list via an LLM-summarized web fetch was caught fabricating rows and discarded, so the crosswalk is never sourced that way. An unmapped court nulls `company_number` rather than falling back to an unscoped, collision-prone bare number. Closing the remaining ~23%, mostly smaller courts whose address city differs from their responsible court's seat, is open follow-up work not yet scheduled.
- **The most recent court wins on merger history.** About 9.5% of rows carry more than one underscore-separated court segment, for example `H1101_R1101_HRB18423`, reflecting an Amtsgericht merger or transfer. The last segment is used.
- **The register id is taken from the raw segment**, not from the separately shard-derived `register_number`. The shard-stage `_registerNummer` extraction drops a trailing court-merger disambiguation suffix, so `HRB1162RZ` shards to `register_number="1162"` and loses `RZ`. That was harmless before court-scoping existed and became a collision source once it did: five distinct companies at one court all reduced to `HRB 1162` after their suffixes were dropped. The composite key uses the raw segment directly, keeping the suffix intact.
- **Placeholder register-number exclusion** (`OFFENEREGISTER_PLACEHOLDER_REGISTER_NUMBER_RANGE`). A historical offeneregister.de scrape batch (Mannheim/Baden-Baden, retrieved 2018-06-30) assigned the same placeholder register number across unrelated companies, and across register types, with HRA, HRB and VR sharing one literal number at the same court. Confirmed as an upstream defect: the raw JSONL was inspected directly and a fresh reacquisition was byte-identical to the original download, ruling out local corruption. It affects ~4% of DE rows (215,039 of 5,305,727). Register ids with a numeric value in this band are excluded from `company_number` entirely rather than court-scoped, since the number was never a real register id. The band boundaries are a heuristic derived from observed cross-court fan-out decay rather than a precise cutover; the constant's docstring carries the full evidence.
- **Known residual, not acted on**: German company numbers can be legitimately reused decades apart once a company is fully deregistered. This produces a small number of duplicate-key pairs even after correct court-scoping and suffix preservation, 141 groups and 282 rows among `currently registered` rows. It is confirmed benign, since all rows in each pair are `removed`, but is not filtered. Whether to exclude historical-reuse duplicates from matching is part of the open `match_ops` duplicate-key discussion (see `src/analysis/gleif_no_match_investigation.md` and `match_ops.py`'s `target_duplicate_keys` check).

## Wikidata

Implemented in `wikidata_projection_helpers.py` (Shard-stage extraction) and `canonical_frame_transform.py` (canonical-stage jurisdiction selection).

- **`company_number` is picked per resolved jurisdiction**, not copied from `lei`. No single Wikidata property covers a company registration number the way P1278 covers `lei`: each jurisdiction has its own. Shard stage extracts each jurisdiction's claim into its own column (`company_number_gb` from P2622, `company_number_fr` from P1616, `company_number_de` from P12012), and canonical stage selects the column matching the row's own resolved `jurisdiction_code`, mirroring GLEIF's registration-authority-per-jurisdiction `CASE` pattern above. A row whose jurisdiction has no confirmed source property resolves `company_number` to null rather than falling back to a different field's value.
- **IE has no confirmed source property.** No Wikidata property for the Companies Registration Office's Irish registration number was found populated on real company entities in the raw dump, so `company_number` is null for every Wikidata row resolved to jurisdiction IE.

## GB, FR, IE

- **Company-type exclusions** (`company_type_mappings/{gb,fr,ie}-exclude.json`). FR excludes company-type code `1000`; GB and IE have no exclusions.
- No jurisdiction-specific `company_number` adjustments are needed for these three: each is a single national registry whose raw registration number is already nationally unique, unlike Germany's per-court numbering.

## Where the underlying investigations live

- GB, FR and IE GLEIF no-match causes: `src/analysis/gleif_no_match_investigation.md`.
- DE offeneregister court-scoping, suffix-preservation and placeholder-band findings: this document and the inline code comments and docstrings are the only write-up.
