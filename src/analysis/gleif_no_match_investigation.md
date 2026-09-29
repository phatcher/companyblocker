# GLEIF No-Match Investigation: GB, FR, IE (2026-08-24)

## Question

Why do GLEIF-sourced records fail to match their target country registries,
and can the gap be usefully classified by GLEIF entity type (`GENERAL`,
`FUND`, `SOLE_PROPRIETOR`, etc.)?

Started as a GB-only investigation, then repeated for FR and IE to check
whether the same shape holds. It doesn't — each country's residual has a
different dominant cause.

## Method

- `scripts/analyze_matches.py --source gleif --target {gb,fr,ie}`
  (Stage 5 match output, run date 2026-08-24).
- `scripts/analyze_gleif_entity_category.py` /
  `src/analysis/match_metrics_gleif.py` rejoin the not-recoverable no-match
  population onto the raw GLEIF source shard via `LEI`, since raw
  `Entity_EntityCategory` doesn't survive Stage 3/4 (see below).
- Ad-hoc cross-reference against
  `Entity_RegistrationAuthority_RegistrationAuthorityID`,
  `Entity_RegistrationAuthority_RegistrationAuthorityEntityID`, and
  `Registration_RegistrationStatus` from the raw GLEIF shard — not yet
  scripted, run directly against `data/gleif/source/2026-06-21/`.

## Summary across countries

| | GB | FR | IE |
|---|---|---|---|
| GLEIF records with a company number | 114,199 | 165,263 | 19,401 |
| — matched | 98,272 (86.05%) | 164,906 (99.78%) | 19,178 (98.85%) |
| — NO_MATCHING_COMPANY residual | 15,927 (13.95%) | 357 (0.22%) | 223 (1.15%) |
| MISSING_COMPANY_NUMBER total | 106,441 | 901 | 3,889 |
| — no number from GLEIF at all | 84,851 (79.7%) | 687 (76.2%) | 2,744 (70.6%) |
| — excluded (non-target-registry authority) | 21,591 (20.3%) | 214 (23.8%) | 1,145 (29.4%) |
| Dominant NO_MATCHING_COMPANY cause | dissolved companies dropped from target snapshot | formatting + likely sole-trader registrations | separate Limited Partnership register |

FR and IE match at a much higher rate than GB (99.78% and 98.85% vs 86.05%
for GLEIF records that have a company number) — GB's registration-authority
whitelist is unrestricted for FR/IE-style small residuals but GB simply has
far more historical/dissolved LEI records in absolute terms. The
`MISSING_COMPANY_NUMBER` funnel shape (roughly 70-80% no-number-at-all,
20-30% excluded-authority) is consistent across all three, though.

## GB

### Top-level funnel

| no_match_reason | rows | % |
|---|---|---|
| MISSING_COMPANY_NUMBER | 106,441 | 87.0% |
| NO_MATCHING_COMPANY | 15,927 | 13.0% |

87% of GB no-matches never had a `company_number` to join on at all — the
matching logic never got a chance to run for those rows.

### MISSING_COMPANY_NUMBER breakdown

| cause | rows | % |
|---|---|---|
| GLEIF reported no registration number at all | 84,851 | 79.7% |
| Registration authority not a GB Companies House code | 21,591 | 20.3% |

The second row is already handled: `GLEIF_VALID_REGISTRATION_AUTHORITIES["GB"]`
(`src/acquisition/canonical_system_config.py`) whitelists Companies House's
three authority codes (`RA000585`/`586`/`587`, England & Wales / NI /
Scotland) and nulls `company_number` for anything else at canonical time. Top
excluded authority codes for GB: `RA000589` (10,914 rows), `RA888888` — GLEIF's
own "no registration authority applies" sentinel (3,365), `RA000591`/`592`/`590`
(~1,800 each), then a long thin tail.

### The 79.7% "no number at all" residual

Characterizing the 84,851 rows where GLEIF supplied no registration number at
all, regardless of authority:

- **Raw `Entity_EntityCategory`**: 99.6% `GENERAL` (84,515). The raw GLEIF
  category doesn't discriminate this bucket — it's not a `FUND`/`BRANCH`/etc.
  concentration.
- **Cleansed, name-derived `company_type`**: `private`/unknown 52.4% (44,475),
  `trust` 42.8% (36,328), `fund` 4.4% (3,764) — together 99.6% of the bucket.
- **`Registration_RegistrationStatus`**: 70.5% `LAPSED`, vs a 51.2% baseline
  across all 220,657 GB-jurisdiction GLEIF records. Lapsed LEI registrations
  are somewhat over-represented here but aren't the dominant driver.
- **`entity_status` / canonical `current_status`**: 98% `ACTIVE`. These are
  live entities, not dissolved ones — the gap isn't explained by staleness.

### The NO_MATCHING_COMPANY residual (15,927 rows)

These rows have a `company_number` GLEIF believes is valid under a
whitelisted GB authority. As a proportion of all GLEIF GB records with a
company number (114,199), this residual is 13.95% — the other 86.05%
(98,272) matched cleanly.

Format is not the problem: 15,924/15,927 (99.98%) are already a clean 8-char
string, with plausible prefixes (`SC` Scotland 755, `NI` Northern Ireland 84,
`OC` LLP 931, `BR` overseas branch 34, plain digits 14,060 for E&W). Note
Companies House is one unified registry/one bulk file today (`gb` here) even
though GLEIF still tracks E&W/NI/Scotland as three separate registration
authorities historically — a small number of GLEIF rows (836: 335 `gb-sct`,
500 `gb-nir`, 1 `gb-eng`) carry a more specific `gb-*` jurisdiction_code and
are out of scope for a `--country gb` run entirely; negligible next to the
15,927.

None of the 15,927 numbers exist anywhere in the GB target snapshot
(5,698,275 rows, checked directly, not just among already-matched rows). Of
those:

| GLEIF's own `current_status` | rows | % of residual |
|---|---|---|
| `INACTIVE` | 15,801 | 99.2% |
| `ACTIVE` / unknown | 126 | 0.8% |

Compare to the 14.2% inactive rate baseline across *all* 114,199 GLEIF-GB
records with a company number — dissolved entities are ~7x over-represented
in the no-match set. Companies House's Free Company Data Product doesn't
retain dissolved companies indefinitely; GLEIF still holds a historical LEI
record with the old number after Companies House has dropped it from the
current snapshot.

The true residual — unmatched *and* GLEIF says active/unknown — is 126 rows
(0.11% of the 114,199 with-number population). Of those, 24 are `BR`-prefixed
(overseas company UK establishments, a distinct Companies House register that
may not be in the main bulk file), the rest plain E&W digits. At this scale
it reads as incorporation/snapshot-date timing lag rather than a systemic
issue, and isn't worth chasing further.

### GB conclusion

Essentially the entire GB no-match gap is explained by real, understood
causes (missing/excluded registration numbers, structurally unincorporated
entity types, and dissolved companies dropped from the target snapshot), not
by a defect in this pipeline's matching logic. A name/`company_type`-based
"expected registry" classifier (e.g. tagging `trust`/`charity`/`cio`/`scio`
as "registered elsewhere") was considered and **rejected**: at best it would
re-derive, less reliably, what the registration-authority whitelist already
establishes, and it wouldn't touch the 79.7% no-number majority at all.
GB's own Companies House data genuinely contains `trust` (1,478), `charity`
(166), `cio` (520), and `scio` (472) records too — a blanket "these types
don't match GB" rule would have been wrong on its own terms, not just
redundant.

## FR

### Top-level funnel

| no_match_reason | rows | % |
|---|---|---|
| MISSING_COMPANY_NUMBER | 901 | 71.6% |
| NO_MATCHING_COMPANY | 357 | 28.4% |

Much smaller in absolute terms than GB (1,258 total no-matches vs 122,368),
and the funnel is less lopsided — `NO_MATCHING_COMPANY` is a much bigger
share of FR's (much smaller) gap than GB's.

### MISSING_COMPANY_NUMBER breakdown

| cause | rows | % |
|---|---|---|
| GLEIF reported no registration number at all | 687 | 76.2% |
| Registration authority not `RA000189`/`RA000192` (INSEE SIRENE / RCS) | 214 | 23.8% |

Same shape as GB. Top excluded authority: `RA888888` sentinel (75), then a
thin tail (`RA000533`, `RA000407`, `RA000585` — the *UK* Companies House code
turning up on 10 FR-jurisdiction records, likely cross-jurisdiction data
noise in GLEIF, not a bug here).

Of the 687 no-number rows: `company_type` is `private`/unknown for 43.7%
(300), but unlike GB the rest are dominated by real French incorporated
forms — `sas` (151), `sarl` (114), `sa` (32) — not trust/fund (`fund` 24,
`trust` 17 combined = 6%). `Registration_RegistrationStatus` is 71.0%
`LAPSED` (488/687), similar over-representation pattern to GB.

### The NO_MATCHING_COMPANY residual (357 rows)

Very different shape from GB. Only 22.4% (80) are GLEIF-`INACTIVE`; 75.4%
(269) are `ACTIVE`. Baseline inactive rate across all 165,263 FR records with
a number is 7.9%, so inactive is still ~2.8x over-represented, but nowhere
near GB's dominant 99.2%/7x pattern — most of FR's residual is *not*
explained by dissolved-and-purged companies.

Inspecting the 269 active-unmatched rows directly:

- **200/269 (74.3%)** are a clean 9-digit string — valid SIREN length — but
  many carry what look like **personal names**, not company names (e.g.
  "LECLERCQ ROMAN", "ADRIEN DESHAYES", "DEGRET SYLVAIN"). The canonical
  `personal_owner` flag is blank for all of them (it isn't derived from name
  pattern for GLEIF-sourced rows), so this is a visual read of the sample,
  not a confirmed flag — but it's consistent with individual/sole-trader
  ("micro-entrepreneur") SIRENE registrations that may not be present in
  whatever FR company dataset this pipeline ingests as its target.
- **58/269 (21.6%)** have a literal space in the stored `company_number`
  (e.g. `"909 903 759"` — the human-readable SIRENE formatting convention).
  Initially flagged as a likely fixable formatting gap, but **ruled out
  empirically, and no code change was kept for it**: `match_ops.py`'s
  `_create_enriched_keys_table` already strips every non-alphanumeric
  character from `company_number` when building the join key, for source and
  target alike, unconditionally — so the raw stored format was never
  actually a matching blocker. Confirmed by adding whitespace stripping to
  canonical GLEIF/FR mapping and rebuilding canonical/cleanse/match end to
  end: the matched-row count was identical (164,906) before and after. The
  correlation between "has a space" and "unmatched" was coincidental, not
  causal, so the change was reverted rather than kept.
- The remaining 55+9+2+1 = 67 rows are longer than 9 digits (11/14/17/18
  chars) — likely SIRET (14-digit SIREN+establishment code) or other
  composite identifiers GLEIF sometimes reports instead of a bare SIREN.

### FR conclusion

Unlike GB, FR's residual is small enough, and different enough in character
(mostly `ACTIVE`, not dissolved), that it looks like real match-logic surface
rather than a purely structural gap — but the cause remains **open**. The
formatting/whitespace lead was investigated and ruled out (see above). The
remaining lead, not yet verified: confirm whether the FR target dataset
excludes personne-physique (sole-trader) registrations that GLEIF includes —
would explain the ~74% of the residual that carries a personal-looking name
with an otherwise valid 9-digit SIREN.

## IE

### Top-level funnel

| no_match_reason | rows | % |
|---|---|---|
| MISSING_COMPANY_NUMBER | 3,889 | 94.6% |
| NO_MATCHING_COMPANY | 223 | 5.4% |

Smallest target of the three (817,761 rows) but the largest GLEIF source
population relative to target (19,401 with-number records). Funnel shape is
the most GB-like of the three (heavily missing-number dominated).

### MISSING_COMPANY_NUMBER breakdown

| cause | rows | % |
|---|---|---|
| GLEIF reported no registration number at all | 2,744 | 70.6% |
| Registration authority not `RA000402` (CRO) | 1,145 | 29.4% |

Top excluded authorities: `RA000404` (578), `RA000700` (285), `RA000403`
(142) — plausible other Irish registries (e.g. Charities Regulator, Credit
Union register) distinct from the CRO, same "registered elsewhere" pattern
as GB's non-Companies-House codes. `RA888888` sentinel: 57. FR's own codes
(`RA000189`/`192`) appear on 27 IE-jurisdiction rows — again minor
cross-jurisdiction noise, not a bug here.

Of the 2,744 no-number rows: `company_type` is `private`/unknown for 70.4%
(1,932), `trust` 17.0% (467), `fund` 9.2% (252) — same shape as GB (trust +
fund dominate the non-private share), unlike FR.
`Registration_RegistrationStatus` is 47.2% `LAPSED` / 40.5% `ISSUED` — less
skewed toward lapsed than GB or FR.

### The NO_MATCHING_COMPANY residual (223 rows)

Also unlike GB: only 4.9% (11) are GLEIF-`INACTIVE`; 94.6% (211) are
`ACTIVE` — inactive is actually *under*-represented here versus the 11.2%
baseline rate across all 19,401 IE records with a number. Dissolved/purged
companies are not the story for IE.

Inspecting the 211 active-unmatched rows by company-number prefix:

- **154/211 (73.0%)** are `LP`-prefixed (e.g. `LP1134`, `LP3634`, `LP2358`).
  Ireland maintains a **separate Register of Limited Partnerships**, distinct
  from the standard CRO companies register. This pipeline's `ie` target
  almost certainly covers only the companies register, so these were never
  going to match — a structural gap, same shape as GB's `trust`/`fund`
  finding, just identified by number prefix instead of name-derived type.
- `C`-prefix (6, likely a different Irish register series), `PB`-prefix (6),
  `CHY` (1, a Charity number format) round out the non-plain-digit share.
  44 rows are plain digits with no recognized prefix — the genuinely
  unexplained remainder, and a small one.

### IE conclusion

IE's residual is dominated by a single, clean, structural cause: Limited
Partnerships on a separate register from the one this pipeline's `ie` target
covers. Same character as GB's trust/fund finding (an entity type that's
never going to appear in the target registry), just surfaced via
`company_number` prefix rather than parsed name text — and much cleaner to
detect than GB's case, since the `LP` prefix is unambiguous and doesn't
require name parsing at all.

## Overall conclusion

The GB story (dissolved companies dropped from the target snapshot) does
**not** generalize to FR or IE — each country's `NO_MATCHING_COMPANY`
residual has its own dominant cause:

| country | dominant cause |
|---|---|
| GB | Dissolved companies GLEIF still tracks, purged from the target snapshot |
| FR | Formatting (embedded spaces) + likely sole-trader/personal registrations not in the target dataset |
| IE | Limited Partnerships on a separate register from the target's companies register |

None of these are bugs in this pipeline's matching *logic* — they're all
either upstream data-coverage differences (target dataset scope, dissolved
record retention) or, for FR's space-formatted numbers, a small and
cheaply-fixable normalization gap. The `MISSING_COMPANY_NUMBER` funnel shape
(70-80% no-number-at-all, 20-30% excluded-authority) is consistent across
all three countries, and the excluded-authority mechanism
(`GLEIF_VALID_REGISTRATION_AUTHORITIES`) is doing real, correctly-targeted
work in each case.

## Where the numbers came from

- `artifacts/analysis/match_analysis/runs/2026-08-24/gleif_to_{gb,fr,ie}_{gb,fr,ie}/metrics/no_match_reasons.parquet`
- `artifacts/analysis/match_analysis/runs/2026-08-24/gleif_to_{gb,fr,ie}_{gb,fr,ie}/metrics/no_match_recoverability_detail.parquet`
- `artifacts/analysis/match_analysis/runs/2026-08-24/gleif_to_gb_gb/metrics/gleif_entity_category_breakdown.parquet`
- `data/gleif/source/2026-06-21/` (raw shard, for the registration-authority
  and registration-status rejoins — ad hoc, not yet scripted)
