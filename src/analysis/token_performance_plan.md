# Token Performance Analysis Plan (No Splink)

## Objective
Build a repeatable analysis workflow for token quality and performance using tokenized data in `data/<system>/`, while keeping implementation and outputs aligned to the existing repository layout (`src/analysis`, `scripts`, `notebooks`, `artifacts`).

## Scope
- In scope:
  - Per-country and global token analysis
  - N-gram analysis (word bigrams/trigrams)
  - Short-name derivation and quality diagnostics
  - IDF and inverse-IDF style metrics
  - Visualization outputs (Python or R)
- Out of scope:
  - Splink/entity resolution
  - Reading from upstream (pre-tokenize) layers under `data/` for this stage

## Principles
- Use `data/` as the source of truth for analysis inputs.
- Keep analysis logic under `src/analysis/` and avoid adding this logic to `acquisition/`.
- Keep runnable entry points in top-level `scripts/` and exploratory work in top-level `notebooks/`.
- Produce deterministic, versioned artifacts for review and reruns.
- Support both automated and manual analysis steps.
- Keep artifact structure stable across phases so reruns and automation do not depend on phase-specific folder reshapes.

---

## Proposed Layout

```text
src/
  analysis/
    token_performance_plan.md
    config/
      analysis_defaults.yaml
      stopwords/
        global.txt
        fr.txt
        gb.txt
        ie.txt
    io_contract.py
    token_metrics.py
    ngram_metrics.py
    short_name_rules.py
    idf_metrics.py
    viz/
      plots.py
      dashboard.py

scripts/
  analyze_tokens.py
  analyze_ngrams.py
  analyze_short_names.py
  analyze_visual_report.py

notebooks/
  token_analysis_exploration.ipynb

artifacts/
  analysis/
    token_performance/
      runs/
        <run_date>/
          profile/
          metrics/
          reports/
          viz/
          run_manifest.json
```

---

## Phase Plan

## Phase 0: Data Contract and Baseline Profiling
Purpose: ensure `data/` inputs are fit for analysis and identify schema drift early.

Tasks:
- [x] Define required columns for analysis input.
- [x] Implement dataset scan over `data/<system>/*.parquet`.
- [x] Produce per-system row counts, null rates, and schema summaries.

Manual checkpoints:
- [x] Review schema drift report and approve/adjust input contract.
- [x] Approve system-specific exceptions (if any).

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/profile/schema_profile.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/schema_profile.md`

Exit criteria:
- [x] All target systems pass required column checks.
- [x] Drift exceptions documented.

## Phase 1: Token Metrics (Per-country + Global)
Purpose: build core token statistics from final data outputs.

Tasks:
- [x] Prepare token inputs for per-country and global analysis views.
- [x] Extract/normalize token lists from selected token fields.
- [x] Compute per-country term frequency (TF) and document frequency (DF).
- [x] Compute global TF/DF across all countries.
- [x] Add country-vs-global comparison metrics (lift/ratio).

Manual checkpoints:
- [ ] Validate token normalization rules on sample records.
- [ ] Approve stopword strategy by country.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/token_stats_country.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/token_stats_global.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/token_summary.md`

Exit criteria:
- [ ] Per-country and global token tables generated.
- [ ] Sanity checks pass for top common and rare tokens.

## Phase 2: IDF and Inverse-IDF Metrics
Purpose: add rarity and commonality scoring for ranking and diagnostics.

Tasks:
- [ ] Compute IDF using global and per-country document counts.
- [ ] Compute inverse-IDF style metric for common-token weighting.
- [ ] Store both raw and normalized variants.
- [ ] Add thresholded candidate lists (rare terms, overly-common terms).

Metric notes:
- IDF: `idf = log(N / max(df, 1))`
- Inverse-IDF (recommended reporting metric): `inv_idf = 1 / max(idf, eps)`
- Also include intuitive commonality metric: `commonality = df / N`

Manual checkpoints:
- [ ] Review top rare terms for noise.
- [ ] Review top common terms for stopword leakage.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/token_idf_metrics.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/token_idf_review.md`

Exit criteria:
- [ ] IDF and inverse/commonality metrics verified on sample systems.

## Phase 2.5: Storage Decision Gate
Purpose: decide when file-native analysis is no longer sufficient and a query engine or database should be introduced.

Decision policy:
- Default mode remains parquet-native analysis (Polars over `data/` and run artifacts).
- Introduce DuckDB first when thresholds indicate analytical query pressure.
- Introduce a managed database only when multi-user governance and service requirements justify it.

Trigger thresholds (evaluate on two consecutive runs):
- Runtime trigger:
  - Standard analysis run (Phases 0-2 scope) exceeds 20 minutes wall-clock on baseline hardware.
- Data volume trigger:
  - Combined input for active systems exceeds 10 million rows or 20 GB parquet footprint.
- Query pattern trigger:
  - More than 5 repeated cross-system joins/aggregations per run, or repeated ad hoc recomputation.
- Collaboration/governance trigger:
  - More than 2 analysts need concurrent query access or auditable role-based access controls.

Decision outcomes:
- If no triggers fire:
  - Stay parquet-native.
- If runtime/data/query triggers fire, but collaboration/governance does not:
  - Introduce DuckDB with external parquet tables and materialized phase summaries under `artifacts/analysis/token_performance/runs/<run_date>/metrics/`.
- If collaboration/governance trigger fires (with any other trigger):
  - Plan migration to managed database (schema + ingestion + retention + access model).

Manual checkpoints:
- [x] Record trigger metrics (runtime, rows, footprint, query count) in run manifest.
- [ ] Approve storage mode for next phase window.
- [x] Document rationale for any storage mode change.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/reports/storage_decision.md`
- `artifacts/analysis/token_performance/runs/<run_date>/run_manifest.json` (with storage decision metrics)

Exit criteria:
- [ ] Storage mode decision captured and approved before Phase 3 starts.

## Phase 3: N-gram Analysis
Purpose: identify phrase-level patterns and candidate phrase features.

Tasks:
- [ ] Generate bigrams and trigrams.
- [ ] Compute TF/DF for n-grams per-country and global.
- [ ] Compute phrase quality metrics (support and PMI/NPMI).
- [ ] Rank n-grams by stability and distinctiveness.

Manual checkpoints:
- [ ] Review top n-grams by country for domain validity.
- [ ] Mark unwanted boilerplate n-grams for suppression list.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/ngram_stats_country.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/ngram_stats_global.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/ngram_review.md`

Exit criteria:
- [ ] N-gram outputs generated and reviewed for all target systems.

## Phase 4: Short-name Derivation and Diagnostics
Purpose: derive short names and measure quality/collision risk.

Tasks:
- [ ] Implement short-name rule pipeline (suffix stripping, normalization, guardrails).
- [ ] Produce candidate short names with confidence scores.
- [ ] Compute per-country/global collision rates and ambiguity diagnostics.
- [ ] Add exception registry for unsafe transformations.

Manual checkpoints:
- [ ] Review false-merge/over-compression samples.
- [ ] Approve rule updates before rerun.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/short_name_candidates.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/metrics/short_name_collisions.parquet`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/short_name_review.md`

Exit criteria:
- [ ] Collision rates within agreed threshold.
- [ ] Approved short-name rule set versioned.

## Phase 5: Visualization and Reporting
Purpose: provide interpretable analysis outputs for technical and manual review.

Tasks:
- [ ] Build static charts for token and n-gram distributions.
- [ ] Build country-vs-global comparison visuals.
- [ ] Build short-name collision visual diagnostics.
- [ ] Publish summary report with decisions and next actions.

Recommended visuals:
- Top tokens by TF/DF per country
- IDF vs commonality scatter
- N-gram heatmaps by country
- Short-name compression and collision histograms

Manual checkpoints:
- [ ] Review visuals for readability and analytical utility.
- [ ] Approve chart set for recurring runs.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/viz/*.png`
- `artifacts/analysis/token_performance/runs/<run_date>/reports/summary_report.md`

Exit criteria:
- [ ] Visual report complete for country and global scope.

## Phase 6: Operationalization and Governance
Purpose: make the analysis repeatable and trackable.

Tasks:
- [ ] Add CLI entry points for each phase.
- [ ] Add run metadata manifest (parameters, data snapshot, runtime).
- [ ] Add unit tests for key metric calculations.
- [ ] Add regression checks for key summary metrics.

Manual checkpoints:
- [ ] Confirm reproducibility across reruns.
- [ ] Sign off runbook and ownership.

Outputs:
- `artifacts/analysis/token_performance/runs/<run_date>/run_manifest.json`
- `src/analysis/runbook.md`

Exit criteria:
- [ ] One-command execution path for standard analysis run.
- [ ] Runbook approved.

---

## Progress Tracker

Use this table to track status as work proceeds.

| Phase | Owner | Status | Started | Target | Completed | Notes |
|---|---|---|---|---|---|---|
| Phase 0: Data Contract/Profile | | Complete | 2026-06-21 | 2026-06-21 | 2026-06-21 | Approved: required-column contract and profile artifacts generated; no schema drift. |
| Phase 1: Token Metrics | | In Progress | 2026-06-22 | 2026-06-22 | | Implemented `token_metrics.py`, `analyze_tokens.py`, tests, and report/plot outputs; full run validation pending on database-backed execution. |
| Phase 2: IDF/Inverse-IDF | | Not Started | | | | |
| Phase 2.5: Storage Decision Gate | | In Progress | 2026-06-22 | 2026-06-22 | | Runtime/data-volume trigger observed; DuckDB backend introduced for Phase 1 pilot. Approval/sign-off pending. |
| Phase 3: N-gram Analysis | | Not Started | | | | |
| Phase 4: Short-name Analysis | | Not Started | | | | |
| Phase 5: Visualization | | Not Started | | | | |
| Phase 6: Operationalization | | Not Started | | | | |

Status values:
- `Not Started`
- `In Progress`
- `Blocked`
- `In Review`
- `Complete`

---

## Manual Review Log Template

Copy this section for each review cycle.

```md
### Review Cycle: <date>
- Scope: <phase(s)>
- Reviewer(s): <names>
- Dataset snapshot: <run_date / source>
- Findings:
  - <finding 1>
  - <finding 2>
- Decisions:
  - <decision 1>
  - <decision 2>
- Follow-up actions:
  - [ ] <action 1>
  - [ ] <action 2>
```

---

## Critique and Risks

Strengths:
- Uses finalized `data/` outputs, reducing stage-contamination risk.
- Clear separation of concerns (`analysis/` vs `acquisition/`).
- Supports both automated and manual quality loops.

Risks:
- Token rarity can over-rank noisy OCR/artifact terms.
- N-gram candidate volume can grow quickly at global scale.
- Short-name compression can inflate collisions if over-aggressive.

Mitigations:
- Add token quality filters and stopword governance.
- Apply minimum support thresholds for n-grams.
- Require manual sign-off for short-name rule changes.

---

## Language and Tooling Recommendations

Primary recommendation: Python.
- Best alignment with current repository and test workflow.
- Strong stack for this plan: DuckDB + Polars + Altair/Plotly.

Optional reporting layer: R.
- Use R/Quarto only for final reporting if preferred by analysts.
- Keep core metric computation in Python for single-source logic.

Backend recommendation:
- DuckDB as the main analytical engine for both per-country and global computations.

---

## Initial Milestone Recommendation

Milestone 1 (Pilot on `fr`):
- Complete Phases 0-3 for `fr`.
- Validate manual review workflow and output formats.
- Then scale to `gb`, `ie`, and global aggregation.
