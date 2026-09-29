# Optimize search methodology: coverage and confidence

This document explains two things about `train_tokenizer.py --mode optimize`
that aren't obvious from the code alone: (1) why the per-seed hash function
matters, and (2) how much of the vocab/min-frequency search space is actually
covered, and how much confidence that coverage buys. Numbers below are real,
pulled from an actual IE run on 2026-08-26 (`artifacts/tokenizers/ie/optimize/run_logs/`),
not hypothetical. A tokenizer's sweeps now sit under
`artifacts/tokenizers/work/<scope>/<tokenizer_id>/optimize/`.

**Correction note (2026-08-26, same day):** the first version of this
document was written from an IE run that turned out to have been executed by
a background process started *before* the hash fix landed on disk -- Python
doesn't hot-reload an already-imported module, so that entire run silently
used the old broken split formula despite the source file being fixed
partway through. The run was re-done after clearing all cached optimize
artifacts for IE (`artifacts/tokenizers/ie/optimize/` -- models, run logs,
and critically the `split_seed_*`/`train_seed_*`/`validation_seed_*` files,
whose reuse check is keyed on `(seed, validation_fraction, corpus_mtime_ns)`
and is completely blind to a code change) and confirming directly that the
new run's own seed-42/seed-43 validation sets overlap ~20% (not the old
76.4%). Every number below reflects the corrected run. The qualitative
finding that motivated this whole investigation -- vocab sizes above ~50K
not reliably beating the 25-40K range -- held up under the fix; only the
*noise estimate* changed.

## Part 1: what "seed" actually varies, and why it must be a real hash

A seed does **not** explore a different point in the vocab/min-frequency
grid -- it selects a different pseudo-random train/validation split of the
*same* corpus (`write_seed_split_artifacts`). Running the same
`(vocab_size, min_frequency)` candidate across 5 seeds is repeated-holdout
validation: it estimates how much the observed `fertility_distance` for that
candidate would vary if you'd happened to draw a different validation
sample, not how good the candidate is compared to others.

That estimate is only meaningful if the seeds actually produce independent
splits. Until 2026-08-26 they didn't: the split assignment was a hand-rolled
linear formula using the classic weak ANSI-C `rand()` LCG constants
(`a=1103515245, c=12345, m=2^31`), with the seed folded in as a pure
additive offset. Measured on a real 817,766-row corpus:

| seed pair | validation-set overlap (old formula) | expected under independence |
|---|---|---|
| 42, 43 | 76.4% | ~20% |
| 42, 46 | 5.6% | ~20% |
| 42, 142 | 0.0% | ~20% |

Replacing the formula with Polars' built-in hash (`pl.col("row_nr").hash(seed=seed)`)
brings every pair back to ~20% overlap, matching genuine independence. The
practical consequence of the old bug: seeds 42-46 (the default sequential
seed list, `base_seed + i`) were not 5 independent samples of validation
noise -- they were mostly duplicates of each other or arbitrarily
unrelated, depending on the specific numeric gap. Multi-seed statistics
computed from them (median fertility distance, pass rate) were noisier and
less trustworthy than the design assumes. This is fixed now (see
`write_seed_split_artifacts` in `company_tokenize/optimize.py`), and
`test_write_seed_split_artifacts_gives_independent_splits_across_seeds`
guards against it regressing.

## Part 2: the search space is a fixed discrete grid, not a continuous search

`--vocab-sizes` and `--min-frequencies` define a **lattice**: the default
grid is 10 vocab points x 6 min-frequency points = 60 cells. There is no
interpolation and no local refinement search between grid points -- a value
like 45,000 or 120,000 is never evaluated unless it's explicitly in
`--vocab-sizes`. "Exploring the search space" here means "covering cells in
this lattice with enough seeds to trust the ranking," not searching a
continuous parameter.

Two-phase coverage (after the 2026-08-26 pilot-phase fix -- see
`update_pilot_combo_cutoffs`'s docstring for why elbow-based pruning no
longer excludes any pilot cell):

- **Pilot**: every cell in the grid gets exactly 1 seed. 60/60 cells
  covered, unconditionally.
- **Expansion**: only cells on the "frontier" (`select_vocab_frontier` --
  the top `--vocab-frontier-top-k` per min-frequency, plus any within
  `--vocab-frontier-distance-tolerance` of the best) get the remaining
  seeds (4 more, for 5 total).

Real numbers from the IE run: **60/60 cells evaluated at all** (confirming
the pilot-phase fix), but only **33/60 (55%) promoted to the full 5-seed
expansion**. The other **27/60 (45%) remain single-seed estimates
permanently** -- their pilot observation is the only evidence that ever
exists for them.

## Part 3: how much confidence does that coverage buy?

For the 33 cells that did get 5 (genuinely independent) seeds, the real
measured spread is: **median standard deviation of `fertility_distance`
across seeds was ~0.001036** in the corrected IE run. Using that as a
working noise estimate, the standard error of the *difference* between two
5-seed cells' means is approximately:

```
SE_diff ~ sigma * sqrt(2/n) ~ 0.001036 * sqrt(2/5) ~ 0.00066
```

So two frontier cells whose mean `fertility_distance` differs by
meaningfully more than ~0.0013 (roughly 2 x SE_diff) can be told apart with
reasonable confidence at 5 seeds each. Cells closer than that are
statistically indistinguishable from this run's noise floor -- picking one
over the other is not a validated result, it's a coin flip that happened to
land somewhere.

Applying this to the actual result: the winning cell (vocab=25,000,
`fertility_distance` ~0.0018) beat the 50K/100K/150K/auto cells (all
~0.038-0.062 for the same min_frequency) by **21 to 110 x sigma** --
nowhere near the noise floor. For IE, "smaller vocab wins" is a real,
confidently-supported result, not noise.

One thing the fix also surfaced that the original (contaminated) analysis
missed: correlation between a cell's mean `fertility_distance` and its
seed-to-seed standard deviation is **0.50** in the corrected run (it looked
like -0.13, i.e. no relationship, under the broken hash). There's real mild
heteroscedasticity -- noisier cells tend to have somewhat larger absolute
noise -- so a single global sigma is a simplification, not an exact model.
It's good enough for the order-of-magnitude comparisons above, but a more
careful treatment would use a per-cell or per-magnitude-band sigma rather
than one number for the whole grid.

**This is a real number now, but it's specific to IE's corpus, corpus size,
and vocabulary composition.** It should not be assumed to hold for FR (16x
more rows), gleif, gb, or offeneregister without checking -- the same
measurement should be repeated per system before trusting a "confident"
ranking there.

## Part 4: what we honestly can't claim

- **The 27 single-seed cells have no variance estimate at all.** If any of
  them is within noise of the frontier cutoff (plausible, since the frontier
  boundary itself is drawn from single-seed pilot data), it could easily be
  a false negative: a genuinely competitive candidate that looked worse than
  it truly is on one noisy draw, and that never gets a second chance to
  prove otherwise. There's no way to bound how often this happens without
  either running every cell to 5 seeds (defeats the point of pruning) or
  spot-checking a sample of excluded cells with extra seeds.
- **We cannot state a calibrated "probability of finding the true optimum"
  in the grid.** That would require knowing the true underlying
  fertility-distance-vs-(vocab, min_frequency) surface, which is exactly
  what we don't have -- we only have noisy point samples of it. What we
  *can* say, honestly: given the measured ~0.0013 minimum-distinguishable
  gap, if the true best and second-best grid cells differ by more than that,
  the current design's frontier selection will very likely surface the
  right one; if they differ by less, the selection among near-ties is
  effectively arbitrary, and this is consistent with the empirical
  observation that motivated this write-up (a 100K-vocab candidate has not
  reliably outperformed a 40K-vocab one in prior experiments -- some of that
  may be genuine near-tied performance, not just noise, and no amount of
  additional seeds resolves a genuine tie).
- **The grid itself could simply miss the true optimum.** Even with perfect
  seed independence and unlimited seeds per cell, if the real optimum sits
  at, say, vocab=75,000 and that value was never in `--vocab-sizes`, it will
  never be found. Widening or densifying the grid (at the cost of more pilot
  jobs, which are cheap relative to expansion) is the only way to reduce
  this risk, and is a separate lever from seed count entirely.

## Part 5: in-sample fit vs. held-out generalization -- selection was optimizing the wrong one

**2026-08-26, same investigation.** The candidate archive's comparison tooling
surfaced something Part 1-4 above don't cover: a candidate's validation-split
`fertility_distance` (what the optimize sweep used to pick a winner) and its
full-corpus `fertility_distance` (what `compare_tokenizer_candidates.py`
reports, and what the promoted tokenizer actually exhibits once `--finalize`
retrains it on the full corpus) can rank candidates *differently* -- not
because of noise, but systematically.

**Root cause.** A WordPiece vocabulary always tokenizes the data it was built
from somewhat more efficiently (lower fertility) than data it never saw --
the vocab's merges are literally chosen to fit train-split frequency
statistics. Measured directly on `ie` (mf=8, all five expansion seeds, very
tight/consistent across seeds -- this is a real effect, not noise):

| vocab | validation-split (held-out) `fertility_distance` | full-corpus / train-split (in-sample) `fertility_distance` |
|---|---|---|
| 21,000 | 0.0165 | 0.0115 |
| 23,000 | 0.0072 | **0.0018** |
| 24,000 | 0.0031 | 0.0025 |
| **25,000 (the original winner)** | **0.0006** | 0.0068 |
| 27,000 | 0.0079 | 0.0142 |
| 29,000 | 0.0128 | 0.0211 |

Both columns are genuine U-shapes (fertility crosses the 1.25 target and
comes back out), but their minima sit ~2,000 vocab apart. The sweep picked
25,000 because that's where the *held-out* column bottoms out; the promoted
tokenizer (trained on 100% of the corpus, no split at all) actually behaves
like the *in-sample* column, whose minimum is at 23,000 -- **~3.8x closer to
target than what got promoted.**

**Which one should selection optimize?** This is a real question, not just a
bug to patch mechanically -- validation-split fertility is a legitimate
measure of generalization to genuinely unseen company names, and in-sample
fertility is a legitimate measure of fit to the deployment corpus, and they
disagree. It was settled by checking how the promoted tokenizer is actually
used, not by picking whichever number looked more rigorous: this pipeline
retrains on the full current cleansed snapshot and, in the same run, applies
that tokenizer to tokenize/validate/match against that *same* snapshot
(`src/acquisition/tokenizer_ops.py`, `scripts/process_companies.py`,
`src/validation/runner.py` all draw from `data/<system>/cleansed`). There is
no live/streaming matching service consuming genuinely novel names in real
time. So the deployed tokenizer's real operating condition is in-sample, and
in-sample fertility is what selection should optimize -- the held-out
validation split, useful as it is for other purposes, was never a proxy for
how this specific pipeline uses the artifact.

One caveat worth watching, not solved here: an operator can skip retraining
and later tokenize a *refreshed* cleansed snapshot with a stale tokenizer
(README's "existing-tokenizer path"), which does introduce real drift toward
genuinely-unseen names over time. Nothing currently measures how often that
happens in practice; if it turns out to be common, the in-sample framing
degrades between retrains and this conclusion should be revisited.

**The fix.** `compute_candidate_selection_score`'s fertility-distance-to-target
term now takes `train_metrics["fertility_distance"]` (in-sample proxy, and
free -- already computed per candidate) instead of
`validation_metrics["fertility_distance"]`. `unk_rate` and the two
token-count terms stay sourced from validation, and rejection gating
(`classify_candidate_rejection`) is untouched -- OOV risk and the
fertility/unk generalization-delta stability terms are genuinely about
held-out behavior, unlike the target-crossing point, so validation still does
real work there. `select_best_pair` also now recomputes `selection_score`
from each run-log row's raw stored metrics under the *current* formula
(`recompute_selection_scores`) rather than trusting a persisted scalar --
the optimize canary/reuse system can skip retraining a candidate whose
config hasn't changed, and a scoring-formula change (like this one) isn't a
config change, so a stale pre-fix score could otherwise silently survive
into a later run's selection.

**Not addressed here, flagged separately:** `min_frequency` selection has
the same blind spot Part 4 raises about the grid generally -- fertility_distance
measures average tokens/word, and has no way to detect whether pruning
low-frequency subword patterns (a `min_frequency=8` winner, versus the naive
baseline's deliberate `min_frequency=1`) costs discriminative power for rare
company names specifically, which are disproportionately the hard cases for
downstream blocking/matching. That needs a real-workload check, not another
fertility-based sweep.

## Part 6: verification run, a correction, and the deeper unresolved problem

**2026-08-26, same investigation, after the fix above landed.** Re-ran `ie`
optimize with the default grid to confirm the fix and re-promote. Result:
`median_train_fertility_distance` reported mid-sweep (0.00738) matched the
actual full-corpus retrain's `fertility_distance` (0.00679) to within ~8% --
direct confirmation the train-split proxy tracks real deployment behavior,
versus the old validation number being off by 11x. Promoted vocab stayed at
25,000 (`min_frequency` moved 8->1, an inconsequential tie-break -- train and
validation `fertility_distance` are numerically identical across
min_frequency 1-8 at this vocab/corpus size; the frequency floor only starts
binding around 12). Archived as `v25000_mf1_train_split_fix_winner`.

**Correction to the vocab=23,000 claim above.** The default grid doesn't
include 23,000 (points are 10000/20000/25000/...), so a follow-up densified
sweep was run to actually reach it -- range derived from the data, not
guessed: fit the local fertility-vs-vocab slope from the directly-measured
points, solved for the fertility=1.25 crossing (~23,412-23,456), then used
the *actual* train-split noise floor from this run's 5 seeds at vocab=25,000
(sigma=0.000339, 2xSE_diff=0.00043) to interpolate where distance crosses
`best_observed + 2xSE_diff` between the real 22K/23K/24K measurements --
giving 22,900-23,600, swept at 100-vocab steps x min_frequency {1,8}.
**Every single candidate in that entire band was rejected**, universally,
on `token_count_p95` (validation p95=7 tokens; the gate limit is <=6) --
not seed noise, a hard deterministic wall across every vocab point, both
min_frequencies, both seeds. The earlier "~3.8x better" number was computed
by hand-evaluating candidates directly against `evaluate_tokenizer_metrics`,
bypassing `classify_candidate_rejection` entirely -- so it never checked
whether those candidates would actually survive the full gate set. They
don't. vocab=25,000 sits right at the p95=6 boundary; nothing between 23,600
and 25,000 has been tested, so a smaller gate-passing vocab might still exist
there, but the easy win evaporated once run through the real pipeline rather
than checked on fertility_distance alone.

**The deeper problem this surfaced, still open.** `git log --all -S` traced
`fertility_target=1.25`, `fertility_tolerance=0.10`, `unk_rate_threshold=0.005`,
`token_count_median_max=4.0`, and `token_count_p95_max=6.0`
(`OPTIMIZE_BASE_DEFAULTS` in `src/training/optimize_execution.py`) back to
their first appearance (commit `d38a490`, "Parallelize tokenizer optimize
candidates", 2026-07-01) -- introduced together as bare literals, no comment,
no commit message discussion. The only validation ever applied to them is
internal *consistency* (`token_count_p95_max >= token_count_median_max`,
`fertility_min < fertility_max`), never a check against anything downstream.
This whole investigation -- Part 1-6, real statistics, a real code fix, a
data-derived search range -- has been getting more and more precise about
finding the vocab that best satisfies a target and gates that were never
themselves validated against what they're supposed to be proxies *for*
(downstream blocking/matching quality). The `token_count_p95<=6` rule that
just rejected an entire vocab band is a reasonable-sounding instinct (don't
let unusual/long names fragment into a large pile of subword pieces) with no
evidence it's calibrated at the right number, or that it matters to
clustering quality at all. No amount of further fertility/token-count
statistics resolves this -- it requires actually running `validate_clustering.py`
against a few already-archived candidates that differ meaningfully on these
proxy metrics (`naive_v40000_mf1`, `v25000_mf8_prior_operational`,
`v25000_mf1_train_split_fix_winner` span a decent range already) and checking
whether real match/cluster quality moves at all. If it doesn't move across
that range, this entire proxy-metric optimization thread has been sharpening
a number that doesn't matter to the actual task.

**Framing corrected by Part 7.** Treating the p95 rejection as this part's
finding -- "the easy win evaporated" -- conceded the argument to the weakest
constant in the system. The gate is not merely unvalidated: it is a step
function whose cap sits at the modal value, and the 23,000 candidate's
fertility advantage was never retracted.

## Part 7: `token_count_p95` is a step function, and Part 6 let it win the argument

**2026-09-05.** Part 6 closed by saying the vocab=23,000 win "evaporated once run
through the real pipeline," having been rejected on `token_count_p95`. Measuring
the gate across every persisted run log says that framing was too generous to the
gate. Three findings, in order of how much they change what to do next.

**It is not a Part 5-style measurement bug.** The obvious hypothesis, given Part 5,
was that the gate reads the validation split while the deployed artifact is
in-sample. It doesn't matter here: across all 493 candidates in `fr`/`gb`/`ie`/
`gleif`/`offeneregister`, `token_count_p95` is identical on train and validation
in 487, differing in 6 (all `gb` at vocab=20,000). Unlike fertility, a coarse
integer tail statistic of name length does not care whether the vocabulary was
fitted to the names it is measured on. The 23,000 band really does exhibit p95=7.
The rejection was arithmetically correct.

**The metric barely responds to the parameter it gates.** Median `token_count_p95`
across each scope's full vocab grid:

| scope | p95 from vocab 10,000 to 150,000 | fertility over the same range |
|---|---|---|
| `gb` | 7, then 6 for all of 20,000-150,000 | 1.133-1.377 |
| `ie` | 7, 7, then 6 from 30,000 up | 1.199-1.365 |
| `fr` | 8, then 7 across 10,000-100,000, 6 at 150,000 | 1.212-1.628 |
| `gleif` | 11 down to 8, never lower | 1.197-1.592 |
| `offeneregister` | 12 down to 9, never lower | 1.110-1.474 |

Two or three integer values across a fifteen-fold span of vocabulary, against a
fertility signal that varies continuously and informatively over the same range.
`token_count_p95` is a step function with one step in it, and the cap is drawn at
the modal value, so moving it by one integer either admits nearly everything or
rejects nearly everything. What the gate costs is therefore decided by where that
single step happens to fall relative to the fertility elbow, which is a property
of the corpus rather than of any candidate: in `gb` the step sits at 10,000-20,000,
far below the elbow near 26,000, and the gate is inert; in `ie` it sits at
20,000-30,000, straight through the middle of the optimum region, and the gate
bisects the answer; in `gleif` and `offeneregister` the floor is 8 and 9, out of
reach at any vocabulary, and the gate voids the whole sweep. The correlation
against `fertility_distance` splits the same way (`gleif` r=0.956, `fr` r=0.814,
`ie` r=0.167), so the gate is close to a duplicate of fertility in the scopes
where it does no damage and independent in the one where it does.

**The premise was sound and the calibration was not.** "Don't let unusual names
fragment into a large pile of subword pieces" is a reasonable instinct, and for
this repo's task it has a mechanism fertility lacks: token count drives candidate-pair
volume in blocking. The error is that a guard against a degenerate vocabulary was
set at 6, the value the winners actually exhibit, rather than somewhere a
degenerate vocabulary alone would reach. That is a tripwire installed inside the
operating range, which makes it a selection criterion by accident. `unk_rate` is
the same kind of metric and is correctly loose enough to have never fired.

So Part 6's conclusion stands corrected: the 23,000 candidate's advantage on
train-split `fertility_distance` (0.0018 against 25,000's 0.0068) was never
retracted, only the hand-evaluation that first found it. What blocked it was the
least-justified constant in the system, sitting one integer step away on a metric
that moves twice across the entire search space.

## Part 8: what a third, refining phase would have to look like

Pilot and expansion answer "which cells are worth more seeds." Neither answers
"where between two adjacent grid points does the optimum sit," and the default
grid's 5,000-vocab spacing near the elbow is coarse enough that the answer is
never on the grid: `ie`'s in-sample minimum interpolates to roughly 23,300 and
`gb`'s to roughly 26,000, and neither is a grid point. A third phase that
refines inside the winning bracket is the missing piece, with four constraints.

**Vary vocab only.** A refinement round over the full `(vocab, min_frequency)`
lattice multiplies out for no gain: Part 6 measured train and validation
`fertility_distance` as numerically identical across `min_frequency` 1-8 at
`ie`'s vocab and corpus size, with the frequency floor only beginning to bind
around 12. Fixing `min_frequency` at the winning value and sweeping vocab alone
keeps the round one-dimensional, which is what makes a denser step affordable
at all.

**Densify rather than bisect.** Bisection converges into the noise, because near
a minimum the difference between two probe points shrinks faster than the noise
does, so the search stops descending and random-walks. It also gives up the
property that saved every existing run log when Part 5 changed the scoring
formula: `recompute_selection_scores` re-scored persisted rows under the new
formula, which works because *which* points were evaluated did not depend on the
old one. A bisection path does depend on it, so a later formula change
invalidates the trajectory rather than just the scores. The cost argument does
not favour bisection either -- an 11-point grid at 1,000-vocab steps across a
10,000-wide bracket is about what bisecting to the same precision costs, and
every grid point stays independently reusable.

**Give every refined point full seeds; do not pilot-prune them.** The 1-seed
pilot discriminates at coarse spacing because the signal there is enormous
(`gb`: 0.0287 at 20,000 against 0.0038 at 25,000, tens of sigma). At 1,000-vocab
spacing neighbouring points differ by order sigma, so pilot pruning would select
a frontier close to at random. The refinement round is a different statistical
regime from the phase it follows, not the same machinery with a finer grid.

**The distance is a V, not a parabola.** Fertility distance is
`|fertility - target|`, and fertility falls smoothly as vocab grows, so where it
crosses the target with a non-zero slope the distance has a kink: near the
crossing it is `|slope| * |vocab - crossing|`, the same steepness on both sides
up to the point. Seed noise softens the tip slightly, since the absolute value of
a noisy number near zero is biased upwards. A parabola fitted through three
points of that curve gets its shape wrong: on `ie`'s bracket 20,000-30,000 at
`min_frequency` 1, 1 of 11 refined points fell within noise of the fitted
parabola.

**Model fertility, and read the distance off it.** Refining interpolates median
in-sample fertility between the three bracket points, linearly in log vocab, and
predicts each refined point's distance as `|fertility - target|`. The same `ie`
points, replayed through this model, all 11 fall within noise of it. The step is
where the V's penalty matches the noise, `d = SE_diff / |slope|`, with the slope
in fertility per vocab at the winner. The noise is the seed spread of fertility,
not of distance: seeds either side of the target fold onto the same distance, so
the distance's spread understates the noise exactly where refining looks. On
`ie` that gives a step of about 50 vocab, below the 100 floor, because near the
crossing fertility moves by one noise unit in about 50 vocab.

**Train the crossing, not the bracket.** The V's point is where interpolated
fertility crosses the target, so a refining round trains that vocab and one step
either side, three points at full seeds, rather than a grid across the bracket:
at a 100-vocab step, `ie`'s 10,000-wide bracket would otherwise be about 100
points. When fertility stays on one side of the target across the bracket, the
round centres on the winning vocab instead. This supersedes the 11-point grid
the paragraph on bisection above assumes.

## Practical takeaway

Trust a "winner" more when: it clears the ~2xSE_diff gap over its nearest
competitor at 5 seeds, and that gap has been measured for *that* system's
corpus (not assumed from IE). Trust it less -- and treat near-ties as
genuinely ambiguous rather than resolved -- when candidates are closer than
that, especially if either one only ever received a single pilot seed.

## Operational note: don't extrapolate per-worker memory from one run either

Not a search-methodology point, but a related "don't trust one number"
lesson from the same two IE runs: peak single-worker memory nearly tripled
between them at identical settings (`--max-workers 6`, same corpus) -- 3.1GB
in the first run vs. 8.9GB in the corrected one -- simply because different
vocab-size candidates happened to land concurrently on the same worker slot.
Median system CPU stayed low (~29%) even at that worker count, meaning the
workload is memory-bound, not CPU-bound. Whatever worker count is chosen for
FR/gb/gleif/offeneregister, don't linearly extrapolate it from a single
IE measurement -- run count *and* memory should both be re-measured per
system, the same way sigma is.
