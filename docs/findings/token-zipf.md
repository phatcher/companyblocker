# Token Zipf findings

Measured with `scripts/analyze_token_zipf.py` on every live system.

- The cleansed-name corpora fit a flatter rank-frequency slope than their general-language reference in every mapped language except German, whose spelled-out legal forms make it steeper.
- Type-level OOV runs far above occurrence-weighted OOV.
- The noise-word trim setting leaves slope and hapax incidence unchanged. Only occurrence-weighted OOV responds, rising as the trim gets more aggressive, which is the trim artefact the report header warns about rather than a property of the corpus.

## A confound in the raw tier

Scoring raw, uncollapsed names against general-language frequency makes the raw tier look more language-like than it is. Spaced initials split into single letters, and single letters are common English words: `wordfreq.zipf_frequency` scores `i`, `b` and `m` at 7.09, 5.35 and 5.39, against 3.93 for the collapsed `ibm`, and `s`, `r` and `l` at 5.86, 5.35 and 5.18 against 2.43 for `srl`. Cleansing's `singlechar` step collapses these runs by the basic tier, so a raw-against-basic comparison either collapses them before scoring or reports the raw tier's figure as confounded by it.
