# Tokenizer findings

## A pooled scope's average hides per-jurisdiction breaches

Measured with `scripts/measure_gleif_jurisdiction_disparity.py` on GLEIF's pooled WordPiece tokenizer, read-only against the cleansed layer. Of the jurisdictions with at least a thousand rows, a third, holding half the rows, individually breach the fertility tolerance the pooled average clears. The English-language jurisdictions are best served.

## Pooling noise words by rule

The global training corpus is sampled in proportion to each system's rows, so the packaged default list's document-frequency percentages are dominated by the largest systems. Pooling `ie`, `gb`, `fr` and `offeneregister` under the `equal` rule shares 8 of `ie`'s own 10 balanced-profile tokens, against 4 for the proportional global list.

## What each noise layer removes

Measured with `scripts/analyze_noise_layers.py`, each layer against the raw tier. The cleanse layer removes between 484 and 3,280 distinct tokens on `ie`, `gb` and `fr`, and 120,182 on `offeneregister`; a noise-word layer removes between 1 and 103. The packaged default list strips `la`, `le`, `les`, `des`, `du`, `societe` and `sarl` from `gb`.

## A pooled vocabulary crowds out the small corpus

A forecast from `scripts/analyze_vocab_shrinkage.py`, not a trained tokenizer. Sampling in proportion to rows leaves `fr` at 12.9 million rows and `ie` at 0.8 million, so at a vocabulary of 100 a count-weighted pool drops 39% of `ie`'s top-100 mass against 14% of `fr`'s. Equal weighting drops 19% of `ie`'s.
