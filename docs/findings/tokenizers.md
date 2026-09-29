# Tokenizer findings

## A pooled scope's average hides per-jurisdiction breaches

Measured with `scripts/measure_gleif_jurisdiction_disparity.py` on GLEIF's pooled WordPiece tokenizer, read-only against the cleansed layer. Of the jurisdictions with at least a thousand rows, a third, holding half the rows, individually breach the fertility tolerance the pooled average clears. The English-language jurisdictions are best served.

## Pooling noise words by rule

The global training corpus is sampled in proportion to each system's rows, so the packaged default list's document-frequency percentages are dominated by the largest systems. Pooling `ie`, `gb`, `fr` and `offeneregister` under the `equal` rule shares 8 of `ie`'s own 10 balanced-profile tokens, against 4 for the proportional global list.
