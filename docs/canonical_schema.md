# Canonical Schema

This project writes country source data into a shared canonical parquet schema in:

- `data/{country}/canonical/{snapshot_date}/{country}-NNN.parquet`

Canonicalization logic is implemented in `acquisition/canonical.py`.

## Columns

Core fields align to the [OpenCorporates company object](https://api.opencorporates.com/documentation/API-Reference#Companies) naming where possible. Columns are listed in output order.

| Column | Type | OpenCorporates | Description |
|---|---|---|---|
| `system_uri` | str, nullable | | Row-level system identifier used to rejoin canonical rows to their source/sidecar records. |
| `jurisdiction_code` | str | Y | The code for the jurisdiction in which the company is incorporated. |
| `company_number` | str | Y | The identifier given to the company by the company register. |
| `name` | str | Y | The legal name of the company. |
| `alternative_names` | list[str], nullable | Y | Alternative names for the company. |
| `previous_names` | list[str], nullable | Y | Previous names of the company. |
| `company_type` | str | Y | The type of company (e.g. LLC, Private Limited Company, GmbH). |
| `current_status` | str | Y | The given description of the filing. |
| `incorporation_date` | str, ISO `YYYY-MM-DD` when parsed | Y | The date the company was incorporated. |
| `dissolution_date` | str, ISO `YYYY-MM-DD` when parsed | Y | The date the company was dissolved. |
| `inactive` | bool | Y | A flag indicating if the company is inactive. |
| `branch` | str | Y | Indicates whether the entry relates to an out-of-jurisdiction branch (`F`), local office (`L`), or neither (null). |
| `branch_status` | str | Y | A descriptive text version of the `branch` flag. |
| `registered_address_in_full` | str | Y | The registered address of the company, as a single string. |
| `registry_url` | str | Y | The url of the company's page in the company register. |
| `opencorporates_url` | str, currently null placeholder | Y | The url of the company on OpenCorporates. |
| `lei` | str, nullable | | GLEIF Legal Entity Identifier, where resolvable. |
| `vat` | str, currently null placeholder | | VAT registration number. |
| `tax_id` | str, currently null placeholder | | Tax identification number. |
| `match_uri` | str, nullable | | Cross-system identifier used for entity matching/linking. |
| `metadata_generated_utc` | str | | Canonical row generation timestamp in UTC. |

`system_uri`, `lei`, `vat`, `tax_id`, `match_uri`, and `metadata_generated_utc` are project additions, not OpenCorporates company object fields. `opencorporates_url`, `vat`, and `tax_id` are reserved placeholders: always null until record-level derivation is implemented for each.

## Source column preservation

Canonical files retain all original source columns after canonical columns. This preserves source fidelity for audit and debugging.

If a source does not contain `CompanyName`, a helper `CompanyName` column is synthesized from the canonical `name`.

## Country-specific mapping

Field mapping candidates are `system_field_candidates` in each system's catalog entry (`src/acquisition/catalog/systems/<system>.json`), typed in `models_plan.py` and resolved by `canonical.py`. A system with none configured fails rather than canonicalizing with an empty mapping.

- `gb` maps from Companies House-style fields (for example `CompanyName`, `CompanyCategory`, `CompanyStatus`).
- `fr` maps from SIRENE-style fields (for example `denominationUniteLegale`, `categorieJuridiqueUniteLegale`).
- `ie` maps from CRO-style fields when present (`company_name`, `company_number`, `status`, etc.).

### Notable transforms

- GB address is composed from:
  - `RegAddress.AddressLine1`
  - `RegAddress.AddressLine2`
  - `RegAddress.PostTown`
  - `RegAddress.County`
  - `RegAddress.PostCode`
- Date parsing accepts `%Y-%m-%d`, `%d/%m/%Y`, `%Y%m%d` and emits ISO date strings.
- `inactive` is derived from `current_status` when no explicit inactive field exists.

## Stability expectations

- Column names above are the contract for downstream cleansing/tokenization.
- New countries should map into existing canonical columns rather than creating country-specific canonical schemas.
- If canonical columns change, update this doc and affected tests in `src/tests/acquisition/test_canonical.py`.
