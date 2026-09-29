"""
Analyze private company token patterns by country.

Generates country_analysis.parquet (raw counts) and country_analysis.txt
(formatted report) in the selected cleansed directory.
"""

import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl

# Default output directory when callers do not provide one.
DEFAULT_OUTPUT_DIR = Path("data/gb/cleansed")

NOISE_WORDS = {
    "UK",
    "EUROPE",
    "BRANCH",
    "INTERNATIONAL",
    "COMPANY",
    "HOLDING",
    "GROUP",
    "GLOBAL",
    "LIMITED",
}


def _resolve_artifact_paths(
    output_dir: Path | str | None,
) -> tuple[Path, Path, Path, Path]:
    base_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
    return (
        base_dir,
        base_dir / "private_country_analysis.parquet",
        base_dir / "private_country_analysis.txt",
        base_dir / "private_country_analysis.md",
    )


def analyze_and_report(
    min_frequency: int = 5,
    *,
    cleansed_dir: Path | str | None = None,
) -> pl.DataFrame:
    """
    Analyze private company tokens and generate reports.

    Args:
    Analyze private company tokens by country and generate reports.

    Creates two reports in data/gb/cleansed/:
        - private_country_analysis.txt: Detailed country-by-token listing
        - private_country_analysis.md: Summary statistics
    """
    # Ensure output directory exists
    output_dir, analysis_file, report_txt, report_md = _resolve_artifact_paths(
        cleansed_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_file}\n"
            "Run the notebook cell that creates private_country_analysis.parquet first."
        )

    # Read analysis data
    print(f"Loading analysis from {analysis_file}...")
    analysis_df = pl.read_parquet(analysis_file)

    # Filter high-frequency tokens
    high_freq = analysis_df.filter(pl.col("quantity") >= min_frequency).sort(
        ["country", "quantity"], descending=[False, True]
    )

    total_countries = analysis_df.select(pl.col("country").n_unique()).item()
    print(
        f"Found {len(high_freq)} high-frequency tokens (>= {min_frequency}) across {total_countries} countries"
    )

    # Generate text report
    print(f"Writing report to {report_txt}...")
    with open(report_txt, "w") as f:
        f.write("=== Countries with Potentially Missed Legal Forms ===\n")
        f.write(f"(Tokens appearing >= {min_frequency} times per country)\n\n")

        current_country = None
        for row in high_freq.to_dicts():
            if row["country"] != current_country:
                current_country = row["country"]
                f.write(f"\n{current_country}:\n")
            f.write(f"  {row['last_token']:30s} - {row['quantity']:4d} companies\n")

    # Generate markdown report with summary statistics
    print(f"Writing summary to {report_md}...")
    with open(report_md, "w") as f:
        f.write("# Private Company Legal Form Analysis\n\n")
        generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"**Date Generated**: {generated_at}\n")
        f.write(f"**Total Records**: {len(analysis_df):,}\n")
        f.write(f"**Total Countries**: {total_countries}\n")
        f.write(f"**High-Frequency Tokens (>= {min_frequency}): {len(high_freq):,}\n\n")

        f.write("## Summary by Country\n\n")

        country_summary = (
            high_freq.group_by("country")
            .agg(
                pl.col("quantity").sum().alias("total_companies"),
                pl.col("last_token").count().alias("unique_tokens"),
                pl.col("last_token").max().alias("top_token"),
                pl.col("quantity").max().alias("top_token_count"),
            )
            .sort("total_companies", descending=True)
        )

        for row in country_summary.to_dicts():
            f.write(f"### {row['country']}\n")
            f.write(f"- Total Companies: {row['total_companies']:,}\n")
            f.write(f"- Unique Tokens: {row['unique_tokens']}\n")
            f.write(
                f"- Top Token: **{row['top_token']}** ({row['top_token_count']:,})\n\n"
            )

    print(f"✓ Report written to {report_txt}")
    print(f"✓ Summary written to {report_md}")
    print("\nTo view high-frequency tokens by country:")
    print(f"  cat {report_txt}")

    return high_freq


def _canonicalize_token(token: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii")
    )
    canonical = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").upper()
    return canonical if canonical else "UNKNOWN"


def _normalize_for_match(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip()).lower()


def _split_rule_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^A-Za-z0-9]+", _normalize_for_match(text))
        if token
    }


def _token_alnum_upper(token: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", _normalize_upper(token))


def _initialism(words: list[str]) -> str:
    chars = []
    for word in words:
        cleaned = re.sub(r"[^A-Z0-9]", "", _normalize_upper(word))
        if cleaned:
            chars.append(cleaned[0])
    return "".join(chars)


def _build_existing_rule_signatures(
    company_type_rules: list[tuple[str, str, str]],
) -> tuple[set[str], set[str]]:
    values = [
        value
        for source, canonical, _country in company_type_rules
        for value in (source, canonical)
    ]

    exact = {_normalize_for_match(value) for value in values}
    exact.update(_canonicalize_token(value) for value in values)

    tokens = (
        set().union(*(_split_rule_tokens(value) for value in values))
        if values
        else set()
    )

    return exact, tokens


def _build_dynamic_long_form_evidence(
    candidates: pl.DataFrame,
    private_records_file: Path,
) -> dict[tuple[str, str], tuple[str, int, int]]:
    """Infer long forms for abbreviation-like tokens from actual private records."""
    if len(candidates) == 0 or not private_records_file.exists():
        return {}

    records = pl.read_parquet(private_records_file)
    if (
        "CountryOfOrigin" not in records.columns
        or "name_cleansed" not in records.columns
    ):
        return {}

    if "company_type" in records.columns:
        records = records.filter(pl.col("company_type") == "PRIVATE")

    candidate_countries = sorted({str(c) for c in candidates["country"].to_list()})
    candidate_tokens = sorted(
        {_normalize_upper(str(t)) for t in candidates["last_token"].to_list()}
    )

    records = (
        records.select(["CountryOfOrigin", "name_cleansed"])
        .filter(
            pl.col("CountryOfOrigin").is_not_null()
            & pl.col("name_cleansed").is_not_null()
        )
        .with_columns(
            pl.col("name_cleansed")
            .cast(pl.Utf8)
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .str.to_uppercase()
            .alias("name_cleansed_u")
        )
        .with_columns(pl.col("name_cleansed_u").str.split(" ").alias("words"))
        .with_columns(pl.col("words").list.last().alias("last_token"))
        .filter(
            pl.col("CountryOfOrigin").cast(pl.Utf8).is_in(candidate_countries)
            & pl.col("last_token").is_in(candidate_tokens)
        )
    )

    phrase_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    pair_totals: Counter = Counter()

    for row in records.iter_rows(named=True):
        country = str(row["CountryOfOrigin"])
        token = str(row["last_token"])
        token_a = _token_alnum_upper(token)
        words = row["words"] or []
        if len(words) < 2 or len(token_a) < 2:
            continue

        pair = (country, token)
        pair_totals[pair] += 1

        max_tail = min(6, len(words))
        for n in range(2, max_tail + 1):
            tail = words[-n:]
            if not tail or tail[-1] != token:
                continue
            prefix_words = tail[:-1]
            if _initialism(prefix_words) == token_a:
                phrase = " ".join(prefix_words)
                phrase_counts[pair][phrase] += 1

    best: dict[tuple[str, str], tuple[str, int, int]] = {}
    for pair, counter in phrase_counts.items():
        phrase, count = counter.most_common(1)[0]
        best[pair] = (phrase, count, pair_totals[pair])
    return best


def _derive_mapping(
    country: str,
    token: str,
    long_form_evidence: dict[tuple[str, str], tuple[str, int, int]],
) -> tuple[str, str | None]:
    evidence = long_form_evidence.get((country, token))
    if evidence:
        phrase, phrase_count, total_rows = evidence
        note = (
            f"Inferred from private records: '{phrase}' + {token} "
            f"({phrase_count}/{total_rows} supporting names)."
        )
        return phrase.title(), note

    return token, None


def _load_company_type_rules_from_source() -> list[tuple[str, str, str]]:
    from company_cleanse.rules import _load_company_type_rules

    return _load_company_type_rules()


def _commentary_notes(token: str, quantity: int, is_existing: bool) -> list[str]:
    notes = []
    if is_existing:
        notes.append(
            "Likely already covered by existing rules; check before adding duplicate."
        )
    else:
        notes.append("Candidate missing legal-form token from private-name analysis.")

    if len(token) <= 2:
        notes.append("Very short token; high ambiguity risk.")

    generic = {
        "FOUNDATION",
        "GROUP",
        "HOLDING",
        "INTERNATIONAL",
        "ESTABLISHMENT",
        "COMPANY",
    }
    if _canonicalize_token(token) in generic:
        notes.append("Generic business term; verify this is a legal form in-country.")

    if quantity >= 20:
        notes.append("High-frequency signal; prioritize review.")
    else:
        notes.append("Medium-frequency signal; review after high-priority items.")

    return notes


def _normalize_upper(token: str) -> str:
    return re.sub(r"\s+", " ", token.strip()).upper()


def _is_likely_noise(country: str, token: str) -> bool:
    token_u = _normalize_upper(token)
    country_u = _normalize_upper(country)
    if token_u in NOISE_WORDS:
        return True
    if token_u == country_u:
        return True
    # Very short generic location fragments are usually noise in this analysis.
    return len(token_u) <= 2


def generate_proposed_rule_additions(
    min_frequency: int = 5,
    high_priority_threshold: int = 20,
    output_path: Path | str | None = None,
    *,
    cleansed_dir: Path | str | None = None,
) -> Path:
    """
    Generate a commentary-rich proposal markdown from private country analysis.

    Writes a separate output file so it can be compared with manually curated notes.
    """
    output_dir, analysis_file, _, _ = _resolve_artifact_paths(cleansed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not analysis_file.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_file}\n"
            "Run the notebook cell that creates private_country_analysis.parquet first."
        )

    if output_path is None:
        output = output_dir / "proposed_rule_template.md"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    analysis_df = pl.read_parquet(analysis_file)
    raw_candidates = analysis_df.filter(pl.col("quantity") >= min_frequency).sort(
        ["country", "quantity"], descending=[False, True]
    )

    rows = []
    for row in raw_candidates.iter_rows(named=True):
        country = str(row["country"])
        token = str(row["last_token"])
        if _is_likely_noise(country, token):
            continue
        rows.append(
            {"country": country, "last_token": token, "quantity": int(row["quantity"])}
        )

    candidates = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame({"country": [], "last_token": [], "quantity": []})
    )

    existing_exact: set[str] = set()
    existing_tokens: set[str] = set()
    try:
        company_type_rules = _load_company_type_rules_from_source()
        existing_exact, existing_tokens = _build_existing_rule_signatures(
            company_type_rules
        )
    except Exception:  # noqa: BLE001
        # If rules cannot be loaded, still produce commentary from frequencies.
        existing_exact, existing_tokens = set(), set()

    private_records_file = output_dir / "private_records.parquet"
    long_form_evidence = _build_dynamic_long_form_evidence(
        candidates, private_records_file
    )

    high_df = candidates.filter(pl.col("quantity") >= high_priority_threshold)
    med_df = candidates.filter(
        (pl.col("quantity") >= min_frequency)
        & (pl.col("quantity") < high_priority_threshold)
    )

    lines: list[str] = []
    lines.append("# Proposed COMPANY_TYPE_RULES Additions\n\n")
    lines.append(
        "Generated from private_country_analysis.parquet with dynamic expansion and filtering heuristics.\n"
    )
    lines.append(f"Threshold: quantity >= {min_frequency}.\n\n")
    lines.append(
        "Likely-noise tokens (country names, generic terms, and short ambiguous fragments) are excluded.\n\n"
    )
    lines.append(
        "Identity mappings (left == right) are handled by preprocessing and are not proposed as addable rules.\n\n"
    )

    def write_section(title: str, frame: pl.DataFrame) -> None:
        lines.append(f"## {title}\n\n")
        if len(frame) == 0:
            lines.append("No candidates.\n\n")
            return

        for country, group in frame.group_by("country", maintain_order=True):
            country_name = country[0] if isinstance(country, tuple) else country
            lines.append(f"### {country_name}\n\n")
            for row in group.iter_rows(named=True):
                token = row["last_token"]
                quantity = int(row["quantity"])
                source_form, inferred_note = _derive_mapping(
                    country_name, token, long_form_evidence
                )
                canonical = _canonicalize_token(token)
                is_existing = (
                    _normalize_for_match(token) in existing_exact
                    or _canonicalize_token(token) in existing_exact
                    or _normalize_for_match(source_form) in existing_exact
                    or _canonicalize_token(source_form) in existing_exact
                    or _normalize_for_match(token) in existing_tokens
                    or _canonicalize_token(token) in existing_tokens
                    or _normalize_for_match(source_form) in existing_tokens
                    or _canonicalize_token(source_form) in existing_tokens
                )
                status = "DUPLICATE?" if is_existing else "NEW"

                lines.append("```python\n")
                lines.append(f'("{source_form}", "{canonical}"),\n')
                lines.append("```\n")
                lines.append(f"- **Count**: {quantity} companies\n")
                lines.append(f"- **Status**: {status}\n")
                if inferred_note and not is_existing:
                    lines.append(f"- **Source**: {inferred_note}\n")
                for note in _commentary_notes(token, quantity, is_existing):
                    lines.append(f"- **Note**: {note}\n")
                lines.append("\n")

    write_section(
        f"High-Priority Additions (Frequency >= {high_priority_threshold})", high_df
    )
    write_section(
        f"Medium-Priority Additions (Frequency {min_frequency}-{high_priority_threshold - 1})",
        med_df,
    )

    lines.append("## Validation Checklist\n\n")
    lines.append("1. Confirm token is a legal form in country context.\n")
    lines.append("2. Avoid duplicates with existing COMPANY_TYPE_RULES entries.\n")
    lines.append("3. Add unit tests for accepted new rules.\n")

    output.write_text("".join(lines), encoding="utf-8")
    print(f"Generated proposal file: {output}")
    return output


def generate_resolved_rule_file(
    template_path: Path | str | None = None,
    output_path: Path | str | None = None,
    *,
    cleansed_dir: Path | str | None = None,
) -> Path:
    """
    Create a resolved working copy from the proposal template.

    The resolved file is intentionally a separate artifact so manual/Copilot updates
    can be reviewed against the untouched template.
    """
    if template_path is None:
        output_dir, _, _, _ = _resolve_artifact_paths(cleansed_dir)
        template = output_dir / "proposed_rule_template.md"
    else:
        template = Path(template_path)

    if output_path is None:
        output_dir, _, _, _ = _resolve_artifact_paths(cleansed_dir)
        output = output_dir / "proposed_rule_resolved.md"
    else:
        output = Path(output_path)

    if not template.exists():
        raise FileNotFoundError(
            f"Template file not found: {template}\n"
            "Run generate_proposed_rule_additions() first."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Generated resolved file: {output}")
    return output


def resolve_proposed_rule_template(
    template_path: Path | str | None = None,
    output_path: Path | str | None = None,
    *,
    cleansed_dir: Path | str | None = None,
) -> Path:
    """
    Create a resolved working copy from the proposed-rule template.

    This intentionally performs no rule inference or hardcoded substitution.
    It simply copies the template so resolution can remain a separate manual step.
    """
    return generate_resolved_rule_file(
        template_path=template_path,
        output_path=output_path,
        cleansed_dir=cleansed_dir,
    )


if __name__ == "__main__":
    analyze_and_report()
