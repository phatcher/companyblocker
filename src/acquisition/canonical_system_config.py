from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalSystemConfig:
    name_mode: str = "default"
    registered_address_mode: str = "default"
    jurisdiction_mode: str = "default"
    registry_url_mode: str = "default"
    jurisdiction_candidates: tuple[str, ...] = ()
    registry_url_candidates: tuple[str, ...] = ()
    alternative_names_mode: str = "default"
    company_number_mode: str = "default"


DEFAULT_SYSTEM_CONFIG = CanonicalSystemConfig()


SYSTEM_CANONICAL_CONFIG: dict[str, CanonicalSystemConfig] = {
    "fr": CanonicalSystemConfig(name_mode="fr"),
    "gb": CanonicalSystemConfig(registered_address_mode="gb"),
    "gleif": CanonicalSystemConfig(
        registered_address_mode="gleif",
        jurisdiction_mode="candidates",
        jurisdiction_candidates=("Entity_LegalJurisdiction", "legal_jurisdiction"),
        registry_url_mode="gleif_lei",
        company_number_mode="gleif_ra_filtered",
    ),
    "dbpedia": CanonicalSystemConfig(
        jurisdiction_mode="candidates",
        jurisdiction_candidates=(
            "jurisdiction_code",
            "jurisdiction",
            "country_code",
            "country",
            "country_iso2",
            "legal_jurisdiction",
        ),
    ),
    "wikidata": CanonicalSystemConfig(
        name_mode="wikidata",
        registered_address_mode="none",
        jurisdiction_mode="wikidata_qid",
        registry_url_mode="wikidata_entity",
        alternative_names_mode="wikidata",
        company_number_mode="wikidata_jurisdiction_scoped",
    ),
    "offeneregister": CanonicalSystemConfig(
        jurisdiction_mode="candidates",
        jurisdiction_candidates=("jurisdiction_code",),
        company_number_mode="offeneregister_authority_scoped",
    ),
}


_WIKIDATA_COUNTRY_QID_TO_ISO2 = {
    "Q16": "CA",
    "Q17": "JP",
    "Q20": "NO",
    "Q27": "IE",
    "Q28": "HU",
    "Q29": "ES",
    "Q30": "US",
    "Q31": "BE",
    "Q32": "LU",
    "Q33": "FI",
    "Q34": "SE",
    "Q35": "DK",
    "Q36": "PL",
    "Q37": "LT",
    "Q38": "IT",
    "Q39": "CH",
    "Q40": "AT",
    "Q41": "GR",
    "Q43": "TR",
    "Q45": "PT",
    "Q55": "NL",
    "Q96": "MX",
    "Q114": "KE",
    "Q115": "ET",
    "Q142": "FR",
    "Q145": "GB",
    "Q148": "CN",
    "Q155": "BR",
    "Q159": "RU",
    "Q183": "DE",
    "Q184": "BY",
    "Q191": "EE",
    "Q211": "LV",
    "Q212": "UA",
    "Q213": "CZ",
    "Q214": "SK",
    "Q215": "SI",
    "Q217": "MD",
    "Q218": "RO",
    "Q219": "BG",
    "Q224": "HR",
    "Q227": "AZ",
    "Q230": "GE",
    "Q252": "ID",
    "Q258": "ZA",
    "Q334": "SG",
    "Q408": "AU",
    "Q414": "AR",
    "Q664": "NZ",
    "Q668": "IN",
    "Q739": "CO",
    "Q794": "IR",
    "Q843": "PK",
    "Q865": "TW",
    "Q878": "AE",
    "Q884": "KR",
    "Q889": "AF",
    "Q902": "BD",
    "Q916": "AO",
    "Q924": "TZ",
    "Q928": "PH",
    "Q948": "TN",
    "Q953": "ZM",
    "Q954": "ZW",
    "Q958": "SS",
    "Q977": "DJ",
    "Q1009": "CM",
    "Q1011": "CV",
    "Q1028": "MA",
    "Q1033": "NG",
    "Q1044": "SL",
    "Q1045": "SO",
}


# GLEIF's Entity_RegistrationAuthority_RegistrationAuthorityID identifies which
# authority issued Entity_RegistrationAuthority_RegistrationAuthorityEntityID.
# Jurisdictions omitted here are left unrestricted. Codes validated by matching
# GLEIF's reported IDs against the real target register (>98% hit rate for the
# codes listed; other codes seen in the same jurisdictions, e.g. fund/charity/tax
# registries, hit ~0% and are excluded).
#
# 2026-08-24 GB/FR/IE no-match investigation (see
# src/analysis/gleif_no_match_investigation.md): for GB, this whitelist only
# accounts for ~20% of MISSING_COMPANY_NUMBER no-matches (rows where GLEIF
# reported a number under an excluded authority) — the other ~80% never had a
# registration number from GLEIF at all, mostly trusts and unparseable
# "private" entities, a structural gap (no UK company number exists for many
# trusts), not a whitelist tuning problem. FR and IE show the same ~70-80%/
# 20-30% split. Don't expect widening this list to close the no-number gap;
# it only ever addresses the excluded-authority share.
GLEIF_VALID_REGISTRATION_AUTHORITIES: dict[str, tuple[str, ...]] = {
    "GB": ("RA000585", "RA000586", "RA000587"),  # Companies House: E&W, NI, Scotland
    "FR": ("RA000189", "RA000192"),  # INSEE SIRENE, RCS
    "IE": ("RA000402",),  # CRO
    # Handelsregister: one code per local court (Amtsgericht), per GLEIF's RA list
    # v1.8.1 (Nov 2024). Excludes BaFin, Bundesanzeiger's Unternehmensregister
    # (different ID format), and non-company DE registries (foundations, lawyers,
    # tax advisors, auditors, political parties) also filed under jurisdiction DE.
    # ~95.6% of matched rows already equal offeneregister's "<TYPE> <number>"
    # format exactly; the remainder carry a GLEIF-only court-disambiguation
    # suffix from historical Amtsgericht mergers (e.g. "HRB 34542 HB") that is
    # not safe to strip — validated against a real snapshot, stripping it
    # collides genuinely different companies onto the same bare number.
    "DE": (
        "RA000197",
        "RA000199",
        "RA000200",
        "RA000201",
        "RA000202",
        "RA000203",
        "RA000204",
        "RA000205",
        "RA000206",
        "RA000207",
        "RA000208",
        "RA000209",
        "RA000210",
        "RA000213",
        "RA000214",
        "RA000215",
        "RA000216",
        "RA000217",
        "RA000218",
        "RA000219",
        "RA000220",
        "RA000221",
        "RA000222",
        "RA000224",
        "RA000225",
        "RA000226",
        "RA000227",
        "RA000228",
        "RA000229",
        "RA000230",
        "RA000231",
        "RA000232",
        "RA000233",
        "RA000234",
        "RA000235",
        "RA000236",
        "RA000238",
        "RA000239",
        "RA000240",
        "RA000241",
        "RA000242",
        "RA000243",
        "RA000244",
        "RA000245",
        "RA000246",
        "RA000247",
        "RA000249",
        "RA000250",
        "RA000251",
        "RA000252",
        "RA000253",
        "RA000255",
        "RA000257",
        "RA000258",
        "RA000259",
        "RA000260",
        "RA000261",
        "RA000262",
        "RA000263",
        "RA000266",
        "RA000267",
        "RA000268",
        "RA000269",
        "RA000271",
        "RA000272",
        "RA000273",
        "RA000274",
        "RA000275",
        "RA000276",
        "RA000277",
        "RA000278",
        "RA000279",
        "RA000280",
        "RA000281",
        "RA000282",
        "RA000283",
        "RA000284",
        "RA000285",
        "RA000286",
        "RA000287",
        "RA000288",
        "RA000289",
        "RA000290",
        "RA000291",
        "RA000293",
        "RA000295",
        "RA000296",
        "RA000297",
        "RA000298",
        "RA000299",
        "RA000300",
        "RA000301",
        "RA000302",
        "RA000303",
        "RA000304",
        "RA000305",
        "RA000306",
        "RA000307",
        "RA000308",
        "RA000309",
        "RA000310",
        "RA000311",
        "RA000313",
        "RA000314",
        "RA000315",
        "RA000316",
        "RA000317",
        "RA000319",
        "RA000320",
        "RA000321",
        "RA000322",
        "RA000324",
        "RA000325",
        "RA000328",
        "RA000329",
        "RA000331",
        "RA000332",
        "RA000333",
        "RA000334",
        "RA000336",
        "RA000337",
        "RA000338",
        "RA000339",
        "RA000342",
        "RA000343",
        "RA000344",
        "RA000345",
        "RA000346",
        "RA000347",
        "RA000348",
        "RA000349",
        "RA000350",
        "RA000351",
        "RA000352",
        "RA000353",
        "RA000354",
        "RA000355",
        "RA000356",
        "RA000358",
        "RA000362",
        "RA000363",
        "RA000364",
        "RA000365",
        "RA000368",
        "RA000369",
        "RA000370",
        "RA000371",
    ),
}

# Jurisdictions where the registration authority's numeric ID is reported without
# leading zeros; padded to the target register's fixed width before matching.
GLEIF_COMPANY_NUMBER_PAD_WIDTH: dict[str, int] = {
    "GB": 8,
}

# Jurisdictions where the bare register number is not unique on its own (Germany's
# ~150 Amtsgerichte each issue independent HRB/HRA sequences from 1), so the
# registration authority code must be folded into company_number as a scoping
# prefix rather than only used to gate/pad it. GB/FR/IE company numbers are
# already nationally unique and are deliberately excluded from this set.
GLEIF_COMPANY_NUMBER_AUTHORITY_SCOPED_JURISDICTIONS: frozenset[str] = frozenset({"DE"})

# 2026-08-25: a specific historical offeneregister.de scrape batch (Mannheim /
# Baden-Baden, retrieved 2018-06-30T01:2x:xxZ) assigned the same placeholder
# register number across unrelated companies and even across register types
# (HRA/HRB/VR sharing one literal number) -- confirmed as a genuine upstream
# defect, not a local download or parsing issue (raw JSONL inspected directly;
# a fresh reacquisition was byte-identical to the original download). Register
# ids in this numeric band repeat across many unrelated courts (up to 38 for
# the worst offenders) well beyond the ~2x baseline fan-out normal for numbers
# this size, tapering back to baseline by the upper bound. The exact
# boundaries are a heuristic derived from the observed fan-out decay, not a
# precise cutover -- affects ~4% of offeneregister's DE rows (215,039 of
# 5,305,727 checked). Register ids in this range are excluded from
# company_number entirely (treated as unreliable) rather than court-scoped,
# since the number itself isn't a real register id to disambiguate.
OFFENEREGISTER_PLACEHOLDER_REGISTER_NUMBER_RANGE: tuple[int, int] = (200_000, 215_000)


def get_system_canonical_config(system_code: str) -> CanonicalSystemConfig:
    return SYSTEM_CANONICAL_CONFIG.get(system_code, DEFAULT_SYSTEM_CONFIG)
